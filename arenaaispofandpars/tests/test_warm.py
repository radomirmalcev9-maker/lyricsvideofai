"""Tests for the warm daemon (:mod:`arena_api.warm`) and its CLI front-end."""

from __future__ import annotations

import asyncio
import contextlib
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from arena_api import warm
from arena_api.exceptions import AuthenticationError, ProtocolError


class FakeSession:
    """Minimal stand-in for :class:`arena_api.warm.WarmSession`."""

    def __init__(
        self,
        deltas: tuple[str, ...] = ("Hel", "lo"),
        error: BaseException | None = None,
        *,
        delay: float = 0.0,
    ) -> None:
        self.deltas = deltas
        self.error = error
        self.delay = delay
        self.prepared = False
        self.closed = False
        self.prompts: list[tuple[str, str | None]] = []

    async def prepare(self) -> None:
        self.prepared = True

    async def ask(self, prompt: str, model: str | None = None) -> Any:
        self.prompts.append((prompt, model))
        if self.delay:
            await asyncio.sleep(self.delay)
        for piece in self.deltas:
            yield piece
        # The real session records these; the daemon forwards them.
        warm.latency.metric("answer.chunks", len(self.deltas))
        warm.latency.metric("answer.chars", sum(len(p) for p in self.deltas))
        if self.error is not None:
            raise self.error

    def describe(self) -> dict[str, Any]:
        return {"warm": True, "last_error": None}

    async def aclose(self) -> None:
        self.closed = True


class ServerThread:
    """Run a real :class:`warm._WarmServer` in its own event loop."""

    def __init__(self, session: FakeSession, state_file: Path) -> None:
        self.session = session
        self.state_file = state_file
        self.token = "t" * 64
        self.state: warm.DaemonState | None = None
        self.server: warm._WarmServer | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=self._main, daemon=True)

    def _main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
            self._failure = exc
            self._ready.set()

    async def _serve(self) -> None:
        # The real daemon always collects phases so `--warm-profile` works.
        warm.latency.enable_collection()
        self.server = warm._WarmServer(
            self.session,
            state_file=self.state_file,
            token=self.token,
            port=0,
            base_url="https://arena.ai",
            idle_timeout=60.0,
        )
        await self.server.start()
        self.state = warm.DaemonState(
            pid=os.getpid(),
            port=self.server.port,
            token=self.token,
            base_url="https://arena.ai",
            api_version=warm.api_version(),
        )
        self._ready.set()
        await self.server.run()

    def __enter__(self) -> ServerThread:
        self._thread.start()
        assert self._ready.wait(10.0), "server did not start"
        if self._failure is not None:
            raise self._failure
        assert self.server is not None
        return self

    def __exit__(self, *_: object) -> None:
        if self.server is not None:
            self.server.request_stop()
        self._thread.join(10.0)

    @property
    def published(self) -> warm.DaemonState:
        published = warm.DaemonState.read(self.state_file)
        assert published is not None
        return published


@pytest.fixture()
def server(tmp_path: Path) -> Any:
    state_file = tmp_path / "warm-daemon.json"
    with ServerThread(FakeSession(), state_file) as thread:
        yield thread


def test_ask_streams_deltas(server: ServerThread) -> None:
    assert list(warm.ask_stream("hi", state=server.published)) == ["Hel", "lo"]
    assert server.session.prompts == [("hi", None)]


def test_ask_forwards_the_model(server: ServerThread) -> None:
    chunks = list(warm.ask_stream("hi", state=server.published, model="claude-opus-4-6"))
    assert chunks == ["Hel", "lo"]
    assert server.session.prompts == [("hi", "claude-opus-4-6")]


def test_ping_and_status(server: ServerThread) -> None:
    state = server.published
    assert warm.ping(state) is True
    info = warm.describe_daemon(server.state_file)
    assert info is not None
    assert info["warm"] is True
    assert info["port"] == state.port
    assert info["base_url"] == "https://arena.ai"


def test_profiles_op_reports_the_last_prompts(server: ServerThread) -> None:
    assert list(warm.ask_stream("hi", state=server.published)) == ["Hel", "lo"]
    payload = warm.fetch_profiles(server.state_file, limit=5)
    assert payload is not None
    assert payload["type"] == "profiles"
    assert payload["total_answered"] == 1
    assert isinstance(payload["asks"], list)
    entry = payload["asks"][-1]
    assert entry["phases"]  # the timeline is always collected by the daemon
    # ... and the numbers that say whether the answer streamed or stalled.
    metrics = entry.get("metrics") or {}
    assert "answer.chunks" in metrics


def test_profile_table_renders_phases() -> None:
    lines = warm.format_profile_table(
        [
            {"phase": "cli.start", "at_ms": 1.0, "delta_ms": 1.0},
            {"phase": "daemon.first_byte", "at_ms": 250.0, "delta_ms": 249.0},
        ],
        title="  ",
    )
    assert lines[0].strip().startswith("phase")
    assert any("daemon.first_byte" in line and "250.0" in line for line in lines)
    assert lines[-1].strip().startswith("total")


def test_profile_main_says_when_no_daemon_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = warm.profile_main(["--warm-profile", "--warm-state", str(tmp_path / "none.json")])
    assert code == 1
    assert "not running" in capsys.readouterr().out


def test_bad_token_is_rejected(server: ServerThread) -> None:
    state = server.published
    forged = warm.DaemonState(
        pid=state.pid, port=state.port, token="x" * 64, base_url=state.base_url
    )
    assert warm.ping(forged) is False
    with pytest.raises(warm.WarmDaemonUnavailable):
        list(warm.ask_stream("hi", state=forged))


def test_unknown_op_is_an_error(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    with ServerThread(FakeSession(), state_file) as thread:
        state = thread.published
        import socket

        sock = socket.create_connection((state.host, state.port), timeout=5.0)
        with sock, sock.makefile("rb") as reader:
            warm._send(
                sock,
                {
                    "op": "hello",
                    "token": state.token,
                    "protocol": warm.PROTOCOL_VERSION,
                    "version": warm.api_version(),
                },
            )
            assert warm._read_message(reader, 5.0)["type"] == "hello"
            warm._send(sock, {"op": "nope"})
            reply = warm._read_message(reader, 5.0)
            assert reply["type"] == "error"
            assert reply["kind"] == "protocol"


def test_error_frames_map_to_library_exceptions(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    session = FakeSession(error=AuthenticationError("session rejected"))
    with ServerThread(session, state_file) as thread:
        with pytest.raises(AuthenticationError):
            list(warm.ask_stream("hi", state=thread.published))


def test_protocol_errors_are_preserved(tmp_path: Path) -> None:
    session = FakeSession(error=ProtocolError("bad stream"))
    with ServerThread(session, tmp_path / "state.json") as thread:
        with pytest.raises(ProtocolError):
            list(warm.ask_stream("hi", state=thread.published))


def test_empty_prompt_is_refused_without_killing_the_daemon(server: ServerThread) -> None:
    state = server.published
    with pytest.raises(warm.WarmDaemonUnavailable):
        list(warm.ask_stream("", state=state))
    # The connection-level error must not take the daemon down.
    assert list(warm.ask_stream("still here", state=server.published)) == ["Hel", "lo"]


def test_closing_a_stream_early_keeps_the_daemon_alive(server: ServerThread) -> None:
    stream = warm.ask_stream("hi", state=server.published)
    assert next(stream) == "Hel"
    stream.close()
    assert list(warm.ask_stream("again", state=server.published)) == ["Hel", "lo"]


def test_state_file_is_private(server: ServerThread) -> None:
    if os.name == "nt":  # pragma: no cover - POSIX permissions
        pytest.skip("POSIX permissions")
    mode = stat.S_IMODE(server.state_file.stat().st_mode)
    assert mode & 0o077 == 0


def test_stop_daemon_removes_the_state_file(server: ServerThread) -> None:
    assert warm.stop_daemon(server.state_file) is True
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and server.state_file.exists():
        time.sleep(0.02)
    assert not server.state_file.exists()


def test_ensure_daemon_reuses_a_live_daemon(
    server: ServerThread, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _explode(**_: Any) -> Any:  # pragma: no cover - must not run
        raise AssertionError("a live daemon must be reused, not restarted")

    monkeypatch.setattr(warm, "spawn_daemon", _explode)
    state = warm.ensure_daemon(
        payload={"argv": []},
        base_url="https://arena.ai",
        state_file=server.state_file,
        start_timeout=5.0,
    )
    assert state.port == server.published.port


def test_ensure_daemon_replaces_an_incompatible_daemon(
    server: ServerThread, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawns: list[str] = []

    def _spawn(**kwargs: Any) -> Any:
        spawns.append(str(kwargs["state_file"]))
        raise warm.WarmDaemonUnavailable("cannot start here")

    monkeypatch.setattr(warm, "spawn_daemon", _spawn)
    with pytest.raises(warm.WarmDaemonUnavailable):
        warm.ensure_daemon(
            payload={"argv": []},
            base_url="https://lmarena.ai",
            state_file=server.state_file,
            start_timeout=0.4,
        )
    assert spawns, "a daemon for another base URL must be replaced"


def test_spawned_daemon_is_reachable(tmp_path: Path) -> None:
    """The real `python -m arena_api.warm` process answers on its state file."""
    state_file = tmp_path / "spawned.json"
    state: warm.DaemonState | None = None
    try:
        state = warm.ensure_daemon(
            payload={
                "argv": [
                    "--no-browser-profiles",
                    "--non-interactive",
                    "--no-verify",
                ]
            },
            base_url="https://arena.ai",
            state_file=state_file,
            start_timeout=60.0,
        )
        assert warm.ping(state)
        info = warm.describe_daemon(state_file)
        assert info is not None
        assert info["protocol"] == warm.PROTOCOL_VERSION
        assert info["base_url"] == "https://arena.ai"
        # No session in this sandbox: the first prompt is refused and the
        # daemon decides to exit instead of lingering for the idle timeout.
        with pytest.raises(AuthenticationError):
            list(warm.ask_stream("hi", state=state, timeout=30.0))
    finally:
        if state is not None:
            warm.stop_daemon(state_file, timeout=5.0)
            if os.name != "nt" and warm._pid_alive(state.pid):
                with contextlib.suppress(OSError):  # last resort
                    os.kill(state.pid, 9)


def test_idle_daemon_stops_itself(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    thread = ServerThread(FakeSession(), state_file)
    thread.server = None
    ready = threading.Event()

    def _main() -> None:
        async def _serve() -> None:
            server = warm._WarmServer(
                FakeSession(),
                state_file=state_file,
                token="z" * 64,
                port=0,
                idle_timeout=0.6,
            )
            await server.start()
            ready.set()
            await server.run()

        asyncio.run(_serve())

    worker = threading.Thread(target=_main, daemon=True)
    worker.start()
    assert ready.wait(10.0)
    worker.join(10.0)
    assert not worker.is_alive()
    assert not state_file.exists()


def test_daemon_profile_is_merged_into_the_caller_timeline(
    server: ServerThread, monkeypatch: pytest.MonkeyPatch
) -> None:
    from arena_api import latency

    monkeypatch.setattr(latency, "_MODE", "1")
    latency.reset()
    try:
        assert list(warm.ask_stream("hi", state=server.published)) == ["Hel", "lo"]
    finally:
        monkeypatch.undo()
    phases = [row["phase"] for row in latency._rows()]
    assert "daemon.warm.stream" in phases
    assert "daemon.warm.stream.end" in phases


# --------------------------------------------------------------------------- #
# CLI front-end
# --------------------------------------------------------------------------- #


def test_scan_argv_splits_the_prompt_from_forwarded_flags() -> None:
    parsed = warm.scan_argv(
        ["--prompt", "hello", "--model", "m", "--proxy", "http://p:1", "--no-verify"]
    )
    assert parsed.prompt == "hello"
    assert parsed.model == "m"
    assert parsed.rest == ["--proxy", "http://p:1", "--no-verify"]
    assert parsed.usable is True
    assert parsed.base_url == "https://arena.ai"


def test_scan_argv_reads_equals_forms_and_warm_flags() -> None:
    parsed = warm.scan_argv(
        [
            "--prompt=hi",
            "--model=x",
            "--base-url=https://lmarena.ai",
            "--warm-idle=120",
            "--warm-state=/tmp/x.json",
            "--no-warm",
        ]
    )
    assert parsed.prompt == "hi"
    assert parsed.model == "x"
    assert parsed.base_url == "https://lmarena.ai"
    assert parsed.idle_timeout == 120.0
    assert parsed.state_file == "/tmp/x.json"
    assert parsed.force is False
    assert parsed.rest == ["--base-url=https://lmarena.ai"]


def test_scan_argv_refuses_a_supplied_verification_token() -> None:
    parsed = warm.scan_argv(["--prompt", "hi", "--verification-token", "abc"])
    assert parsed.usable is False


def test_scan_argv_accepts_auto_verification_token() -> None:
    parsed = warm.scan_argv(["--prompt", "hi", "--verification-token", "auto"])
    assert parsed.usable is True
    assert parsed.rest == ["--verification-token", "auto"]


def test_warm_enabled_reads_the_environment() -> None:
    assert warm.warm_enabled({}) is True
    assert warm.warm_enabled({"ARENA_WARM": "0"}) is False
    assert warm.warm_enabled({"ARENA_WARM": "off"}) is False
    assert warm.warm_enabled({"ARENA_WARM": "1"}) is True


def test_fast_prompt_main_falls_back_when_the_daemon_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unavailable(**kwargs: Any) -> Any:
        raise warm.WarmDaemonUnavailable("nope")

    monkeypatch.delenv("ARENA_WARM_SKIP", raising=False)
    monkeypatch.setattr(warm, "ensure_daemon", _unavailable)
    assert warm.fast_prompt_main(["--prompt", "hi"]) is None
    # The heavy path in the same process must not try the daemon again.
    assert warm.skip_for_process() is True


def test_fast_prompt_main_prints_the_stream(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = warm.DaemonState(pid=1, port=2, token="t")
    monkeypatch.setattr(warm, "ensure_daemon", lambda **_: state)
    monkeypatch.setattr(
        warm, "ask_stream", lambda *a, **k: iter(["Hel", "lo", ""])
    )
    assert warm.fast_prompt_main(["--prompt", "hi", "--model", "m"]) == 0
    out = capsys.readouterr().out
    assert "model             : m" in out
    assert out.strip().endswith("answer            : Hello")


def test_fast_prompt_main_reports_library_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = warm.DaemonState(pid=1, port=2, token="t")

    def _boom(*_: Any, **__: Any) -> Any:
        raise AuthenticationError("session rejected")

    monkeypatch.setattr(warm, "ensure_daemon", lambda **_: state)
    monkeypatch.setattr(warm, "ask_stream", _boom)
    assert warm.fast_prompt_main(["--prompt", "hi"]) == 1
    assert "session rejected" in capsys.readouterr().err


def test_start_main_reports_an_unavailable_daemon(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _unavailable(**_: Any) -> Any:
        raise warm.WarmDaemonUnavailable("no browser")

    monkeypatch.setattr(warm, "ensure_daemon", _unavailable)
    assert warm.start_main(["--warm-start"]) == 1
    assert "could not start" in capsys.readouterr().out


def test_scan_argv_drops_daemon_commands() -> None:
    parsed = warm.scan_argv(["--warm-start", "--warm-status", "--warm-stop"])
    assert parsed.rest == []
    assert parsed.prompt is None


def test_fast_prompt_main_honours_no_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[Any] = []
    monkeypatch.setattr(warm, "DaemonState", _FakeState)
    monkeypatch.setattr(warm, "stop_daemon", lambda *a, **k: stopped.append(a) or True)
    assert warm.fast_prompt_main(["--prompt", "hi", "--no-warm"]) is None
    # The daemon keeps the shared Chromium profile open, so --no-warm retires it.
    assert stopped


class _FakeState:
    @staticmethod
    def read(_path: Any) -> Any:
        return object()


def test_fast_prompt_main_defers_to_a_supplied_env_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARENA_VERIFICATION_TOKEN", "some-token")
    assert warm.fast_prompt_main(["--prompt", "hi"]) is None


def test_fast_prompt_main_honours_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARENA_WARM", "0")
    assert warm.fast_prompt_main(["--prompt", "hi"]) is None


def test_fast_prompt_main_can_be_forced(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARENA_WARM", "0")
    state = warm.DaemonState(pid=1, port=2, token="t")
    monkeypatch.setattr(warm, "ensure_daemon", lambda **_: state)
    monkeypatch.setattr(warm, "ask_stream", lambda *a, **k: iter(["ok"]))
    assert warm.fast_prompt_main(["--prompt", "hi", "--warm"]) == 0
    assert "ok" in capsys.readouterr().out


def test_cli_routes_one_shot_prompts_to_the_warm_path() -> None:
    from arena_api import _cli

    assert _cli._wants_warm_prompt(["--prompt", "hi"]) is True
    assert _cli._wants_warm_prompt(["--prompt=hi", "--model", "m"]) is True
    assert _cli._wants_warm_prompt(["--prompt", "hi", "--models"]) is False
    assert _cli._wants_warm_prompt(["--login", "--prompt", "hi"]) is False
    assert _cli._wants_warm_prompt(["--status"]) is False
    assert _cli._wants_warm_prompt([]) is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--warm-stop"],
        ["--warm-status"],
    ],
)
def test_cli_handles_daemon_commands_without_a_session(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from arena_api import _cli

    calls: list[str] = []
    monkeypatch.setattr(warm, "stop_main", lambda _argv: calls.append("stop") or 0)
    monkeypatch.setattr(warm, "status_main", lambda _argv: calls.append("status") or 0)
    assert _cli.main(argv) == 0
    assert calls == ["stop" if argv[0] == "--warm-stop" else "status"]


# --------------------------------------------------------------------------- #
# The daemon answers with one session; a different one must not reuse it
# --------------------------------------------------------------------------- #


def test_session_signature_reads_the_flags_that_pick_a_session() -> None:
    argv = [
        "--prompt",
        "hi",
        "--session",
        "firefox",
        "--profile=work",
        "--warm-idle",
        "600",
    ]
    signature = warm.session_signature(argv)
    assert "--session=firefox" in signature
    assert "--profile=work" in signature
    # A prompt or an idle timeout does not change which session is read.
    assert "hi" not in signature
    assert "600" not in signature
    assert warm.session_signature([]) == ""


def test_daemon_state_round_trips_the_session(tmp_path: Path) -> None:
    path = tmp_path / "warm-daemon.json"
    state = warm.DaemonState(
        pid=1, port=2, token="t", session="--session=firefox"
    )
    state.write(path)
    restored = warm.DaemonState.read(path)
    assert restored is not None
    assert restored.session == "--session=firefox"


def test_a_daemon_with_another_session_is_not_reused() -> None:
    state = warm.DaemonState(
        pid=1,
        port=2,
        token="t",
        api_version=warm.api_version(),
        base_url="https://arena.ai",
        session="--session=firefox",
    )
    assert warm._compatible(state, base_url="https://arena.ai", session="--session=firefox")
    # Same URL, another *explicit* source: the caller asked for a different
    # profile, so the running daemon (which holds a browser and cookies) cannot
    # answer for it.
    assert not warm._compatible(
        state, base_url="https://arena.ai", session="--session=firefox:work"
    )
    # No source named means "the usual discovery order", and a daemon that is
    # already warm with any source is exactly what that caller wants: restarting
    # it here cost a full browser launch and looked like a slow request.
    assert warm._compatible(state, base_url="https://arena.ai", session="")


def test_old_state_files_without_a_session_still_work() -> None:
    state = warm.DaemonState.from_json({"pid": 5, "port": 6, "token": "t"})
    assert state is not None
    assert state.session == ""


def test_an_unforced_caller_reuses_a_forced_daemon() -> None:
    state = warm.DaemonState(
        pid=1,
        port=2,
        token="t",
        api_version=warm.api_version(),
        base_url="https://arena.ai",
        session="--session=firefox",
    )
    assert warm._compatible(state, base_url="https://arena.ai", session="")


def test_warm_session_asks_for_an_idle_moment_before_archiving() -> None:
    from dataclasses import replace

    from arena_api.auto import AutoOptions

    session = warm.WarmSession(AutoOptions())
    guard = session.options.archive_if_idle
    assert callable(guard)
    assert guard() is True
    session._waiting = 2  # a second prompt is already queued
    assert guard() is False
    session._waiting = 0
    assert guard() is True
    # The rest of the options are untouched: this is the same session config.
    assert replace(session.options, archive_if_idle=None) == AutoOptions()
