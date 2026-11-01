"""A persistent warm browser behind one-shot Direct Chat CLI calls.

Nearly all of the wall-clock time between ``arena-api --prompt ...`` and the
first streamed token is spent on work that is *identical for every prompt*:
reading the browser session, launching a browser, loading the Direct Chat page,
warming the composer and minting a reCAPTCHA v3 assertion.  Repeating that from
scratch for each invocation costs seconds; the answer itself usually starts
much sooner than that.

This module keeps that state alive in a small local daemon:

* :func:`ensure_daemon` starts ``python -m arena_api.warm`` on first use (a
  detached process with its own warm browser) and returns its address;
* :func:`ask_stream` sends one prompt over a loopback socket and yields the
  answer deltas exactly as the in-process client would;
* :func:`stop_daemon` / :func:`describe_daemon` manage and inspect it;
* ``python -m arena_api.warm`` is the daemon itself.

Everything is bound to ``127.0.0.1`` and gated by a random token written to a
``0600`` state file, so nothing is exposed to the network.  The daemon exits by
itself once it has been idle (default 15 minutes).  If it cannot be started —
no browser, a locked profile, a stale state file — the caller falls back to the
existing in-process path, which is exactly as fast (or as slow) as before.

The module imports only the standard library at import time; the heavy client
machinery is imported lazily inside the daemon, and the one-shot fast path
(:func:`fast_prompt_main`) never imports :mod:`arena_api.auto` at all.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import hmac
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import latency
from .constants import DEFAULT_BASE_URL
from .exceptions import AuthenticationError

#: Bumped whenever the wire protocol changes; a daemon of another version is
#: replaced instead of being talked to.
PROTOCOL_VERSION = 2

DEFAULT_IDLE_TIMEOUT = 900.0
DEFAULT_START_TIMEOUT = 90.0
DEFAULT_ASK_TIMEOUT = 900.0
DEFAULT_STATE_NAME = "warm-daemon.json"

_WARM_ENV = "ARENA_WARM"
_STATE_ENV = "ARENA_WARM_STATE"
_IDLE_ENV = "ARENA_WARM_IDLE"
_START_ENV = "ARENA_WARM_TIMEOUT"
_DEBUG_ENV = "ARENA_WARM_DEBUG"
_SKIP_ENV = "ARENA_WARM_SKIP"

_FALSE = {"0", "false", "no", "off", "disable", "disabled", "none", ""}


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() not in _FALSE


def warm_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the warm daemon may be used at all (``ARENA_WARM=0`` opts out)."""
    env = os.environ if environ is None else environ
    return _truthy(env.get(_WARM_ENV, "1"))


def skip_for_process() -> bool:
    """``True`` once this process has already tried and lost the daemon.

    The light CLI front-end falls back to :func:`arena_api.auto.main` in the
    very same process; without this marker the heavy path would start (and
    fail) a second daemon for the same prompt.
    """
    return _truthy(os.environ.get(_SKIP_ENV, "0"))


def _mark_skipped() -> None:
    os.environ[_SKIP_ENV] = "1"


def supplied_verification_token(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``ARENA_VERIFICATION_TOKEN`` holds a token the daemon cannot mint.

    The daemon always mints its own assertion; a fixed token supplied by the
    caller must keep taking the ordinary path where it is actually used.
    """
    env = os.environ if environ is None else environ
    value = str(env.get("ARENA_VERIFICATION_TOKEN", "") or "").strip()
    return bool(value) and value.lower() != "auto"


def _debug(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return _truthy(env.get(_DEBUG_ENV, "0"))


def log(message: str) -> None:
    """Write one ``[arena-api]`` line to stderr (daemon log when detached)."""
    try:
        sys.stderr.write(f"[arena-api] {message}\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - logging must never raise
        pass


def _abs_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def state_path(explicit: str | Path | None = None) -> Path:
    """Where the daemon publishes how to reach it (``state/warm-daemon.json``)."""
    if explicit:
        return _abs_path(explicit)
    env = os.environ.get(_STATE_ENV)
    if env:
        return _abs_path(env)
    return _abs_path(Path("state") / DEFAULT_STATE_NAME)


def _idle_timeout(value: float | None = None) -> float:
    if value is not None:
        return float(value)
    env = os.environ.get(_IDLE_ENV)
    if env:
        try:
            parsed = float(env)
        except ValueError:
            parsed = DEFAULT_IDLE_TIMEOUT
        if parsed > 0:
            return parsed
    return DEFAULT_IDLE_TIMEOUT


def _start_timeout(value: float | None = None) -> float:
    if value is not None:
        return float(value)
    env = os.environ.get(_START_ENV)
    if env:
        try:
            parsed = float(env)
        except ValueError:
            parsed = DEFAULT_START_TIMEOUT
        if parsed > 0:
            return parsed
    return DEFAULT_START_TIMEOUT


_API_VERSION: str | None = None


def api_version() -> str:
    """Installed ``arena-api`` version, used to retire an older daemon."""
    global _API_VERSION
    if _API_VERSION is None:
        try:
            from importlib.metadata import version

            _API_VERSION = version("arena-api")
        except Exception:  # noqa: BLE001 - running from a plain checkout
            _API_VERSION = "unknown"
    return _API_VERSION


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class WarmDaemonUnavailable(RuntimeError):
    """The daemon could not be reached, started or stayed usable.

    Callers treat this as "use the ordinary in-process path" rather than as a
    failure of the request itself.
    """


# --------------------------------------------------------------------------- #
# State file
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DaemonState:
    """How to reach one running daemon (never logged, never world-readable)."""

    pid: int
    port: int
    token: str
    host: str = "127.0.0.1"
    started: float = 0.0
    protocol: int = PROTOCOL_VERSION
    api_version: str = ""
    base_url: str = DEFAULT_BASE_URL
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT
    python: str = ""
    #: Which session the daemon answers with (see :func:`session_signature`).
    #: A daemon reading Firefox must not be reused by a caller that asked for a
    #: different profile or for the automatic ladder.
    session: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "port": self.port,
            "token": self.token,
            "host": self.host,
            "started": self.started,
            "protocol": self.protocol,
            "api_version": self.api_version,
            "base_url": self.base_url,
            "idle_timeout": self.idle_timeout,
            "python": self.python,
            "session": self.session,
        }

    @classmethod
    def from_json(cls, payload: Any) -> DaemonState | None:
        if not isinstance(payload, Mapping):
            return None
        try:
            pid = int(payload["pid"])
            port = int(payload["port"])
            token = str(payload["token"])
        except (KeyError, TypeError, ValueError):
            return None
        if pid <= 0 or not 0 < port < 65536 or not token:
            return None
        try:
            protocol = int(payload.get("protocol", PROTOCOL_VERSION))
        except (TypeError, ValueError):
            return None
        try:
            started = float(payload.get("started") or 0.0)
        except (TypeError, ValueError):
            started = 0.0
        try:
            idle_timeout = float(payload.get("idle_timeout") or DEFAULT_IDLE_TIMEOUT)
        except (TypeError, ValueError):
            idle_timeout = DEFAULT_IDLE_TIMEOUT
        return cls(
            pid=pid,
            port=port,
            token=token,
            host=str(payload.get("host") or "127.0.0.1"),
            started=started,
            protocol=protocol,
            api_version=str(payload.get("api_version") or ""),
            base_url=str(payload.get("base_url") or DEFAULT_BASE_URL),
            idle_timeout=idle_timeout,
            python=str(payload.get("python") or ""),
            session=str(payload.get("session") or ""),
        )

    def write(self, path: Path) -> None:
        _write_private_json(path, self.to_json())

    @classmethod
    def read(cls, path: Path) -> DaemonState | None:
        try:
            raw = path.read_text(encoding="utf-8")
            return cls.from_json(json.loads(raw))
        except (OSError, ValueError, TypeError):
            return None


def _write_private_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` so only the current user can read it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


# --------------------------------------------------------------------------- #
# Wire protocol (newline-delimited JSON over loopback TCP)
# --------------------------------------------------------------------------- #


def _send(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    sock.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")


def _read_message(reader: Any, timeout: float) -> dict[str, Any]:
    """Read one JSON object; raises :class:`WarmDaemonUnavailable` on EOF."""
    try:
        line = reader.readline()
    except (socket.timeout, TimeoutError) as exc:
        raise WarmDaemonUnavailable("warm daemon did not answer in time") from exc
    except OSError as exc:
        raise WarmDaemonUnavailable(f"warm daemon connection failed: {exc}") from exc
    if not line:
        raise WarmDaemonUnavailable("warm daemon closed the connection")
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise WarmDaemonUnavailable("warm daemon sent a malformed frame") from exc
    if not isinstance(message, dict):
        raise WarmDaemonUnavailable("warm daemon sent a malformed frame")
    return message


def _open(
    state: DaemonState, *, timeout: float, handshake_timeout: float | None = None
) -> tuple[socket.socket, Any]:
    handshake = handshake_timeout if handshake_timeout is not None else timeout
    try:
        sock = socket.create_connection((state.host, state.port), timeout=handshake)
    except OSError as exc:
        raise WarmDaemonUnavailable(f"cannot reach {state.host}:{state.port}: {exc}") from exc
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    reader = sock.makefile("rb")
    try:
        _send(
            sock,
            {
                "op": "hello",
                "token": state.token,
                "protocol": PROTOCOL_VERSION,
                "version": api_version(),
            },
        )
        reply = _read_message(reader, handshake)
        if reply.get("type") != "hello":
            message = str(reply.get("message") or reply.get("type") or "handshake failed")
            raise WarmDaemonUnavailable(message)
        sock.settimeout(timeout)
        return sock, reader
    except BaseException:
        with contextlib.suppress(Exception):
            reader.close()
        with contextlib.suppress(Exception):
            sock.close()
        raise


def ping(state: DaemonState, *, timeout: float = 1.0) -> bool:
    """``True`` when the daemon answers on its published port."""
    try:
        sock, reader = _open(state, timeout=timeout, handshake_timeout=timeout)
    except WarmDaemonUnavailable:
        return False
    try:
        _send(sock, {"op": "ping"})
        return _read_message(reader, timeout).get("type") == "pong"
    except (WarmDaemonUnavailable, OSError):
        return False
    finally:
        with contextlib.suppress(Exception):
            reader.close()
        with contextlib.suppress(Exception):
            sock.close()


def describe_daemon(
    state_path_value: str | Path | None = None, *, timeout: float = 2.0
) -> dict[str, Any] | None:
    """Return the daemon's own status payload, or ``None`` when it is not up."""
    path = state_path(state_path_value)
    state = DaemonState.read(path)
    if state is None:
        return None
    try:
        sock, reader = _open(state, timeout=timeout, handshake_timeout=timeout)
    except WarmDaemonUnavailable:
        return None
    try:
        _send(sock, {"op": "status"})
        reply = _read_message(reader, timeout)
        if reply.get("type") != "status":
            return None
        reply.setdefault("pid", state.pid)
        reply.setdefault("state_file", str(path))
        return reply
    except (WarmDaemonUnavailable, OSError):
        return None
    finally:
        with contextlib.suppress(Exception):
            reader.close()
        with contextlib.suppress(Exception):
            sock.close()


def _ask_payload(
    prompt: str,
    model: str | None = None,
    config: Mapping[str, Any] | None = None,
    archive: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"op": "ask", "prompt": prompt, "model": model}
    if config:
        payload["config"] = dict(config)
    if archive is not None:
        payload["archive"] = bool(archive)
    return payload


def ask_stream(
    prompt: str,
    *,
    state: DaemonState,
    model: str | None = None,
    config: Mapping[str, Any] | None = None,
    archive: bool | None = None,
    timeout: float = DEFAULT_ASK_TIMEOUT,
) -> Generator[str, None, None]:
    """Yield answer deltas for ``prompt`` from the warm daemon.

    The connection is opened eagerly, so a dead daemon surfaces as
    :class:`WarmDaemonUnavailable` on the first :func:`next` call — before the
    caller has printed anything and can still fall back cleanly.
    """
    sock, reader = _open(state, timeout=timeout, handshake_timeout=min(timeout, 10.0))
    finished = False
    anchor = time.perf_counter()
    try:
        _send(sock, _ask_payload(prompt, model, config, archive=archive))
        while True:
            message = _read_message(reader, timeout)
            kind = message.get("type")
            if kind == "profile":
                # The daemon's own phases (browser start, page load, submit,
                # first byte) belong on the same timeline as ours.
                latency.extend(
                    message.get("phases"),
                    anchor=anchor,
                    prefix="daemon.startup." if message.get("kind") == "startup" else "daemon.",
                )
                latency.set_metrics(message.get("metrics"))
                continue
            if kind == "delta":
                text = message.get("text")
                if isinstance(text, str) and text:
                    yield text
            elif kind == "done":
                finished = True
                return
            elif kind == "start":
                continue
            elif kind == "error":
                raise error_from_frame(message)
            else:
                raise WarmDaemonUnavailable(
                    f"unexpected frame {kind!r} from the warm daemon"
                )
    finally:
        if not finished:
            # Let the daemon stop generating as soon as the caller walks away.
            with contextlib.suppress(Exception):
                _send(sock, {"op": "cancel"})
        with contextlib.suppress(Exception):
            reader.close()
        with contextlib.suppress(Exception):
            sock.close()


class WarmConnection:
    """Persistent, reusable loopback TCP connection to a warm daemon.

    ``_WarmServer._dispatch`` already serves multiple ``ask`` frames over a
    single authenticated socket.  Reusing that socket across requests avoids
    per-turn ``warm-daemon.json`` disk reads, TCP handshakes, ``hello`` auth
    round-trips, and pre-flight ``ping`` connections.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: DaemonState | None = None
        self._sock: socket.socket | None = None
        self._reader: Any | None = None

    def _drop(self) -> None:
        sock, reader = self._sock, self._reader
        self._sock = None
        self._reader = None
        self._state = None
        if reader is not None:
            with contextlib.suppress(Exception):
                reader.close()
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _connect(
        self, state: DaemonState, *, timeout: float
    ) -> tuple[socket.socket, Any]:
        if (
            self._sock is not None
            and self._reader is not None
            and self._state is not None
            and self._state.host == state.host
            and self._state.port == state.port
            and self._state.token == state.token
        ):
            self._sock.settimeout(timeout)
            return self._sock, self._reader
        self._drop()
        sock, reader = _open(
            state, timeout=timeout, handshake_timeout=min(timeout, 10.0)
        )
        self._sock = sock
        self._reader = reader
        self._state = state
        return sock, reader

    def ask_stream(
        self,
        prompt: str,
        *,
        ensure_state: Any,
        model: str | None = None,
        config: Mapping[str, Any] | None = None,
        archive: bool | None = None,
        timeout: float = DEFAULT_ASK_TIMEOUT,
    ) -> Generator[str, None, None]:
        """Stream one prompt, reusing the cached socket or reconnecting cleanly."""
        if not self._lock.acquire(blocking=False):
            # Concurrent caller while another stream owns the persistent socket:
            # open an independent stream rather than blocking or interleaving.
            fresh_state = ensure_state(force_refresh=False)
            yield from ask_stream(
                prompt,
                state=fresh_state,
                model=model,
                config=config,
                archive=archive,
                timeout=timeout,
            )
            return

        finished = False
        try:
            state = self._state or ensure_state(force_refresh=False)
            for attempt in range(2):
                try:
                    sock, reader = self._connect(state, timeout=timeout)
                    _send(sock, _ask_payload(prompt, model, config, archive=archive))
                    first_msg = _read_message(reader, timeout)
                    break
                except (WarmDaemonUnavailable, OSError):
                    self._drop()
                    if attempt == 1:
                        raise
                    state = ensure_state(force_refresh=True)

            anchor = time.perf_counter()
            pending: dict[str, Any] | None = first_msg
            while True:
                message = (
                    pending
                    if pending is not None
                    else _read_message(reader, timeout)
                )
                pending = None
                kind = message.get("type")
                if kind == "profile":
                    latency.extend(
                        message.get("phases"),
                        anchor=anchor,
                        prefix=(
                            "daemon.startup."
                            if message.get("kind") == "startup"
                            else "daemon."
                        ),
                    )
                    latency.set_metrics(message.get("metrics"))
                    continue
                if kind == "delta":
                    text = message.get("text")
                    if isinstance(text, str) and text:
                        yield text
                elif kind == "done":
                    finished = True
                    return
                elif kind == "start":
                    continue
                elif kind == "error":
                    # Server stayed synchronized after writing a single error frame.
                    finished = True
                    raise error_from_frame(message)
                else:
                    self._drop()
                    raise WarmDaemonUnavailable(
                        f"unexpected frame {kind!r} from the warm daemon"
                    )
        finally:
            if not finished:
                # Abandoned mid-stream: close the socket so unread delta frames
                # never bleed into the next request.
                if self._sock is not None:
                    with contextlib.suppress(Exception):
                        _send(self._sock, {"op": "cancel"})
                self._drop()
            self._lock.release()


def error_from_frame(message: Mapping[str, Any]) -> BaseException:
    """Map a daemon ``error`` frame onto the library's exception type."""
    detail = str(message.get("message") or "warm daemon error")
    kind = str(message.get("kind") or "")
    try:
        from .exceptions import (
            ArenaError,
            AuthenticationError,
            ChallengeRequiredError,
            ModelUnavailableError,
            ProtocolError,
            QueueTimeoutError,
            RateLimitError,
        )
    except Exception:  # noqa: BLE001 - keep the daemon self-contained
        return WarmDaemonUnavailable(detail)
    mapping: dict[str, type[Exception]] = {
        "auth": AuthenticationError,
        "challenge": ChallengeRequiredError,
        "protocol": ProtocolError,
        "rate_limit": RateLimitError,
        "timeout": QueueTimeoutError,
        "model": ModelUnavailableError,
        "usage": WarmDaemonUnavailable,
    }
    return mapping.get(kind, ArenaError)(detail)


def error_kind(exc: BaseException) -> str:
    """Classify an exception for the wire (``error_from_frame`` reverses it)."""
    try:
        from .exceptions import (
            AuthenticationError,
            ChallengeRequiredError,
            ModelUnavailableError,
            ProtocolError,
            QueueTimeoutError,
            RateLimitError,
        )
    except Exception:  # noqa: BLE001
        return "other"
    if isinstance(exc, AuthenticationError):
        return "auth"
    if isinstance(exc, ChallengeRequiredError):
        return "challenge"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, QueueTimeoutError):
        return "timeout"
    if isinstance(exc, ModelUnavailableError):
        return "model"
    if isinstance(exc, ProtocolError):
        return "protocol"
    if isinstance(exc, UsageError):
        return "usage"
    return "other"


class UsageError(RuntimeError):
    """The daemon could not make sense of the arguments it was given."""


# --------------------------------------------------------------------------- #
# Starting, stopping and reusing the daemon
# --------------------------------------------------------------------------- #


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _options_path(path: Path) -> Path:
    return path.with_name(path.name + ".options.json")


def _log_path(path: Path) -> Path:
    return path.with_name(path.name + ".log")


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check; ``False`` means "do not trust the state"."""
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) terminates the process on Windows -- never call it.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextlib.contextmanager
def _spawn_lock(path: Path, *, stale_after: float) -> Iterator[bool]:
    """Serialise first starts so two parallel CLI calls launch one browser."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            age = 0.0
        if age > stale_after:
            with contextlib.suppress(OSError):
                path.unlink()
        yield False
        return
    try:
        with contextlib.suppress(OSError):
            os.write(fd, str(os.getpid()).encode("ascii"))
        yield True
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            path.unlink()


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH")
    parts = [root]
    if existing:
        parts.extend(part for part in existing.split(os.pathsep) if part)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


#: Children we started in this process.  Holding them keeps ``Popen.__del__``
#: from firing at an arbitrary moment (it warns about a running child) and lets
#: us reap an exited daemon instead of leaving a zombie behind.
_SPAWNED: list[subprocess.Popen[bytes]] = []


def _reap_spawned() -> None:
    for child in list(_SPAWNED):
        if child.poll() is not None:
            _SPAWNED.remove(child)


atexit.register(_reap_spawned)


def spawn_daemon(
    *,
    state_file: Path,
    payload: Mapping[str, Any],
    idle_timeout: float,
) -> subprocess.Popen[bytes]:
    """Launch ``python -m arena_api.warm`` detached from this process."""
    options_file = _options_path(state_file)
    _write_private_json(options_file, payload)
    log_file = _log_path(state_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "arena_api.warm",
        "--state",
        str(state_file),
        "--options-file",
        str(options_file),
        "--idle-timeout",
        repr(float(idle_timeout)),
    ]
    handle = open(log_file, "ab", buffering=0)  # noqa: SIM115 - handed to the child
    kwargs: dict[str, Any] = {
        "cwd": os.getcwd(),
        "env": _child_env(),
        "stdin": subprocess.DEVNULL,
        "stdout": handle,
        "stderr": handle,
        "close_fds": True,
    }
    if os.name == "nt":
        creationflags = 0
        for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP", "CREATE_NO_WINDOW"):
            creationflags |= int(getattr(subprocess, name, 0))
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    _reap_spawned()
    try:
        child = subprocess.Popen(command, **kwargs)  # noqa: S603 - fixed argv
        _SPAWNED.append(child)
        return child
    finally:
        handle.close()


#: Command-line flags that decide *which* session the daemon uses.
_SESSION_FLAGS = ("--session", "--profile", "--profile-root", "--cookies", "--storage")


def _argv_value(argv: Sequence[str], flag: str) -> str | None:
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return str(argv[index + 1])
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def session_signature(argv: Sequence[str]) -> str:
    """The part of a command line that decides which session is read.

    The daemon holds a browser profile and a signed-in session, so two callers
    that disagree about Firefox/profile/cookies must not share one daemon.
    """
    parts: list[str] = []
    for flag in _SESSION_FLAGS:
        value = _argv_value(argv, flag)
        if value:
            parts.append(f"{flag}={value}")
    return " ".join(parts)


def _compatible(state: DaemonState, *, base_url: str, session: str = "") -> bool:
    """Whether ``state`` can answer for a caller with this session signature.

    A caller that names no source (``session == ""``) is happy with any warm
    daemon: it wanted the discovery ladder, and a daemon started with an
    explicit source already holds *a* usable session.  Being strict here cost a
    full cold start (browser launch included) every time the CLI and the Claude
    Code shim alternated, which looked exactly like "sending takes forever".
    """
    return (
        state.protocol == PROTOCOL_VERSION
        and state.api_version == api_version()
        and (not base_url or state.base_url == base_url)
        and (not session or state.session == session)
    )


def _tail(path: Path, limit: int = 800) -> str:
    try:
        data = path.read_bytes()[-limit:]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").strip()


def ensure_daemon(
    *,
    payload: Mapping[str, Any],
    base_url: str = DEFAULT_BASE_URL,
    state_file: str | Path | None = None,
    idle_timeout: float | None = None,
    start_timeout: float | None = None,
) -> DaemonState:
    """Return a reachable daemon, starting one when necessary.

    ``payload`` is forwarded to the daemon verbatim (the current CLI
    arguments), so the daemon builds its options with the very same parser the
    caller would have used.
    """
    path = state_path(state_file)
    idle = _idle_timeout(idle_timeout)
    budget = _start_timeout(start_timeout)
    raw_argv = payload.get("argv") or []
    wanted = session_signature(
        [str(item) for item in raw_argv] if isinstance(raw_argv, Sequence) else []
    )

    current = DaemonState.read(path)
    if current is not None:
        if _compatible(current, base_url=base_url, session=wanted) and ping(current):
            latency.mark("warm.reuse")
            return current
        if current.pid and _pid_alive(current.pid):
            # Something answered with incompatible settings: retire it.
            with contextlib.suppress(WarmDaemonUnavailable):
                stop_daemon(path, timeout=5.0)
        with contextlib.suppress(OSError):
            path.unlink()

    deadline = time.monotonic() + budget
    spawned: subprocess.Popen[bytes] | None = None
    poll_sleep = 0.005
    while time.monotonic() < deadline:
        candidate = DaemonState.read(path)
        if candidate is not None and _compatible(
            candidate, base_url=base_url, session=wanted
        ):
            if ping(candidate):
                latency.mark("warm.started")
                return candidate
        if spawned is not None and spawned.poll() is not None:
            detail = _tail(_log_path(path))
            suffix = f": {detail.splitlines()[-1]}" if detail else ""
            with contextlib.suppress(OSError):
                _options_path(path).unlink()
            raise WarmDaemonUnavailable(
                f"the warm daemon exited with code {spawned.returncode}{suffix}"
            )
        if spawned is None:
            with _spawn_lock(_lock_path(path), stale_after=budget) as won:
                if won:
                    # Re-check under the lock: a parallel caller may have won.
                    candidate = DaemonState.read(path)
                    if candidate is not None and ping(candidate):
                        return candidate
                    with contextlib.suppress(OSError):
                        path.unlink()
                    spawned = spawn_daemon(
                        state_file=path, payload=payload, idle_timeout=idle
                    )
        time.sleep(poll_sleep)
        poll_sleep = min(0.025, poll_sleep * 1.5)
    detail = _tail(_log_path(path))
    suffix = f" (log: {detail.splitlines()[-1]})" if detail else ""
    raise WarmDaemonUnavailable(
        f"the warm daemon did not become ready in {budget:g}s{suffix}"
    )


def _close_quietly(reader: Any, sock: socket.socket) -> None:
    with contextlib.suppress(Exception):
        reader.close()
    with contextlib.suppress(Exception):
        sock.close()


def stop_daemon(
    state_file: str | Path | None = None, *, timeout: float = 10.0
) -> bool:
    """Ask the daemon to shut down; ``True`` when one was running."""
    path = state_path(state_file)
    state = DaemonState.read(path)
    stopped = False
    if state is not None:
        try:
            sock, reader = _open(state, timeout=2.0, handshake_timeout=2.0)
        except WarmDaemonUnavailable:
            sock = None
        if sock is not None:
            try:
                _send(sock, {"op": "shutdown"})
                _read_message(reader, 2.0)
                stopped = True
            except (WarmDaemonUnavailable, OSError):
                stopped = True
            finally:
                _close_quietly(reader, sock)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and _pid_alive(state.pid) and ping(state, timeout=0.2):
            time.sleep(0.015)
    for path_to_remove in (
        path,
        _options_path(path),
        _lock_path(path),
    ):
        with contextlib.suppress(OSError):
            path_to_remove.unlink()
    return stopped


# --------------------------------------------------------------------------- #
# Daemon: a browser kept warm by a local session wrapper
# --------------------------------------------------------------------------- #


class WarmSession:
    """Owns one warm :class:`AsyncDirectArenaClient` and serialises prompts."""

    def __init__(self, options: Any) -> None:
        #: How many prompts are queued on the lock.  Claude Code sends the next
        #: request the moment an answer ends, so an archive step must not sit
        #: between them: it is skipped while somebody is waiting.
        self._waiting = 0
        self.options = replace(
            options, archive_if_idle=lambda: self._waiting <= 1
        )
        self._client: Any | None = None
        self._lock = asyncio.Lock()
        self._last_error: str | None = None

    async def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        from .auto import auto_async_direct_client

        with latency.span("warm.session.open"):
            client = await auto_async_direct_client(
                self.options, verification_token="auto", prewarm=True
            )
        self._client = client
        return client

    async def prepare(self) -> None:
        """Start the browser before the first prompt arrives."""
        async with self._lock:
            with contextlib.suppress(Exception):
                await self._client_or_create()

    async def _reset(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()

    _accepts_config: bool = True

    async def ask(
        self,
        prompt: str,
        model: str | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        archive: bool | None = None,
    ) -> AsyncIterator[str]:
        self._waiting += 1
        try:
            async for delta in self._ask_locked(
                prompt, model=model, config=config, archive=archive
            ):
                yield delta
        finally:
            self._waiting -= 1

    async def _ask_locked(
        self,
        prompt: str,
        *,
        model: str | None,
        config: Mapping[str, Any] | None = None,
        archive: bool | None = None,
    ) -> AsyncIterator[str]:
        from .exceptions import ArenaError, AuthenticationError, ChallengeRequiredError
        from .models import GenerationConfig

        gen_config: GenerationConfig | None = None
        if config:
            cfg_kwargs: dict[str, Any] = {}
            if isinstance(config.get("max_tokens"), int):
                cfg_kwargs["max_tokens"] = int(config["max_tokens"])
            if isinstance(config.get("temperature"), (int, float)):
                cfg_kwargs["temperature"] = float(config["temperature"])
            if isinstance(config.get("top_p"), (int, float)):
                cfg_kwargs["top_p"] = float(config["top_p"])
            if cfg_kwargs:
                gen_config = GenerationConfig(**cfg_kwargs)

        async with self._lock:
            for attempt in range(2):
                try:
                    client = await self._client_or_create()
                    tok: Any = getattr(client, "_verification_token", None)
                    prev_tok_options = (
                        getattr(tok, "options", None)
                        if archive is not None and tok is not None
                        else None
                    )
                    if prev_tok_options is not None and tok is not None:
                        with contextlib.suppress(Exception):
                            tok.options = replace(
                                prev_tok_options,
                                archive_after_answer=bool(archive),
                            )
                    # How much of the answer time was the server *thinking*, and
                    # how much was it trickling tokens?  Recorded per request and
                    # shown by `arena-api --warm-profile`.
                    chunks = 0
                    chars = 0
                    longest_gap = 0.0
                    asked_at = time.monotonic()
                    first_at: float | None = None
                    last_at = asked_at
                    stream_kwargs: dict[str, Any] = {"model": model}
                    if gen_config is not None:
                        stream_kwargs["config"] = gen_config
                    try:
                        with latency.span("warm.ask"):
                            async for chunk in client.astream(prompt, **stream_kwargs):
                                now = time.monotonic()
                                if first_at is None:
                                    first_at = now
                                longest_gap = max(longest_gap, now - last_at)
                                last_at = now
                                chunks += 1
                                chars += len(chunk.delta or "")
                                if chunk.delta:
                                    yield chunk.delta
                    finally:
                        if prev_tok_options is not None and tok is not None:
                            with contextlib.suppress(Exception):
                                tok.options = prev_tok_options
                    latency.metric("answer.chunks", chunks)
                    latency.metric("answer.chars", chars)
                    latency.metric("answer.longest_gap_ms", round(longest_gap * 1000.0, 1))
                    # Where the wall clock really goes: the wait for the first
                    # token (Arena thinking) and the generation speed after it.
                    # The streaming rate is the server's, not ours -- knowing it
                    # keeps "make it faster" an honest conversation.
                    if first_at is not None:
                        latency.metric(
                            "answer.first_delta_ms",
                            round((first_at - asked_at) * 1000.0, 1),
                        )
                        generation = max(1e-6, last_at - first_at)
                        latency.metric(
                            "answer.chars_per_s", round(chars / generation, 1)
                        )
                    self._last_error = None
                    return
                except (AuthenticationError, ChallengeRequiredError) as exc:
                    self._last_error = str(exc)
                    await self._reset()
                    if attempt:
                        raise
                except ArenaError:
                    raise

    def describe(self) -> dict[str, Any]:
        return {
            "warm": self._client is not None,
            "last_error": self._last_error,
        }

    async def aclose(self) -> None:
        await self._reset()


class _WarmServer:
    """Loopback JSON-lines server around a *session* object.

    ``session`` only needs ``ask(prompt, model)`` (an async iterator),
    ``aclose()``, and optionally ``prepare()``/``describe()``; the tests pass a
    fake, the daemon passes :class:`WarmSession`.
    """

    def __init__(
        self,
        session: Any,
        *,
        state_file: Path,
        token: str,
        host: str = "127.0.0.1",
        port: int = 0,
        base_url: str = DEFAULT_BASE_URL,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        session_identity: str = "",
    ) -> None:
        self.session = session
        self.state_file = state_file
        self.token = token
        self.host = host
        self.port = port
        self.base_url = base_url
        self.idle_timeout = idle_timeout
        self.session_identity = session_identity
        self._server: asyncio.AbstractServer | None = None
        self._stop = asyncio.Event()
        self._active = 0
        self._answered = 0
        self._last_activity = time.monotonic()
        self._background: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._started = time.time()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._startup_phases: list[dict[str, Any]] | None = None
        self._ask_profiles: deque[dict[str, Any]] = deque(maxlen=10)
        self._boot_marker = latency.marker()

    # -- lifecycle --------------------------------------------------------- #

    async def start(self) -> DaemonState:
        self._loop = asyncio.get_running_loop()
        self._server = await asyncio.start_server(
            self._handle,
            self.host,
            self.port,
            reuse_address=os.name != "nt",
        )
        sockets = self._server.sockets or ()
        if sockets:
            self.port = int(sockets[0].getsockname()[1])
        state = DaemonState(
            pid=os.getpid(),
            port=self.port,
            token=self.token,
            host=self.host,
            started=time.time(),
            protocol=PROTOCOL_VERSION,
            api_version=api_version(),
            base_url=self.base_url,
            idle_timeout=self.idle_timeout,
            python=sys.executable,
            session=self.session_identity,
        )
        state.write(self.state_file)
        self._spawn(self._watchdog())
        prepare = getattr(self.session, "prepare", None)
        if callable(prepare):
            self._spawn(self._prepare(prepare))
        return state

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _prepare(self, prepare: Any) -> None:
        try:
            await prepare()
        except Exception as exc:  # noqa: BLE001 - surfaced on the first prompt
            log(f"warm browser pre-start failed: {exc}")
        finally:
            # Browser launch + first page load happen once per daemon; report
            # them so a slow "cold" prompt can be told apart from a warm one.
            self._startup_phases = latency.rows_since(self._boot_marker) or None

    def request_stop(self) -> None:
        """Ask the server to stop; safe from any thread or signal handler."""
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._stop.set)
                return
            except RuntimeError:  # pragma: no cover - loop already closing
                pass
        self._stop.set()

    async def run(self) -> None:
        if self._server is None:
            await self.start()
        try:
            await self._stop.wait()
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        for task in list(self._background):
            task.cancel()
        for task in list(self._background):
            with contextlib.suppress(BaseException):
                await task
        self._background.clear()
        with contextlib.suppress(OSError):
            self.state_file.unlink()
        with contextlib.suppress(OSError):
            _options_path(self.state_file).unlink()
        with contextlib.suppress(Exception):
            await self.session.aclose()

    async def _watchdog(self) -> None:
        interval = max(1.0, min(30.0, self.idle_timeout / 4.0))
        while not self._stop.is_set():
            await asyncio.sleep(interval)
            if self._active:
                continue
            if time.monotonic() - self._last_activity >= self.idle_timeout:
                log(f"idle for {self.idle_timeout:g}s, shutting down")
                self._stop.set()
                return

    # -- connection handling ---------------------------------------------- #

    def _touch(self) -> None:
        self._last_activity = time.monotonic()

    async def _write(self, writer: asyncio.StreamWriter, payload: Mapping[str, Any]) -> None:
        writer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        await writer.drain()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        raw_sock = writer.get_extra_info("socket")
        if raw_sock is not None:
            with contextlib.suppress(OSError, AttributeError):
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            await self._dispatch(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        except Exception as exc:  # noqa: BLE001 - one bad client must not stop us
            log(f"connection failed: {exc}")
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def _dispatch(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        line = await reader.readline()
        if not line:
            return
        try:
            hello = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            await self._write(
                writer, {"type": "error", "kind": "protocol", "message": "bad handshake"}
            )
            return
        if not isinstance(hello, Mapping) or hello.get("op") != "hello":
            await self._write(
                writer, {"type": "error", "kind": "protocol", "message": "bad handshake"}
            )
            return
        token = str(hello.get("token") or "")
        if not hmac.compare_digest(token, self.token):
            await self._write(
                writer, {"type": "error", "kind": "auth", "message": "invalid daemon token"}
            )
            return
        self._touch()
        await self._write(
            writer,
            {
                "type": "hello",
                "pid": os.getpid(),
                "protocol": PROTOCOL_VERSION,
                "version": api_version(),
                "base_url": self.base_url,
            },
        )
        while True:
            line = await reader.readline()
            if not line:
                return
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                await self._write(
                    writer,
                    {"type": "error", "kind": "protocol", "message": "malformed frame"},
                )
                continue
            if not isinstance(message, Mapping):
                continue
            self._touch()
            op = message.get("op")
            if op == "ping":
                await self._write(writer, {"type": "pong"})
            elif op == "status":
                await self._write(writer, self._status_payload())
            elif op == "profiles":
                await self._write(writer, self._profiles_payload(message))
            elif op == "shutdown":
                await self._write(writer, {"type": "bye"})
                self._stop.set()
                return
            elif op == "ask":
                if not await self._ask(message, writer):
                    return
            elif op == "cancel":
                continue
            else:
                await self._write(
                    writer,
                    {
                        "type": "error",
                        "kind": "protocol",
                        "message": f"unknown op {op!r}",
                    },
                )

    async def _send_startup_profile(self, writer: asyncio.StreamWriter) -> None:
        """Hand the browser/page startup timeline to the caller once."""
        phases = self._startup_phases
        self._startup_phases = None
        if phases:
            with contextlib.suppress(Exception):
                await self._write(
                    writer, {"type": "profile", "kind": "startup", "phases": phases}
                )

    async def _send_ask_profile(
        self, writer: asyncio.StreamWriter, marker: int
    ) -> None:
        """Report where this prompt spent its time (best effort)."""
        phases = latency.rows_since(marker)
        if not phases:
            return
        collected = latency.metrics()
        self._ask_profiles.append(
            {"at": time.time(), "phases": phases, "metrics": collected}
        )
        with contextlib.suppress(Exception):
            await self._write(
                writer,
                {
                    "type": "profile",
                    "kind": "ask",
                    "phases": phases,
                    "metrics": latency.metrics(),
                },
            )

    def _profiles_payload(self, message: Mapping[str, Any]) -> dict[str, Any]:
        try:
            limit = int(message.get("limit") or 3)
        except (TypeError, ValueError):
            limit = 3
        entries = list(self._ask_profiles)[-max(1, limit) :]
        return {
            "type": "profiles",
            "startup": self._startup_phases or [],
            "asks": entries,
            "total_answered": self._answered,
        }

    def _status_payload(self) -> dict[str, Any]:
        describe = getattr(self.session, "describe", None)
        detail: dict[str, Any] = {}
        if callable(describe):
            with contextlib.suppress(Exception):
                detail = dict(describe() or {})
        payload: dict[str, Any] = {
            "type": "status",
            "pid": os.getpid(),
            "port": self.port,
            "protocol": PROTOCOL_VERSION,
            "version": api_version(),
            "base_url": self.base_url,
            "idle_timeout": self.idle_timeout,
            "idle_for": round(time.monotonic() - self._last_activity, 1),
            "active": self._active,
            "started": self._started,
        }
        payload.update(detail)
        return payload

    async def _ask(
        self, message: Mapping[str, Any], writer: asyncio.StreamWriter
    ) -> bool:
        prompt = message.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            await self._write(
                writer,
                {"type": "error", "kind": "usage", "message": "empty prompt"},
            )
            return True
        model = message.get("model")
        model_value = model if isinstance(model, str) and model else None
        raw_config = message.get("config")
        config_value = (
            dict(raw_config)
            if isinstance(raw_config, Mapping) and raw_config
            else None
        )
        raw_archive = message.get("archive")
        archive_value = raw_archive if isinstance(raw_archive, bool) else None
        self._active += 1
        extra_ask_kwargs: dict[str, Any] = {}
        if config_value is not None:
            extra_ask_kwargs["config"] = config_value
        if archive_value is not None:
            extra_ask_kwargs["archive"] = archive_value
        if extra_ask_kwargs and getattr(
            self.session, "_accepts_config", False
        ):
            stream = self.session.ask(
                prompt, model_value, **extra_ask_kwargs
            )
        else:
            stream = self.session.ask(prompt, model_value)
        try:
            await self._write(writer, {"type": "start"})
            await self._send_startup_profile(writer)
            latency.reset()
            ask_marker = latency.marker()
            pieces: list[str] = []
            with latency.span("warm.stream"):
                async for delta in stream:
                    if not delta:
                        continue
                    pieces.append(delta)
                    await self._write(writer, {"type": "delta", "text": delta})
            # The profile goes out before "done": the caller stops reading as
            # soon as the answer is complete and closes the socket.
            await self._send_ask_profile(writer, ask_marker)
            await self._write(writer, {"type": "done", "text": "".join(pieces)})
            self._answered += 1
            return True
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            detail = str(exc) or type(exc).__name__
            kind = error_kind(exc)
            with contextlib.suppress(Exception):
                await self._write(
                    writer,
                    {"type": "error", "kind": kind, "message": detail},
                )
            if kind == "auth" and not self._answered:
                # Nothing ever worked with this session: do not linger for the
                # whole idle timeout with a browser nobody can use.  The caller
                # now falls back to the ordinary path, which explains why.
                log(f"first prompt failed with {detail}; shutting down")
                self.request_stop()
            return True
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    await close()
            self._active -= 1
            self._touch()


# --------------------------------------------------------------------------- #
# Daemon entry point: python -m arena_api.warm
# --------------------------------------------------------------------------- #


def _options_from_payload(payload: Mapping[str, Any]) -> Any:
    """Rebuild the CLI's own :class:`AutoOptions` from the forwarded arguments."""
    from .auto import _options_from_args, build_arg_parser

    argv = payload.get("argv") or []
    if not isinstance(argv, Sequence):
        raise UsageError("argv must be a list")
    parser = build_arg_parser()
    try:
        namespace = parser.parse_args([str(item) for item in argv])
    except SystemExit as exc:
        raise UsageError("invalid command line forwarded to the warm daemon") from exc
    options = _options_from_args(namespace)
    # A daemon cannot ask a human for anything, and the verification probe only
    # delays the first prompt: the request itself reports a rejected session.
    return replace(options, interactive=False, verify=False)


async def _serve(
    options: Any,
    *,
    state_file: Path,
    idle_timeout: float,
    host: str,
    port: int,
    forwarded_argv: Sequence[str] = (),
) -> int:
    session = WarmSession(options)
    published = session_signature(list(forwarded_argv))
    server = _WarmServer(
        session,
        state_file=state_file,
        token=secrets.token_hex(32),
        host=host,
        port=port,
        base_url=str(options.base_url),
        idle_timeout=idle_timeout,
        session_identity=published,
    )
    loop = asyncio.get_running_loop()
    installed: list[int] = []
    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, server.request_stop)
            installed.append(sig)
    try:
        await server.run()
    finally:
        for sig in installed:
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(sig)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m arena_api.warm``: the daemon process itself."""
    latency.enable_collection()
    parser = argparse.ArgumentParser(
        prog="python -m arena_api.warm",
        description=(
            "Keep a Direct Chat browser warm for one-shot arena-api calls. "
            "Started automatically; run `arena-api --warm-stop` to stop it."
        ),
    )
    parser.add_argument("--state", default=None, help="state file to publish")
    parser.add_argument("--options-file", default=None, help="JSON with the CLI arguments")
    parser.add_argument("--idle-timeout", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload: dict[str, Any] = {}
    if args.options_file:
        try:
            payload = json.loads(Path(args.options_file).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log(f"cannot read {args.options_file}: {exc}")
            return 2
    try:
        options = _options_from_payload(payload)
    except UsageError as exc:
        log(str(exc))
        return 2
    finally:
        if args.options_file:
            # The forwarded command line may carry a cookie header; once it is
            # parsed there is no reason to keep it on disk.
            with contextlib.suppress(OSError):
                Path(args.options_file).unlink()
    path = state_path(args.state)
    idle = _idle_timeout(args.idle_timeout)
    latency.reset()
    log(
        f"aiming at {options.base_url}; idle timeout {idle:g}s; "
        f"state {path}"
    )
    try:
        return asyncio.run(
            _serve(
                options,
                state_file=path,
                idle_timeout=idle,
                host=args.host,
                port=args.port,
                forwarded_argv=[
                    str(item) for item in (payload.get("argv") or [])
                ],
            )
        )
    except KeyboardInterrupt:
        return 0
    finally:
        for leftover in (path, _options_path(path)):
            with contextlib.suppress(OSError):
                leftover.unlink()


# --------------------------------------------------------------------------- #
# Lightweight CLI front-end
# --------------------------------------------------------------------------- #


@dataclass
class _WarmArgs:
    prompt: str | None = None
    model: str | None = None
    fast_model: bool | None = None
    archive_after_answer: bool | None = None
    rest: list[str] = field(default_factory=list)
    base_url: str = DEFAULT_BASE_URL
    state_file: str | None = None
    idle_timeout: float | None = None
    start_timeout: float | None = None
    force: bool | None = None
    usable: bool = True


def scan_argv(argv: Sequence[str]) -> _WarmArgs:
    """Split the warm-relevant flags off a one-shot ``--prompt`` command line.

    Everything else is forwarded to the daemon unchanged, which rebuilds
    :class:`AutoOptions` with the real CLI parser — one source of truth for
    flag semantics.
    """
    parsed = _WarmArgs()
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            parsed.usable = False
            return parsed
        if arg == "--prompt":
            parsed.prompt = argv[index + 1] if index + 1 < len(argv) else ""
            index += 2
            continue
        if arg.startswith("--prompt="):
            parsed.prompt = arg.split("=", 1)[1]
            index += 1
            continue
        if arg == "--model":
            parsed.model = argv[index + 1] if index + 1 < len(argv) else None
            index += 2
            continue
        if arg.startswith("--model="):
            parsed.model = arg.split("=", 1)[1]
            index += 1
            continue
        if arg in ("--warm", "--no-warm"):
            parsed.force = arg == "--warm"
            index += 1
            continue
        if arg in ("--fast-model", "--no-fast-model"):
            parsed.fast_model = arg == "--fast-model"
            parsed.rest.append(arg)
            index += 1
            continue
        if arg in ("--archive-after-answer", "--no-archive-after-answer"):
            parsed.archive_after_answer = arg == "--archive-after-answer"
            parsed.rest.append(arg)
            index += 1
            continue
        if arg in ("--warm-start", "--warm-stop", "--warm-status", "--warm-profile"):
            # Daemon commands, never forwarded to the daemon's own parser.
            index += 1
            continue
        if arg.startswith("--warm-profile="):
            index += 1
            continue
        if arg in ("--warm-idle", "--warm-timeout", "--warm-state"):
            value = argv[index + 1] if index + 1 < len(argv) else None
            _store_warm_option(parsed, arg, value)
            index += 2
            continue
        if arg.startswith(("--warm-idle=", "--warm-timeout=", "--warm-state=")):
            name, value = arg.split("=", 1)
            _store_warm_option(parsed, name, value)
            index += 1
            continue
        if arg in ("--base-url", "--verification-token"):
            value = argv[index + 1] if index + 1 < len(argv) else None
            if arg == "--base-url" and value:
                parsed.base_url = value
            if arg == "--verification-token" and value not in (None, "", "auto"):
                # A caller-supplied assertion cannot be reproduced by the
                # daemon, which mints its own: use the regular path.
                parsed.usable = False
            parsed.rest.extend([arg, value] if value is not None else [arg])
            index += 2
            continue
        if arg.startswith("--verification-token="):
            value = arg.split("=", 1)[1]
            if value not in ("", "auto"):
                parsed.usable = False
            parsed.rest.append(arg)
            index += 1
            continue
        if arg.startswith("--base-url="):
            parsed.base_url = arg.split("=", 1)[1]
            parsed.rest.append(arg)
            index += 1
            continue
        parsed.rest.append(arg)
        index += 1
    # `--warm-profile 5` takes a value; drop it from the forwarded arguments.
    cleaned: list[str] = []
    skip_next = False
    for item in parsed.rest:
        if skip_next:
            skip_next = False
            continue
        if item == "--warm-profile":
            skip_next = True
            continue
        cleaned.append(item)
    parsed.rest = cleaned
    if parsed.model is not None and parsed.model.strip().lower() == "auto":
        parsed.model = None
    return parsed


def _store_warm_option(parsed: _WarmArgs, name: str, value: str | None) -> None:
    if value is None:
        parsed.usable = False
        return
    if name == "--warm-state":
        parsed.state_file = value
        return
    try:
        number = float(value)
    except ValueError:
        parsed.usable = False
        return
    if name == "--warm-idle":
        parsed.idle_timeout = number
    else:
        parsed.start_timeout = number


def profile_main(argv: Sequence[str]) -> int:
    """``--warm-profile``: where the daemon's last prompts spent their time."""
    parsed = scan_argv(list(argv))
    limit = 3
    for index, arg in enumerate(argv):
        if arg == "--warm-profile" and index + 1 < len(argv):
            try:
                limit = max(1, int(argv[index + 1]))
            except ValueError:
                pass
        elif arg.startswith("--warm-profile="):
            try:
                limit = max(1, int(arg.split("=", 1)[1]))
            except ValueError:
                pass
    payload = fetch_profiles(parsed.state_file, limit=limit)
    if payload is None:
        print(f"warm daemon       : not running ({state_path(parsed.state_file)})")
        print("hint              : run a prompt first, then ask again")
        return 1
    startup = payload.get("startup") or []
    if startup:
        print("daemon startup (browser launch + first page load):")
        for line in format_profile_table(startup, title="  "):
            print(line)
        print()
    asks = payload.get("asks") or []
    if not asks:
        print("no prompts answered yet")
        return 0
    print(f"last {len(asks)} prompt(s):")
    for number, entry in enumerate(asks, start=1):
        phases = entry.get("phases") if isinstance(entry, Mapping) else None
        when = entry.get("at") if isinstance(entry, Mapping) else None
        stamp = time.strftime("%H:%M:%S", time.localtime(when)) if when else "?"
        print(f"\n  #{number} at {stamp}")
        for line in format_profile_table(phases or [], title="    "):
            print(line)
        metrics = entry.get("metrics") if isinstance(entry, Mapping) else None
        if isinstance(metrics, Mapping) and metrics:
            rendered = ", ".join(
                f"{key}={value:g}"
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            )
            if rendered:
                print(f"    metrics: {rendered}")
    return 0


def _stdout_write(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = str(getattr(sys.stdout, "encoding", "") or "utf-8") or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding))
    sys.stdout.flush()


def _payload_for(parsed: _WarmArgs) -> dict[str, Any]:
    return {"argv": parsed.rest, "base_url": parsed.base_url}


def _ensure_for_cli(parsed: _WarmArgs) -> DaemonState | None:
    try:
        return ensure_daemon(
            payload=_payload_for(parsed),
            base_url=parsed.base_url,
            state_file=parsed.state_file,
            idle_timeout=parsed.idle_timeout,
            start_timeout=parsed.start_timeout,
        )
    except WarmDaemonUnavailable as exc:
        _mark_skipped()
        log(f"warm daemon unavailable ({exc}); using the one-shot path")
        return None


def _stop_if_running(parsed: _WarmArgs) -> None:
    """Retire the daemon so it cannot hold the shared browser profile."""
    if DaemonState.read(state_path(parsed.state_file)) is None:
        return
    with contextlib.suppress(Exception):
        stop_daemon(parsed.state_file)


_DEFAULT_ENSURE_DAEMON = ensure_daemon
_DEFAULT_ASK_STREAM = ask_stream


def _optimistic_live_state(parsed: _WarmArgs) -> DaemonState | None:
    """Read a live compatible daemon state from disk without a TCP ping round-trip."""
    path = state_path(parsed.state_file)
    current = DaemonState.read(path)
    if current is None:
        return None
    wanted = session_signature(parsed.rest)
    if not _compatible(current, base_url=parsed.base_url, session=wanted):
        return None
    if current.pid and not _pid_alive(current.pid):
        return None
    return current


def fast_prompt_main(argv: Sequence[str]) -> int | None:
    """Answer ``--prompt`` through the daemon; ``None`` means "fall back".

    Returning ``None`` is always safe: the caller then runs the ordinary
    in-process path, which is what every version before the daemon did.
    """
    parsed = scan_argv(argv)
    if parsed.force is False:
        # The daemon keeps the Direct Chat Chromium profile open; a second
        # browser on the same profile cannot start.  ``--no-warm`` therefore
        # retires it instead of failing on the lock.
        _stop_if_running(parsed)
        return None
    if not parsed.usable or not parsed.prompt:
        return None
    if parsed.force is not True and not warm_enabled():
        return None
    if parsed.force is not True and supplied_verification_token():
        return None
    wire_model = parsed.model
    if wire_model is None and parsed.fast_model is not None:
        wire_model = "fast" if parsed.fast_model else "max"

    unpatched = (
        ensure_daemon is _DEFAULT_ENSURE_DAEMON
        and ask_stream is _DEFAULT_ASK_STREAM
    )
    used_optimistic = False
    state: DaemonState | None = None
    if unpatched:
        state = _optimistic_live_state(parsed)
        if state is not None:
            used_optimistic = True
            latency.mark("warm.reuse")
    if state is None:
        state = _ensure_for_cli(parsed)
        if state is None:
            return None
    latency.mark("cli.warm.ready")
    try:
        stream: Iterator[str] | None = None
        first = ""
        for connect_attempt in range(2 if used_optimistic else 1):
            ask_kwargs: dict[str, Any] = {
                "state": state,
                "model": wire_model,
            }
            if parsed.archive_after_answer is not None:
                ask_kwargs["archive"] = parsed.archive_after_answer
            stream = ask_stream(parsed.prompt, **ask_kwargs)
            try:
                first = next(stream)
                break
            except StopIteration:
                first = ""
                break
            except WarmDaemonUnavailable as exc:
                close_fn = getattr(stream, "close", None)
                if callable(close_fn):
                    close_fn()
                if connect_attempt == 0 and used_optimistic:
                    state = _ensure_for_cli(parsed)
                    if state is None:
                        return None
                    continue
                _mark_skipped()
                reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                log(
                    f"warm daemon could not start the answer ({reason}); using the one-shot path"
                )
                return None
            except AuthenticationError:
                close_fn = getattr(stream, "close", None)
                if callable(close_fn):
                    close_fn()
                _mark_skipped()
                log(
                    "warm daemon could not start the answer (no usable session); using the one-shot path"
                )
                return None
        assert stream is not None
        latency.mark("cli.first_delta")
        if parsed.model:
            print(f"model             : {parsed.model}")
        sys.stdout.write("answer            : ")
        sys.stdout.flush()
        _stdout_write(first)
        for delta in stream:
            _stdout_write(delta)
        print()
        latency.mark("cli.done")
        return 0
    except WarmDaemonUnavailable as exc:
        _mark_skipped()
        log(f"warm daemon went away ({exc}); using the one-shot path")
        return None
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - same contract as arena_api.auto.main
        print(f"[arena-api] {exc}", file=sys.stderr)
        return 1


def start_main(argv: Sequence[str]) -> int:
    """``--warm-start``: bring the daemon up now, for a later prompt."""
    parsed = scan_argv(list(argv))
    if not parsed.usable:
        print("warm daemon       : cannot start (unsupported command line)")
        return 2
    try:
        state = ensure_daemon(
            payload=_payload_for(parsed),
            base_url=parsed.base_url,
            state_file=parsed.state_file,
            idle_timeout=parsed.idle_timeout,
            start_timeout=parsed.start_timeout,
        )
    except WarmDaemonUnavailable as exc:
        print(f"warm daemon       : could not start ({exc})")
        return 1
    print(f"warm daemon       : running (pid {state.pid}, port {state.port})")
    print(f"state file        : {state_path(parsed.state_file)}")
    return 0


def fetch_profiles(
    state_path_value: str | Path | None = None,
    *,
    limit: int = 3,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """Ask the daemon for the timelines of its last prompts."""
    path = state_path(state_path_value)
    state = DaemonState.read(path)
    if state is None:
        return None
    try:
        sock, reader = _open(state, timeout=timeout, handshake_timeout=timeout)
    except WarmDaemonUnavailable:
        return None
    try:
        _send(sock, {"op": "profiles", "limit": limit})
        reply = _read_message(reader, timeout)
        if reply.get("type") != "profiles":
            return None
        return reply
    except (WarmDaemonUnavailable, OSError):
        return None
    finally:
        _close_quietly(reader, sock)


def format_profile_table(
    phases: Sequence[Mapping[str, Any]], *, title: str = ""
) -> list[str]:
    """Render phases the same way ``ARENA_PROFILE=1`` does."""
    rows = [
        (str(row.get("phase")), float(row.get("at_ms") or 0.0), float(row.get("delta_ms") or 0.0))
        for row in phases
        if isinstance(row, Mapping)
    ]
    if not rows:
        return []
    width = max(len("phase"), *(len(name) for name, _, _ in rows))
    lines = [f"{title}{'phase':<{width}} {'delta ms':>10} {'total ms':>10}".rstrip()]
    for name, at_ms, delta_ms in rows:
        lines.append(f"{name:<{width}} {delta_ms:>10.1f} {at_ms:>10.1f}")
    lines.append(f"{'total':<{width}} {'':>10} {rows[-1][1]:>10.1f}")
    return lines


def stop_main(argv: Sequence[str]) -> int:
    """``--warm-stop``: shut the daemon down and report what happened."""
    parsed = scan_argv(list(argv))
    stopped = stop_daemon(parsed.state_file)
    path = state_path(parsed.state_file)
    if stopped:
        print(f"warm daemon       : stopped ({path})")
        return 0
    print(f"warm daemon       : not running ({path})")
    return 0


def status_main(argv: Sequence[str]) -> int:
    """``--warm-status``: print where the daemon is and how warm it is."""
    parsed = scan_argv(list(argv))
    path = state_path(parsed.state_file)
    info = describe_daemon(parsed.state_file)
    if info is None:
        print(f"warm daemon       : not running ({path})")
        return 1
    print(f"warm daemon       : running (pid {info.get('pid')}, port {info.get('port')})")
    print(f"state file        : {path}")
    print(f"browser warmed    : {'yes' if info.get('warm') else 'starting'}")
    print(f"idle for          : {info.get('idle_for')}s (timeout {info.get('idle_timeout')}s)")
    print(f"active prompts    : {info.get('active')}")
    print(f"base url          : {info.get('base_url')}")
    if info.get("last_error"):
        print(f"last error        : {info['last_error']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - `python -m arena_api.warm`
    raise SystemExit(main())
