"""Tests for ``tools/bench_cli.py`` (the cold-vs-warm measurement script)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "bench_cli.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("bench_cli", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load()


def _report(phases: list[tuple[str, float]]) -> str:
    payload = {
        "total_ms": phases[-1][1] if phases else 0.0,
        "phases": [{"phase": n, "at_ms": at, "delta_ms": 0.0} for n, at in phases],
    }
    return f"[arena-api latency] {json.dumps(payload)}"


def test_parse_first_token_reads_the_merged_daemon_phases() -> None:
    stderr = "\n".join(
        [
            "[arena-api] session: Firefox profile ...",
            _report([("cli.start", 4.0), ("daemon.first_byte", 900.0), ("cli.exit", 8000.0)]),
        ]
    )
    assert bench.parse_first_token(stderr) == 0.9


def test_parse_first_token_takes_the_earliest_marker() -> None:
    stderr = _report([("first_byte", 700.0), ("first_delta", 950.0), ("cli.exit", 5000.0)])
    assert bench.parse_first_token(stderr) == 0.7


def test_parse_first_token_without_a_report() -> None:
    assert bench.parse_first_token("[arena-api] nothing") is None
    assert bench.parse_first_token("[arena-api latency] {broken") is None


def _strict(*args: Any, **kwargs: Any) -> Any:
    """Fail the test if the benchmark ever sleeps between runs."""
    raise AssertionError(f"the benchmark must not delay a run ({args})")


def test_no_delays_are_used_and_failures_do_not_abort(
    monkeypatch: Any, capsys: Any
) -> None:
    calls: list[list[str]] = []

    def _run(_python: str, argv: Any) -> tuple[float, int, str, float | None]:
        argv = list(argv)
        calls.append(argv)
        if "--warm-stop" in argv:
            return 0.1, 0, "", None
        if "--no-warm" in argv and len(calls) == 2:
            # The user's case: one prompt is rejected (captcha not cleared).
            return 8.5, 1, "[arena-api] Arena answered HTTP 429. No verification challenge was found", None
        cold = "--no-warm" in argv
        return (8.8 if cold else 1.2), 0, "", (5.0 if cold else 0.6)

    monkeypatch.setattr(bench, "_run", _run)
    monkeypatch.setattr(bench.time, "sleep", _strict)

    assert bench.main(["--prompt", "hi", "--runs", "3"]) == 0
    out = capsys.readouterr()
    assert "failed" in out.out
    assert "speed-up on repeated prompts" in out.out
    # Three cold runs, one of which failed, and three warm runs were all tried.
    assert sum(1 for argv in calls if "--no-warm" in argv) == 3
    warm_calls = [a for a in calls if "--prompt" in a and "--no-warm" not in a]
    assert len(warm_calls) == 4  # the first (cold-starting) one plus --runs


def test_stops_only_when_nothing_was_measured(
    monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(
        bench,
        "_run",
        lambda *a, **k: (1.0, 1, "[arena-api] no session found for arena.ai", None),
    )
    monkeypatch.setattr(bench.time, "sleep", _strict)
    assert bench.main(["--prompt", "hi", "--runs", "2"]) == 2
    assert "arena-api --status" in capsys.readouterr().err


def test_reports_when_only_one_side_was_measured(
    monkeypatch: Any, capsys: Any
) -> None:
    def _run(_python: str, argv: Any) -> tuple[float, int, str, float | None]:
        argv = list(argv)
        if "--warm-stop" in argv:
            return 0.1, 0, "", None
        if "--no-warm" in argv:
            return 8.8, 0, "", 4.0
        return 2.0, 1, "[arena-api] Arena answered HTTP 429.", None

    monkeypatch.setattr(bench, "_run", _run)
    monkeypatch.setattr(bench.time, "sleep", _strict)
    assert bench.main(["--prompt", "hi", "--runs", "1"]) == 1
    out = capsys.readouterr()
    assert "warm side did not produce a measurement" in out.out


def test_first_warm_run_is_reported_apart(
    monkeypatch: Any, capsys: Any
) -> None:
    times = {"cold": 9.0, "warm": 1.5}

    def _run(_python: str, argv: Any) -> tuple[float, int, str, float | None]:
        argv = list(argv)
        if "--warm-stop" in argv:
            return 0.1, 0, "", None
        if "--no-warm" in argv:
            return times["cold"], 0, "", 6.0
        value = times["warm"]
        times["warm"] = 1.0  # every following warm call is faster
        return value, 0, "", 0.5

    monkeypatch.setattr(bench, "_run", _run)
    monkeypatch.setattr(bench.time, "sleep", _strict)
    assert bench.main(["--prompt", "hi", "--runs", "2"]) == 0
    out = capsys.readouterr().out
    assert "first warm run (daemon start): 1.50s" in out
    assert "time to first token" in out
    assert "speed-up on first token" in out
    assert "speed-up on repeated prompts: 9.0x" in out


def test_rejects_nonsense_options() -> None:
    import pytest

    with pytest.raises(SystemExit):
        bench.main(["--runs", "0"])
