"""Tests for the opt-in latency profiler and the light CLI front-end."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from arena_api import _cli, latency


@pytest.fixture()
def profiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latency, "_MODE", "1")
    latency.reset()
    yield
    latency.reset()


def test_disabled_by_default() -> None:
    latency.reset()
    latency.mark("never")
    assert latency.report() == ""
    assert latency.enabled() is False


def test_marks_and_report(profiling: None) -> None:
    latency.mark("a")
    latency.mark("b")
    text = latency.report()
    assert "phase" in text
    assert "a" in text and "b" in text
    assert "total" in text


def test_span_marks_entry_and_exit(profiling: None) -> None:
    with latency.span("phase"):
        pass
    phases = [row["phase"] for row in latency._rows()]
    assert phases == ["phase", "phase.end"]


def test_span_records_failures(profiling: None) -> None:
    with pytest.raises(RuntimeError):
        with latency.span("phase"):
            raise RuntimeError("boom")
    assert [row["phase"] for row in latency._rows()] == ["phase", "phase.error"]


def test_json_report_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latency, "_MODE", "json")
    latency.reset()
    latency.mark("a")
    latency.mark("b")
    payload = json.loads(latency.as_json(prefix="") or "{}")
    assert payload["total_ms"] >= 0
    assert [phase["phase"] for phase in payload["phases"]] == ["a", "b"]


def test_metrics_are_reported_next_to_phases(profiling: None) -> None:
    latency.mark("a")
    latency.metric("answer.chunks", 12)
    latency.metric("answer.longest_gap_ms", 830.5)
    text = latency.report()
    assert "answer.chunks = 12" in text
    assert "answer.longest_gap_ms = 830.5" in text
    payload = json.loads(latency.as_json(prefix="") or "{}")
    assert payload["metrics"]["answer.chunks"] == 12


def test_set_metrics_imports_the_daemon_numbers(profiling: None) -> None:
    latency.mark("daemon.first_byte")
    latency.set_metrics({"answer.chars": 4, "not_a_number": "x"})
    assert latency.metrics() == {"answer.chars": 4.0}


def test_reset_clears_metrics(profiling: None) -> None:
    latency.metric("answer.chunks", 1)
    latency.reset()
    assert latency.metrics() == {}


def test_dump_writes_to_the_given_stream(profiling: None) -> None:
    buffer = io.StringIO()
    latency.mark("a")
    latency.dump(stream=buffer)
    assert "a" in buffer.getvalue()


def test_dump_is_silent_when_disabled() -> None:
    latency.reset()
    buffer = io.StringIO()
    latency.dump(stream=buffer)
    assert buffer.getvalue() == ""


def test_mode_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert latency._mode_from_env({"ARENA_PROFILE": "1"}) == "1"
    assert latency._mode_from_env({"ARENA_PROFILE": "JSON"}) == "json"
    assert latency._mode_from_env({"ARENA_PROFILE": "0"}) == ""


# --------------------------------------------------------------------------- #
# The fast CLI route must not drag the heavy bootstrap in
# --------------------------------------------------------------------------- #


def test_cli_front_end_imports_without_the_auto_module() -> None:
    code = (
        "import sys, arena_api._cli;"
        "assert 'arena_api.auto' not in sys.modules, 'auto imported eagerly';"
        "assert 'httpx' not in sys.modules, 'httpx imported eagerly';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_warm_module_imports_without_the_heavy_stack() -> None:
    code = (
        "import sys, arena_api.warm;"
        "assert 'httpx' not in sys.modules, 'httpx imported eagerly';"
        "assert 'arena_api.auto' not in sys.modules, 'auto imported eagerly';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_cli_help_still_reaches_the_real_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _cli.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--prompt" in captured.out
    assert "--warm-stop" in captured.out
