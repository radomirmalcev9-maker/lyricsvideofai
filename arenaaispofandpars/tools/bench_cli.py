#!/usr/bin/env python3
"""Measure the warm-daemon speed-up with the CLI you actually use.

The daemon only removes *overhead*: session discovery, browser launch, page
load, composer warm-up.  How much wall-clock time that is depends on your
machine and profile, so the honest way to claim a factor is to measure it:

    python tools/bench_cli.py --prompt "Ответь одним словом: ок"

The script runs the same prompt N times with a cold start (``--no-warm``, and
the daemon stopped before each run) and N times through the warm daemon, and
prints the medians plus the resulting speed-up.  It needs a working session
(``arena-api --status`` must pass); otherwise every run fails the same way and
there is nothing to compare.

Two numbers are reported per variant: the whole command (generation included)
and the time to the first streamed token, read from ``ARENA_PROFILE=json``.
The first token is what the warm daemon actually removes.

Notes:
* nothing is delayed or retried between runs: the script measures back-to-back
  invocations, exactly like calling the CLI in a loop;
* a failed run is reported and skipped; the benchmark only stops when a whole
  variant produced no measurement at all;
* the first warm run pays for starting the daemon, so it is reported apart;
* the prompt should produce a short answer — the benchmark measures the whole
  command, and a long generation dominates both variants equally.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence


#: Phase names meaning "the first piece of the answer arrived".
_FIRST_TOKEN_SUFFIXES = ("first_delta", "first_byte")
_PROFILE_PREFIX = "[arena-api latency]"


def parse_first_token(stderr: str) -> float | None:
    """Seconds until the first streamed token, from an ``ARENA_PROFILE=json`` line.

    The daemon forwards its own phases (``daemon.first_byte``, ...), so the same
    report tells us how long the *whole* path took, not just this process.
    """
    best: float | None = None
    for line in stderr.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_PROFILE_PREFIX):
            continue
        try:
            report = json.loads(stripped[len(_PROFILE_PREFIX) :].strip())
        except ValueError:
            continue
        phases = report.get("phases")
        if not isinstance(phases, list):
            continue
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            name = phase.get("phase")
            at_ms = phase.get("at_ms")
            if not isinstance(name, str) or not isinstance(at_ms, (int, float)):
                continue
            if not name.endswith(_FIRST_TOKEN_SUFFIXES):
                continue
            seconds = float(at_ms) / 1000.0
            if best is None or seconds < best:
                best = seconds
    return best


def _run(python: str, argv: Sequence[str]) -> tuple[float, int, str, float | None]:
    started = time.monotonic()
    env = dict(os.environ)
    env["ARENA_PROFILE"] = "json"
    result = subprocess.run(
        [python, "-m", "arena_api", *argv],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    stderr = (result.stderr or "").strip()
    return (
        time.monotonic() - started,
        result.returncode,
        stderr,
        parse_first_token(stderr),
    )


def _label(seconds: float) -> str:
    return f"{seconds:6.2f} s"


def _format_runs(name: str, runs: list[float]) -> str:
    rendered = ", ".join(f"{value:.2f}" for value in runs)
    median = statistics.median(runs) if runs else 0.0
    return f"{name:<26}{rendered:<32}{_label(median):>10}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a cold CLI start with the warm daemon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        default="Ответь ровно одним словом: ок",
        help="prompt used for every run (keep the answer short)",
    )
    parser.add_argument("--model", default=None, help="forwarded --model")
    parser.add_argument("--runs", type=int, default=3, help="runs per variant (default 3)")
    parser.add_argument(
        "--python", default=sys.executable, help="interpreter used for the runs"
    )
    parser.add_argument(
        "--stop-first",
        action="store_true",
        help="stop an already running daemon before measuring (recommended)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    common = ["--prompt", args.prompt]
    if args.model:
        common += ["--model", args.model]

    if args.stop_first:
        _run(args.python, ["--warm-stop"])

    print(f"prompt: {args.prompt!r}")
    print(f"runs  : {args.runs} per variant\n")

    failed: dict[str, int] = {"cold": 0, "warm": 0}
    first_tokens: dict[str, list[float]] = {"cold": [], "warm": []}

    def measure(label: str, variant: str, argv: Sequence[str]) -> float | None:
        seconds, code, err, first = _run(args.python, argv)
        if code != 0:
            failed[variant] += 1
            first_line = err.splitlines()[0] if err else f"exit code {code}"
            print(f"  {label}: failed — {first_line}")
            return None
        if first is not None:
            first_tokens[variant].append(first)
            print(f"  {label}: {seconds:.2f}s (first token {first:.2f}s)")
        else:
            print(f"  {label}: {seconds:.2f}s")
        return seconds

    cold: list[float] = []
    for index in range(args.runs):
        _run(args.python, ["--warm-stop"])
        seconds = measure(f"cold run {index + 1}", "cold", [*common, "--no-warm"])
        if seconds is not None:
            cold.append(seconds)

    first_warm = measure("first warm run (daemon start)", "warm", common)
    warm: list[float] = []
    for index in range(args.runs):
        seconds = measure(f"warm run {index + 1}", "warm", common)
        if seconds is not None:
            warm.append(seconds)

    _run(args.python, ["--warm-stop"])

    if not cold and not warm:
        print(
            "\nNo run succeeded. Start with `arena-api --status` to check the "
            "session, then retry.",
            file=sys.stderr,
        )
        return 2

    print()
    print(f"{'variant':<26}{'runs (s)':<32}{'median':>10}{'failed':>8}")
    print("-" * 76)
    print(_format_runs("cold (--no-warm)", cold), f"{failed['cold']:>8}")
    print(_format_runs("warm (daemon reused)", warm), f"{failed['warm']:>8}")
    if first_warm is not None:
        print(f"{'first warm run (start)':<26}{'':<32}{_label(first_warm):>10}")
    cold_first = first_tokens["cold"]
    warm_first = first_tokens["warm"]
    if cold_first or warm_first:
        print()
        print("time to first token (from the ARENA_PROFILE=json timeline)")
        print(f"{'variant':<26}{'runs (s)':<32}{'median':>10}")
        print("-" * 68)
        print(_format_runs("cold (--no-warm)", cold_first))
        print(_format_runs("warm (daemon reused)", warm_first))
        if cold_first and warm_first:
            print(
                "speed-up on first token "
                f": {statistics.median(cold_first) / statistics.median(warm_first):.1f}x"
            )
    if cold and warm:
        print(f"\nspeed-up on repeated prompts: {statistics.median(cold) / statistics.median(warm):.1f}x")
        if first_warm:
            print(f"first call vs cold          : {statistics.median(cold) / first_warm:.2f}x")
    elif not cold:
        print("\nThe cold side did not produce a measurement, so there is no ratio.")
    elif not warm:
        print("\nThe warm side did not produce a measurement, so there is no ratio.")
    print(
        "\nReminder: both variants include the model's own generation time; "
        "only the pre-answer overhead is removed,\nso the factor grows with "
        "shorter answers and shrinks on a slow network."
    )
    return 0 if cold and warm else 1


if __name__ == "__main__":
    raise SystemExit(main())
