"""Console entry point, with a fast route for one-shot Direct Chat prompts.

``arena-api --prompt "..."`` is by far the most common invocation, and starting
a browser for it is the slow part.  When the command line is exactly that, the
:mod:`arena_api.warm` daemon answers it over loopback and this module never
imports the heavy bootstrap (:mod:`arena_api.auto` pulls in ``httpx``, the
session machinery and the Playwright bridge, which costs ~150 ms before any
work starts).

Everything else — ``--login``, ``--status``, ``--models``, ``--diagnose``,
``--help``, an explicit ``--verification-token``, ``--no-warm`` — falls through
to :func:`arena_api.auto.main` unchanged, so the CLI behaves exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Sequence

from . import latency

#: Anything that makes the command line something other than a single prompt.
_DISQUALIFYING = frozenset(
    {
        "-h",
        "--help",
        "--version",
        "--login",
        "--fresh",
        "--headless",
        "--diagnose",
        "--json",
        "--explain",
        "--status",
        "--models",
    }
)


def _wants_warm_prompt(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    if any(item in _DISQUALIFYING for item in argv):
        return False
    return any(item == "--prompt" or item.startswith("--prompt=") for item in argv)


def _package_version() -> str:
    """The installed distribution version, or ``"unknown"`` for a loose tree."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return "unknown"
    try:
        return version("arena-api")
    except PackageNotFoundError:
        return "unknown"


def _has_anthropic_shim() -> bool:
    """Whether this build can serve ``--claude`` (i.e. is not stale)."""
    from importlib.util import find_spec

    try:
        return find_spec("arena_api.compat.anthropic_server") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def version_report() -> list[str]:
    """Where this build lives and what it can do.

    Printed by ``--version``: a stale install looks exactly like a code bug --
    a new flag comes back as "unrecognized arguments" -- so the first question
    is always *which* copy of the package answered.
    """
    import platform
    import sys

    import arena_api

    module_dir = str(Path(str(arena_api.__file__)).parent)
    shim = (
        "yes (arena-api --claude)"
        if _has_anthropic_shim()
        else "no -- this build predates --claude, upgrade it"
    )
    try:
        from .auto import AutoOptions

        sessions = (
            "yes (--session firefox, firefox:<profile>, storage, chrome, auto)"
            if "session_source" in getattr(AutoOptions, "__dataclass_fields__", {})
            else "NOT supported, upgrade the package"
        )
    except ModuleNotFoundError as exc:
        # Not a broken build but an environment without the extras: name the
        # missing package and the command that installs it.
        sessions = (
            f"cannot import arena_api.auto: {exc} "
            '(install the dependencies: pip install -e ".[auto]")'
        )
    except Exception as exc:  # noqa: BLE001 - a broken install is the answer
        sessions = f"cannot import arena_api.auto: {exc}"
    return [
        f"arena-api {_package_version()}",
        f"module       : {module_dir}",
        f"python       : {sys.executable} ({platform.python_version()})",
        f"anthropic shim: {shim}",
        f"sessions     : {sessions}",
    ]


def _wants_claude_server(args: Sequence[str]) -> bool:
    """``--claude`` (or a leading ``claude``) starts the Anthropic shim."""
    if "--claude" in args or "--serve-anthropic" in args:
        return True
    return bool(args) and args[0] == "claude"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; ``argv`` defaults to ``sys.argv[1:]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    latency.mark("cli.entry")
    if "--version" in args:
        print("\n".join(version_report()))
        return 0
    if any(
        flag in args or any(item.startswith(f"{flag}=") for item in args)
        for flag in ("--warm-stop", "--warm-status", "--warm-start", "--warm-profile")
    ):
        from .warm import profile_main, start_main, status_main, stop_main

        if "--warm-stop" in args:
            return stop_main(args)
        if "--warm-status" in args:
            return status_main(args)
        if "--warm-start" in args:
            return start_main(args)
        return profile_main(args)
    if _wants_claude_server(args):
        from .compat import main as claude_main

        return claude_main(args)
    if _wants_warm_prompt(args):
        from .warm import fast_prompt_main

        code = fast_prompt_main(args)
        if code is not None:
            return code
    from .auto import main as auto_main

    return auto_main(args)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
