"""``python -m arena_api`` — the zero-configuration bootstrap CLI.

Thin wrapper around :func:`arena_api._cli.main`, which routes a one-shot
``--prompt`` through the warm daemon (see :mod:`arena_api.warm`) and hands every
other command line to :func:`arena_api.auto.main`.  ``python -m arena_api.auto``
keeps working for anyone who already typed it.
"""

from __future__ import annotations

from ._cli import main

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
