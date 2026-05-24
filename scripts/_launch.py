"""Shared launcher used by every wrapper in scripts/.

Adds the repo root and the vendored ``interphyre/`` source dir to PYTHONPATH
so that the wrappers work even before ``pip install ./interphyre`` has been
run (good for ``--help`` and import sanity checks). Forwards argv unchanged
to the underlying module's ``__main__``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import NoReturn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_module(module: str, extra_argv: list[str] | None = None) -> NoReturn:
    """Run ``python -m <module>`` with the wrapper's argv forwarded."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (ROOT, os.path.join(ROOT, "interphyre"), env.get("PYTHONPATH", "")) if p
    )
    argv = list(sys.argv[1:]) + (extra_argv or [])
    sys.exit(subprocess.run(
        [sys.executable, "-m", module, *argv],
        cwd=ROOT,
        env=env,
    ).returncode)
