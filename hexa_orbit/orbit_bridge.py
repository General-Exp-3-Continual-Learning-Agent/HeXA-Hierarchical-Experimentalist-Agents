"""Locate and import ORBIT's environment code in place.

ORBIT is imported, never forked or vendored: its maze generator, transition
rules, termination conditions and episode/success semantics are the single
source of truth for anything a HExA-vs-ORBIT comparison depends on, and a
reimplemented wall-collision or shortest-path filter would silently invalidate
every number (RECON §7, and the plan's Phase 1 constraint).

ORBIT's modules use flat absolute imports rooted at the repo top level
(``from envs.maze_env_adapter import MazeEnvAdapter``, ``from prompts.prompt
import reflection_prompt_2``), so the repo *root* goes on ``sys.path`` -- not a
``src/`` subdirectory as with CORE.

Resolution order for the repo location:
  1. ``$ORBIT_REPO`` -- if set, it is *authoritative*: a value that does not look
     like an ORBIT checkout raises rather than falling back, so a stale export
     surfaces here instead of as a confusing ImportError later.
  2. sibling directory ``../ORBIT`` relative to this repo.

Importing ORBIT requires the rLLM shim (:mod:`hexa_orbit.rllm_shim`, D4), which
:func:`install_orbit_on_path` installs first. Everything is cached, so repeated
calls are free and the recorded git revision is stable within a process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path

from . import rllm_shim

_THIS_REPO = Path(__file__).resolve().parent.parent
_SIBLING_DEFAULT = _THIS_REPO.parent / "ORBIT"

# Files every usable ORBIT checkout must have. Checking several guards against
# the failure that actually happens: $ORBIT_REPO pointing at a partial clone.
# Note none of these live under third_party/ -- the submodules are empty in a
# fresh clone (RECON §0a) and Design B does not need them, so requiring them
# would reject a perfectly usable checkout.
_REQUIRED_MARKERS = (
    Path("envs") / "maze_env_adapter.py",
    Path("envs") / "multi_episode_env.py",
    Path("configs") / "eval_config.yaml",
    Path("prompts") / "prompt.py",
)


class OrbitRepoNotFound(RuntimeError):
    pass


def _missing_markers(root: Path) -> list[str]:
    return [str(m) for m in _REQUIRED_MARKERS if not (root / m).exists()]


def _reject_reason(candidate: Path) -> str | None:
    """Why *candidate* is not a usable ORBIT checkout, or ``None`` if it is."""
    if not candidate.is_dir():
        return "no such directory"
    missing = _missing_markers(candidate)
    if missing:
        return f"exists but is missing {', '.join(missing)}"
    return None


@lru_cache(maxsize=1)
def orbit_repo_root() -> Path:
    """The ORBIT checkout to import from. Raises :class:`OrbitRepoNotFound`."""
    env_value = os.environ.get("ORBIT_REPO")
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        reason = _reject_reason(candidate)
        if reason is not None:
            raise OrbitRepoNotFound(
                f"$ORBIT_REPO={env_value!r} is not an ORBIT checkout: {reason}. "
                "Unset it to fall back to the sibling directory."
            )
        return candidate

    candidate = _SIBLING_DEFAULT.resolve()
    reason = _reject_reason(candidate)
    if reason is not None:
        raise OrbitRepoNotFound(
            f"No ORBIT checkout found. Tried {candidate} ({reason}). "
            "Set $ORBIT_REPO to the repository root."
        )
    return candidate


@lru_cache(maxsize=1)
def orbit_git_revision() -> str:
    """Short git revision of the ORBIT checkout, or ``"unknown"``.

    Recorded in run manifests: which ORBIT commit produced an instance pool is
    part of that pool's identity, since a change to the maze generator or the
    shortest-path filter would silently redraw every instance.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(orbit_repo_root()), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _register_envs_package(root: Path) -> None:
    """Register ``envs`` as a path-only package, skipping ORBIT's ``__init__``.

    ``ORBIT/envs/__init__.py`` eagerly re-exports every adapter, including
    ``GEMEnvAdapter`` -- whose module raises ``ImportError`` at import time when
    ``gem`` is absent (``envs/gem_env_adapter.py:15-20``), and ``gem`` lives in
    an uninitialised submodule (RECON §0a). Design B is Maze-only (D1), so that
    import is pure cost.

    Skipping it is safe because that ``__init__`` is nothing but re-exports plus
    a GEM-only Minesweeper registration -- no state, no side effects the maze
    path depends on. What matters is that the *submodules* stay importable under
    their real dotted names, because ORBIT imports them that way internally:
    ``multi_episode_env`` does ``from envs.maze_env_adapter import
    MazeEnvAdapter`` inside ``_is_maze_inner_env`` (:_is_maze_inner_env, :477)
    under a bare ``except Exception: return False``. If that import failed we
    would lose ``maze/unique_visited_states`` -- the exploration signal the
    reward mapping wants (RECON §5) -- **silently**. Giving the package a
    ``__path__`` keeps it working; ``hexa_orbit.tests.test_adapter`` asserts it.

    If a real ``envs`` module is already imported (someone ran from inside the
    ORBIT checkout), it is left alone.
    """
    if "envs" in sys.modules:
        return
    pkg = types.ModuleType("envs")
    pkg.__path__ = [str(root / "envs")]  # type: ignore[attr-defined]
    pkg.__doc__ = (
        "Path-only stand-in for ORBIT's envs package, installed by "
        "hexa_orbit.orbit_bridge to avoid its GEM-dependent __init__."
    )
    pkg.__orbit_init_skipped__ = True  # type: ignore[attr-defined]
    sys.modules["envs"] = pkg


@lru_cache(maxsize=1)
def install_orbit_on_path() -> str:
    """Put ORBIT's root on ``sys.path`` and install the rLLM shim.

    Returns the rLLM provenance string (``"shim"`` or ``"real"``) so callers can
    record it. Appends rather than prepends: ORBIT's top-level package names
    (``envs``, ``configs``, ``prompts``, ``scripts``, ``tests``) are generic
    enough to shadow something of ours, and this repo has a ``scripts/`` and a
    ``tests/`` of its own.
    """
    provenance = rllm_shim.install()
    root = orbit_repo_root()
    if str(root) not in sys.path:
        sys.path.append(str(root))
    _register_envs_package(root)
    return provenance


def manifest() -> dict[str, str]:
    """Provenance block for a run record."""
    return {
        "orbit_repo": str(orbit_repo_root()),
        "orbit_git_revision": orbit_git_revision(),
        "rllm": install_orbit_on_path(),
    }
