"""Locate and import the CORE (`core-reasoning`) codebase in place.

CORE is imported, never forked or vendored. Its verifiers, generators, prompt
templates and inference client are the single source of truth for anything that
a HExA-vs-CORE comparison depends on.

CORE's ``src/`` uses a mix of package-relative imports (``from ..base import``)
and flat absolute ones (``from reflection.single_turn_prompt_templates import``),
so ``<core_repo>/src`` has to be on ``sys.path`` and the task modules imported as
``tasks.single_turn.<module>``.

Resolution order for the repo location:
  1. ``$CORE_REPO`` — if set, it is *authoritative*: a value that does not look
     like a CORE checkout raises rather than falling back.
  2. sibling directory ``../core-reasoning`` relative to this repo
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

_THIS_REPO = Path(__file__).resolve().parent.parent
_SIBLING_DEFAULT = _THIS_REPO.parent / "core-reasoning"

# Files/dirs every usable CORE checkout must have. Checking more than one guards
# against the failure mode that actually happens: $CORE_REPO pointing at a stale
# or partial clone, which would otherwise surface much later as an ImportError
# from inside a task module.
_REQUIRED_MARKERS = (
    Path("src") / "tasks" / "single_turn",
    Path("src") / "reflection" / "single_turn_prompt_templates.py",
    Path("scripts") / "run_core.py",
)


class CoreRepoNotFound(RuntimeError):
    pass


def _missing_markers(root: Path) -> list[str]:
    return [str(m) for m in _REQUIRED_MARKERS if not (root / m).exists()]


def _reject_reason(candidate: Path) -> str | None:
    """Why *candidate* is not a usable CORE checkout, or ``None`` if it is."""
    if not candidate.is_dir():
        return "no such directory"
    missing = _missing_markers(candidate)
    if missing:
        return f"exists but is missing {', '.join(missing)}"
    return None


@lru_cache(maxsize=1)
def core_repo_root() -> Path:
    """Return the CORE repo root, verifying it actually looks like CORE.

    An explicitly set ``$CORE_REPO`` is authoritative and is never silently
    replaced by the sibling checkout. Setting it is an instruction about *which*
    CORE the numbers are measured against; falling back on a typo'd or stale
    value would resolve to a different tree and write the wrong
    :func:`core_git_revision` into every run manifest — provenance corrupted with
    no visible symptom. A wrong ``$CORE_REPO`` therefore fails here, loudly.
    """
    env = os.environ.get("CORE_REPO")
    if env:
        candidate = Path(env).expanduser().resolve()
        reason = _reject_reason(candidate)
        if reason is None:
            return candidate
        raise CoreRepoNotFound(
            f"$CORE_REPO is set to {env!r} (resolved to {candidate}), which is not a "
            f"usable core-reasoning checkout: {reason}.\n  "
            "Refusing to fall back to another checkout, because $CORE_REPO is how a "
            "run declares which CORE it was measured against. Fix or unset it "
            f"(unset falls back to {_SIBLING_DEFAULT})."
        )

    reason = _reject_reason(_SIBLING_DEFAULT)
    if reason is None:
        return _SIBLING_DEFAULT

    raise CoreRepoNotFound(
        "Could not locate a usable core-reasoning checkout. Set $CORE_REPO to its "
        f"root, or place it at {_SIBLING_DEFAULT}.\n  "
        f"{_SIBLING_DEFAULT}: {reason}"
    )


@lru_cache(maxsize=1)
def install_core_on_path() -> Path:
    """Put ``<core_repo>/src`` on ``sys.path`` (idempotent). Returns the repo root.

    Also sets ``sys.dont_write_bytecode``. CORE is imported in place out of a repo
    we are forbidden to modify, and importing it writes ``.pyc`` into
    ``core-reasoning/src/**/__pycache__/``. Those files are gitignored *there*
    (``core-reasoning/.gitignore:1``), so ``git -C ../core-reasoning status`` reports
    a clean tree either way -- the check every stage of this project used as
    evidence is a false negative. Setting the flag here enforces in code what was
    previously only a ``PYTHONDONTWRITEBYTECODE=1`` convention in test docstrings,
    which nobody running ``scripts/run_repro_gate.py`` would have known to set.

    The flag is process-global and costs a recompile per import; that is the right
    trade against mutating a read-only dependency.
    """
    root = core_repo_root()
    sys.dont_write_bytecode = True
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def core_scripts_dir() -> Path:
    """Path to CORE's ``scripts/`` — needed to import ``run_core`` internals."""
    return core_repo_root() / "scripts"


def core_git_revision() -> str | None:
    """Best-effort CORE git SHA, recorded in run manifests for provenance.

    Never raises: provenance must not be able to break a run.
    """
    root = core_repo_root()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    # Fallback for environments without a git binary. Handles the common
    # "ref: refs/heads/main" indirection but not packed-refs.
    head = core_repo_root() / ".git" / "HEAD"
    try:
        content = head.read_text().strip()
    except OSError:
        return None
    if content.startswith("ref: "):
        ref = (core_repo_root() / ".git" / content[5:]).resolve()
        try:
            return ref.read_text().strip()
        except OSError:
            return None
    return content


def core_git_dirty() -> bool | None:
    """Whether the CORE checkout has uncommitted changes. ``None`` if unknowable.

    Recorded next to :func:`core_git_revision` because a SHA alone cannot support
    the claim the whole comparison rests on -- that CORE was run *unmodified*. A
    manifest showing a clean-looking revision while the working tree carries edits
    is precisely the provenance failure that would invalidate the numbers, so the
    two must be read together.

    ``-uall`` so an untracked file counts: a stray module shadowing a CORE import
    changes behaviour without touching a tracked file.
    """
    try:
        root = core_repo_root()
    except CoreRepoNotFound:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-uall"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def core_provenance() -> dict[str, object]:
    """Which CORE checkout produced this run's tasks, and was it modified.

    Every claim in this project rests on CORE supplying the splits, prompts and
    verifiers *unmodified*; a revision alone cannot support that, because a dirty tree
    at the same revision is a different experiment. Both travel together.

    This lived in ``logging_schema.RunManifest`` until that module was archived with
    the rest of the CORE-parity machinery. The manifest was for emitting CORE's own
    ``validation.jsonl`` so their plotting code could read our runs -- which we no
    longer need -- but the provenance it carried is not CORE-comparison-specific and
    belongs with the git helpers it calls.
    """
    return {
        "core_git_revision": core_git_revision(),
        "core_git_dirty": core_git_dirty(),
        "core_repo_root": str(core_repo_root()),
    }
