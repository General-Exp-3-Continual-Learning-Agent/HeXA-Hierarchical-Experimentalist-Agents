"""Send the teacher prompt on stdin instead of argv.

`skillrl.distillation.distill.call_teacher` invokes the Claude CLI as
`claude -p <prompt> ...`, passing the whole prompt as a single command-line
ARGUMENT. Linux caps one argument at MAX_ARG_STRLEN = 128 KiB (independently of
the much larger ARG_MAX), so a prompt past that fails before the process starts:

    OSError: [Errno 7] Argument list too long: 'claude'

That is not a corner case here. Round-1 contrastive distillation sends 5
successes + 5 failures, and a DiscoverPhysics trajectory carries real numeric
observations (up to 1400 chars per step, a dozen steps), so the measured
round-1 prompt for the pooled global_term bootstrap is **139,023 bytes** --
over the limit with 10 trajectories, which is simply the default configuration.
InterPhyre trajectories are far smaller, which is why this never surfaced there.

The CLI reads the prompt from stdin when no prompt argument is given, which has
no size limit worth worrying about (verified live at 150 KB). This module swaps
in a `call_teacher` that does exactly that and is otherwise byte-for-byte the
same contract: same signature, same return of `""` on failure, so every caller's
error handling (including `evolve_skill_bank`'s empty-bank guard) is unchanged.

`patch()` must rebind BOTH modules: `evolving_distill` does
`from skillrl.distillation.distill import call_teacher` -- by value, at import --
so rebinding only `distill` leaves the evolve path on the argv version, which is
the path with the *larger* prompt (it sends the current bank as well).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Optional

#: Single-argument limit on Linux (MAX_ARG_STRLEN). Prompts above this size are
#: the reason this module exists; logged when crossed so a run says out loud that
#: it would have died under the stock implementation.
ARGV_LIMIT = 128 * 1024

DEFAULT_TIMEOUT = 1800


def call_teacher(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Drop-in replacement for `distill.call_teacher`, prompt passed on stdin.

    Returns "" on any failure, matching the stock contract — callers treat an
    empty string as "the teacher produced nothing" and (since the evolve guard)
    keep the previous bank rather than wiping it.
    """
    if not model.startswith("claude-"):
        # Same rule cross_world_loop enforces up front; repeated here because
        # this function is also reachable directly.
        raise ValueError(
            f"teacher model {model!r} must start with 'claude-' to route "
            "through the Claude CLI"
        )

    size = len(prompt.encode("utf-8", errors="ignore"))
    if size > ARGV_LIMIT:
        print(f"  [teacher] prompt {size:,} bytes (> {ARGV_LIMIT:,} argv limit) "
              f"— sending on stdin")

    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--no-session-persistence",
    ]
    start = time.time()
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            env=os.environ.copy(), timeout=timeout,
        )
    except FileNotFoundError:
        print("  [teacher] 'claude' CLI not found")
        return ""
    except subprocess.TimeoutExpired:
        print(f"  [teacher] timeout after {time.time() - start:.0f}s")
        return ""

    if result.returncode != 0:
        print(f"  [teacher] exit {result.returncode}: {(result.stderr or '')[:300]}")
        return ""

    try:
        return json.loads(result.stdout).get("result", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return (result.stdout or "").strip()


def patch() -> None:
    """Install the stdin-based teacher into distill + evolving_distill."""
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving_distill

    _distill.call_teacher = call_teacher
    # Imported by value at module load — needs its own rebind, and it is the
    # path with the bigger prompt (trajectories AND the current bank).
    _evolving_distill.call_teacher = call_teacher

    # Cross-level synthesis does `from ...distill import call_teacher` too, and
    # its prompt carries whole source banks plus scene blocks — squarely over the
    # argv limit. Imported lazily so this module stays usable without it.
    try:
        import skillrl.distillation.cross_level_synthesis as _xl
        _xl.call_teacher = call_teacher
    except ImportError:
        pass
