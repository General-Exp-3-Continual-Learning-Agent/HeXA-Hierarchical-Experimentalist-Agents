"""Play one DiscoverPhysics episode with a retrieved-memory block injected.

The analogue of `baselines/orbit/rollout.py`.  Only the text of the injected
block differs between arms; the system prompt, the world, the noise seed, the
round budget, the actor and the judge are the no-bank baseline's.

    HeXA           the distilled skill bank
    Episodic RAG   the nearest past episode transcripts, verbatim
    no bank        nothing

The injection point is `discovery_runner.run_episode(prompt_block=...)`, which
appends to `agent._system` and nowhere else -- byte-for-byte the seam the skill
bank uses, so a memory block costs context exactly where a bank does.

`run_episode` already returns a HeXA-schema trajectory dict carrying `success`,
`reward`, `iterations` and a `trajectory` of thought/action/observation steps,
which is the shape `baselines/memory/rag.py` consumes.  What this module adds
is the two dict-first fields that module would otherwise compute the InterPhyre
way (`simulator_calls`, `verifier_feedback`) plus the accounting a memory run
logs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

_DISCOVERY = Path(__file__).resolve().parents[2] / "discovery-agents"
if str(_DISCOVERY) not in sys.path:
    sys.path.insert(0, str(_DISCOVERY))

from baselines.discovery.blocks import count_tokens

_PATCHED = False


@contextmanager
def _cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def ensure_cli(cli_model: str) -> None:
    """Route actor and judge through the Claude Code CLI, once per process.

    `cli_client.patch_llm_client` is idempotent but NOT re-parameterisable: the
    first override wins for the lifetime of the process (see the note in
    `hexa_discovery/cli_client.py`). One arm per process, therefore, and the
    flag here makes the single call explicit rather than incidental.
    """
    global _PATCHED
    if _PATCHED:
        return
    from hexa_discovery.cli_client import patch_llm_client

    patch_llm_client(model_override=cli_model)
    _PATCHED = True


def verifier_feedback(result: dict) -> str:
    """The evaluator's verdict on one episode, as a retrieved memory shows it.

    Observable-to-the-agent facts plus the two graded numbers HeXA also lets its
    teacher see: `norm_mse` and the explanation score.

    What is excluded is `expl_reasoning`, the judge's critique. The judge is
    shown the world's ground-truth explanation, so the critique states what the
    agent failed to discover -- on oscillator it named "a cos(wt) with the right
    w~pi/2" outright. HeXA's own pre-registration closed that channel to the
    teacher; leaving it open to the actor would be strictly worse, because the
    actor could act on it in the episode it is solving.
    """
    nm = result.get("norm_mse")
    nm_text = f"{nm:.3g}" if isinstance(nm, (int, float)) else "n/a"
    return "\n".join(
        [
            f"Evaluator: {'PASS' if result.get('success') else 'FAIL'} "
            f"(needs normalised MSE < 0.1 AND explanation score >= 0.75)",
            f"Normalised trajectory MSE: {nm_text}",
            f"Explanation score: {result.get('expl_score')}",
            f"Shaped reward: {float(result.get('reward') or 0.0):+.2f}",
            f"Experiment rounds used: {result.get('iterations', '?')} "
            f"of {result.get('max_rounds', '?')}",
        ]
    )


def play_and_record(
    world: str,
    seed: int,
    memory_block: Optional[str],
    *,
    out_path: Optional[Path] = None,
    retrieval_diagnostics: Optional[dict] = None,
    **episode_kwargs,
) -> Optional[dict]:
    """Run one episode and return the result dict. `None` means it errored.

    A transport or executor failure is recorded and excluded downstream rather
    than folded in as a failed episode -- `run_batch_discovery`'s convention,
    and the right one: it is evidence about the harness, not about the agent.

    An existing `out_path` is reused, so an interrupted arm resumes from disk
    the way every other DiscoverPhysics run does.
    """
    if out_path is not None and out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass  # corrupted -- re-run

    from hexa_discovery import _paths
    from hexa_discovery.discovery_runner import run_episode

    started = time.perf_counter()
    try:
        with _cwd(_paths.DISCOVERY_ROOT):
            result = run_episode(
                world, seed, None, prompt_block=memory_block or None,
                **episode_kwargs
            )
    except Exception as exc:                       # noqa: BLE001 -- see docstring
        print(f"  seed {seed}: ERRORED {type(exc).__name__}: {str(exc)[:200]}")
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.with_suffix(".error.json").write_text(
                json.dumps({"world": world, "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}"}, indent=2)
            )
        return None

    block = memory_block or ""
    result["simulator_calls"] = int(result.get("iterations") or 0)
    result["verifier_feedback"] = verifier_feedback(result)
    result["added_context_tokens"] = count_tokens(block)
    result["added_context_chars"] = len(block)
    result["elapsed_time"] = time.perf_counter() - started
    result["retrieval_diagnostics"] = retrieval_diagnostics or {}
    result["memory_block"] = block

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
    return result
