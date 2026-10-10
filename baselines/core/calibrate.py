"""Calibrate MemRL's similarity constants directly on a CORE task's train pool.

Same statistic as `baselines/calibrate.py` (pairwise cosine similarity of
problem embeddings; delta = the top-20%-quantile threshold the paper's appendix
specifies) but keyed on `Problem.problem_text` rather than InterPhyre's
`IntentCache`. CORE problems already carry per-item discriminating text -- a
distinct word problem, grid, or disk state per item -- so unlike InterPhyre's
templated scenes (`baselines/memory/features.py`'s whole reason for existing),
no numeric-geometry fallback is needed here: measured below, ZebraLogic and
MathGAP intents are not near-degenerate the way InterPhyre's are.

Run automatically by `scripts/run_core_memory_baseline.py --method memrl` when
`--delta` etc. are not all given explicitly; also runnable standalone to inspect
the similarity distribution before committing to a run.
"""

from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path
from typing import Sequence

from baselines.calibrate import quantile
from baselines.memory.embedder import DEFAULT_MODEL, Embedder
from baselines.memory.store import cosine


def calibration_path(cache_dir: Path, task_name: str, embedding_model: str) -> Path:
    slug = embedding_model.replace("/", "__").replace(":", "_")
    return Path(cache_dir) / f"calibration_core_{task_name}_{slug}.json"


def calibrate_core_task(
    task_name: str,
    texts: Sequence[str],
    *,
    embedding_model: str = DEFAULT_MODEL,
    cache_dir: Path,
    embedder: Embedder | None = None,
) -> dict:
    """Fit (delta, sim_norm_mean, sim_norm_std) on this task's `problem_text` pool.

    ``embedder`` may be passed in to reuse one already constructed (and its warm
    cache) for the run's own retrieval; otherwise one is built here.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    if len(texts) < 2:
        raise ValueError(f"{task_name}: need >=2 train items to calibrate, got {len(texts)}")

    embedder = embedder or Embedder(embedding_model, cache_dir=cache_dir)
    vectors = embedder.embed(texts)

    sims = [
        cosine(vectors[i], vectors[j])
        for i, j in itertools.combinations(range(len(vectors)), 2)
    ]
    sims_sorted = sorted(sims)
    mean = statistics.fmean(sims)
    std = statistics.pstdev(sims) if len(sims) > 1 else 1.0
    delta = quantile(sims_sorted, 0.80)

    result = {
        "task": task_name,
        "embedding_model": embedder.name,
        "num_items": len(texts),
        "num_pairs": len(sims),
        "sim_mean": mean,
        "sim_std": std,
        "sim_min": sims_sorted[0],
        "sim_max": sims_sorted[-1],
        "recommended": {"delta": delta, "sim_norm_mean": mean, "sim_norm_std": std},
    }
    path = calibration_path(cache_dir, task_name, embedder.name)
    path.write_text(json.dumps(result, indent=2))
    print(
        f"  [calibrate] {task_name}: delta={delta:.4f} sim_mean={mean:.4f} "
        f"sim_std={std:.4f} (n={len(texts)} items, {len(sims)} pairs) -> {path}"
    )
    if std < 0.01:
        print(
            f"  [calibrate] WARNING: {task_name} similarities are nearly degenerate "
            f"(std < 0.01) -- Phase A will admit all-or-nothing."
        )
    return result
