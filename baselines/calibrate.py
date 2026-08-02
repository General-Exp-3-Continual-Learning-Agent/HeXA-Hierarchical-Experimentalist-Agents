"""Calibrate MemRL's similarity constants for a level. Run this FIRST.

MemRL's Phase-A threshold is dataset-dependent.  From the paper's appendix:

    "The similarity threshold delta is adaptive to the dataset density;
    specifically, we determine delta by calculating the pairwise cosine
    similarity distribution of task descriptions within each benchmark and
    selecting the threshold at the top 20% quantile."

and its similarity z-normalization uses precomputed dataset statistics
(`memory.sim_norm_mean` / `sim_norm_std` in the official configs), not per-pool
statistics.  All three constants therefore have to be measured on InterPhyre
intents rather than copied from the ALFWorld config.

This also answers the design question the port hinges on: if within-level
similarities are near-degenerate (everything ~1.0), Phase A admits everything
or nothing and the intent text needs to carry more discriminating geometry.
The printed spread tells you which regime you are in.

Usage:
    python -m baselines.calibrate --level pass_the_parcel --seeds 1-105
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from baselines.memory.embedder import DEFAULT_MODEL, Embedder
from baselines.memory.features import (
    FeatureScaler,
    NumericEmbedder,
    calibration_path,
    parse_features,
    rank_agreement,
    scaler_path,
)
from baselines.memory.intent import DEFAULT_MODE, MODES, IntentCache, cache_path
from baselines.memory.store import cosine


def parse_seed_spec(spec: str) -> list[int]:
    """Parse "1-105" or "1,2,5-9" into a list of seeds."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def calibrate(
    level: str,
    seeds: list[int],
    *,
    embedding_model: str = DEFAULT_MODEL,
    cache_dir: Path,
    intent_mode: str = DEFAULT_MODE,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    intents = IntentCache(cache_path(cache_dir, level, intent_mode), intent_mode)
    embedder = (
        None
        if embedding_model == NumericEmbedder.name
        else Embedder(embedding_model, cache_dir=cache_dir)
    )

    texts = []
    kept_seeds = []
    for seed in seeds:
        try:
            texts.append(intents.get(level, seed).text)
            kept_seeds.append(seed)
        except Exception as exc:                    # unsolvable/absent seed
            print(f"  [warn] seed {seed} skipped: {type(exc).__name__}: {exc}")
    if len(kept_seeds) < 2:
        raise SystemExit(f"Need >=2 usable seeds for {level}, got {len(kept_seeds)}")

    # Numeric geometry keys, fitted on this seed pool. Always computed: they are
    # the ground truth for the rank-agreement diagnostic below, and they are the
    # retrieval key itself when --embedding-model numeric is used.
    numeric_vectors: list[list[float]] = []
    scaler = None
    if intent_mode == "geometry":
        raw = [parse_features(t) for t in texts]
        try:
            scaler = FeatureScaler.fit(raw, level=level, intent_mode=intent_mode)
            numeric_vectors = [scaler.transform(v) for v in raw]
            scaler.save(scaler_path(cache_dir, level, intent_mode))
            print(
                f"  Numeric key: {scaler.dim} features/seed, scaler written to "
                f"{scaler_path(cache_dir, level, intent_mode)}"
            )
        except ValueError as exc:
            print(f"  [warn] numeric key unavailable: {exc}")

    if embedding_model == NumericEmbedder.name:
        if scaler is None:
            raise SystemExit(
                "--embedding-model numeric requires --intent-mode geometry and a "
                "consistent object set across seeds."
            )
        print(f"  Keying on the numeric geometry ({scaler.dim} dims) ...")
        vectors = numeric_vectors
    else:
        print(f"  Embedding {len(texts)} intents with {embedding_model} ...")
        vectors = embedder.embed(texts)

    sims = [
        cosine(vectors[i], vectors[j])
        for i, j in itertools.combinations(range(len(vectors)), 2)
    ]
    sims_sorted = sorted(sims)

    mean = statistics.fmean(sims)
    # Population stdev, matching MemRL's use of pstdev for its norm statistics.
    std = statistics.pstdev(sims) if len(sims) > 1 else 1.0
    # "top 20% quantile" = the value only the most similar 20% of pairs exceed.
    delta = quantile(sims_sorted, 0.80)

    result = {
        "level": level,
        "embedding_model": embedding_model,
        "intent_mode": intent_mode,
        "num_seeds": len(kept_seeds),
        "seeds": kept_seeds,
        "num_pairs": len(sims),
        "intent_chars_mean": statistics.fmean(len(t) for t in texts),
        "sim_mean": mean,
        "sim_std": std,
        "sim_min": sims_sorted[0],
        "sim_max": sims_sorted[-1],
        "quantiles": {
            f"p{int(q * 100)}": quantile(sims_sorted, q)
            for q in (0.05, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95)
        },
        "recommended": {
            "delta": delta,
            "sim_norm_mean": mean,
            "sim_norm_std": std,
        },
    }

    # Does this key rank instances the way the geometry does? Only meaningful
    # when a numeric ground truth exists and is not the key itself.
    if numeric_vectors and embedding_model != NumericEmbedder.name:
        result["rank_agreement_vs_geometry"] = rank_agreement(
            vectors, numeric_vectors
        )

    out_path = calibration_path(cache_dir, level, intent_mode, embedding_model)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n  Pairwise cosine similarity over {len(sims)} pairs:")
    print(f"    mean {mean:.4f}   std {std:.4f}   "
          f"min {sims_sorted[0]:.4f}   max {sims_sorted[-1]:.4f}")
    for name, value in result["quantiles"].items():
        print(f"    {name}: {value:.4f}")
    print(f"\n  Recommended MemRL constants for {level} (intent mode: {intent_mode}):")
    print(f"    --delta {delta:.4f}")
    print(f"    --sim-norm-mean {mean:.4f}")
    print(f"    --sim-norm-std {std:.4f}")
    if std < 0.01:
        hint = (
            "      Try --intent-mode geometry, which drops the constant prose "
            "and keeps only per-seed geometry."
            if intent_mode == "full"
            else "      Even geometry-only intents do not discriminate here; a "
            "numeric retrieval key is needed rather than an embedded string."
        )
        print(
            "\n  [!] Similarities are nearly degenerate (std < 0.01): Phase A "
            f"will admit all-or-nothing.\n{hint}"
        )
    agreement = result.get("rank_agreement_vs_geometry")
    if agreement:
        rho = agreement.get("spearman", float("nan"))
        print("\n  Rank agreement with the numeric geometry:")
        print(f"    spearman rho = {rho:+.3f}")
        for name, value in (agreement.get("top_k_overlap") or {}).items():
            print(f"    {name} retrieval overlap = {value:.2%}")
        if rho == rho and abs(rho) < 0.3:
            print(
                "\n  [!] This key barely tracks scene similarity, so Phase A "
                "would retrieve near-arbitrary memories.\n      Prefer "
                "--embedding-model numeric (keys on the standardized geometry), "
                "and re-run this calibration with it."
            )

    print(f"\n  Written to {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate MemRL similarity constants on InterPhyre intents"
    )
    parser.add_argument("--level", required=True)
    parser.add_argument(
        "--seeds", default="1-105",
        help='Seed spec, e.g. "1-105" or "1,2,6-55" (default: 1-105)',
    )
    parser.add_argument(
        "--embedding-model", default=DEFAULT_MODEL,
        help="Text embedder, or 'numeric' to key on the standardized geometry "
             "parsed from a geometry-mode intent",
    )
    parser.add_argument(
        "--intent-mode", default=DEFAULT_MODE, choices=MODES,
        help="geometry = per-seed geometry only (default); full = level prose "
             "plus the whole scene dump (measured degenerate within a level)",
    )
    parser.add_argument(
        "--cache-dir", default="baselines/data/cache",
        help="Where intent/embedding caches and the calibration JSON go",
    )
    args = parser.parse_args()

    calibrate(
        args.level,
        parse_seed_spec(args.seeds),
        embedding_model=args.embedding_model,
        cache_dir=Path(args.cache_dir),
        intent_mode=args.intent_mode,
    )


if __name__ == "__main__":
    main()
