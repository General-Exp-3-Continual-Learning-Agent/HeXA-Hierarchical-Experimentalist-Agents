"""Numeric geometry keys, and the diagnostic that justifies using them.

Language-model embeddings of InterPhyre intents are nearly collinear: within a
level every intent is the same template with different numbers, so cosine lands
at 0.999 ± 0.0005 whether the scene is similar or not (measured on 105 seeds of
`catapult` and `pass_the_parcel`, in both intent modes).

A tiny spread is not by itself fatal — MemRL's delta is a *quantile* of that
distribution, and z-normalization rescales whatever variance exists — so the
question that actually matters is whether the embedding *ranking* tracks real
geometric similarity.  `rank_agreement` below answers it, using the numeric
geometry as ground truth.

If the answer is no, the honest fix is to retrieve on the geometry itself:
`--embedding-model numeric` keys memory on the standardized feature vector
parsed straight out of the geometry intent.  That is a documented deviation from
"embed the task description" — recorded in every run manifest — but it preserves
what Phase A is *for*: finding instances that genuinely resemble the current one.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Iterable, Optional, Sequence

#: Signed fixed-point numbers as rendered by `intent.py` in geometry mode.
_NUMBER = re.compile(r"[+-]\d+\.\d+")


def parse_features(text: str) -> list[float]:
    """Extract the feature vector from a geometry-mode intent.

    The geometry intent is a fixed-format rendering (same objects, same order,
    same fields for a given level), so the signed numbers in it *are* the
    feature vector — positions, extents, angles, and derived offsets.
    """
    return [float(m.group()) for m in _NUMBER.finditer(text)]


class FeatureScaler:
    """Per-dimension standardization fitted on a level's seed pool.

    Raw coordinates are not comparable dimension to dimension (an angle spans
    360, a radius spans 0.5), and cosine over unstandardized vectors is
    dominated by whichever dimension happens to be largest.  Standardizing first
    makes "similar scene" mean similar in every dimension equally.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float], *, level: str = "", intent_mode: str = ""):
        self.mean = list(mean)
        self.std = [s if s and s > 1e-9 else 1.0 for s in std]
        self.level = level
        self.intent_mode = intent_mode

    @property
    def dim(self) -> int:
        return len(self.mean)

    @classmethod
    def fit(
        cls,
        vectors: Sequence[Sequence[float]],
        *,
        level: str = "",
        intent_mode: str = "",
    ) -> "FeatureScaler":
        if not vectors:
            raise ValueError("Cannot fit a scaler on zero vectors")
        dims = {len(v) for v in vectors}
        if len(dims) != 1:
            raise ValueError(
                f"Inconsistent feature dimensionality across seeds: {sorted(dims)}. "
                "The level's object set must be identical across seeds for a "
                "numeric key to be well-defined."
            )
        n = dims.pop()
        mean = [statistics.fmean(v[i] for v in vectors) for i in range(n)]
        std = [
            statistics.pstdev([v[i] for v in vectors]) if len(vectors) > 1 else 1.0
            for i in range(n)
        ]
        return cls(mean, std, level=level, intent_mode=intent_mode)

    def transform(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.dim:
            raise ValueError(
                f"Feature vector has {len(vector)} dims, scaler expects {self.dim}. "
                "Re-run baselines.calibrate for this level and intent mode."
            )
        return [
            (value - mu) / sigma
            for value, mu, sigma in zip(vector, self.mean, self.std)
        ]

    # ── persistence ─────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "level": self.level,
                    "intent_mode": self.intent_mode,
                    "dim": self.dim,
                    "mean": self.mean,
                    "std": self.std,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        data = json.loads(Path(path).read_text())
        return cls(
            data["mean"],
            data["std"],
            level=data.get("level", ""),
            intent_mode=data.get("intent_mode", ""),
        )


def scaler_path(cache_dir: Path, level: str, intent_mode: str) -> Path:
    return Path(cache_dir) / f"features_{level}_{intent_mode}.json"


def key_slug(embedding_model: str) -> str:
    """Filesystem-safe tag for a retrieval key."""
    return embedding_model.replace("/", "__").replace(":", "-")


def calibration_path(
    cache_dir: Path, level: str, intent_mode: str, embedding_model: str
) -> Path:
    """Calibration is per (level, intent mode, key).

    All three change the similarity distribution, so they must not share a file —
    concurrent calibration jobs for different keys would otherwise overwrite each
    other and a run could silently load constants measured on a different key.
    """
    return (
        Path(cache_dir)
        / f"calibration_{level}_{intent_mode}_{key_slug(embedding_model)}.json"
    )


class NumericEmbedder:
    """Embedder-compatible numeric key: parse geometry, then standardize.

    Drop-in for `baselines.memory.embedder.Embedder` — same `embed` / `embed_one`
    surface, so the stores, retrievers and stream loop are unchanged.
    """

    name = "numeric"
    provider = "numeric"

    def __init__(self, scaler: FeatureScaler):
        self.scaler = scaler

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.scaler.transform(parse_features(text))


# ── diagnostics ─────────────────────────────────────────────────────────

def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, so ties do not bias the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation, without pulling in scipy."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(
        sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)
    )
    return num / den if den > 0 else float("nan")


def top_k_overlap(
    sims_a: Sequence[Sequence[float]],
    sims_b: Sequence[Sequence[float]],
    k: int,
) -> float:
    """Mean |top-k(a) ∩ top-k(b)| / k over rows — retrieval-level agreement.

    Spearman weighs the whole ranking; retrieval only ever looks at the head, so
    this reports the part MemRL's Phase A actually depends on.
    """
    if not sims_a or k <= 0:
        return float("nan")
    scores = []
    for row_a, row_b in zip(sims_a, sims_b):
        idx = range(len(row_a))
        top_a = set(sorted(idx, key=lambda i: row_a[i], reverse=True)[:k])
        top_b = set(sorted(idx, key=lambda i: row_b[i], reverse=True)[:k])
        scores.append(len(top_a & top_b) / float(k))
    return statistics.fmean(scores) if scores else float("nan")


def rank_agreement(
    embedding_vectors: Sequence[Sequence[float]],
    numeric_vectors: Sequence[Sequence[float]],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict:
    """Does the embedding rank instances the way the geometry does?

    Returns the Spearman correlation over all off-diagonal pairs plus top-k
    retrieval overlap.  Interpretation: near 0 means the embedding key is noise
    with respect to scene similarity and a numeric key is warranted; high values
    mean the embedding is a faithful stand-in despite its compressed range.

    Both sides are scored with **cosine**, the metric retrieval actually uses.
    Scoring the geometry side with Euclidean distance instead would compare two
    different metrics and understate agreement — identical vectors would score
    well below 1.
    """
    from .store import cosine

    n = len(embedding_vectors)
    if n < 3 or n != len(numeric_vectors):
        return {"spearman": float("nan"), "top_k_overlap": {}}

    emb_rows, num_rows, flat_emb, flat_num = [], [], [], []
    for i in range(n):
        emb_row, num_row = [], []
        for j in range(n):
            if i == j:
                # Exclude self-similarity: it is 1.0 by construction in both
                # spaces and would inflate the agreement.
                emb_row.append(float("-inf"))
                num_row.append(float("-inf"))
                continue
            e = cosine(embedding_vectors[i], embedding_vectors[j])
            g = cosine(numeric_vectors[i], numeric_vectors[j])
            emb_row.append(e)
            num_row.append(g)
            if j > i:
                flat_emb.append(e)
                flat_num.append(g)
        emb_rows.append(emb_row)
        num_rows.append(num_row)

    return {
        "spearman": spearman(flat_emb, flat_num),
        "top_k_overlap": {
            f"top{k}": top_k_overlap(emb_rows, num_rows, k)
            for k in ks
            if k < n
        },
    }
