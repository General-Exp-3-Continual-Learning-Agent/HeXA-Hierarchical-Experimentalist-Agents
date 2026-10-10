"""Retrieval-key backends for ORBIT, and how to choose one.

`baselines/memory/embedder.py` handles the two the papers name -- a local
SentenceTransformers model (the CORE paper's jina-v2, for Episodic RAG) and
OpenAI `text-embedding-3-large` (MemRL's own config).  This adds one more and
records which of them actually runs here.

**jina-v2 does not load under transformers >= 5.**  It ships its model code via
`trust_remote_code`, and that code imports `transformers.onnx`, which was
removed in 5.0.  The environment these tasks run in has transformers 5.0, so the
CORE paper's exact embedder is unavailable without pinning the whole environment
back.  `sentence-transformers/all-MiniLM-L6-v2` is the substitute:
architecturally plain, no remote code, and it loads.  Recorded as a deviation on
every run rather than quietly swapped.

`hashing` is a dependency-free alternative and not a stand-in for a neural
embedder.  It is a bag of character n-grams hashed into a fixed vector -- an
honest lexical similarity, deterministic, needing no download and no GPU.  On
Maze that is a defensible key and arguably a better-conditioned one than a
sentence embedder: a maze intent is a coordinate pair and four wall/path tokens,
which is a *string pattern* rather than a sentence, and sentence embedders are
not built to separate "(4, 3)" from "(5, 2)".  Use it when the cluster has no
egress, or to check that a result does not hinge on the embedder.

It changes nothing about Mastermind, where every intent is the same string and
every key backend returns the same vector.  That degeneracy is a property of the
task, not of the embedder, and no choice here fixes it.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Optional, Sequence

#: The CORE paper's choice for Episodic RAG. Kept as the documented default so
#: a run states what it is deviating from, even where it cannot load.
PAPER_DEFAULT = "jinaai/jina-embeddings-v2-base-en"

#: What loads in this environment. See the module docstring.
WORKING_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

HASHING = "hashing"


class HashingEmbedder:
    """Character n-gram hashing into a unit-normalised fixed-width vector.

    The scheme is `sklearn.feature_extraction.text.HashingVectorizer`'s, done in
    a dozen lines so it needs no scikit-learn: lowercase, collapse whitespace,
    take every character n-gram in `ngram_range`, hash it into one of `dim`
    buckets with a sign from the same digest (signed hashing keeps collisions
    from systematically inflating similarity), then L2-normalise so the store's
    cosine is a plain dot product.
    """

    name = HASHING

    def __init__(self, dim: int = 512, ngram_range: tuple[int, int] = (3, 5)):
        self.dim = int(dim)
        self.ngram_range = ngram_range

    def _normalise(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower()).strip()

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        normalised = self._normalise(text)
        low, high = self.ngram_range
        for n in range(low, high + 1):
            for i in range(len(normalised) - n + 1):
                digest = hashlib.blake2b(
                    normalised[i : i + n].encode("utf-8"), digest_size=8
                ).digest()
                value = int.from_bytes(digest, "big")
                vector[value % self.dim] += 1.0 if (value >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm <= 0.0:
            # An empty or all-whitespace intent. Returning zeros is right: the
            # store's `cosine` scores a zero vector 0.0 against everything, which
            # is the honest "this key says nothing" answer rather than a
            # fabricated similarity.
            return vector
        return [v / norm for v in vector]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def build_embedder(name: str, *, cache_dir: Optional[Path] = None):
    """Resolve `--embedding-model` to a backend with `embed` / `embed_one`."""
    if name == HASHING:
        return HashingEmbedder()

    from baselines.memory.embedder import Embedder  # noqa: PLC0415

    return Embedder(name, cache_dir=cache_dir)


def embedder_note(name: str) -> str:
    """One line for the run manifest saying what the key actually is."""
    if name == HASHING:
        return (
            "retrieval key is signed character-n-gram hashing (no neural "
            "embedder); lexical similarity only"
        )
    if name == PAPER_DEFAULT:
        return (
            f"retrieval key is {name}, the CORE paper's embedder. It requires "
            f"transformers < 5 (its trust_remote_code imports transformers.onnx)"
        )
    return f"retrieval key is the text embedder {name}"
