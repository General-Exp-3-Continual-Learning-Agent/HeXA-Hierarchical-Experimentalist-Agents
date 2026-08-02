"""JSON-backed memory store shared by Episodic RAG and MemRL.

One record per memory entry.  Episodic RAG only ever appends and never touches
`q`; MemRL maintains `q` / `q_visits` on the same records, so a single store
serves both and the two methods stay directly comparable.

The store is written after every episode so an interrupted run resumes from
disk, matching the resume behaviour of `skillrl/runner/run_skill_agent.py`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

SUCCESS = "success"
FAILURE = "failure"


@dataclass
class MemoryRecord:
    memory_id: str
    level: str
    seed: int
    kind: str                      # SUCCESS | FAILURE
    intent: str                    # embedded text (the retrieval key)
    content: str                   # what gets injected into the prompt
    success: bool
    iterations: int
    simulator_calls: int
    reward: float                  # shaped reward in [-1, 1]
    embedding: list[float] = field(default_factory=list)
    q: float = 0.0
    q_visits: int = 0
    last_reward: Optional[float] = None
    created_step: int = 0          # position in the seed stream
    source: str = "online"         # "offline" for preloaded trajectories
    meta: dict[str, Any] = field(default_factory=dict)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class MemoryStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.records: list[MemoryRecord] = []
        self._counter = 0
        if self.path and self.path.exists():
            self.load()

    # ── persistence ─────────────────────────────────────────────────────
    def load(self) -> None:
        raw = json.loads(self.path.read_text())
        self.records = [MemoryRecord(**r) for r in raw.get("records", [])]
        self._counter = int(raw.get("counter", len(self.records)))

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "counter": self._counter,
            "num_records": len(self.records),
            "records": [asdict(r) for r in self.records],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    # ── mutation ────────────────────────────────────────────────────────
    def next_id(self, prefix: str = "mem") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:05d}"

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self.records.append(record)
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        for r in self.records:
            if r.memory_id == memory_id:
                return r
        return None

    # ── views ───────────────────────────────────────────────────────────
    def for_level(self, level: Optional[str]) -> list[MemoryRecord]:
        if level is None:
            return list(self.records)
        return [r for r in self.records if r.level == level]

    def similarities(
        self,
        query_embedding: list[float],
        *,
        level: Optional[str] = None,
        exclude_seeds: Iterable[int] = (),
    ) -> list[tuple[MemoryRecord, float]]:
        """(record, cosine) pairs sorted by descending similarity.

        `exclude_seeds` keeps an episode's own past memories out of its
        retrieval, so a seed is never solved by replaying itself.
        """
        excluded = set(exclude_seeds)
        scored = [
            (r, cosine(query_embedding, r.embedding))
            for r in self.for_level(level)
            if r.seed not in excluded
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def stats(self) -> dict[str, Any]:
        qs = [r.q for r in self.records if r.q_visits > 0]
        return {
            "num_records": len(self.records),
            "num_success": sum(1 for r in self.records if r.kind == SUCCESS),
            "num_failure": sum(1 for r in self.records if r.kind == FAILURE),
            "num_with_q_updates": len(qs),
            "mean_q": sum(qs) / len(qs) if qs else 0.0,
            "max_q": max(qs) if qs else 0.0,
            "min_q": min(qs) if qs else 0.0,
        }
