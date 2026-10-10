"""A memory store whose ties break towards the most recent memory.

`MemoryStore.similarities` sorts with Python's stable sort, so records that
score equally keep insertion order and the OLDEST wins.  On InterPhyre and
ORBIT that is immaterial -- instances differ, so exact ties are rare.  On
within-world DiscoverPhysics it decides the whole arm.

A DiscoverPhysics seed reshuffles observation noise but not the hidden law, so
every episode of one world has the *same* problem embedding: retrieval is an
exact tie across the entire memory.  Under the stock ordering Episodic RAG would
lock onto the first success and the first failure it ever saw and inject those
two forever -- the build pass would grow a memory it never reads, which is a
strawman rather than a baseline.  Preferring the most recent memory instead
lets accumulated experience reach the actor, which is the property the arm is
supposed to be testing.

Recorded as a deviation on every run: the CORE paper specifies "the most similar
past successful rollout", and states no tie-break because its problems differ.

Nothing else changes -- `for_level` is the only ordering seam, and both
`rag.py` and `memrl.py` read the store exclusively through it.
"""

from __future__ import annotations

from typing import Optional

from baselines.memory.store import MemoryRecord, MemoryStore


class RecencyTieStore(MemoryStore):
    def for_level(self, level: Optional[str]) -> list[MemoryRecord]:
        return list(reversed(super().for_level(level)))
