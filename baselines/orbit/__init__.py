"""Episodic RAG and MemRL on ORBIT's Maze and Mastermind.

`baselines/` implements both methods against InterPhyre: one seed is one
episode, the actor is `ReactAgent` over `InterphyreToolkit`, and the retrieval
key is the scene geometry.  None of that transfers.  An ORBIT instance is a
*fixed hidden task played for three episodes of cumulative chat*, driven by
`hexa_orbit.actor.MazeActor` against a vLLM endpoint, and the agent never sees
the thing that makes one instance differ from another — the maze layout, the
secret code.

This package is the adapter, and it is deliberately thin.  `memory/rag.py` and
`memory/memrl.py` are **not** touched: the methods stay the ports they were, and
everything ORBIT-specific lives here.

    tasks.py      the two tasks behind one interface
    intent.py     retrieval keys, built only from what the agent can observe
    rollout.py    one instance played with a memory block injected
    stream.py     memory-building pass, then the frozen-memory eval
    calibrate.py  MemRL's delta and similarity z-norm constants

Four things the port has to decide, recorded here because each one is a claim a
reader may want to disagree with:

**The memory unit is the instance, not the episode.**  Within an instance ORBIT
already carries memory — the chat is cumulative across all three episodes, and
that is ORBIT's own mechanism, which the no-bank baseline keeps.  What Episodic
RAG and MemRL add is *cross-instance* memory, which is exactly what HeXA's skill
bank adds.  So one memory record is one three-episode instance, retrieved once
at instance start.

**The block goes where the bank goes.**  `MazeActor(guidelines=...)` appends to
the system prompt and nowhere else, so every added token is attributable and the
per-turn observations stay byte-identical to the no-bank arm's.  Retrieval is
per instance, so the block differs per instance where HeXA's is fixed for a
round; `added_tokens` is recorded per instance either way.

**Retrieval keys use only observable information.**  See `intent.py`.  Keying on
the maze layout or the secret code would hand the baseline the answer.

**Memory is frozen for the eval pass.**  It is built over the training pool and
read-only over the held-out eval pool, mirroring `hexa_orbit.loop.frozen_eval`,
so these arms are paired per `instance_id` with the no-bank and Design B arms
already in `MAZE_RESULTS.md` and `MASTERMIND_RESULTS.md`.
"""
