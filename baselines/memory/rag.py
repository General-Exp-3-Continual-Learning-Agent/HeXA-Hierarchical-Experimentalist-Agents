"""Episodic RAG — retrieval of entire past rollouts.

Follows the CORE paper's definition verbatim:

    "Our RAG implementation is a setting where all rollouts are added to a
    rollout memory and indexed according to their problem embedding.  For any
    model inference during training and testing, we retrieve the most similar
    past successful rollout and the most similar past unsuccessful rollout to
    place in-context alongside their verifier feedback.  If the rollout memory
    does not yet contain any successful or unsuccessful rollouts, then we
    retrieve similar rollouts without accounting for success."

No summarization, no utility, no consolidation: memory grows monotonically and
every entry is a raw trajectory.  In the paper this is deliberately the
context-hungry baseline (33.6k added tokens per eval item, vs 0.92k for CORE);
`baselines/memory/blocks.py` measures the equivalent on InterPhyre.

One deviation: we keep only the last `max_steps` (default 12) steps of each
rollout rather than all of them.  The rollouts are still raw — no LLM touches
them — but they are no longer complete.  See RAGConfig.max_steps for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .store import FAILURE, SUCCESS, MemoryRecord, MemoryStore
from .trajectory_fmt import (
    count_simulator_calls,
    format_trajectory,
    format_verifier_feedback,
    trajectory_reward,
)


@dataclass
class RAGConfig:
    """One success + one failure is the paper's specification."""

    k_success: int = 1
    k_failure: int = 1
    max_obs_chars: int = 600
    # The CORE paper stores rollouts verbatim (None). We truncate to the last 12
    # steps, matching MemRL's --max-traj-steps, because gpt-oss-120b must use
    # eager attention and the untruncated block OOMs a 4x A100 node on the first
    # episode. Holding trajectory length equal also isolates the axis under test
    # (summarized vs raw memory) from sheer context volume. Pass
    # --max-traj-steps 0 or a large value to restore verbatim storage.
    max_steps: Optional[int] = 12


@dataclass
class Retrieved:
    records: list[MemoryRecord]
    diagnostics: dict


class EpisodicRAG:
    def __init__(self, store: MemoryStore, config: Optional[RAGConfig] = None):
        self.store = store
        self.cfg = config or RAGConfig()

    # ── retrieval ───────────────────────────────────────────────────────
    def retrieve(
        self,
        query_embedding: list[float],
        *,
        level: Optional[str] = None,
        exclude_seeds: Iterable[int] = (),
    ) -> Retrieved:
        scored = self.store.similarities(
            query_embedding, level=level, exclude_seeds=exclude_seeds
        )
        if not scored:
            return Retrieved([], {"mode": "empty", "num_candidates": 0})

        successes = [(r, s) for r, s in scored if r.kind == SUCCESS]
        failures = [(r, s) for r, s in scored if r.kind == FAILURE]

        # Paper's fallback: if either outcome class is empty, retrieve the most
        # similar rollouts without accounting for success at all.
        if not successes or not failures:
            want = self.cfg.k_success + self.cfg.k_failure
            picked = scored[:want]
            return Retrieved(
                [r for r, _ in picked],
                {
                    "mode": "similarity_only",
                    "num_candidates": len(scored),
                    "similarities": [round(s, 4) for _, s in picked],
                },
            )

        picked = successes[: self.cfg.k_success] + failures[: self.cfg.k_failure]
        return Retrieved(
            [r for r, _ in picked],
            {
                "mode": "success_and_failure",
                "num_candidates": len(scored),
                "top_success_similarity": round(successes[0][1], 4),
                "top_failure_similarity": round(failures[0][1], 4),
                "similarities": [round(s, 4) for _, s in picked],
            },
        )

    # ── write-back ──────────────────────────────────────────────────────
    def add_episode(
        self,
        *,
        result: dict,
        level: str,
        seed: int,
        intent: str,
        embedding: list[float],
        step_index: int,
        source: str = "online",
    ) -> MemoryRecord:
        """Store one rollout verbatim, with its verifier feedback attached."""
        success = bool(result.get("success"))
        body = format_trajectory(
            result,
            max_steps=self.cfg.max_steps,
            max_obs_chars=self.cfg.max_obs_chars,
        )
        content = "\n".join(
            [
                f"Level: {level}  |  Seed: {seed}",
                format_verifier_feedback(result),
                "",
                "Rollout:",
                body,
            ]
        )
        record = MemoryRecord(
            memory_id=self.store.next_id("rag"),
            level=level,
            seed=seed,
            kind=SUCCESS if success else FAILURE,
            intent=intent,
            content=content,
            success=success,
            iterations=int(result.get("iterations") or 0),
            simulator_calls=count_simulator_calls(result),
            reward=trajectory_reward(result),
            embedding=list(embedding),
            created_step=step_index,
            source=source,
        )
        return self.store.add(record)

    # Episodic RAG has no utility signal; the hook exists so the stream loop
    # can treat both baselines identically.
    def update_utilities(self, retrieved: Retrieved, result: dict) -> dict:
        del retrieved, result
        return {"updates": []}
