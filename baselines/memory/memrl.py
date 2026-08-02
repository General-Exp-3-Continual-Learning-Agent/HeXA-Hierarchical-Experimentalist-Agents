"""MemRL — value-aware episodic memory (Intent-Experience-Utility triplets).

Ported from the official release (MemTensor/MemRL, arXiv:2601.03192) to the
InterPhyre episode interface, using its **ALFWorld** configuration
(`configs/rl_alf_config.yaml`) — the closest analogue to InterPhyre: sequential,
exploration-heavy, environment-verified.  Defaults below are that file's values.

Mapping to the official code:

  triplet (z, e, Q)      MemoryRecord.intent / .content / .q          (store.py)
  Phase A                `retrieve_query`: top-k1 by cosine, sim > delta
                         (`memrl/service/retrievers.py:QueryRetriever`)
  Phase B                score = w_sim * z(sim) + w_q * z(Q), take top-k2
                         (`memrl/service/memory_service.py:1439`)
  z-norm                 similarity uses *precomputed dataset* mean/std;
                         Q uses *per-candidate-pool* mean/std clipped to +-3
                         (`memory_service.py:_normalize_similarity/_normalize_q`)
  utility update         Q <- (1-alpha) Q + alpha (r + gamma max Q'), gamma = 0
                         (`memrl/service/value_driven.py:QValueUpdater.update`)
  reward                 +1 success / -1 failure  (`rl_config.success_reward`)
  build (success)        "SCRIPT:\\n{script}\\n\\nTRAJECTORY:\\n{traj}"
                         (`memrl/service/builders.py:ProceduralizationBuilder`)
  update (failure)       reflection note, appended as a new memory
                         (`memrl/service/updater.py:AdjustmentUpdater`)
  novelty                near-duplicate intents update in place instead of
                         adding (`add_similarity_threshold`), and a merge-target
                         gets a scaled reward attribution (`reward_merge_gain`)

Deviations from the official config, all logged in every run manifest:
  * embeddings default to local jina-v2-base-en rather than
    text-embedding-3-large (pass `openai:text-embedding-3-large` to match);
  * with the Claude actor, memory-building goes through the `claude -p` CLI,
    which exposes no temperature knob, so the paper's 0.0 / 0.3 split cannot be
    set.  The local-model backend does honour it — see
    `baselines/memory/summarize.py`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .store import FAILURE, SUCCESS, MemoryRecord, MemoryStore
from .trajectory_fmt import (
    count_simulator_calls,
    format_trajectory,
    trajectory_reward,
)


@dataclass
class MemRLConfig:
    # ── Phase A: similarity recall ──
    k1: int = 5                      # memory.k_retrieve
    delta: float = 0.62              # rl_config.sim_threshold (CALIBRATE per level)
    # ── Phase B: value-aware selection ──
    k2: int = 3                      # rl_config.topk
    weight_sim: float = 0.5          # rl_config.weight_sim
    weight_q: float = 0.5            # rl_config.weight_q
    epsilon: float = 0.0             # rl_config.epsilon
    q_min_threshold: Optional[float] = -10.0
    # z-normalization of similarity uses precomputed dataset statistics
    sim_norm_mean: float = 0.5187    # memory.sim_norm_mean (CALIBRATE per level)
    sim_norm_std: float = 0.1203     # memory.sim_norm_std  (CALIBRATE per level)
    use_z_score_normalization: bool = True
    # ── utility update ──
    alpha: float = 0.3               # rl_config.alpha
    gamma: float = 0.0               # rl_config.gamma
    q_init: float = 0.0              # rl_config.q_init_pos
    success_reward: float = 1.0      # rl_config.success_reward
    failure_reward: float = -1.0     # rl_config.failure_reward
    use_shaped_reward: bool = False  # True -> HeXA's iteration-aware reward
    # ── curation ──
    add_similarity_threshold: float = 0.90   # memory.add_similarity_threshold
    novelty_threshold: float = 0.85          # rl_config.novelty_threshold
    reward_merge_gain: float = 0.1           # rl_config.reward_merge_gain
    # ── experience content ──
    max_traj_steps: Optional[int] = 12
    max_obs_chars: int = 600
    seed: int = 42                           # experiment.random_seed


@dataclass
class Retrieved:
    records: list[MemoryRecord] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class MemRL:
    """Two-phase retrieval + Monte-Carlo utility updates over episodic memory.

    `summarizer` produces the stored experience text.  It is called as
    `summarizer(kind, intent, trajectory_text)` where kind is "script" (success)
    or "reflection" (failure) — see `baselines/memory/summarize.py`.
    """

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[MemRLConfig] = None,
        summarizer: Optional[Callable[[str, str, str], str]] = None,
    ):
        import random

        self.store = store
        self.cfg = config or MemRLConfig()
        self.summarizer = summarizer
        self._rng = random.Random(self.cfg.seed)

    # ── normalization ───────────────────────────────────────────────────
    def _norm_sim(self, sim: float) -> float:
        if not self.cfg.use_z_score_normalization:
            return float(sim)
        std = self.cfg.sim_norm_std if self.cfg.sim_norm_std > 1e-9 else 1.0
        return (float(sim) - self.cfg.sim_norm_mean) / std

    def _norm_q(self, q: float, mean: float, std: float) -> float:
        if not self.cfg.use_z_score_normalization:
            return float(q)
        std = std if std and std > 1e-9 else 1.0
        return _clip((float(q) - mean) / std, -3.0, 3.0)

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
            return Retrieved([], {"phase_a": "empty_store", "num_candidates": 0})

        simmax = scored[0][1]

        # Phase A — similarity recall above delta, capped at k1.
        pool = [(r, s) for r, s in scored if s > self.cfg.delta][: self.cfg.k1]
        if not pool:
            # Paper: "If C(s) is empty, MemRL relies solely on the frozen LLM
            # for exploration."  No memory is injected.
            return Retrieved(
                [],
                {
                    "phase_a": "below_delta",
                    "num_candidates": len(scored),
                    "simmax": round(simmax, 4),
                    "delta": self.cfg.delta,
                },
            )

        # Optional Q floor, as in the official selector.
        if self.cfg.q_min_threshold is not None:
            pool = [(r, s) for r, s in pool if r.q >= self.cfg.q_min_threshold]
        if not pool:
            return Retrieved(
                [],
                {
                    "phase_a": "q_floor_filtered",
                    "num_candidates": len(scored),
                    "simmax": round(simmax, 4),
                },
            )

        # Phase B — value-aware selection on the composite z-scored score.
        qs = [r.q for r, _ in pool]
        mean_q = float(statistics.fmean(qs))
        std_q = float(statistics.pstdev(qs)) if len(qs) > 1 else 1.0

        enriched = []
        for record, sim in pool:
            sim_z = self._norm_sim(sim)
            q_z = self._norm_q(record.q, mean_q, std_q)
            enriched.append(
                {
                    "record": record,
                    "similarity": sim,
                    "similarity_z": sim_z,
                    "q": record.q,
                    "q_z": q_z,
                    "score": self.cfg.weight_sim * sim_z + self.cfg.weight_q * q_z,
                }
            )
        enriched.sort(key=lambda c: c["score"], reverse=True)

        k2 = min(self.cfg.k2, len(enriched))
        if self.cfg.epsilon > 0 and self._rng.random() < self.cfg.epsilon:
            selected = self._rng.sample(enriched, k2)
        else:
            selected = enriched[:k2]

        return Retrieved(
            [c["record"] for c in selected],
            {
                "phase_a": "ok",
                "num_candidates": len(scored),
                "pool_size": len(enriched),
                "simmax": round(simmax, 4),
                "delta": self.cfg.delta,
                "selected": [
                    {
                        "memory_id": c["record"].memory_id,
                        "seed": c["record"].seed,
                        "kind": c["record"].kind,
                        "similarity": round(c["similarity"], 4),
                        "similarity_z": round(c["similarity_z"], 4),
                        "q": round(c["q"], 4),
                        "q_z": round(c["q_z"], 4),
                        "score": round(c["score"], 4),
                    }
                    for c in selected
                ],
            },
        )

    # ── utility update ──────────────────────────────────────────────────
    def episode_reward(self, result: dict) -> float:
        if self.cfg.use_shaped_reward:
            return trajectory_reward(result)
        return (
            self.cfg.success_reward
            if result.get("success")
            else self.cfg.failure_reward
        )

    def update_utilities(self, retrieved: Retrieved, result: dict) -> dict:
        """Monte-Carlo update of every memory that was injected this episode."""
        reward = self.episode_reward(result)
        updates = []
        for record in retrieved.records:
            old_q = record.q
            target = reward + self.cfg.gamma * 0.0      # gamma = 0: terminal
            record.q = (1.0 - self.cfg.alpha) * old_q + self.cfg.alpha * target
            record.q_visits += 1
            record.last_reward = reward
            updates.append(
                {
                    "memory_id": record.memory_id,
                    "old_q": round(old_q, 4),
                    "new_q": round(record.q, 4),
                    "reward": reward,
                    "visits": record.q_visits,
                }
            )
        return {"reward": reward, "updates": updates}

    # ── write-back ──────────────────────────────────────────────────────
    def _experience_text(self, intent: str, result: dict, level: str, seed: int) -> tuple[str, str]:
        """Return (kind, content) for the new triplet.

        Success -> proceduralization: an LLM script plus the trajectory.
        Failure -> adjustment: an LLM reflection plus the failed approach.
        """
        traj = format_trajectory(
            result,
            max_steps=self.cfg.max_traj_steps,
            max_obs_chars=self.cfg.max_obs_chars,
        )
        header = f"Task: {level} (seed {seed})"

        if result.get("success"):
            script = ""
            if self.summarizer is not None:
                script = self.summarizer("script", intent, traj).strip()
            content = f"{header}\n\nSCRIPT:\n{script}\n\nTRAJECTORY:\n{traj}"
            return SUCCESS, content

        reflection = ""
        if self.summarizer is not None:
            reflection = self.summarizer("reflection", intent, traj).strip()
        content = (
            f"TASK REFLECTION:\n{header}\n\n"
            f"What went wrong:\n{reflection}\n\nFailed approach:\n{traj}\n"
        )
        return FAILURE, content

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
    ) -> Optional[MemoryRecord]:
        """Write back a new triplet, or fold into a near-duplicate intent.

        Mirrors `add_memories`' novelty handling: an intent whose nearest
        existing entry is above `add_similarity_threshold` refreshes that entry's
        content rather than growing the bank, and a merge target above
        `novelty_threshold` receives a scaled reward attribution.
        """
        kind, content = self._experience_text(intent, result, level, seed)
        success = bool(result.get("success"))
        reward = self.episode_reward(result)

        # Near-duplicate intent of the *same outcome class* -> update in place.
        nearest = self.store.similarities(embedding, level=level, exclude_seeds=[seed])
        same_kind = [(r, s) for r, s in nearest if r.kind == kind]
        if same_kind and same_kind[0][1] >= self.cfg.add_similarity_threshold:
            target, sim = same_kind[0]
            target.content = content
            target.meta.setdefault("merged_seeds", []).append(seed)
            target.meta["last_merge_similarity"] = round(sim, 4)
            return target

        # Reward attribution to a highly similar existing memory (merge target).
        if nearest and nearest[0][1] >= self.cfg.novelty_threshold:
            merge_target, sim = nearest[0]
            gain = self.cfg.reward_merge_gain * reward
            merge_target.q = (
                1.0 - self.cfg.alpha
            ) * merge_target.q + self.cfg.alpha * gain
            merge_target.q_visits += 1
            merge_target.meta.setdefault("attributed_from_seeds", []).append(seed)

        record = MemoryRecord(
            memory_id=self.store.next_id("memrl"),
            level=level,
            seed=seed,
            kind=kind,
            intent=intent,
            content=content,
            success=success,
            iterations=int(result.get("iterations") or 0),
            simulator_calls=count_simulator_calls(result),
            reward=trajectory_reward(result),
            embedding=list(embedding),
            q=self.cfg.q_init,
            created_step=step_index,
            source=source,
        )
        return self.store.add(record)
