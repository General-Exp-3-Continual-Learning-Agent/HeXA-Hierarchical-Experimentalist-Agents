"""ReasoningBank — append-only memory of distilled reasoning items.

One `MemoryRecord` per past experience; its distilled items (<=3) are rendered
into `.content` for injection and kept structured in `.meta["items"]`. Retrieval
is pure embedding-cosine top-k over experiences (the paper's default k=1; its
Figure 13 shows more retrieved experiences hurt). Consolidation is a plain
append — no capacity cap, no merge, no utility/forgetting. That append-only,
flat design is exactly what the paper leaves to future work and what HeXA's
bounded, hierarchical, evolve/merge bank provides, so running the two on the same
actor / env / seeds / eval isolates the memory mechanism.

The success/failure label that selects the extraction prompt is the InterPhyre
environment's verdict (`result["success"]`), not the paper's LLM-as-judge: with a
real reward available, using it keeps the supervision identical to HeXA's and
removes a confound. `make_rb_extractor`/`make_cli_extractor` in extract.py
produce the items; this class owns retrieval, rendering, and write-back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from baselines.memory.store import FAILURE, SUCCESS, MemoryRecord, MemoryStore
from baselines.memory.trajectory_fmt import (
    count_simulator_calls,
    format_trajectory,
    trajectory_reward,
)


@dataclass
class ReasoningBankConfig:
    #: Number of past experiences retrieved. The paper's default is 1 (each
    #: carries up to `max_items` items); raise it to widen the injected budget,
    #: e.g. to match HeXA's 6-skill/4-mistake prompt real-estate.
    k: int = 1
    #: Items extracted per trajectory (paper: at most 3).
    max_items: int = 3
    #: Trajectory formatting handed to the extractor. Truncation matches the
    #: other memory baselines so extraction sees comparable context volume.
    max_traj_steps: Optional[int] = 12
    max_obs_chars: int = 600


@dataclass
class Retrieved:
    records: list[MemoryRecord] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class ReasoningBank:
    """Append-only distilled-reasoning memory with top-k cosine retrieval.

    `extractor(success, intent, trajectory_text) -> list[{title, description,
    content}]` produces the stored items — see `baselines/reasoning_bank/extract.py`.
    """

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[ReasoningBankConfig] = None,
        extractor: Optional[Callable[[bool, str, str], list[dict]]] = None,
    ):
        self.store = store
        self.cfg = config or ReasoningBankConfig()
        self.extractor = extractor

    # ── retrieval ───────────────────────────────────────────────────────
    def retrieve(
        self,
        query_embedding: list[float],
        *,
        level: Optional[str] = None,
        exclude_seeds: Iterable[int] = (),
    ) -> Retrieved:
        """Top-k experiences by cosine similarity (no success/failure split).

        The paper retrieves the most similar past experiences regardless of
        outcome; each retrieved experience contributes its own items (extracted
        from a success or a failure), so both kinds still reach the prompt.
        """
        scored = self.store.similarities(
            query_embedding, level=level, exclude_seeds=exclude_seeds
        )
        if not scored:
            return Retrieved([], {"mode": "empty", "num_candidates": 0})

        picked = scored[: self.cfg.k]
        return Retrieved(
            [r for r, _ in picked],
            {
                "mode": "topk_similarity",
                "k": self.cfg.k,
                "num_candidates": len(scored),
                "similarities": [round(s, 4) for _, s in picked],
                "picked": [r.memory_id for r, _ in picked],
            },
        )

    # ReasoningBank has no utility signal; the hook exists so the stream loop can
    # treat every memory baseline identically.
    def update_utilities(self, retrieved: Retrieved, result: dict) -> dict:
        del retrieved, result
        return {"updates": []}

    # ── rendering ───────────────────────────────────────────────────────
    @staticmethod
    def render_items(items: list[dict], *, success: bool) -> str:
        """Render distilled items into the text stored in `MemoryRecord.content`.

        Successes become strategies, failures become lessons — the label the
        agent reads to know whether an item is a thing to do or a thing to avoid.
        """
        tag = "Strategy" if success else "Lesson"
        lines: list[str] = []
        for item in items:
            title = item.get("title") or "(untitled)"
            content = item.get("content") or ""
            description = item.get("description") or ""
            lines.append(f"- [{tag}: {title}] {content}".rstrip())
            if description:
                lines.append(f"  Use when: {description}")
        return "\n".join(lines)

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
    ) -> Optional[MemoryRecord]:
        """Distill one trajectory into <=3 items and append them as one record.

        Append-only: every episode that yields items grows the bank by exactly
        one record. No merging, pruning, capacity cap, or utility update — the
        contrast with HeXA's evolve/merge loop.
        """
        success = bool(result.get("success"))
        traj = format_trajectory(
            result,
            max_steps=self.cfg.max_traj_steps,
            max_obs_chars=self.cfg.max_obs_chars,
        )

        items: list[dict] = []
        if self.extractor is not None:
            items = self.extractor(success, intent, traj)
        if not items:
            # Nothing usable distilled; add nothing rather than an empty record.
            print(
                f"  [ReasoningBank] seed {seed}: no items extracted, memory unchanged",
                flush=True,
            )
            return None

        content = self.render_items(items, success=success)
        record = MemoryRecord(
            memory_id=self.store.next_id("rb"),
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
            meta={"items": items, "num_items": len(items)},
        )
        return self.store.add(record)
