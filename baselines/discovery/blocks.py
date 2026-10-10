"""The retrieved-rollouts block, in DiscoverPhysics' terms.

Same shape as `baselines/memory/blocks.py` and `baselines/orbit/blocks.py`
-- header, one framed record each, footer -- and different only in wording,
for the reason the ORBIT module gives: a footer that tells the agent to "match
on the geometry" names something that does not exist here.

Two things are deliberately absent, both matching the other two block modules:

* any instruction to cite a retrieved rollout.  Citation is HeXA-specific
  scaffolding and asking a baseline for it would confound the comparison.
* the judge's critique.  `expl_reasoning` is written by a judge that reads the
  world's ground-truth explanation, so it is oracle feedback -- HeXA's own
  pre-registration excludes it from the teacher (PREREG_within_oscillator_v2.md
  item 1), and it must not reach the actor through a memory either.  The
  explanation *score* is shown, which is the same line HeXA draws.

The block is appended to `agent._system` via `discovery_runner.run_episode`'s
`prompt_block`, which is the exact seam the skills block uses.
"""

from __future__ import annotations

from typing import Sequence

from baselines.discovery.intent import world_label
from baselines.memory.blocks import count_tokens  # noqa: F401  (re-exported)
from baselines.memory.store import SUCCESS, MemoryRecord

RAG_HEADER = (
    "## Retrieved Past Attempts (your own transcripts from earlier discovery episodes)"
)
RAG_FOOTER = (
    "Each block above is one complete past attempt at discovering a hidden law "
    "of motion, with the evaluator's verdict on it. Numeric literals in the "
    "reasoning are masked as <n>, so nothing above hands you a constant, and "
    "the worlds are labelled anonymously. Read them for which experiments "
    "produced a clean measurement and which wasted rounds, and verify every "
    "hypothesis against the data you observe here."
)


def _outcome_label(record: MemoryRecord) -> str:
    return "SUCCESS" if record.kind == SUCCESS else "FAILURE"


def _world_ref(record: MemoryRecord) -> str:
    """The anonymous label, never the world's real name -- see `intent.world_label`.

    `record.level` is the retrieval scope ("discovery" for the cross-world arm,
    the world name for a within-world one) and is not safe to print either way.
    """
    return world_label(str(record.meta.get("world") or record.level))


def format_rag_block(records: Sequence[MemoryRecord]) -> str:
    """Episodic RAG: verbatim past episodes, each with its verdict."""
    if not records:
        return ""
    lines = ["", RAG_HEADER, ""]
    for i, record in enumerate(records, 1):
        lines.append(
            f"### Past attempt {i} — {_outcome_label(record)} "
            f"(world {_world_ref(record)}, seed {record.seed})"
        )
        lines.append(record.content)
        lines.append("")
    lines.append(RAG_FOOTER)
    return "\n".join(lines) + "\n"


MEMRL_HEADER = "## Archived Memories (from your own past discovery episodes)"
MEMRL_FOOTER = (
    "These were retrieved because they have been useful before on tasks like "
    "this one. They describe experimental strategy, not answers: the hidden law "
    "of each past world differs from this one's, and every world's controls "
    "differ too. Apply the reasoning where this world's controls allow it, and "
    "verify everything against the data you observe here."
)


def format_memrl_block(records: Sequence[MemoryRecord]) -> str:
    """MemRL: summarised experiences, ordered by their composite score."""
    if not records:
        return ""
    lines = ["", MEMRL_HEADER, ""]
    for i, record in enumerate(records, 1):
        lines.append(
            f"### Memory {i} — {_outcome_label(record)} "
            f"(world {_world_ref(record)}, seed {record.seed}, "
            f"utility Q={record.q:+.3f}, updates={record.q_visits})"
        )
        lines.append(record.content)
        lines.append("")
    lines.append(MEMRL_FOOTER)
    return "\n".join(lines) + "\n"
