"""Prompt blocks for retrieved memories, in ORBIT's terms.

Structurally identical to `baselines/memory/blocks.py` -- same header/body/footer
shape, same per-record framing, same `count_tokens` accounting -- and different
only in wording.  That difference is not cosmetic and is why these exist rather
than the originals being reused: the InterPhyre footers instruct the agent to
"use the successful rollout as a template where the geometry matches" and to
"adapt them to this seed's geometry", and there is no geometry in a maze the
agent cannot see or a code it has to guess.  A retrieved memory whose footer
tells the agent to match on something that does not exist is a worse baseline,
not a more faithful one.

The block is appended to ORBIT's system prompt in exactly the place
`hexa_orbit.loop.render_block` puts the skill bank, so added-token counts are
comparable across HeXA, both memory baselines, and the no-bank arm.

What is deliberately *not* here, matching the InterPhyre blocks: any instruction
to cite the retrieved memory.  Citation is HeXA-specific scaffolding, it is what
the audit pass in `hexa_orbit.citations` checks, and asking a baseline for it
would confound the comparison.
"""

from __future__ import annotations

from typing import Sequence

from baselines.memory.blocks import count_tokens  # noqa: F401  (re-exported)
from baselines.memory.store import SUCCESS, MemoryRecord

RAG_HEADER = "## Retrieved Past Attempts (your own transcripts on other instances of this task)"
RAG_FOOTER = (
    "Each block above is a complete past attempt at a different instance of "
    "this same task, played over the same three episodes you have now, with the "
    "environment's verdict. The hidden part of the task differs from instance to "
    "instance, so nothing above tells you this instance's answer -- read them for "
    "which search behaviour paid off and which wasted episodes, and do not "
    "replay a past action sequence."
)

MEMRL_HEADER = "## Archived Memories (from your own past experiences on this task)"
MEMRL_FOOTER = (
    "These were retrieved because they have been useful before on instances "
    "like this one. They describe strategy, not answers: the hidden part of "
    "each past instance was different from this one's. Apply the reasoning, not "
    "the specific moves."
)


def _outcome_label(record: MemoryRecord) -> str:
    return "SUCCESS" if record.kind == SUCCESS else "FAILURE"


def _instance_ref(record: MemoryRecord) -> str:
    """How a record names itself in the block.

    The real instance id, not `MemoryRecord.seed` -- that is a truncated hash
    (`intent.seed_key`) carried only to satisfy the store's `int` field, and
    printing it would put a number in the prompt that matches nothing a reader
    of the run files can look up.
    """
    return str(record.meta.get("instance_id") or record.seed)


def format_rag_block(records: Sequence[MemoryRecord]) -> str:
    """Episodic RAG: verbatim past instances, each with its verdict."""
    if not records:
        return ""
    lines = ["", RAG_HEADER, ""]
    for i, record in enumerate(records, 1):
        lines.append(
            f"### Past attempt {i} — {_outcome_label(record)} "
            f"(task {record.level}, instance {_instance_ref(record)})"
        )
        lines.append(record.content)
        lines.append("")
    lines.append(RAG_FOOTER)
    return "\n".join(lines) + "\n"


def format_memrl_block(records: Sequence[MemoryRecord]) -> str:
    """MemRL: summarized experiences, ordered by their composite score."""
    if not records:
        return ""
    lines = ["", MEMRL_HEADER, ""]
    for i, record in enumerate(records, 1):
        lines.append(
            f"### Memory {i} — {_outcome_label(record)} "
            f"(instance {_instance_ref(record)}, utility Q={record.q:+.3f}, "
            f"updates={record.q_visits})"
        )
        lines.append(record.content)
        lines.append("")
    lines.append(MEMRL_FOOTER)
    return "\n".join(lines) + "\n"
