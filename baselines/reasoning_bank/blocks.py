"""Prompt block for retrieved ReasoningBank memories.

Appended to the agent's system prompt at the same seam as Episodic RAG's and
MemRL's blocks (`baselines/runner/memory_runner.py`), so the base prompt stays
byte-identical across methods and the only difference is the injected text. Each
record's `.content` already holds its rendered items (see `rb.render_items`); this
just wraps the retrieved records with a header and footer.
"""

from __future__ import annotations

from typing import Sequence

from baselines.memory.store import SUCCESS, MemoryRecord

RB_HEADER = "## Learned Reasoning Memory (distilled from your own past attempts on this level)"
RB_FOOTER = (
    "These are generalizable strategies (from past successes) and lessons (from "
    "past failures). Apply the ones whose 'Use when' matches this scene, adapting "
    "them to this seed's geometry rather than copying coordinates blindly."
)


def _outcome_label(record: MemoryRecord) -> str:
    return "SUCCESS" if record.kind == SUCCESS else "FAILURE"


def format_reasoning_bank_block(records: Sequence[MemoryRecord]) -> str:
    """ReasoningBank: distilled reasoning items from the top-k retrieved experiences."""
    if not records:
        return ""
    lines = ["", RB_HEADER, ""]
    for i, record in enumerate(records, 1):
        lines.append(
            f"### Memory {i} — distilled from a {_outcome_label(record)} "
            f"(seed {record.seed})"
        )
        lines.append(record.content)
        lines.append("")
    lines.append(RB_FOOTER)
    return "\n".join(lines) + "\n"
