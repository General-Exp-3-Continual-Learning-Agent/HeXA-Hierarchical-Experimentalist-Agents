"""Prompt blocks for retrieved memories, plus added-context accounting.

The block is appended to the agent's system prompt exactly where HeXA appends
its `LEARNED PHYSICS SKILLS` block (`skillrl/runner/augmented_runner.py:163`),
so the base prompt stays byte-identical across methods.  HeXA's
`DIAGNOSTIC_ADDENDUM` is deliberately *not* used: it instructs the agent to cite
skills, which is HeXA-specific scaffolding and would confound the baselines.

`count_tokens` is the context-efficiency axis from the CORE paper (Figure 3:
added tokens per evaluation item).  It is logged for every episode.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .store import SUCCESS, MemoryRecord

RAG_HEADER = "## Retrieved Past Rollouts (from your own past attempts on this level)"
RAG_FOOTER = (
    "These are complete past attempts with the environment's verdict. Use the "
    "successful rollout as a template where the geometry matches, and avoid "
    "repeating what the unsuccessful rollout did."
)

MEMRL_HEADER = "## Archived Memories (from your own past experiences)"
MEMRL_FOOTER = (
    "These experiences were retrieved because they are similar to the current "
    "scene and have been useful in the past. Adapt them to this seed's geometry "
    "rather than copying coordinates blindly."
)


def _outcome_label(record: MemoryRecord) -> str:
    return "SUCCESS" if record.kind == SUCCESS else "FAILURE"


def format_rag_block(records: Sequence[MemoryRecord]) -> str:
    """Episodic RAG: verbatim rollouts, each with its verifier feedback."""
    if not records:
        return ""
    lines = ["", RAG_HEADER, ""]
    for i, record in enumerate(records, 1):
        lines.append(
            f"### Past rollout {i} — {_outcome_label(record)} "
            f"(level {record.level}, seed {record.seed})"
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
            f"(seed {record.seed}, utility Q={record.q:+.3f}, "
            f"updates={record.q_visits})"
        )
        lines.append(record.content)
        lines.append("")
    lines.append(MEMRL_FOOTER)
    return "\n".join(lines) + "\n"


_ENCODER = None


def count_tokens(text: str, *, encoding: str = "cl100k_base") -> Optional[int]:
    """Token count of an injected block, or None if tiktoken is unavailable."""
    global _ENCODER
    if not text:
        return 0
    if _ENCODER is None:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding(encoding)
        except Exception:
            _ENCODER = False
    if _ENCODER is False:
        return None
    return len(_ENCODER.encode(text))
