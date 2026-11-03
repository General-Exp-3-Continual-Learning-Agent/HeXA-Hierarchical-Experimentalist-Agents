"""ACE Curator with UPDATE / MERGE / DELETE (`--edit-ops`), a deviation from the
reference code.

The paper (Sec. 3.2) describes a playbook that is refined in place: bullets are
updated, merged and pruned.  The released code (github.com/ace-agent/ace @
82709de) implements ADD only -- its Curator prompt forbids "UPDATE, MERGE,
DELETE, or CREATE_META", and `playbook_utils.apply_curator_operations` leaves
the three as a TODO stub.  Under ADD only the playbook can never shrink except
through the cosine >= 0.9 dedup, which on catapult never fired (151 bullets,
~25k tokens, none removed).

This module fills in the TODO stub's operations with the fields it names
(UPDATE: bullet_id + content; MERGE: source_ids; DELETE: bullet_id) and gives
the Curator a prompt that offers them.  The prompt is the reference
CURATOR_PROMPT with only the ADD-specific lines changed; the vendored
`playbook.py` is left untouched so the ADD-only path stays verbatim.

Merge semantics follow the dedup merge (`dedup.py`): the merged bullet keeps the
ID of its first source and the summed helpful/harmful counts, so a merge never
erases the evidence the Reflector has recorded.  An UPDATE keeps the ID and the
counts.  Ops that name an unknown ID are skipped (counted, not fatal), since the
playbook may have changed between the Curator's read and an earlier op.
"""
from __future__ import annotations

from baselines.ace import playbook as pb

CURATOR_PROMPT_EDIT = """You are a master curator of knowledge. Your job is to keep an existing playbook accurate, compact and useful, based on a reflection from a previous attempt: add what is missing, and fix, merge or remove what is wrong or redundant.

**Context:**
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used.
- Each bullet shows helpful=X harmful=Y: how often it was judged helpful or harmful on past attempts.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- ADD only NEW insights, strategies, or mistakes that are MISSING from the current playbook
- UPDATE a bullet when the reflection shows its advice is incomplete, imprecise, or wrong
- MERGE bullets that give overlapping or redundant advice into one stronger bullet
- DELETE bullets that are wrong, contradicted by the evidence, redundant, or have proven harmful (harmful count clearly above helpful)
- Prefer refining an existing bullet over adding a near-duplicate of it
- Do NOT regenerate the entire playbook - only provide the operations needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- If nothing needs to change, return an empty list for the operations field
- Be concise and specific - each bullet should be actionable


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Task Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook, applied in order

**Available Operations:**
1. ADD: Create a new bullet point with a fresh ID
    - section: the section to add the new bullet to
    - content: the content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.
2. UPDATE: Rewrite an existing bullet (its ID and counts are kept)
    - bullet_id: the ID of the bullet to rewrite, e.g. "sai-00012"
    - content: the new content of the bullet
3. MERGE: Replace two or more existing bullets with one (keeps the first ID and the summed counts)
    - source_ids: the IDs of the bullets to merge, e.g. ["sai-00003", "err-00007"]
    - content: the content of the merged bullet
4. DELETE: Remove an existing bullet
    - bullet_id: the ID of the bullet to remove

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{"type": "ADD", "section": "formulas_and_calculations", "content": "[New calculation method...]"}},
    {{"type": "UPDATE", "bullet_id": "xmpl-00001", "content": "[Corrected advice...]"}},
    {{"type": "MERGE", "source_ids": ["xmpl-00002", "xmpl-00003"], "content": "[Combined advice...]"}},
    {{"type": "DELETE", "bullet_id": "xmpl-00004"}}
  ]
}}

---
"""

#: Fields each op needs; a missing field rejects the whole response, as the
#: reference does for ADD (`ace/core/curator.py:_extract_and_validate_operations`).
REQUIRED = {
    "ADD": {"section", "content"},
    "UPDATE": {"bullet_id", "content"},
    "MERGE": {"source_ids", "content"},
    "DELETE": {"bullet_id"},
}


def validate_edit_operations(ops: list) -> list[dict]:
    """Field checks for the four op types; raises ValueError like the reference."""
    for i, op in enumerate(ops):
        kind = op.get("type")
        if kind not in REQUIRED:
            continue        # unknown types are ignored downstream (counted)
        missing = REQUIRED[kind] - set(op)
        if missing:
            raise ValueError(f"{kind} operation {i} missing fields: {sorted(missing)}")
        if kind == "MERGE" and (not isinstance(op["source_ids"], list) or len(op["source_ids"]) < 2):
            raise ValueError(f"MERGE operation {i} needs a list of >= 2 source_ids")
    return ops


def _norm(section: str) -> str:
    return section.lower().strip().replace(" ", "_").replace("&", "and")


def _insert(lines: list[str], section: str, line: str) -> None:
    """Append `line` at the end of `section` (OTHERS if the section is unknown,
    as in the reference ADD)."""
    headers = {_norm(l.strip()[2:]): i for i, l in enumerate(lines) if l.strip().startswith("##")}
    if section not in headers:
        section = "others"
    if section not in headers:
        lines.append(line)
        return
    i = headers[section] + 1
    end = i
    while i < len(lines) and not lines[i].strip().startswith("##"):
        if lines[i].strip():
            end = i + 1
        i += 1
    lines.insert(end, line)


def apply_edit_operations(playbook_text: str, operations: list, next_id: int):
    """Apply ADD / UPDATE / MERGE / DELETE in order.

    Returns (playbook, next_id, stats) where stats counts what was applied and
    what was skipped because an ID was not in the playbook.
    """
    lines = playbook_text.strip().split("\n")
    stats = {"added": 0, "updated": 0, "merged_sources_removed": 0, "merges": 0,
             "deleted": 0, "skipped_unknown_id": 0, "skipped_unknown_type": 0}

    def find(bid):
        for i, l in enumerate(lines):
            p = pb.parse_playbook_line(l)
            if p and p["id"] == bid:
                return i, p
        return None, None

    for op in operations:
        kind = op.get("type")
        if kind == "ADD":
            section = _norm(op.get("section", "others"))
            bid = f"{pb.get_section_slug(section if section in _sections(lines) else 'others')}-{next_id:05d}"
            next_id += 1
            _insert(lines, section, pb.format_playbook_line(bid, 0, 0, op["content"]))
            stats["added"] += 1
        elif kind == "UPDATE":
            i, p = find(str(op["bullet_id"]).strip("[] "))
            if i is None:
                stats["skipped_unknown_id"] += 1
                continue
            lines[i] = pb.format_playbook_line(p["id"], p["helpful"], p["harmful"], op["content"])
            stats["updated"] += 1
        elif kind == "DELETE":
            i, _ = find(str(op["bullet_id"]).strip("[] "))
            if i is None:
                stats["skipped_unknown_id"] += 1
                continue
            del lines[i]
            stats["deleted"] += 1
        elif kind == "MERGE":
            found = []
            for bid in op["source_ids"]:
                i, p = find(str(bid).strip("[] "))
                if i is not None and all(p["id"] != q["id"] for _, q in found):
                    found.append((i, p))
            if len(found) < 2:
                stats["skipped_unknown_id"] += 1
                continue
            first_i, first = found[0]
            helpful = sum(p["helpful"] for _, p in found)
            harmful = sum(p["harmful"] for _, p in found)
            lines[first_i] = pb.format_playbook_line(first["id"], helpful, harmful, op["content"])
            for i, _ in sorted(found[1:], reverse=True):
                del lines[i]
            stats["merges"] += 1
            stats["merged_sources_removed"] += len(found) - 1
        else:
            stats["skipped_unknown_type"] += 1
    return "\n".join(lines), next_id, stats


def _sections(lines: list[str]) -> set:
    return {_norm(l.strip()[2:]) for l in lines if l.strip().startswith("##")}
