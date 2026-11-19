"""Ablation: Joint synthesis vs. independent transfer.

Tests whether combining source knowledge *during* synthesis helps beyond
independently adapting the same knowledge and presenting it together.

Two conditions, built from identical source banks and the identical factual
target description (no target trajectories or outcomes ever enter synthesis):

  JOINT (the existing HExA pipeline)
    All source banks + target description -> ONE synthesis call -> joint bank.

  INDEPENDENT (this ablation)
    Each source bank alone + target description -> a target-adapted bank
    (n_sources=1 per call). The per-source banks are POOLED verbatim, with no
    rewriting or merging of entries.

Both conditions then pass through the SAME confidence-free selection-only
procedure under the SAME final skill / mistake / token budgets. Selection
returns IDs; entries are copied verbatim into a frozen bank. Confidence is
neutralised in the frozen bank so it plays no role in eval-time retrieval
(the existing confidence rule favours multi-source entries, which would
confound the comparison). Source citations and transfer rationales are
preserved in the audit sidecars for inspection.

The original pipeline (`synthesize_target_bank`) is untouched; this module
only orchestrates it and adds the pooling + selection steps.

Outputs under ``<out-dir>/rep<r>/``:
  joint/skill_bank_xl_<target>.json            raw joint synthesis
  independent/<source>/skill_bank_xl_<target>.json   per-source adaptations
  bank_pooled_independent.json (+ _audit)      verbatim pool
  bank_joint_selected.json     (+ _selection_log)   FROZEN, evaluate this
  bank_independent_selected.json (+ _selection_log) FROZEN, evaluate this

Run commands are in the module docstring of
``skillrl/analysis/analyze_joint_vs_independent.py`` and in
``jobs/ablation_joint_vs_independent_catapult.sh``.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from skillrl.core.config import TEACHER_MODEL
from skillrl.core.skill_bank import Mistake, Skill, SkillBank
from skillrl.distillation.distill import _parse_json_object, call_teacher
from skillrl.distillation.cross_level_synthesis import (
    synthesize_target_bank,
    _parse_source_bank_arg,
)


# ── token estimate (deterministic cost proxy) ──────────────────────────

def estimate_tokens(text: str) -> int:
    """Best-effort token count; falls back to a chars/4 heuristic."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ── confidence-free selection-only procedure (shared by both conditions) ─

SELECTION_PROMPT = """\
You are selecting the most useful entries from a CANDIDATE skill bank for a
physics puzzle agent. You are NOT allowed to rewrite, merge, or edit any
entry — only to choose which existing entries to keep, by id.

TARGET LEVEL: {target_level}

{target_block}

═══ BUDGET ═══
- Keep AT MOST {max_skills} skills.
- Keep AT MOST {max_mistakes} mistakes.
- The kept skills, rendered together, should stay within about
  {token_budget} tokens of injected context. Prefer fewer, denser,
  non-overlapping entries over many near-duplicates.

═══ SELECTION CRITERIA ═══
- Coverage: prefer a set that covers distinct mechanisms over one that
  repeats the same idea.
- Specificity: prefer entries that name a concrete mechanism and a clear
  when-to-apply condition over vague ones.
- Target fit: prefer entries whose principle plainly applies to the target
  scene above.
Do NOT use any notion of author confidence or how many sources an entry
cites — judge each entry only on its own content and its fit to the target.

═══ CANDIDATE SKILLS ═══
{skills_block}

═══ CANDIDATE MISTAKES ═══
{mistakes_block}

═══ OUTPUT (single JSON object, nothing else) ═══
```json
{{
  "skill_ids": ["<id>", "..."],
  "mistake_ids": ["<id>", "..."],
  "reason": "<2-3 sentences on why this set, in terms of coverage and target fit>"
}}
```
Return only ids that appear in the candidates above. Respect the budget caps.
"""


def _render_candidate_skills(bank: SkillBank, target_level: str) -> str:
    out = []
    for s in bank.level_skills.get(target_level, []):
        out.append(
            f"- id: {s.skill_id}\n"
            f"  title: {s.title}\n"
            f"  principle: {s.principle}\n"
            f"  when_to_apply: {s.when_to_apply}"
            + (f"\n  example: {s.example}" if s.example else "")
        )
    return "\n".join(out) if out else "(none)"


def _render_candidate_mistakes(bank: SkillBank, target_level: str) -> str:
    out = []
    for m in bank.level_mistakes.get(target_level, []):
        out.append(
            f"- id: {m.mistake_id}\n"
            f"  description: {m.description}\n"
            f"  why_it_happens: {m.why_it_happens}\n"
            f"  how_to_avoid: {m.how_to_avoid}"
        )
    return "\n".join(out) if out else "(none)"


def select_under_budget(
    candidate_bank_path: Path,
    target_level: str,
    target_block: str,
    out_dir: Path,
    tag: str,
    max_skills: int,
    max_mistakes: int,
    token_budget: int,
    teacher_model: str = TEACHER_MODEL,
    neutral_confidence: float = 0.5,
) -> tuple[Path, dict]:
    """Confidence-free LLM selection; copy chosen entries verbatim into a frozen bank.

    Identical procedure for both conditions. Returns (frozen_bank_path, log).
    """
    cand = SkillBank.load(candidate_bank_path)
    skills_by_id = {s.skill_id: s for s in cand.level_skills.get(target_level, [])}
    mistakes_by_id = {m.mistake_id: m for m in cand.level_mistakes.get(target_level, [])}

    prompt = SELECTION_PROMPT.format(
        target_level=target_level,
        target_block=target_block,
        max_skills=max_skills,
        max_mistakes=max_mistakes,
        token_budget=token_budget,
        skills_block=_render_candidate_skills(cand, target_level),
        mistakes_block=_render_candidate_mistakes(cand, target_level),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"_selection_prompt_{tag}.txt").write_text(prompt)

    response = call_teacher(prompt, model=teacher_model)
    (out_dir / f"_selection_raw_{tag}.txt").write_text(response or "")
    parsed = _parse_json_object(response) if response else None
    if not parsed:
        raise RuntimeError(
            f"Selection call for '{tag}' returned unparseable output. "
            f"See {out_dir / f'_selection_raw_{tag}.txt'}."
        )

    # Keep only ids that exist; preserve the model's order; enforce caps in code.
    chosen_skill_ids = [i for i in parsed.get("skill_ids", []) if i in skills_by_id][:max_skills]
    chosen_mistake_ids = [i for i in parsed.get("mistake_ids", []) if i in mistakes_by_id][:max_mistakes]

    frozen = SkillBank()
    for sid in chosen_skill_ids:
        s = copy.deepcopy(skills_by_id[sid])     # verbatim copy of content
        s.confidence = neutral_confidence        # neutralise: no confidence signal at eval
        frozen.add_skill(s)
    for mid in chosen_mistake_ids:
        frozen.add_mistake(copy.deepcopy(mistakes_by_id[mid]))

    frozen_path = out_dir / f"bank_{tag}_selected.json"
    frozen.save(frozen_path)

    injected_block = frozen.format_skills_as_prompt(target_level)
    log = {
        "tag": tag,
        "candidate_bank": str(candidate_bank_path),
        "target_level": target_level,
        "budget": {"max_skills": max_skills, "max_mistakes": max_mistakes,
                   "token_budget": token_budget},
        "n_candidates": {"skills": len(skills_by_id), "mistakes": len(mistakes_by_id)},
        "chosen_skill_ids": chosen_skill_ids,
        "chosen_mistake_ids": chosen_mistake_ids,
        "selection_reason": parsed.get("reason", ""),
        "injected_tokens": estimate_tokens(injected_block),
        "frozen_bank": str(frozen_path),
        "confidence_neutralised_to": neutral_confidence,
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"_selection_log_{tag}.json").write_text(json.dumps(log, indent=2))
    print(f"  [select:{tag}] kept {len(chosen_skill_ids)} skills, "
          f"{len(chosen_mistake_ids)} mistakes  (~{log['injected_tokens']} tok injected)")
    return frozen_path, log


# ── independent synthesis + verbatim pooling ───────────────────────────

def synthesize_independent(
    target_level: str,
    source_bank_paths: dict[str, Path],
    out_dir: Path,
    teacher_model: str = TEACHER_MODEL,
    **synth_kwargs,
) -> dict[str, Path]:
    """One single-source synthesis call per source bank (n_sources=1 each)."""
    per_source: dict[str, Path] = {}
    for source_level, bank_path in source_bank_paths.items():
        sdir = out_dir / "independent" / source_level
        print(f"\n[independent] {source_level} -> {target_level}")
        synthesize_target_bank(
            target_level=target_level,
            source_bank_paths={source_level: bank_path},   # n_sources = 1
            output_dir=sdir,
            teacher_model=teacher_model,
            **synth_kwargs,
        )
        per_source[source_level] = sdir / f"skill_bank_xl_{target_level}.json"
    return per_source


def pool_independent(
    target_level: str,
    per_source_banks: dict[str, Path],
    out_dir: Path,
) -> Path:
    """Pool per-source adapted banks verbatim — no rewrite, no merge.

    Entry ids are re-prefixed with the source level so each pooled entry is
    unique and traceable to the single-source bank it came from. All other
    fields (principle, when_to_apply, example, confidence, source citations in
    the audit) are preserved unchanged.
    """
    pooled = SkillBank()
    audit_skills: list[dict] = []
    audit_mistakes: list[dict] = []

    for source_level, bank_path in per_source_banks.items():
        bank = SkillBank.load(bank_path)
        # per-source audit for citations/rationales
        audit_path = bank_path.parent / f"skill_bank_xl_{target_level}_audit.json"
        src_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        aud_skill = {a["skill_id"]: a for a in src_audit.get("skills", [])}
        aud_mis = {a["mistake_id"]: a for a in src_audit.get("mistakes", [])}
        prefix = source_level[:3]

        for s in bank.level_skills.get(target_level, []):
            new = copy.deepcopy(s)
            new.skill_id = f"{prefix}__{s.skill_id}"
            pooled.add_skill(new)
            a = aud_skill.get(s.skill_id, {})
            audit_skills.append({
                "skill_id": new.skill_id,
                "pooled_from_source": source_level,
                "original_skill_id": s.skill_id,
                "title": s.title,
                "confidence": s.confidence,
                "source_skills": a.get("source_skills", []),
                "transfer_rationale": a.get("transfer_rationale", ""),
            })
        for m in bank.level_mistakes.get(target_level, []):
            new = copy.deepcopy(m)
            new.mistake_id = f"{prefix}__{m.mistake_id}"
            pooled.add_mistake(new)
            a = aud_mis.get(m.mistake_id, {})
            audit_mistakes.append({
                "mistake_id": new.mistake_id,
                "pooled_from_source": source_level,
                "original_mistake_id": m.mistake_id,
                "description": m.description,
                "source_skills": a.get("source_skills", []),
                "transfer_rationale": a.get("transfer_rationale", ""),
            })

    pooled_path = out_dir / "bank_pooled_independent.json"
    pooled.save(pooled_path)
    audit = {
        "target_level": target_level,
        "pooled_from": {k: str(v) for k, v in per_source_banks.items()},
        "n_skills": len(audit_skills),
        "n_mistakes": len(audit_mistakes),
        "note": "verbatim pool; ids re-prefixed by source; no rewrite or merge",
        "skills": audit_skills,
        "mistakes": audit_mistakes,
        "pooled_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "bank_pooled_independent_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\n[pool] {len(audit_skills)} skills + {len(audit_mistakes)} mistakes "
          f"-> {pooled_path}")
    return pooled_path


def synthesize_joint(
    target_level: str,
    source_bank_paths: dict[str, Path],
    out_dir: Path,
    teacher_model: str = TEACHER_MODEL,
    **synth_kwargs,
) -> Path:
    """Existing joint pipeline: all sources in one synthesis call."""
    jdir = out_dir / "joint"
    print(f"\n[joint] {list(source_bank_paths)} -> {target_level}")
    synthesize_target_bank(
        target_level=target_level,
        source_bank_paths=source_bank_paths,
        output_dir=jdir,
        teacher_model=teacher_model,
        **synth_kwargs,
    )
    return jdir / f"skill_bank_xl_{target_level}.json"


# ── driver ─────────────────────────────────────────────────────────────

def _target_block_for(target_level: str) -> str:
    from skillrl.distillation.cross_level_synthesis import TARGET_LEVEL_BLOCKS
    if target_level not in TARGET_LEVEL_BLOCKS:
        raise ValueError(f"No TARGET_LEVEL_BLOCKS entry for '{target_level}'.")
    return TARGET_LEVEL_BLOCKS[target_level]


def run_replicate(
    target_level: str,
    source_bank_paths: dict[str, Path],
    out_dir: Path,
    condition: str,
    max_skills: int,
    max_mistakes: int,
    token_budget: int,
    teacher_model: str = TEACHER_MODEL,
    **synth_kwargs,
) -> dict:
    """Build the frozen bank(s) for one replicate. condition in {joint,independent,both}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_block = _target_block_for(target_level)
    result: dict = {"target_level": target_level, "out_dir": str(out_dir),
                    "condition": condition}

    if condition in ("joint", "both"):
        joint_bank = synthesize_joint(target_level, source_bank_paths, out_dir,
                                      teacher_model, **synth_kwargs)
        _, jlog = select_under_budget(
            joint_bank, target_level, target_block, out_dir, tag="joint",
            max_skills=max_skills, max_mistakes=max_mistakes,
            token_budget=token_budget, teacher_model=teacher_model)
        result["joint"] = jlog

    if condition in ("independent", "both"):
        per_source = synthesize_independent(target_level, source_bank_paths, out_dir,
                                            teacher_model, **synth_kwargs)
        pooled_bank = pool_independent(target_level, per_source, out_dir)
        _, ilog = select_under_budget(
            pooled_bank, target_level, target_block, out_dir, tag="independent",
            max_skills=max_skills, max_mistakes=max_mistakes,
            token_budget=token_budget, teacher_model=teacher_model)
        result["independent"] = ilog

    (out_dir / "replicate_summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Joint-synthesis vs independent-transfer ablation (synthesis side).")
    p.add_argument("--target", required=True, help="Target level, e.g. catapult")
    p.add_argument("--source-bank", action="append", required=True,
                   metavar="level=path",
                   help="Repeatable. Source level and its evolved bank path.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Replicate output dir (e.g. .../ablation/rep0).")
    p.add_argument("--condition", choices=["joint", "independent", "both"],
                   default="both")
    p.add_argument("--max-skills", type=int, default=6,
                   help="Final skill budget (matches eval --max-specific-skills).")
    p.add_argument("--max-mistakes", type=int, default=4,
                   help="Final mistake budget (matches eval --max-mistakes).")
    p.add_argument("--token-budget", type=int, default=1200,
                   help="Approx injected-context token budget for selection.")
    p.add_argument("--teacher-model", default=TEACHER_MODEL)
    # passthrough synthesis flags (kept identical across conditions)
    p.add_argument("--transfer-guard", action="store_true")
    p.add_argument("--detailed-scenes", action="store_true")
    args = p.parse_args()

    source_bank_paths = dict(_parse_source_bank_arg(s) for s in args.source_bank)
    run_replicate(
        target_level=args.target,
        source_bank_paths=source_bank_paths,
        out_dir=args.out_dir,
        condition=args.condition,
        max_skills=args.max_skills,
        max_mistakes=args.max_mistakes,
        token_budget=args.token_budget,
        teacher_model=args.teacher_model,
        transfer_guard=args.transfer_guard,
        detailed_scenes=args.detailed_scenes,
    )


if __name__ == "__main__":
    main()
