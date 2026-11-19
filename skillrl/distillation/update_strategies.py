"""Skill-bank update ablations for the HExA evolving loop (``--update-strategy``).

Full HExA (``evolution``) rewrites the whole bank each round in ONE teacher call
that sees the previous bank and the new trajectories (it can merge, rewrite,
re-weight, drop and add). The two ablations below restrict what the update may
do, while every other part of the loop (K0, training seeds, reward, retriever,
M/N, bank caps, actor prompt) stays the same:

``append_only``
    1. EXTRACT: one teacher call that sees ONLY the new trajectories (never the
       bank) proposes new skills and mistakes.
    2. DEDUP: one judge call drops candidates that duplicate an existing entry
       (or an earlier candidate). The judge only labels; it writes no content.
    3. APPEND the survivors; existing entries are never edited (text, confidence
       and order unchanged).
    4. EVICT to the same caps as full HExA: lowest-confidence skills first
       (ties: older generation, then earlier position). Mistakes carry no
       confidence, so the oldest mistakes are evicted first.

``single_parent``
    0. REVISE: all existing SKILLS in ONE batched teacher call and all MISTAKES
       in ONE batched call (2 calls per round). The prompt asks for keep /
       revise / delete per id and forbids merging; code enforces one decision
       per existing id (missing/invalid -> keep), so an entry can yield at most
       one revised child and nothing is merged. Each revised entry gets a new id and its
       ``parent_id`` is logged.
    1-4. Same extraction / dedup / append / eviction as ``append_only``, with
       dedup run against the post-revision bank.

``summary`` (the "summary bank" baseline, no structure at all)
    The memory is ONE free-text block of insights/lessons -- no separate skills or
    mistakes, no confidence/reward weighting, no retrieval, no ids. Each round the
    teacher sees the PREVIOUS summary and the new trajectories and writes the NEW
    summary (capped at ``SUMMARY_MAX_WORDS``). Round 1 writes the first summary from
    the warm-start trajectories (empty previous summary). The summary is stored in
    ``SkillBank.summary_text`` and injected whole in place of the skills/mistakes block.

Per-round audit: ``<bank>_update_log.json`` next to each saved bank (strategy,
candidates, dedup verdicts, revisions with parent_id, evictions, raw outputs).
New-entry confidence uses the same reward-based ``_compute_initial_confidence``
as new skills in full HExA; trajectories are rendered with the same
``format_trajectories_block(..., max_trajs=5)``.
"""

from __future__ import annotations

import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from skillrl.core.config import LEVEL_DESCRIPTIONS, TEACHER_MODEL
from skillrl.core.skill_bank import Mistake, Skill, SkillBank
from skillrl.distillation.distill import (
    _compute_initial_confidence,
    _parse_json_object,
    call_teacher,
    load_trajectories,
)
from skillrl.distillation.teacher_prompts import format_trajectories_block

UPDATE_STRATEGIES = ("evolution", "append_only", "single_parent", "summary")
# Budget = size of the block HExA INJECTS into the actor prompt (6 skills + 4 mistakes, as rendered):
# catapult run2/4/5 final banks = 922/1023/1079 words (1.5k/2.2k/2.3k cl100k tokens), mean ~1010 words.
SUMMARY_MAX_WORDS = 1000
REVISION_WORKERS = 4  # concurrent per-entry teacher calls (Claude CLI processes)


class TeacherCallFailed(RuntimeError):
    """A teacher call returned nothing (CLI error / usage limit / server down).

    Raised BEFORE the round's bank is saved, so the run can simply be resumed:
    the loop recomputes this round's update instead of saving a silently
    degraded bank (unparseable revisions would otherwise count as "keep")."""


def _require(raw, what):
    if not (raw or "").strip():
        raise TeacherCallFailed(f"empty teacher output for {what}; not saving this round's bank")


def _level_block(level_name: str) -> str:
    if level_name == "catapult":
        from skillrl.distillation.teacher_prompts_catapult import CATAPULT_LEVEL_BLOCK
        return CATAPULT_LEVEL_BLOCK
    return LEVEL_DESCRIPTIONS.get(level_name, "No description available.")


_HEADER = """\
**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5].

**Level: {level_name}**
{level_block}
"""

EXTRACT_PROMPT = """\
You are a physics teacher extracting skills for a puzzle-solving agent from its latest attempts.

""" + _HEADER + """
NEW TRAJECTORIES (from the latest round):
Successes: {n_successes}
Failures: {n_failures}

{new_trajectories_block}

YOUR TASK: Extract the physics skills and the mistakes that THESE trajectories support.

RULES:
1. Ground every entry in the trajectories above. Include "source_seeds": the seed numbers it
   was observed in (required for confidence calibration).
2. At most {max_skills} skills and at most {max_mistakes} mistakes. Do not pad: if the
   trajectories support fewer, output fewer (an empty list is allowed).
3. No duplicate entries.

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence physics insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<optional concrete coordinate example>",
      "source_seeds": [<seed numbers>]
    }}
  ],
  "mistakes": [
    {{
      "description": "<what the mistake is>",
      "why_it_happens": "<why agents make this mistake>",
      "how_to_avoid": "<actionable fix>",
      "source_seeds": [<seed numbers>]
    }}
  ]
}}

Be concise but precise. Focus on physics insights grounded in the trajectories.
"""

DEDUP_PROMPT = """\
You are checking candidate entries for a skill bank against the entries it already has.
Compare skills only with skills and mistakes only with mistakes.
A candidate is a DUPLICATE only if an existing entry (or an EARLIER candidate) of the SAME
kind already makes the same claim AND would lead the agent to the same action -- i.e. the
agent would gain nothing from reading the candidate as well. It is NOT a duplicate if it
adds anything specific the other entry lacks: a different or narrower condition, a
parameter value or range, a direction of adjustment, a different mechanism, a procedure or
tool-use step, or a different fix. Sharing a topic (e.g. both about radius) is not enough.
When in doubt, it is NOT a duplicate. Judge only; do not rewrite anything.

EXISTING ENTRIES:
{existing}

CANDIDATES:
{candidates}

Output ONLY a JSON object:
{{"verdicts": [{{"candidate": "<candidate id>", "duplicate_of": "<existing id or earlier candidate id, or null>", "why": "<short>"}}]}}
"""

REVISE_PROMPT = """\
You are a physics teacher maintaining ONE entry of a skill bank for a puzzle-solving agent.

""" + _HEADER + """
THE ENTRY ({kind}):
{entry}

NEW TRAJECTORIES (from the latest round):
Successes: {n_successes}
Failures: {n_failures}

{new_trajectories_block}

YOUR TASK: Decide what to do with THIS entry given the new trajectories.
- "keep": the entry is still correct and useful as written (also choose this if the new
  trajectories say nothing about it).
- "revise": the new trajectories show the entry should be corrected, sharpened or updated.
  Output the full revised entry. Revise only this entry; do not fold in other insights.
- "delete": the new trajectories contradict it, or it is redundant/too specific to be useful.
{confidence_rule}
OUTPUT JSON OBJECT:
{{"decision": "keep|revise|delete", "reason": "<1 sentence>", "revised": {revised_schema}}}
("revised" is null unless decision is "revise".)
"""

BATCH_REVISE_PROMPT = """\
You are a physics teacher maintaining the {kind_upper} list of a skill bank for a puzzle-solving agent.

""" + _HEADER + """
CURRENT {kind_upper} (each has an id):
{entries}

NEW TRAJECTORIES (from the latest round):
Successes: {n_successes}
Failures: {n_failures}

{new_trajectories_block}

YOUR TASK: For EACH {kind} above, decide separately what to do given the new trajectories:
- "keep": still correct and useful as written (also if the trajectories say nothing about it).
- "revise": correct or sharpen THIS {kind}. Rewrite it from its own content and the
  trajectories only.
- "delete": contradicted by the trajectories, or no longer useful.
Treat every {kind} independently, as if it were the only one you were shown.
Never merge {kind_plural}, never combine content from two {kind_plural}, never move content
from one {kind} into another, never add new {kind_plural} here. Give exactly one decision per id.
{confidence_rule}
OUTPUT JSON OBJECT:
{{"decisions": [{{"id": "<{kind} id>", "decision": "keep|revise|delete", "reason": "<1 sentence>",
  "revised": {schema} or null}}]}}
"""

_SKILL_SCHEMA = ('{"title": "...", "principle": "...", "when_to_apply": "...", "example": "...", '
                 '"source_seeds": [<seed numbers>], "confidence": <float in [0.1, 1.0]>}')
_MISTAKE_SCHEMA = ('{"description": "...", "why_it_happens": "...", "how_to_avoid": "...", '
                   '"source_seeds": [<seed numbers>]}')
_SKILL_CONF_RULE = ("For a revised skill, give \"confidence\" in [0.1, 1.0]: keep the current "
                    "confidence unless the new trajectories clearly raise or lower it.\n")


# ── rendering helpers ────────────────────────────────────────────────────────

def _render_skill(s: Skill) -> str:
    out = [f"[{s.skill_id}] ({s.confidence * 100:.0f}%) {s.title}",
           f"    principle: {s.principle}"]
    if s.when_to_apply:
        out.append(f"    apply when: {s.when_to_apply}")
    if s.example:
        out.append(f"    example: {s.example}")
    return "\n".join(out)


def _render_mistake(m: Mistake) -> str:
    out = [f"[{m.mistake_id}] {m.description}"]
    if m.why_it_happens:
        out.append(f"    why it happens: {m.why_it_happens}")
    if m.how_to_avoid:
        out.append(f"    how to avoid: {m.how_to_avoid}")
    return "\n".join(out)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _prev_gen(bank: SkillBank, level: str) -> int:
    gens = [s.generation for s in bank.level_skills.get(level, [])]
    gens += [m.generation for m in bank.level_mistakes.get(level, [])]
    return max(gens, default=0)


def _seeds(raw: dict) -> list[int]:
    out = []
    for x in raw.get("source_seeds", []) or []:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            pass
    return out


# ── step 1: extraction (bank never shown) ────────────────────────────────────

def _extract(level, successes, failures, traj_block, max_skills, max_mistakes, teacher_model, log):
    prompt = EXTRACT_PROMPT.format(
        level_name=level, level_block=_level_block(level),
        n_successes=len(successes), n_failures=len(failures),
        new_trajectories_block=traj_block, max_skills=max_skills, max_mistakes=max_mistakes)
    raw = call_teacher(prompt, model=teacher_model)
    log["extract_raw"] = raw
    _require(raw, "extraction")
    parsed = _parse_json_object(raw or "") or {}
    if not parsed:
        print("  [update] extraction output unparseable -> no new entries this round")
    skills = [s for s in (parsed.get("skills") or []) if isinstance(s, dict) and s.get("title")]
    mistakes = [m for m in (parsed.get("mistakes") or []) if isinstance(m, dict) and m.get("description")]
    return skills[:max_skills], mistakes[:max_mistakes]


# ── step 2: dedup against the (current) bank ─────────────────────────────────

def _dedup(bank, level, cand_skills, cand_mistakes, teacher_model, log):
    if not cand_skills and not cand_mistakes:
        return cand_skills, cand_mistakes
    existing = "\n".join(
        [_render_skill(s) for s in bank.level_skills.get(level, [])]
        + [_render_mistake(m) for m in bank.level_mistakes.get(level, [])]) or "(none)"
    cid_s = [f"cand_skill_{i}" for i in range(len(cand_skills))]
    cid_m = [f"cand_mistake_{i}" for i in range(len(cand_mistakes))]
    cands = "\n".join(
        [f"[{cid_s[i]}] {c.get('title', '')}\n    principle: {c.get('principle', '')}\n"
         f"    apply when: {c.get('when_to_apply', '')}" for i, c in enumerate(cand_skills)]
        + [f"[{cid_m[i]}] {c.get('description', '')}\n    how to avoid: {c.get('how_to_avoid', '')}"
           for i, c in enumerate(cand_mistakes)])
    raw = call_teacher(DEDUP_PROMPT.format(existing=existing, candidates=cands), model=teacher_model)
    log["dedup_raw"] = raw
    _require(raw, "dedup")
    parsed = _parse_json_object(raw or "") or {}
    # A verdict counts only if it points at an entry of the SAME kind: an existing
    # entry, or an EARLIER candidate. Cross-kind or unknown targets are ignored.
    same_kind = {}
    ex_s = {s.skill_id for s in bank.level_skills.get(level, [])}
    ex_m = {m.mistake_id for m in bank.level_mistakes.get(level, [])}
    for i, c in enumerate(cid_s):
        same_kind[c] = ex_s | set(cid_s[:i])
    for i, c in enumerate(cid_m):
        same_kind[c] = ex_m | set(cid_m[:i])
    dup, rejected = {}, []
    for v in parsed.get("verdicts", []) or []:
        if isinstance(v, dict) and v.get("candidate") and v.get("duplicate_of") not in (None, "", "null"):
            cand, tgt = str(v["candidate"]), str(v["duplicate_of"])
            if cand in same_kind and tgt in same_kind[cand]:
                dup[cand] = (tgt, v.get("why", ""))
            else:
                rejected.append({"candidate": cand, "duplicate_of": tgt, "why": v.get("why", "")})
    log["dedup_rejected_verdicts"] = rejected
    if not parsed:
        # Fallback: exact (normalised) title / description match only.
        print("  [update] dedup output unparseable -> exact-text dedup fallback")
        seen = {_norm(s.title) for s in bank.level_skills.get(level, [])}
        for i, c in enumerate(cand_skills):
            k = _norm(c.get("title", ""))
            if k in seen:
                dup[cid_s[i]] = ("exact-title", "fallback")
            seen.add(k)
        seen = {_norm(m.description) for m in bank.level_mistakes.get(level, [])}
        for i, c in enumerate(cand_mistakes):
            k = _norm(c.get("description", ""))
            if k in seen:
                dup[cid_m[i]] = ("exact-text", "fallback")
            seen.add(k)
    log["dedup"] = [{"candidate": k, "duplicate_of": v[0], "why": v[1]} for k, v in dup.items()]
    keep_s = [c for i, c in enumerate(cand_skills) if cid_s[i] not in dup]
    keep_m = [c for i, c in enumerate(cand_mistakes) if cid_m[i] not in dup]
    print(f"  [update] dedup: skills {len(cand_skills)}->{len(keep_s)}, "
          f"mistakes {len(cand_mistakes)}->{len(keep_m)}")
    return keep_s, keep_m


# ── step 0 (single_parent): one call per existing entry ──────────────────────

def _revise_one(kind, entry, level, n_s, n_f, traj_block, teacher_model):
    prompt = REVISE_PROMPT.format(
        level_name=level, level_block=_level_block(level), kind=kind,
        entry=_render_skill(entry) if kind == "skill" else _render_mistake(entry),
        n_successes=n_s, n_failures=n_f, new_trajectories_block=traj_block,
        confidence_rule=_SKILL_CONF_RULE if kind == "skill" else "",
        revised_schema=_SKILL_SCHEMA if kind == "skill" else _MISTAKE_SCHEMA)
    raw = call_teacher(prompt, model=teacher_model)
    parsed = _parse_json_object(raw or "") or {}
    decision = str(parsed.get("decision", "")).lower().strip()
    if decision not in ("keep", "revise", "delete"):
        decision = "keep"  # unparseable -> leave the entry untouched
        parsed["reason"] = f"unparseable output; kept unchanged ({parsed.get('reason', '')})"
    rev = parsed.get("revised") if decision == "revise" else None
    if decision == "revise" and not isinstance(rev, dict):
        decision, rev = "keep", None
        parsed["reason"] = "revise without a revised entry; kept unchanged"
    return {"decision": decision, "reason": parsed.get("reason", ""), "revised": rev, "raw": raw}


def _revise_batch(kind, entries, level, n_s, n_f, traj_block, teacher_model):
    """ONE call for all entries of one kind; per-id keep/revise/delete enforced in code
    (no merging: each existing entry yields at most one revised child)."""
    if not entries:
        return []
    is_skill = kind == "skill"
    raw = call_teacher(BATCH_REVISE_PROMPT.format(
        level_name=level, level_block=_level_block(level), kind=kind,
        kind_upper="SKILLS" if is_skill else "MISTAKES",
        kind_plural="skills" if is_skill else "mistakes",
        entries="\n".join((_render_skill if is_skill else _render_mistake)(e) for e in entries),
        n_successes=n_s, n_failures=n_f, new_trajectories_block=traj_block,
        confidence_rule=_SKILL_CONF_RULE if is_skill else "",
        schema=_SKILL_SCHEMA if is_skill else _MISTAKE_SCHEMA), model=teacher_model)
    _require(raw, f"{kind}s batch revision")
    parsed = _parse_json_object(raw or "") or {}
    by_id = {}
    for d in parsed.get("decisions", []) or []:
        if isinstance(d, dict) and str(d.get("id")) not in by_id:
            by_id[str(d.get("id"))] = d
    out = []
    for e in entries:
        d = by_id.get(e.skill_id if is_skill else e.mistake_id, {})
        dec = str(d.get("decision", "")).lower().strip()
        rev = d.get("revised") if dec == "revise" else None
        reason = d.get("reason", "")
        if dec not in ("keep", "revise", "delete"):
            dec, reason = "keep", "no valid decision for this id; kept unchanged"
        elif dec == "revise" and not isinstance(rev, dict):
            dec, rev, reason = "keep", None, "revise without a revised entry; kept unchanged"
        out.append({"decision": dec, "reason": reason, "revised": rev,
                    "raw": raw if e is entries[0] else f"(see first {kind})"})
    return out


def _revise_all(bank, level, successes, failures, traj_block, teacher_model, round_num, gen, log):
    """Skills: ONE batched call. Mistakes: ONE batched call. (2 calls per round.)"""
    skills = list(bank.level_skills.get(level, []))
    mistakes = list(bank.level_mistakes.get(level, []))
    s_results = _revise_batch("skill", skills, level, len(successes), len(failures),
                              traj_block, teacher_model)
    m_results = _revise_batch("mistake", mistakes, level, len(successes), len(failures),
                              traj_block, teacher_model)
    jobs = [("skill", s) for s in skills] + [("mistake", m) for m in mistakes]
    results = s_results + m_results
    prefix = level[:3]
    new_skills, new_mistakes, revlog = [], [], []
    k = 0
    for (kind, e), r in zip(jobs, results):
        pid = e.skill_id if kind == "skill" else e.mistake_id
        entry_log = {"kind": kind, "parent_id": pid, "decision": r["decision"],
                     "reason": r["reason"], "raw": r["raw"]}
        if r["decision"] == "delete":
            pass
        elif r["decision"] == "keep":
            (new_skills if kind == "skill" else new_mistakes).append(e)
        else:
            rv = r["revised"]
            if kind == "skill":
                conf = rv.get("confidence", e.confidence)
                try:
                    conf = max(0.1, min(1.0, float(conf)))
                except (TypeError, ValueError):
                    conf = e.confidence
                nid = f"{prefix}_sp_r{round_num}_{k:03d}"
                new_skills.append(Skill(
                    skill_id=nid, title=rv.get("title", e.title),
                    principle=rv.get("principle", e.principle),
                    when_to_apply=rv.get("when_to_apply", e.when_to_apply),
                    example=rv.get("example", e.example), source_level=level,
                    source_type="single_parent_revised",
                    source_seeds=sorted(set(e.source_seeds) | set(_seeds(rv))),
                    generation=gen, confidence=conf))
            else:
                nid = f"{prefix}_sp_err_r{round_num}_{k:03d}"
                new_mistakes.append(Mistake(
                    mistake_id=nid, description=rv.get("description", e.description),
                    why_it_happens=rv.get("why_it_happens", e.why_it_happens),
                    how_to_avoid=rv.get("how_to_avoid", e.how_to_avoid), source_level=level,
                    source_seeds=sorted(set(e.source_seeds) | set(_seeds(rv))), generation=gen))
            k += 1
            entry_log["new_id"] = nid
        revlog.append(entry_log)
        print(f"    [revise] {kind} {pid}: {r['decision'].upper()}"
              + (f" -> {entry_log['new_id']}" if "new_id" in entry_log else ""))
    log["revisions"] = revlog
    bank.level_skills[level] = new_skills
    bank.level_mistakes[level] = new_mistakes


# ── step 3-4: append + evict ─────────────────────────────────────────────────

def _append_and_evict(bank, level, keep_s, keep_m, all_trajs, round_num, gen,
                      max_skills, max_mistakes, log):
    prefix = level[:3]
    added = []
    for i, c in enumerate(keep_s):
        seeds = _seeds(c)
        sk = Skill(skill_id=f"{prefix}_new_r{round_num}_{i:03d}", title=c.get("title", ""),
                   principle=c.get("principle", ""), when_to_apply=c.get("when_to_apply", ""),
                   example=c.get("example", ""), source_level=level, source_type="appended",
                   source_seeds=seeds, generation=gen,
                   confidence=_compute_initial_confidence(seeds, all_trajs, fallback_confidence=0.5))
        bank.add_skill(sk)
        added.append(sk.skill_id)
    for i, c in enumerate(keep_m):
        m = Mistake(mistake_id=f"{prefix}_new_err_r{round_num}_{i:03d}",
                    description=c.get("description", ""), why_it_happens=c.get("why_it_happens", ""),
                    how_to_avoid=c.get("how_to_avoid", ""), source_level=level,
                    source_seeds=_seeds(c), generation=gen)
        bank.add_mistake(m)
        added.append(m.mistake_id)
    log["added"] = added

    evicted = []
    skills = bank.level_skills.get(level, [])
    if len(skills) > max_skills:
        order = sorted(range(len(skills)), key=lambda j: (skills[j].confidence, skills[j].generation, j))
        drop = set(order[:len(skills) - max_skills])
        evicted += [{"id": skills[j].skill_id, "confidence": skills[j].confidence} for j in sorted(drop)]
        bank.level_skills[level] = [s for j, s in enumerate(skills) if j not in drop]
    mistakes = bank.level_mistakes.get(level, [])
    if len(mistakes) > max_mistakes:
        order = sorted(range(len(mistakes)), key=lambda j: (mistakes[j].generation, j))
        drop = set(order[:len(mistakes) - max_mistakes])
        evicted += [{"id": mistakes[j].mistake_id, "rule": "oldest"} for j in sorted(drop)]
        bank.level_mistakes[level] = [m for j, m in enumerate(mistakes) if j not in drop]
    log["evicted"] = evicted
    for a in added:
        print(f"    [append] {a}")
    for e in evicted:
        print(f"    [evict] {e['id']}")

# ── summary bank (compact-summary baseline) ──────────────────────────────────

SUMMARY_PROMPT = """\
You are maintaining a compact running summary of insights and lessons for a puzzle-solving agent.

""" + _HEADER + """
PREVIOUS SUMMARY:
{prev_summary}

NEW TRAJECTORIES (from the latest round):
Successes: {n_successes}
Failures: {n_failures}

{new_trajectories_block}

YOUR TASK: Write the NEW summary: the insights and lessons about this level that the agent
should know, given the previous summary and the new trajectories.
The summary must be at most {max_words} words.

Output ONLY the new summary text.
"""

CONDENSE_PROMPT = """\
Condense the following summary of insights and lessons to at most {max_words} words. Keep the
most useful, specific, actionable content (values, directions of adjustment, procedures).
Output ONLY the condensed summary text.

{summary}
"""


def _write_summary(level, prev_summary, successes, failures, teacher_model, max_words, log):
    # Summary bank: no reward anywhere in what the evolver sees (header line or ordering).
    traj_block = format_trajectories_block(successes + failures, max_trajs=5, show_reward=False)
    prompt = SUMMARY_PROMPT.format(
        level_name=level, level_block=_level_block(level),
        prev_summary=prev_summary.strip() or "(none yet)",
        n_successes=len(successes), n_failures=len(failures),
        new_trajectories_block=traj_block, max_words=max_words)
    raw = call_teacher(prompt, model=teacher_model)
    log["summary_raw"] = raw
    _require(raw, "summary")
    text = raw.strip()
    n = len(text.split())
    log["summary_words_first_pass"] = n
    if n > max_words * 1.1:
        raw2 = call_teacher(CONDENSE_PROMPT.format(max_words=max_words, summary=text), model=teacher_model)
        log["condense_raw"] = raw2
        _require(raw2, "summary condense")
        text = raw2.strip()
    log["summary_words"] = len(text.split())
    return text


def seed_summary_bank(level_name, initial_traj_dir, output_path, teacher_model=TEACHER_MODEL,
                      max_words=SUMMARY_MAX_WORDS) -> SkillBank:
    """Round 1 of the summary baseline: first summary from the warm-start trajectories."""
    output_path = Path(output_path)
    successes, failures = load_trajectories(Path(initial_traj_dir), level_name)
    print(f"\n  [summary] seeding from warm start: {len(successes)} successes, {len(failures)} failures")
    log = {"strategy": "summary", "round": 1, "max_words": max_words}
    bank = SkillBank()
    bank.summary_text = _write_summary(level_name, "", successes, failures, teacher_model, max_words, log)
    bank.save(output_path)
    output_path.with_name(output_path.stem + "_update_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"  [summary] saved {output_path} ({log['summary_words']} words)")
    return bank


def update_summary_bank(level_name, prev_bank, new_trajs_dir, output_path, round_num,
                        teacher_model=TEACHER_MODEL, max_words=SUMMARY_MAX_WORDS) -> SkillBank:
    output_path = Path(output_path)
    successes, failures = load_trajectories(new_trajs_dir, level_name)
    print(f"\n  [update:summary] {len(successes)} successes, {len(failures)} failures")
    log = {"strategy": "summary", "round": round_num, "max_words": max_words,
           "prev_summary": prev_bank.summary_text}
    bank = SkillBank()
    bank.summary_text = _write_summary(level_name, prev_bank.summary_text, successes, failures,
                                       teacher_model, max_words, log)
    bank.save(output_path)
    output_path.with_name(output_path.stem + "_update_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"  [update] saved {output_path} ({log['summary_words']} words)")
    return bank


# ── entry point ──────────────────────────────────────────────────────────────

def update_skill_bank(
    strategy: str,
    level_name: str,
    prev_bank: SkillBank,
    new_trajs_dir: Path,
    output_path: Path,
    round_num: int,
    max_skills: int = 10,
    max_mistakes: int = 5,
    teacher_model: str = TEACHER_MODEL,
    summary_max_words: int = SUMMARY_MAX_WORDS,
) -> SkillBank:
    """Apply the ``append_only`` / ``single_parent`` / ``summary`` update and save the bank."""
    if strategy not in ("append_only", "single_parent", "summary"):
        raise ValueError(f"update_skill_bank handles append_only/single_parent/summary, got {strategy!r}")
    output_path = Path(output_path)
    if strategy == "summary":
        return update_summary_bank(level_name, prev_bank, new_trajs_dir, output_path, round_num,
                                   teacher_model, summary_max_words)
    successes, failures = load_trajectories(new_trajs_dir, level_name)
    all_trajs = successes + failures
    traj_block = format_trajectories_block(all_trajs, max_trajs=5)
    print(f"\n  [update:{strategy}] {len(successes)} successes, {len(failures)} failures")

    bank = copy.deepcopy(prev_bank)  # never mutate the caller's previous bank

    gen = _prev_gen(bank, level_name) + 1
    log: dict = {"strategy": strategy, "round": round_num, "generation": gen,
                 "prev_skill_ids": [s.skill_id for s in bank.level_skills.get(level_name, [])],
                 "prev_mistake_ids": [m.mistake_id for m in bank.level_mistakes.get(level_name, [])],
                 "max_skills": max_skills, "max_mistakes": max_mistakes}

    if strategy == "single_parent":
        print(f"  [update] revising {len(log['prev_skill_ids'])} skills (one batched call) + "
              f"{len(log['prev_mistake_ids'])} mistakes (one batched call)")
        _revise_all(bank, level_name, successes, failures, traj_block, teacher_model,
                    round_num, gen, log)

    cand_s, cand_m = _extract(level_name, successes, failures, traj_block,
                              max_skills, max_mistakes, teacher_model, log)
    log["candidates"] = {"skills": cand_s, "mistakes": cand_m}
    print(f"  [update] extracted {len(cand_s)} skills, {len(cand_m)} mistakes (bank not shown)")
    keep_s, keep_m = _dedup(bank, level_name, cand_s, cand_m, teacher_model, log)
    _append_and_evict(bank, level_name, keep_s, keep_m, all_trajs, round_num, gen,
                      max_skills, max_mistakes, log)

    log["final_skill_ids"] = [s.skill_id for s in bank.level_skills.get(level_name, [])]
    log["final_mistake_ids"] = [m.mistake_id for m in bank.level_mistakes.get(level_name, [])]
    bank.save(output_path)
    output_path.with_name(output_path.stem + "_update_log.json").write_text(
        json.dumps(log, indent=2, default=str))
    print(f"  [update] saved {output_path}: {bank}")
    return bank
