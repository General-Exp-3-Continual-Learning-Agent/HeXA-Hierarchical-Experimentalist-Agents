"""Runner for the two-measure skill-lineage audit.

Measures (never summed, never reported under one heading):
  A  INCORPORATION  -- a later entry carries content traceable to a SECOND
                       earlier entry that is not its own predecessor.
                       This is the composition measure.
  B  ABSTRACTION    -- a later entry is a more general form of its SINGLE
                       predecessor. A separate, weaker claim.

Conditions
  calib         8 transitions of unconditioned_collection/catapult/run5 where
                the teacher's own narration survives in logs/uncond_collect_cat-*.out.
                RUN THIS FIRST -- it is the only ground truth available.
  main          48 transitions, HeXA_catapult_claude_run{2,4,5}, k=1
  main-k4/k8    long-range pairs (generalisation often spans many rounds)
  neg-append    update_strategy/catapult/append_only/run5 -- that update rule
                never edits existing entries, so every measure should be ~0
  neg-mismatch  round N of run2 vs round N+1 of run4 -- unrelated banks
  history       Measure A with donors from ANY earlier round (per final skill)
  dirblind      verbosity control: which version is more general, order hidden
  verify        pass-2 adversarial check of every proposed incorporation

Every call is cached under --out-dir by a hash of (prompt, model), so reruns
cost nothing and a crashed run resumes.

Quote grounding: every quote the judge returns is checked character-for-
character against the entry it is attributed to. Ungrounded events are dropped
and counted separately -- this is the cheapest hallucination filter available.

Usage (login node is fine for --dry-run; use the job script to actually call):
    python -m skillrl.analysis.lineage_audit --condition calib --dry-run
    python -m skillrl.analysis.lineage_audit --condition calib
    python -m skillrl.analysis.lineage_audit --condition main --verify
    python -m skillrl.analysis.lineage_audit --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from skillrl.core.config import TEACHER_MODEL
from skillrl.distillation.distill import call_teacher, _parse_json_object
from skillrl.analysis.lineage_audit_prompts import (
    TRANSITION_AUDIT_PROMPT,
    HISTORY_ATTRIBUTION_PROMPT,
    VERIFY_EVENT_PROMPT,
    VERIFY_ABSORPTION_PROMPT,
    DIRECTION_BLIND_PROMPT,
    format_entries,
)

LEVEL = "catapult"
MAIN_BASE = PROJ / "skillrl/RESULTS/catapult_claude/HeXA_catapult"
MAIN_RUNS = {
    "run2": MAIN_BASE / "HeXA_catapult_claude_run2",
    "run4": MAIN_BASE / "HeXA_catapult_claude_run4",
    "run5": MAIN_BASE / "HeXA_catapult_claude_run5",
}
APPEND_DIR = PROJ / "skillrl/data/ablation/update_strategy/catapult/append_only/run5"
UNCOND_DIR = PROJ / "skillrl/data/ablation/unconditioned_collection/catapult/run5"
CALIB_LOGS = sorted((PROJ / "logs").glob("uncond_collect_cat-*.out"))

SEED = 20261109  # fixed so anonymisation/shuffling is reproducible


# ── bank loading ───────────────────────────────────────────────────────

def bank_path(run_dir: Path, n: int) -> Path:
    return run_dir / f"skill_bank_evolving_{n}.json"


def load_skills(run_dir: Path, n: int) -> list[dict]:
    p = bank_path(run_dir, n)
    if not p.exists():
        return []
    return json.load(open(p))["level_skills"].get(LEVEL, [])


def n_rounds(run_dir: Path) -> int:
    ns = [int(m.group(1)) for p in run_dir.glob("skill_bank_evolving_*.json")
          if (m := re.search(r"evolving_(\d+)\.json$", p.name))]
    return max(ns) if ns else 0


# ── blinding contract ──────────────────────────────────────────────────

FIELDS = ("title", "principle", "when_to_apply", "example", "source_seeds")


def anonymise(skills: list[dict], prefix: str, rng: random.Random):
    """Strip ids, drop confidence/generation, relabel, shuffle. Returns
    (entries_for_prompt, label -> real skill_id, label -> entry)."""
    idx = list(range(len(skills)))
    rng.shuffle(idx)
    entries, mapping, by_label = [], {}, {}
    for new_pos, i in enumerate(idx, start=1):
        s = skills[i]
        label = f"{prefix}{new_pos}"
        e = {"label": label}
        e.update({f: s.get(f) for f in FIELDS})
        entries.append(e)
        mapping[label] = s.get("skill_id")
        by_label[label] = e
    return entries, mapping, by_label


def entry_text(e: dict) -> str:
    return " ".join(str(e.get(f) or "") for f in FIELDS)


# ── quote grounding ────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def quote_ok(quote: str, entry: dict | None) -> bool:
    if entry is None or not quote:
        return False
    return _norm(quote) in _norm(entry_text(entry))


def ground(parsed: dict, by_label: dict) -> tuple[dict, dict]:
    """Drop events whose quotes are not verbatim. Returns (kept, dropped)."""
    kept = {k: v for k, v in parsed.items()}
    dropped = defaultdict(list)

    def filt(key, pairs):
        out = []
        for ev in parsed.get(key) or []:
            ok = all(quote_ok(ev.get(q), by_label.get(ev.get(lab)))
                     for q, lab in pairs)
            (out if ok else dropped[key]).append(ev)
        kept[key] = out

    filt("incorporations", [("quote_donor", "donor"), ("quote_later", "later")])
    # Absorptions: only ABSORBED_INTO claims carry quotes and get grounded.
    # DELETED dispositions are kept as-is -- they are the counter-direction and
    # must stay visible in the count.
    abs_keep, abs_drop = [], []
    for ev in parsed.get("absorptions") or []:
        if (ev.get("disposition") or "").upper() != "ABSORBED_INTO":
            abs_keep.append(ev)
            continue
        ok = (quote_ok(ev.get("quote_removed"), by_label.get(ev.get("removed_entry")))
              and quote_ok(ev.get("quote_covering"), by_label.get(ev.get("covering_entry"))))
        (abs_keep if ok else abs_drop).append(ev)
    kept["absorptions"] = abs_keep
    if abs_drop:
        dropped["absorptions"] = abs_drop
    filt("abstractions", [("quote_predecessor", "predecessor"), ("quote_later", "later")])
    filt("narrowings", [("quote_predecessor", "predecessor"), ("quote_later", "later")])
    return kept, dict(dropped)


# ── cached model call ──────────────────────────────────────────────────

def ask(prompt: str, model: str, out_dir: Path, tag: str, dry: bool) -> dict:
    h = hashlib.sha1((model + "\x00" + prompt).encode()).hexdigest()[:16]
    cache = out_dir / "raw" / f"{tag}.{h}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.load(open(cache))["parsed"]
    if dry:
        (out_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (out_dir / "prompts" / f"{tag}.txt").write_text(prompt)
        return {}
    # The claude CLI timeout is 300 s (skillrl/distillation/distill.py:120) and
    # is shared with the live training path, so it is not ours to raise. A big
    # 10x10 transition occasionally exceeds it; timeouts are transient, so
    # retry here instead.
    parsed = {}
    for attempt in range(1, 4):
        raw = call_teacher(prompt, model=model)
        parsed = _parse_json_object(raw)
        if parsed:
            break
        print(f"  [!] {tag}: empty/unparseable reply (attempt {attempt}/3)")
    if not parsed:
        # Never cache an empty result, or the failure is frozen in and a later
        # rerun silently reuses it as a real zero.
        print(f"  [!] {tag}: giving up after 3 attempts -- not cached")
        return {}
    json.dump({"tag": tag, "model": model, "raw": raw, "parsed": parsed},
              open(cache, "w"), indent=1)
    return parsed


# ── one transition ─────────────────────────────────────────────────────

def audit_transition(earlier: list[dict], later: list[dict], tag: str,
                     model: str, out_dir: Path, dry: bool) -> dict:
    rng = random.Random(f"{SEED}:{tag}")
    e_ent, e_map, e_by = anonymise(earlier, "P", rng)
    l_ent, l_map, l_by = anonymise(later, "C", rng)
    by_label = {**e_by, **l_by}
    prompt = TRANSITION_AUDIT_PROMPT.format(
        earlier_block=format_entries(e_ent), later_block=format_entries(l_ent))
    parsed = ask(prompt, model, out_dir, tag, dry)
    if not parsed:
        return {"tag": tag, "parsed": {}, "dropped": {}, "maps": {**e_map, **l_map}}
    kept, dropped = ground(parsed, by_label)
    return {"tag": tag, "parsed": kept, "dropped": dropped,
            "maps": {**e_map, **l_map},
            "n_earlier": len(earlier), "n_later": len(later)}


# ── condition builders ─────────────────────────────────────────────────

def pairs_adjacent(run_dir: Path, k: int = 1):
    N = n_rounds(run_dir)
    return [(n, n + k) for n in range(1, N - k + 1)]


def build(condition: str, k: int):
    """Yield (tag, earlier_skills, later_skills)."""
    if condition == "calib":
        for a, b in pairs_adjacent(UNCOND_DIR):
            yield f"calib.uncond_run5.{a}-{b}", load_skills(UNCOND_DIR, a), load_skills(UNCOND_DIR, b)
    elif condition == "main":
        for run, d in MAIN_RUNS.items():
            for a, b in pairs_adjacent(d, k):
                yield f"main.{run}.k{k}.{a}-{b}", load_skills(d, a), load_skills(d, b)
    elif condition == "neg-append":
        for a, b in pairs_adjacent(APPEND_DIR):
            yield f"negappend.run5.{a}-{b}", load_skills(APPEND_DIR, a), load_skills(APPEND_DIR, b)
    elif condition == "neg-mismatch":
        d2, d4 = MAIN_RUNS["run2"], MAIN_RUNS["run4"]
        for a, b in pairs_adjacent(d2):
            yield f"negmismatch.r2_{a}-r4_{b}", load_skills(d2, a), load_skills(d4, b)
    else:
        raise SystemExit(f"unknown condition for pair-building: {condition}")


# ── lexical chain tracing (for history / dirblind) ─────────────────────
#
# Used ONLY to decide which earlier entries belong to a skill's own revision
# history and which go in the donor pool. It is a weak matcher and is never
# itself reported as evidence: the judge re-decides attribution from text.

_STOP = {"the", "a", "an", "to", "of", "and", "or", "for", "with", "at", "in",
         "on", "is", "as", "by", "when", "then", "x", "y", "r", "it", "this",
         "that", "be", "are", "if", "from"}
CHAIN_TAU = 0.30


def _toks(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower())
            if w not in _STOP and len(w) > 1}


def _jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b)) if (a or b) else 0.0


def _lex(p: dict, c: dict) -> float:
    return (0.50 * _jac(_toks(p.get("title")), _toks(c.get("title")))
            + 0.35 * _jac(_toks(p.get("principle")), _toks(c.get("principle")))
            + 0.15 * _jac(_toks(p.get("when_to_apply")), _toks(c.get("when_to_apply"))))


def trace_chain(run_dir: Path, final_skill: dict, N: int) -> list[tuple[int, dict]]:
    """Earliest-first chain of versions of one final skill."""
    chain = [(N, final_skill)]
    cur = final_skill
    for n in range(N - 1, 0, -1):
        ps = load_skills(run_dir, n)
        best, bs = None, 0.0
        for q in ps:
            l = _lex(q, cur)
            if l > bs:
                best, bs = q, l
        if best is None or bs < CHAIN_TAU:
            break
        chain.append((n, best))
        cur = best
    return list(reversed(chain))


def build_history(run: str):
    """Yield (tag, final_skill, own_history, pool) per final-round skill."""
    d = MAIN_RUNS[run]
    N = n_rounds(d)
    for i, fs in enumerate(load_skills(d, N)):
        chain = trace_chain(d, fs, N)
        chain_texts = {_norm(entry_text(s)) for _, s in chain}
        raw, seen = [], set()
        for n in range(1, N):
            for q in load_skills(d, n):
                t = _norm(entry_text(q))
                if t in chain_texts or t in seen:
                    continue
                seen.add(t)
                raw.append((n, q))
        yield f"history.{run}.{i:02d}", fs, [s for _, s in chain], collapse_pool(raw)


def collapse_pool(raw: list[tuple[int, dict]]) -> list[dict]:
    """The raw pool holds every version of every OTHER skill -- ~140 entries,
    mostly near-duplicates of each other. Cluster them into other-skill
    lineages and keep the FIRST and LAST version of each: early versions can
    donate content that the late version has since dropped, and late versions
    carry everything accumulated since. Cuts the pool to ~20-30 entries without
    choosing which skills the judge may consider."""
    clusters: list[list[tuple[int, dict]]] = []
    for n, q in raw:
        placed = False
        for cl in clusters:
            if _lex(cl[-1][1], q) >= CHAIN_TAU:
                cl.append((n, q))
                placed = True
                break
        if not placed:
            clusters.append([(n, q)])
    out = []
    for cl in clusters:
        cl.sort(key=lambda t: t[0])
        picks = [cl[0]] if len(cl) == 1 else [cl[0], cl[-1]]
        for _, q in picks:
            out.append(q)
    return out


def build_dirblind(run: str):
    """Yield (tag, root_version, final_version) for chains whose text changed."""
    d = MAIN_RUNS[run]
    N = n_rounds(d)
    for i, fs in enumerate(load_skills(d, N)):
        chain = trace_chain(d, fs, N)
        if len(chain) < 2:
            continue
        root, final = chain[0][1], chain[-1][1]
        if _norm(entry_text(root)) == _norm(entry_text(final)):
            continue
        yield f"dirblind.{run}.{i:02d}", root, final


# ── MEASURE A over a whole run history ─────────────────────────────────

def audit_history(final_skill, own_history, pool, tag, model, out_dir, dry):
    rng = random.Random(f"{SEED}:{tag}")
    f_ent, _, f_by = anonymise([final_skill], "F", rng)
    h_ent, h_by = [], {}
    for j, s in enumerate(own_history, start=1):          # chronological, not shuffled
        e = {"label": f"V{j}"}
        e.update({k: s.get(k) for k in FIELDS})
        h_ent.append(e)
        h_by[e["label"]] = e
    p_ent, p_map, p_by = anonymise(pool, "D", rng)
    by_label = {**f_by, **h_by, **p_by}

    prompt = HISTORY_ATTRIBUTION_PROMPT.format(
        final_block=format_entries(f_ent),
        own_history_block=format_entries(h_ent),
        pool_block=format_entries(p_ent))
    parsed = ask(prompt, model, out_dir, tag, dry)

    kept, dropped = [], []
    for ev in (parsed.get("contributions") or []):
        ok = (quote_ok(ev.get("quote_donor"), by_label.get(ev.get("donor")))
              and quote_ok(ev.get("quote_final"), f_by["F1"]))
        (kept if ok else dropped).append(ev)
    return {"tag": tag, "parsed": {"contributions": kept,
                                   "verdict": parsed.get("verdict")},
            "dropped": {"contributions": dropped},
            "maps": p_map, "_by_label": by_label,
            "n_history": len(own_history), "n_pool": len(pool)}


# ── DIRECTION-BLIND control ────────────────────────────────────────────

def audit_dirblind(root, final, tag, model, out_dir, dry):
    rng = random.Random(f"{SEED}:{tag}")
    flip = rng.random() < 0.5                 # True -> VERSION ONE is the LATER one
    one, two = (final, root) if flip else (root, final)
    truth = "ONE" if flip else "TWO"
    mk = lambda s, lab: [{"label": lab, **{k: s.get(k) for k in FIELDS}}]
    prompt = DIRECTION_BLIND_PROMPT.format(
        version_one_block=format_entries(mk(one, "VERSION ONE")),
        version_two_block=format_entries(mk(two, "VERSION TWO")))
    parsed = ask(prompt, model, out_dir, tag, dry)
    wc = lambda s: len(entry_text(s).split())
    return {"tag": tag, "parsed": parsed, "truth_later": truth,
            "picked": parsed.get("more_general"),
            "correct": (parsed.get("more_general") == truth) if parsed.get("more_general") in ("ONE", "TWO") else None,
            "picked_longer": (parsed.get("more_general") == ("ONE" if wc(one) > wc(two) else "TWO"))
                             if parsed.get("more_general") in ("ONE", "TWO") else None,
            "wc_root": wc(root), "wc_final": wc(final), "dropped": {}}


def dirblind_report(results, out_dir: Path):
    scored = [r for r in results if r.get("correct") is not None]
    eq = sum(1 for r in results if r.get("picked") == "EQUAL")
    if not scored:
        print("\n=== DIRBLIND: no scorable answers (all EQUAL or empty) ===")
        return None
    acc = sum(1 for r in scored if r["correct"]) / len(scored)
    lng = sum(1 for r in scored if r["picked_longer"]) / len(scored)
    print("\n=== DIRBLIND control: is the generality verdict reading verbosity? ===")
    print(f"  scorable {len(scored)} / {len(results)}   (EQUAL answers: {eq})")
    print(f"  picked the TRUE LATER version : {acc:.2f}   (chance = 0.50)")
    print(f"  picked the LONGER version     : {lng:.2f}   (chance = 0.50)")
    if acc < 0.60:
        print("  >> AT OR NEAR CHANCE. The abstraction measure is not detecting")
        print("     direction of generality. Do not report Measure B.")
    if lng > 0.75:
        print("  >> STRONGLY TRACKS LENGTH. Treat Measure B as a verbosity proxy.")
    json.dump(results, open(out_dir / "dirblind_report.json", "w"), indent=1)
    return acc


# ── CALIB ground truth ─────────────────────────────────────────────────

def calib_ground_truth() -> dict[int, dict]:
    """round N (the LATER bank) -> {reasoning, merge_sentences, quoted_titles}."""
    # NB: do not split on "." -- the narration is full of decimals (0.25, r=1.4).
    # Take a character window around each keyword instead.
    MERGE_KW = re.compile(
        r"\b(merg\w+|subsum\w+|consolidat\w+|absorb\w+|now (?:also )?cover\w*|"
        r"folded into|covered more broadly|incorporat\w+)\b", re.I)
    WINDOW = 220
    out = {}
    for log in CALIB_LOGS:
        txt = log.read_text(errors="ignore")
        blocks = re.split(r"\n\s*Teacher reasoning:", txt)
        for blk in blocks[1:]:
            m = re.search(r"skill_bank_evolving_(\d+)\.json", blk)
            if not m:
                continue
            rnd = int(m.group(1))
            reasoning = blk.split("Evolved skill bank saved")[0].strip()
            sents = []
            for m in MERGE_KW.finditer(reasoning):
                a = max(0, m.start() - WINDOW // 2)
                b = min(len(reasoning), m.end() + WINDOW // 2)
                sents.append(re.sub(r"\s+", " ", reasoning[a:b]).strip())
            titles = re.findall(r"'([^']{6,90})'", reasoning)
            out[rnd] = {"reasoning": reasoning, "merge_sentences": sents,
                        "quoted_titles": sorted(set(titles))}
    return out


def calib_report(results: list[dict], out_dir: Path):
    gt = calib_ground_truth()
    # The keyword scan OVER-TRIGGERS: "incorporate seed 23's evidence" and
    # "consolidate the INVALID ACTION failures" are about evidence and about
    # mistakes, not about one skill absorbing another. Hand-label once and the
    # file below takes over as the reference.
    labels_path = out_dir / "calib_labels.json"
    hand = {}
    if labels_path.exists():
        hand = {int(k): bool(v) for k, v in json.load(open(labels_path)).items()}
    else:
        tmpl = {str(r): bool(gt[r]["merge_sentences"]) for r in sorted(gt)}
        json.dump(tmpl, open(out_dir / "calib_labels.template.json", "w"), indent=1)
        print(f"\n  [!] no hand labels found. A template seeded from the keyword scan\n"
              f"      is at {out_dir}/calib_labels.template.json -- read the\n"
              f"      sentences below, correct it, save as calib_labels.json, rerun\n"
              f"      --condition calib (cached, free) for the corrected recall.")
    rows = []
    for r in results:
        rnd = int(r["tag"].split(".")[-1].split("-")[1])
        g = gt.get(rnd, {})
        inc = r["parsed"].get("incorporations") or []
        absb = [e for e in (r["parsed"].get("absorptions") or [])
                if (e.get("disposition") or "").upper() == "ABSORBED_INTO"]
        inc = inc + absb          # Measure A = element transfer OR absorption
        rem = r["parsed"].get("removed") or []
        gt_merge = hand.get(rnd, bool(g.get("merge_sentences")))
        rows.append({
            "round": rnd,
            "teacher_claims_merge": gt_merge,
            "judge_incorporations": len(inc),
            "judge_removed": len(rem),
            "teacher_sentence": (g.get("merge_sentences") or [""])[0][:160],
            "judge_elements": [(e.get("element") or e.get("what_it_taught") or "")[:60]
                               for e in inc],
            "judge_absorptions": [f"{e.get('removed_entry')}->{e.get('covering_entry')}"
                                  for e in absb],
        })
    rows.sort(key=lambda x: x["round"])
    tp = sum(1 for x in rows if x["teacher_claims_merge"] and x["judge_incorporations"])
    fn = sum(1 for x in rows if x["teacher_claims_merge"] and not x["judge_incorporations"])
    fp = sum(1 for x in rows if not x["teacher_claims_merge"] and x["judge_incorporations"])
    tn = sum(1 for x in rows if not x["teacher_claims_merge"] and not x["judge_incorporations"])
    print("\n=== CALIB: judge vs teacher's own narration "
          "(unconditioned_collection/catapult/run5) ===")
    for x in rows:
        flag = "OK " if x["teacher_claims_merge"] == bool(x["judge_incorporations"]) else "XX "
        print(f" {flag}round {x['round']:>2}  teacher_merge={x['teacher_claims_merge']!s:<5} "
              f"judge_inc={x['judge_incorporations']}  {x['teacher_sentence'][:90]}")
    rec = tp / max(1, tp + fn)
    print(f"\n  round-level agreement: TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"  recall on teacher-stated merges = {rec:.2f}  ({tp}/{tp+fn})")
    print(f"  reference labels: {'hand-corrected' if hand else 'KEYWORD SCAN (over-triggers)'}")
    print("\n  NOTE: this is round-level agreement, not event-level. The teacher's")
    print("  narration is itself a claim, not a gold label. Read the sentences")
    print("  above next to the judge's elements before trusting the main run.")
    json.dump(rows, open(out_dir / "calib_report.json", "w"), indent=1)
    return rec


# ── pass-2 verification ────────────────────────────────────────────────

def verify_all(results: list[dict], model: str, out_dir: Path, dry: bool):
    """Adversarially re-check each grounded incorporation. Mutates results."""
    for r in results:
        inc = r["parsed"].get("incorporations") or []
        if not inc:
            continue
        tag_base = r["tag"]
        rng = random.Random(f"{SEED}:{tag_base}")
        # rebuild the same anonymisation to recover entry text by label
        raw = (out_dir / "raw")
        by_label = r.get("_by_label") or {}
        for i, ev in enumerate(inc):
            if not by_label:
                ev["verify"] = {"verdict": "SKIPPED", "reason": "entries not cached"}
                continue
            prompt = VERIFY_EVENT_PROMPT.format(
                predecessor_block=format_entries([by_label[ev["predecessor"]]]),
                donor_block=format_entries([by_label[ev["donor"]]]),
                later_block=format_entries([by_label[ev["later"]]]),
                element=ev.get("element", ""),
                quote_donor=ev.get("quote_donor", ""),
                quote_later=ev.get("quote_later", ""))
            ev["verify"] = ask(prompt, model, out_dir, f"verify.{tag_base}.{i}", dry)

        # Pass-2 for ABSORBED_INTO: the counter-hypothesis is plain deletion.
        for j, ev in enumerate(r["parsed"].get("absorptions") or []):
            if (ev.get("disposition") or "").upper() != "ABSORBED_INTO":
                continue
            cov = ev.get("covering_entry")
            rem = ev.get("removed_entry")
            if not by_label or cov not in by_label or rem not in by_label:
                ev["verify"] = {"verdict": "SKIPPED", "reason": "entry not found"}
                continue
            before = _match_earlier(by_label[cov], by_label)
            prompt = VERIFY_ABSORPTION_PROMPT.format(
                removed_block=format_entries([by_label[rem]]),
                covering_before_block=format_entries([before]) if before
                    else "(no clear earlier form of this entry)",
                covering_after_block=format_entries([by_label[cov]]),
                what_it_taught=ev.get("what_it_taught", ""),
                quote_removed=ev.get("quote_removed", ""),
                quote_covering=ev.get("quote_covering", ""))
            ev["verify"] = ask(prompt, model, out_dir, f"verifyabs.{tag_base}.{j}", dry)


def _match_earlier(later_entry: dict, by_label: dict) -> dict | None:
    """Best lexical match for a later entry among the earlier (P*) entries.
    Used only to show pass-2 the covering entry's before/after forms."""
    best, bs = None, 0.0
    for lab, e in by_label.items():
        if not lab.startswith("P"):
            continue
        v = _lex(e, later_entry)
        if v > bs:
            best, bs = e, v
    return best if bs >= CHAIN_TAU else None


def verify_history(results, model: str, out_dir: Path, dry: bool):
    """Pass-2 on history contributions. The 'predecessor' is the latest version
    of the entry's OWN history before the final one -- that is exactly what
    condition 3 of the history prompt asks about."""
    for r in results:
        contribs = r["parsed"].get("contributions") or []
        by_label = r.get("_by_label") or {}
        if not contribs or not by_label:
            continue
        vs = sorted((k for k in by_label if k.startswith("V")),
                    key=lambda k: int(k[1:]))
        pred_label = vs[-2] if len(vs) >= 2 else (vs[-1] if vs else None)
        for i, ev in enumerate(contribs):
            if pred_label is None or ev.get("donor") not in by_label:
                ev["verify"] = {"verdict": "SKIPPED", "reason": "no predecessor/donor entry"}
                continue
            prompt = VERIFY_EVENT_PROMPT.format(
                predecessor_block=format_entries([by_label[pred_label]]),
                donor_block=format_entries([by_label[ev["donor"]]]),
                later_block=format_entries([by_label["F1"]]),
                element=ev.get("element", ""),
                quote_donor=ev.get("quote_donor", ""),
                quote_later=ev.get("quote_final", ""))
            ev["verify"] = ask(prompt, model, out_dir, f"verify.{r['tag']}.{i}", dry)


# ── aggregate report ───────────────────────────────────────────────────

def summarise(out_dir: Path):
    files = sorted(out_dir.glob("results.*.json"))
    if not files:
        raise SystemExit("no results.*.json in out-dir; run a condition first")
    print(f"\n{'condition':<15}{'pairs':>6}{'INC':>5}{'inc_ok':>7}{'ABSORB':>8}"
          f"{'abs_ok':>7}{'DELETED':>8}{'ABSTR(B)':>9}{'NARROW':>7}{'FRAG':>6}{'drop':>6}")
    print("-" * 84)
    specials = []
    for f in files:
        cond = f.stem.split(".", 1)[1]
        rs = json.load(open(f))
        if cond == "dirblind":
            sc = [r for r in rs if r.get("correct") is not None]
            acc = (sum(1 for r in sc if r["correct"]) / len(sc)) if sc else float("nan")
            lng = (sum(1 for r in sc if r.get("picked_longer")) / len(sc)) if sc else float("nan")
            specials.append(f"dirblind       {len(rs):>6} pairs   picked-true-later {acc:.2f}   "
                            f"picked-longer {lng:.2f}   (chance 0.50)")
            continue
        if cond == "history":
            con = sum(len(r["parsed"].get("contributions") or []) for r in rs)
            cf = sum(1 for r in rs for e in (r["parsed"].get("contributions") or [])
                     if (e.get("verify") or {}).get("verdict") == "CONFIRMED")
            drp = sum(len(v) for r in rs for v in (r.get("dropped") or {}).values())
            specials.append(f"history        {len(rs):>6} skills  INCORP(A) {con:<4} "
                            f"confirmed {cf:<4} dropped {drp}")
            continue
        inc = sum(len(r["parsed"].get("incorporations") or []) for r in rs)
        conf = sum(1 for r in rs for e in (r["parsed"].get("incorporations") or [])
                   if (e.get("verify") or {}).get("verdict") == "CONFIRMED")
        allabs = [e for r in rs for e in (r["parsed"].get("absorptions") or [])]
        absb = sum(1 for e in allabs if (e.get("disposition") or "").upper() == "ABSORBED_INTO")
        dele = sum(1 for e in allabs if (e.get("disposition") or "").upper() == "DELETED")
        absc = sum(1 for e in allabs if (e.get("verify") or {}).get("verdict") == "CONFIRMED")
        abs_ = sum(len(r["parsed"].get("abstractions") or []) for r in rs)
        nar = sum(len(r["parsed"].get("narrowings") or []) for r in rs)
        fra = sum(len(r["parsed"].get("fragmentations") or []) for r in rs)
        drp = sum(sum(len(v) for v in (r.get("dropped") or {}).values()) for r in rs)
        print(f"{cond:<15}{len(rs):>6}{inc:>5}{conf:>7}{absb:>8}{absc:>7}{dele:>8}"
              f"{abs_:>9}{nar:>7}{fra:>6}{drp:>6}")
    for line in specials:
        print(line)
    print("\nMeasure A (incorporation = composition) and Measure B (abstraction)")
    print("are separate claims. Do not sum them. Report each against the")
    print("neg-append floor.")


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["calib", "main", "neg-append",
                                            "neg-mismatch", "history", "dirblind"])
    ap.add_argument("--k", type=int, default=1, help="round gap for --condition main")
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument("--out-dir", default=str(PROJ / "skillrl/data/ablation/lineage_audit"))
    ap.add_argument("--dry-run", action="store_true",
                    help="write prompts to <out-dir>/prompts, make no model calls")
    ap.add_argument("--verify", action="store_true", help="run pass-2 on incorporations")
    ap.add_argument("--report", action="store_true", help="summarise existing results")
    ap.add_argument("--force", action="store_true",
                    help="run a non-calib condition without a calib report present")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.report:
        summarise(out)
        return
    if not a.condition:
        raise SystemExit("pass --condition or --report")

    if (a.condition != "calib" and not a.dry_run
            and not (out / "calib_report.json").exists() and not a.force):
        raise SystemExit(
            "CALIB has not been run. The calibration set is the only ground truth\n"
            "available (8 rounds where the teacher narrated its own merges).\n"
            "Run:  --condition calib\n"
            "then read the agreement table before spending calls on the rest.\n"
            "Override with --force if you really mean to.")

    results = []

    if a.condition == "history":
        for run in MAIN_RUNS:
            for tag, fs, hist, pool in build_history(run):
                r = audit_history(fs, hist, pool, tag, a.model, out, a.dry_run)
                results.append(r)
                print(f"  {tag:<24} hist={r['n_history']:<3} pool={r['n_pool']:<4} "
                      f"A={len(r['parsed']['contributions']):<3} "
                      f"dropped={len(r['dropped']['contributions'])}")

    elif a.condition == "dirblind":
        for run in MAIN_RUNS:
            for tag, root, final in build_dirblind(run):
                r = audit_dirblind(root, final, tag, a.model, out, a.dry_run)
                results.append(r)
                print(f"  {tag:<24} truth={r['truth_later']} picked={r['picked']} "
                      f"wc {r['wc_root']}->{r['wc_final']}")

    else:
        for tag, earlier, later in build(a.condition, a.k):
            if not earlier or not later:
                print(f"  skip {tag}: empty bank")
                continue
            r = audit_transition(earlier, later, tag, a.model, out, a.dry_run)
            # keep anonymised entries for pass-2 and for hand-checking
            rng = random.Random(f"{SEED}:{tag}")
            e_ent, _, e_by = anonymise(earlier, "P", rng)
            l_ent, _, l_by = anonymise(later, "C", rng)
            r["_by_label"] = {**e_by, **l_by}
            results.append(r)
            inc = len(r["parsed"].get("incorporations") or [])
            abst = len(r["parsed"].get("abstractions") or [])
            absb = sum(1 for e in (r["parsed"].get("absorptions") or [])
                       if (e.get("disposition") or "").upper() == "ABSORBED_INTO")
            dele = sum(1 for e in (r["parsed"].get("absorptions") or [])
                       if (e.get("disposition") or "").upper() == "DELETED")
            print(f"  {tag:<34} A={inc:<3} ABS={absb:<3} DEL={dele:<3} B={abst:<3} "
                  f"dropped={sum(len(v) for v in (r.get('dropped') or {}).values())}")

    if a.verify and not a.dry_run and a.condition not in ("dirblind",):
        if a.condition == "history":
            verify_history(results, a.model, out, a.dry_run)
        else:
            verify_all(results, a.model, out, a.dry_run)

    json.dump(results, open(out / f"results.{a.condition}.json", "w"), indent=1)
    print(f"\nwrote {out}/results.{a.condition}.json  ({len(results)} pairs)")

    if a.condition == "calib" and not a.dry_run:
        calib_report(results, out)
    if a.condition == "dirblind" and not a.dry_run:
        dirblind_report(results, out)
    if a.dry_run:
        print(f"DRY RUN -- prompts in {out}/prompts, no model calls made")


if __name__ == "__main__":
    main()
