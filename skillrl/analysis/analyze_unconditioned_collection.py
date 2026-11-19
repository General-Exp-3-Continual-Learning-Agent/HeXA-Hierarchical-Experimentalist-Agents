#!/usr/bin/env python
"""Report the "skill-unconditioned collection" ablation vs full HExA (catapult).

For a paired build (one of run2/run4/run5), compares:
  - Full HExA  (control) : <full_run>/frozen_eval_seeds57-106/catapult
  - Ablation            : <abl_run>/eval_seeds57-106/catapult
on the SAME held-out seeds, and reports, per arm:
    held-out success (paired per-seed), average turns, injected tokens (the
    skills/mistakes block fed to the eval actor), and the training-time
    construction cost (environment calls + actor turns + teacher evolution
    calls) actually logged on disk.

Usage:
    python -m skillrl.analysis.analyze_unconditioned_collection --run run5
    python -m skillrl.analysis.analyze_unconditioned_collection --run run5 --eval-seeds 57-106
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

from skillrl.core.skill_bank import SkillBank
from skillrl.core.retriever import retrieve_skills

PROJ = Path(__file__).resolve().parents[2]
FULL_BASE = PROJ / "skillrl/RESULTS/catapult_claude/HeXA_catapult"
ABL_BASE = PROJ / "skillrl/data/ablation/unconditioned_collection/catapult"
LEVEL = "catapult"


# ─── token estimate ──────────────────────────────────────────────────────────
def _approx_tokens(text: str) -> int:
    """~tokens; use tiktoken if available, else chars/4 (GPT rule of thumb)."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return round(len(text) / 4)


def injected_block(bank_path: Path, max_spec: int = 6, max_mis: int = 4) -> tuple[str, int]:
    """The skills+mistakes block the eval actor receives for this bank, and its
    approx token count. Same for every seed (fixed bank, no eval-time updates)."""
    if not bank_path.exists():
        return "", 0
    bank = SkillBank.load(bank_path)
    skills = retrieve_skills(bank, LEVEL, max_general=0, max_specific=max_spec)
    mistakes = bank.get_mistakes_for_level(LEVEL)[:max_mis]
    block = bank.format_skills_as_prompt(LEVEL, skills=skills, mistakes=mistakes) or ""
    return block, _approx_tokens(block)


# ─── trajectory loading / counting ───────────────────────────────────────────
def _seed_of(fname: str) -> int | None:
    m = re.search(r"seed(\d+)", os.path.basename(fname))
    return int(m.group(1)) if m else None


def _tool_of(action_call: str) -> str | None:
    """'simulate({...})' -> 'simulate'."""
    if not isinstance(action_call, str):
        return None
    m = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", action_call)
    return m.group(1) if m else None


# Physics-engine tools. simulate_action is the true environment rollout; the
# others are lighter predictive/inspection queries against the engine.
_ENV_SIM = {"simulate_action"}
_ENV_QUERY = {"predict_first_contact", "trace_green_ball"}


def count_calls(traj: dict) -> dict:
    """Count tool-turns, env (simulate_action) calls, and other physics queries."""
    turns = traj.get("trajectory") or []
    tool_turns = sim_calls = query_calls = 0
    for t in turns:
        if isinstance(t, list) and len(t) >= 2:
            tool = _tool_of(t[1])
            if tool:
                tool_turns += 1
                if tool in _ENV_SIM:
                    sim_calls += 1
                elif tool in _ENV_QUERY:
                    query_calls += 1
    return {"tool_turns": tool_turns, "sim_calls": sim_calls, "query_calls": query_calls}


def load_eval(eval_level_dir: Path, seeds: list[int]) -> dict[int, dict]:
    out = {}
    if not eval_level_dir.exists():
        return out
    for f in eval_level_dir.glob("trajectory_seed*_skillrl.json"):
        s = _seed_of(f.name)
        if s is None or s not in seeds:
            continue
        try:
            out[s] = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return out


def training_cost(run_dir: Path) -> dict:
    """Sum environment calls + actor turns over all training round trajectories,
    and count teacher evolution calls (= saved banks beyond K0)."""
    episodes = 0
    tool_turns = sim_calls = query_calls = iters = 0
    for f in sorted(run_dir.glob(f"round_*/{LEVEL}/trajectory_seed*_skillrl.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        episodes += 1
        iters += int(d.get("iterations") or 0)
        c = count_calls(d)
        tool_turns += c["tool_turns"]
        sim_calls += c["sim_calls"]
        query_calls += c["query_calls"]
    banks = sorted(run_dir.glob("skill_bank_evolving_*.json"))
    n_banks = len(banks)
    # teacher is called once to SEED K0 (round 1 distill) and once per evolution
    # (rounds 2..N). With n_banks saved, that's n_banks teacher constructions.
    return {
        "episodes": episodes,
        "actor_turns": tool_turns,
        "actor_iterations": iters,
        "sim_calls": sim_calls,
        "query_calls": query_calls,
        "banks_saved": n_banks,
        "teacher_constructions": n_banks,
    }


# ─── report ──────────────────────────────────────────────────────────────────
def arm_summary(name: str, bank_path: Path, eval_dir: Path, train_dir: Path,
                seeds: list[int]) -> dict:
    ev = load_eval(eval_dir / LEVEL, seeds)
    done = [s for s in seeds if s in ev]
    succ = {s: bool(ev[s].get("success")) for s in done}
    n = len(done)
    nsucc = sum(succ.values())
    avg_turns = (sum(int(ev[s].get("iterations") or 0) for s in done) / n) if n else 0.0
    block, inj_tok = injected_block(bank_path)
    tc = training_cost(train_dir)
    return {
        "name": name,
        "bank": str(bank_path),
        "eval_dir": str(eval_dir),
        "n_eval": n,
        "successes": nsucc,
        "success_rate": (nsucc / n) if n else 0.0,
        "avg_turns": avg_turns,
        "injected_tokens": inj_tok,
        "injected_chars": len(block),
        "success_by_seed": succ,
        "training": tc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run2 | run4 | run5")
    ap.add_argument("--eval-seeds", default="57-106",
                    help="held-out range, e.g. 57-86; also names the ablation eval dir")
    ap.add_argument("--rounds", type=int, default=17,
                    help="which evolving round is the ablation FINAL bank (e.g. 9)")
    args = ap.parse_args()

    lo, hi = (args.eval_seeds.split("-") + [args.eval_seeds])[:2]
    seeds = list(range(int(lo), int(hi) + 1))

    full = FULL_BASE / f"HeXA_catapult_claude_{args.run}"
    abl = ABL_BASE / args.run
    r = args.rounds

    full_arm = arm_summary(
        "FULL (control)",
        full / f"skill_bank_evolving_{r}.json",
        full / "frozen_eval_seeds57-106",
        full, seeds,
    )
    abl_arm = arm_summary(
        "ABLATION (unconditioned collection)",
        abl / f"skill_bank_evolving_{r}.json",
        abl / f"eval_seeds{args.eval_seeds}",
        abl, seeds,
    )

    print(f"\n{'='*78}")
    print(f"  SKILL-UNCONDITIONED COLLECTION ABLATION — catapult — {args.run}")
    print(f"  held-out seeds {args.eval_seeds}  (paired, same seeds, bank frozen at eval)")
    print(f"{'='*78}")
    for a in (full_arm, abl_arm):
        print(f"\n  [{a['name']}]")
        print(f"    bank            : {a['bank']}")
        print(f"    held-out success: {a['successes']}/{a['n_eval']} = {a['success_rate']:.1%}")
        print(f"    avg turns       : {a['avg_turns']:.2f}")
        print(f"    injected tokens : {a['injected_tokens']}  ({a['injected_chars']} chars; eval block, per seed)")
        t = a["training"]
        print(f"    TRAIN cost      : {t['episodes']} episodes, {t['sim_calls']} env (simulate_action) calls, "
              f"{t['query_calls']} physics queries, {t['actor_turns']} actor tool-turns, {t['actor_iterations']} actor iters")
        print(f"                      {t['teacher_constructions']} teacher constructions "
              f"({t['banks_saved']} banks saved)")

    # paired diff on the intersection of completed seeds
    both = [s for s in seeds if s in full_arm["success_by_seed"] and s in abl_arm["success_by_seed"]]
    if both:
        f_s = sum(full_arm["success_by_seed"][s] for s in both)
        a_s = sum(abl_arm["success_by_seed"][s] for s in both)
        s2f = [s for s in both if full_arm["success_by_seed"][s] and not abl_arm["success_by_seed"][s]]
        f2s = [s for s in both if not full_arm["success_by_seed"][s] and abl_arm["success_by_seed"][s]]
        print(f"\n  PAIRED on {len(both)} common seeds:")
        print(f"    full {f_s}/{len(both)} ({f_s/len(both):.1%})  vs  ablation {a_s}/{len(both)} ({a_s/len(both):.1%})")
        print(f"    diff (ablation - full) = {(a_s-f_s)/len(both):+.1%}")
        print(f"    full-only wins (full S, abl F): {s2f}")
        print(f"    abl-only  wins (full F, abl S): {f2s}")
    else:
        print("\n  [paired diff] ablation eval not found yet — run the job, then re-run this.")

    print(f"\n  NOTE: injected tokens are estimated (tiktoken cl100k if available, else chars/4).")
    print(f"        Per-call teacher/actor TOKEN counts are not logged on disk; 'teacher")
    print(f"        constructions' and actor turn/sim counts are the available construction-cost")
    print(f"        proxies. Purpose: does bank-guided collection yield a better FINAL bank?\n")


if __name__ == "__main__":
    main()
