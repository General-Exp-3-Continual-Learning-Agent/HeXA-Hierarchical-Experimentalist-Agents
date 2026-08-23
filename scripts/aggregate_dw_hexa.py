"""Aggregate the HeXA-vs-dynamic-workflow catapult comparison into one report.

Rebuilds `results.json` + `REPORT.md` from the raw artifacts, so adding seeds is
just: drop the new workflow output into `runs/`, re-run this.

Ground truth comes from the simulator, not from what an agent claimed: every
recorded placement is re-simulated here and `verified_success` is what counts in
the stats. `claim_success` is kept alongside so the two can be compared.

HeXA is re-scored under the DW's per-attempt budget (`--cap`), because HeXA was
allowed up to 24 rollouts and several of its successes arrived after rollout 12.
Comparing raw success rates across unequal budgets is the mistake this guards
against.

    python scripts/aggregate_dw_hexa.py
    python scripts/aggregate_dw_hexa.py --cap 12 --no-verify
"""

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(_REPO, "Direct Results", "DW_vs_HeXA_benchmark")
DEFAULT_HEXA = os.path.expanduser(
    "~/Scalar_Research/physics-reasoning-agents/skillrl/RESULTS/"
    "catapult_claude/HeXA_catapult"
)
PY = os.path.expanduser("~/.conda/envs/phyreagent/bin/python")

# Tools that step the physics engine, i.e. consume one environment rollout.
ROLLOUT_TOOLS = {
    "simulate_action", "simulate_partial", "simulate_with_trace",
    "predict_first_contact", "trace_green_ball",
}

# Branches proven contaminated by transcript audit: these read
# `interphyre/validation/oracles/catapult.py`, a solution generator whose
# docstring names the working zones. Excluded from every statistic.
VOIDED = {
    ("wjf0fjwne", 7, "geometry"),
    ("wjf0fjwne", 33, "geometry"),
}


def _run_id(path: str) -> str:
    m = re.search(r"run_(\w+?)_", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def load_dw_runs(runs_dir: str) -> list:
    """Flatten every workflow output in `runs_dir` into per-branch records."""
    out = []
    for fp in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(fp) as f:
            blob = json.load(f)
        res = blob.get("result") or blob
        rid = _run_id(fp)
        budget = res.get("budget_per_agent")
        # Agents-per-seed defines the experimental design as much as the budget
        # does: 2 agents x 12 rollouts spends twice as much per seed as 1 x 12.
        # Derive it from the data rather than trusting a field the earlier run
        # scripts did not emit.
        agents = res.get("agents_per_seed") or max(
            (len(e.get("branches") or []) for e in (res.get("per_seed") or {}).values()),
            default=1)
        for seed_str, entry in (res.get("per_seed") or {}).items():
            for br in entry.get("branches") or []:
                seed = br.get("seed", int(seed_str))
                lens = br.get("lens")
                # Two branch schemas exist across runs: the earlier one emits
                # `placement`/`success`/`reward`, the later `best_x`/`best_success`.
                if br.get("best_x") is not None:
                    placement = [br.get("best_x"), br.get("best_y"),
                                 br.get("best_radius")]
                else:
                    placement = list(br.get("placement") or [None, None, None])
                claim = br.get("best_success")
                if claim is None:
                    claim = br.get("success")
                reward = br.get("best_reward")
                if reward is None:
                    reward = br.get("reward")
                out.append({
                    "run_id": rid,
                    "source_file": os.path.basename(fp),
                    "seed": seed,
                    "lens": lens,
                    "budget_per_agent": budget,
                    "agents_per_seed": agents,
                    "rollouts_used": br.get("rollouts_used"),
                    "placement": placement,
                    "claim_success": bool(claim),
                    "claim_reward": reward,
                    "closest_gap": br.get("closest_gap"),
                    "hit_budget_cap": br.get("hit_budget_cap"),
                    "declared_read_files": br.get("read_any_source_or_data_file"),
                    "files_inspected": br.get("files_inspected") or [],
                    "voided_for_leakage": (rid, seed, lens) in VOIDED,
                    "hypothesis": br.get("hypothesis"),
                })
    return out


def verify(level: str, seed: int, placement: list) -> dict:
    """Re-simulate a placement. This is the ground truth for `success`."""
    x, y, r = placement
    if None in placement:
        return {"verified_success": None, "verified_reward": None,
                "note": "incomplete placement"}
    cmd = [PY, os.path.join(_REPO, "scripts", "interphyre_probe.py"),
           "--level", level, "--seed", str(seed),
           "simulate", "--x", str(x), "--y", str(y), "--radius", str(r)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"verified_success": None, "verified_reward": None,
                "note": "timeout"}
    obs = p.stdout
    m = re.search(r"Reward:\s*(-?\d+\.?\d*)", obs)
    return {
        "verified_success": "VERDICT: SUCCESS" in obs,
        "verified_reward": float(m.group(1)) if m else None,
        "note": None,
    }


def load_hexa(hexa_dir: str) -> dict:
    """seed -> list of (success, rollouts) across all HeXA runs/rounds."""
    per = {}
    pat = os.path.join(hexa_dir, "**", "trajectory_seed*_skillrl.json")
    for fp in glob.glob(pat, recursive=True):
        m = re.search(r"seed(\d+)", fp)
        if not m:
            continue
        seed = int(m.group(1))
        with open(fp) as f:
            d = json.load(f)
        steps = d.get("trajectory") or d.get("steps") or []
        n = 0
        for stp in steps:
            # HeXA trajectory steps are [thought, action, observation] lists,
            # not dicts.
            if isinstance(stp, list) and len(stp) > 1:
                act = stp[1]
            elif isinstance(stp, dict):
                act = stp.get("action", "")
            else:
                act = ""
            if any(t in str(act) for t in ROLLOUT_TOOLS):
                n += 1
        per.setdefault(seed, []).append(
            {"success": bool(d.get("success")), "rollouts": n})
    return per


def summarise(dw: list, hexa: dict, cap: int) -> dict:
    """Per-attempt and per-seed rates for both harnesses on the shared seeds."""
    live = [b for b in dw if not b["voided_for_leakage"]]
    # Only compare on seeds where the DW ran at this per-agent budget.
    by_budget = {}
    for b in live:
        by_budget.setdefault(
            (b["budget_per_agent"], b.get("agents_per_seed") or 1), []).append(b)

    blocks = {}
    for (budget, agents), branches in sorted(by_budget.items(),
                                             key=lambda kv: (kv[0][0] or 0, kv[0][1])):
        seeds = sorted({b["seed"] for b in branches})
        succ = [b for b in branches if b.get("verified_success")]
        seeds_solved = sorted({b["seed"] for b in succ})
        rolls = [b["rollouts_used"] for b in succ if b["rollouts_used"]]

        h_att = h_att_cap = h_tot = 0
        h_seeds = set()
        h_seeds_cap = set()
        h_rolls_cap = []
        for s in seeds:
            for run in hexa.get(s, []):
                h_tot += 1
                if run["success"]:
                    h_att += 1
                    h_seeds.add(s)
                    if run["rollouts"] <= cap:
                        h_att_cap += 1
                        h_seeds_cap.add(s)
                        h_rolls_cap.append(run["rollouts"])

        blocks[f"{budget}x{agents}"] = {
            "dw_budget_per_agent": budget,
            "dw_agents_per_seed": agents,
            "dw_rollouts_per_seed": (budget or 0) * agents,
            "seeds": seeds,
            "dw_attempts": len(branches),
            "dw_attempts_solved": len(succ),
            "dw_attempt_rate": round(len(succ) / len(branches), 4) if branches else None,
            "dw_seeds_solved": seeds_solved,
            "dw_seed_rate": round(len(seeds_solved) / len(seeds), 4) if seeds else None,
            "dw_mean_rollouts_on_success": round(statistics.mean(rolls), 2) if rolls else None,
            "dw_median_rollouts_on_success": statistics.median(rolls) if rolls else None,
            "hexa_attempts": h_tot,
            "hexa_attempts_solved_uncapped": h_att,
            "hexa_attempt_rate_uncapped": round(h_att / h_tot, 4) if h_tot else None,
            "hexa_attempts_solved_capped": h_att_cap,
            "hexa_attempt_rate_capped": round(h_att_cap / h_tot, 4) if h_tot else None,
            "hexa_cap_applied": cap,
            "hexa_seeds_solved_uncapped": sorted(h_seeds),
            "hexa_seeds_solved_capped": sorted(h_seeds_cap),
            "hexa_mean_rollouts_on_success_capped":
                round(statistics.mean(h_rolls_cap), 2) if h_rolls_cap else None,
        }
    return blocks


def write_report(path: str, data: dict) -> None:
    L = []
    A = L.append
    A("# HeXA vs Claude-Code dynamic workflow — catapult\n")
    A(f"Generated by `scripts/aggregate_dw_hexa.py`. Level: `{data['level']}`. "
      f"DW actor: `{data['dw_actor']}`.\n")
    A("Ground truth is the simulator: every placement below was re-simulated at "
      "aggregation time.\n")

    A("\n## Headline\n")
    A("| DW design | seeds | DW per-attempt | HeXA per-attempt @cap | "
      "HeXA per-attempt uncapped |")
    A("|---|---|---|---|---|")
    for _, b in sorted(data["summary"].items(),
                       key=lambda kv: (kv[1]['dw_budget_per_agent'] or 0,
                                       kv[1]['dw_agents_per_seed'])):
        A(f"| {b['dw_budget_per_agent']} x {b['dw_agents_per_seed']} "
          f"= {b['dw_rollouts_per_seed']}/seed | {len(b['seeds'])} | "
          f"{b['dw_attempts_solved']}/{b['dw_attempts']} = "
          f"{100*(b['dw_attempt_rate'] or 0):.1f}% | "
          f"{b['hexa_attempts_solved_capped']}/{b['hexa_attempts']} = "
          f"{100*(b['hexa_attempt_rate_capped'] or 0):.1f}% | "
          f"{b['hexa_attempts_solved_uncapped']}/{b['hexa_attempts']} = "
          f"{100*(b['hexa_attempt_rate_uncapped'] or 0):.1f}% |")

    A("\n## Per-seed detail\n")
    A("| seed | DW budget | lens | rollouts | placement | claimed | verified | capped |")
    A("|---|---|---|---|---|---|---|---|")
    for b in sorted(data["branches"], key=lambda z: (z["budget_per_agent"] or 0,
                                                     z.get("agents_per_seed") or 1,
                                                     z["seed"], z["lens"] or "")):
        if b["voided_for_leakage"]:
            continue
        p = b["placement"]
        ps = (f"({p[0]}, {p[1]}, r={p[2]})" if None not in p else "-")
        v = b.get("verified_success")
        A(f"| {b['seed']} | {b['budget_per_agent']} | {b['lens']} | "
          f"{b['rollouts_used']} | {ps} | {b['claim_success']} | "
          f"{'**YES**' if v else ('no' if v is False else '?')} | "
          f"{b['hit_budget_cap']} |")

    mism = [b for b in data["branches"]
            if not b["voided_for_leakage"]
            and b.get("verified_success") is not None
            and bool(b["claim_success"]) != bool(b["verified_success"])]
    A(f"\n**Claim-vs-verified mismatches: {len(mism)}**"
      + ("" if not mism else " — listed below, treat with suspicion"))
    for b in mism:
        A(f"- seed {b['seed']} / {b['lens']}: claimed {b['claim_success']}, "
          f"verified {b['verified_success']}")

    A("\n## HeXA reference (all runs, this level)\n")
    A("| seed | solved | rollouts per run |")
    A("|---|---|---|")
    for seed in sorted(data["hexa"], key=int):
        runs = data["hexa"][seed]
        s = sum(1 for r in runs if r["success"])
        A(f"| {seed} | {s}/{len(runs)} | {[r['rollouts'] for r in runs]} |")

    if data["voided"]:
        A("\n## Voided branches (contaminated)\n")
        A("Excluded from all statistics. These read the solution oracle.\n")
        for b in data["voided"]:
            A(f"- run `{b['run_id']}` seed {b['seed']} / {b['lens']}")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--hexa-dir", default=DEFAULT_HEXA)
    p.add_argument("--level", default="catapult")
    p.add_argument("--cap", type=int, default=12,
                   help="Per-attempt rollout cap to re-score HeXA under.")
    p.add_argument("--actor", default="claude-opus-5")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip re-simulation; trust recorded claims (faster).")
    a = p.parse_args()

    branches = load_dw_runs(os.path.join(a.dir, "runs"))
    if not branches:
        print(f"no runs found in {a.dir}/runs", file=sys.stderr)
        return 1

    if not a.no_verify:
        for i, b in enumerate(branches, 1):
            if b["voided_for_leakage"]:
                b.update({"verified_success": None, "verified_reward": None,
                          "note": "voided"})
                continue
            b.update(verify(a.level, b["seed"], b["placement"]))
            print(f"  [{i}/{len(branches)}] seed {b['seed']:>3} {str(b['lens']):<9} "
                  f"claim={str(b['claim_success']):<5} "
                  f"verified={b['verified_success']}")
    else:
        for b in branches:
            b["verified_success"] = (None if b["voided_for_leakage"]
                                     else b["claim_success"])
            b["verified_reward"] = b["claim_reward"]

    hexa = load_hexa(a.hexa_dir)
    data = {
        "level": a.level,
        "dw_actor": a.actor,
        "hexa_cap": a.cap,
        "verified": not a.no_verify,
        "summary": summarise(branches, hexa, a.cap),
        "branches": branches,
        "voided": [b for b in branches if b["voided_for_leakage"]],
        "hexa": {str(k): v for k, v in sorted(hexa.items())},
    }

    rj = os.path.join(a.dir, "results.json")
    with open(rj, "w") as f:
        json.dump(data, f, indent=2)
    rm = os.path.join(a.dir, "REPORT.md")
    write_report(rm, data)

    print()
    for _, b in sorted(data["summary"].items(),
                       key=lambda kv: (kv[1]['dw_budget_per_agent'] or 0,
                                       kv[1]['dw_agents_per_seed'])):
        print(f"{b['dw_budget_per_agent']} rollouts x {b['dw_agents_per_seed']} agent(s) "
              f"= {b['dw_rollouts_per_seed']}/seed | seeds {b['seeds']}")
        print(f"  DW   per-attempt {b['dw_attempts_solved']}/{b['dw_attempts']} "
              f"= {100*(b['dw_attempt_rate'] or 0):.1f}%   "
              f"seeds {len(b['dw_seeds_solved'])}/{len(b['seeds'])}")
        print(f"  HeXA per-attempt {b['hexa_attempts_solved_capped']}/{b['hexa_attempts']} "
              f"= {100*(b['hexa_attempt_rate_capped'] or 0):.1f}% (cap {a.cap})   "
              f"uncapped {100*(b['hexa_attempt_rate_uncapped'] or 0):.1f}%")
    print(f"\nwrote {rj}\nwrote {rm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
