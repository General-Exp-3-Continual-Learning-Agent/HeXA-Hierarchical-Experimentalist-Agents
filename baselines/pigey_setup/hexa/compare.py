"""Paired comparison of two frozen-evaluation arms: with a HeXA bank and without.

Both arms ran the same (task, init state) episodes through ``slurm/run_trial.sbatch``.
This reads every ``trial.json`` under each results directory and reports

* success per arm on the episodes both arms finished, and the difference;
* discordant episodes (solved by one arm only) with an exact two-sided McNemar p;
* per-task counts, Claude cost and mean LLM steps per arm;
* an audit of the with-bank arm: every episode received the bank, nothing was
  dropped for the token budget, and no injected entry cites the episode it was
  injected into (``parent_overlap``, HeXA's contamination check).

Episodes missing from either arm, or ending in a backend error, are listed rather
than compared. When an episode was rerun, the latest finished attempt counts.

    python hexa/compare.py --with-bank DIR --no-bank DIR [--suite libero_goal_task] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

SETUP_ROOT = Path(__file__).resolve().parents[1]  # baselines/pigey_setup
if str(SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(SETUP_ROOT))

from hexa.trajectory import seed_of  # noqa: E402


def _count(value) -> int:
    return len(value) if isinstance(value, (list, tuple, set, dict)) else int(value or 0)


def load_arm(root: str | Path, suite: str | None = None) -> dict[tuple[int, int], dict]:
    """The latest finished trial per (task, episode); a backend error is kept only when nothing finished."""
    trials: dict[tuple[int, int], dict] = {}
    for path in sorted(Path(root).rglob("trial.json")):
        try:
            trial = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if suite and trial.get("suite") != suite:
            continue
        key = (int(trial["task_id"]), int(trial["episode"]))
        finished = trial.get("status") != "backend_error" and trial.get("success") is not None
        previous = trials.get(key)
        if previous is not None and previous["finished"] and not finished:
            continue
        injection = trial.get("skill_injection")
        trials[key] = {
            "task": key[0], "episode": key[1], "finished": finished,
            "success": bool(trial.get("success")) if finished else None,
            "cost": sum(call.get("total_cost_usd") or 0 for call in trial.get("claude_code_calls", [])),
            "llm_steps": int(trial.get("llm_steps") or 0),
            "description": trial.get("task_description", ""),
            "injection": None if injection is None else {
                "skills": len(injection.get("skill_ids", [])), "mistakes": len(injection.get("mistake_ids", [])),
                "dropped": _count(injection.get("dropped_skills")) + _count(injection.get("dropped_mistakes")),
                "parent_overlap": _count(injection.get("parent_overlap")), "tokens": injection.get("block_tokens")},
            "path": str(path),
        }
    return trials


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(with_bank: dict, no_bank: dict) -> dict:
    paired = sorted(key for key in with_bank
                    if key in no_bank and with_bank[key]["finished"] and no_bank[key]["finished"])
    seeds = [seed_of(*key) for key in paired]
    wins_bank = sum(with_bank[key]["success"] for key in paired)
    wins_none = sum(no_bank[key]["success"] for key in paired)
    only_bank = [seed_of(*key) for key in paired if with_bank[key]["success"] and not no_bank[key]["success"]]
    only_none = [seed_of(*key) for key in paired if no_bank[key]["success"] and not with_bank[key]["success"]]
    per_task = {}
    for task in sorted({key[0] for key in paired}):
        keys = [key for key in paired if key[0] == task]
        per_task[task] = {"episodes": len(keys), "with_bank": sum(with_bank[k]["success"] for k in keys),
                          "no_bank": sum(no_bank[k]["success"] for k in keys),
                          "description": with_bank[keys[0]]["description"]}

    def arm_stats(arm):
        return {"cost_usd": round(sum(arm[k]["cost"] for k in paired), 2),
                "mean_llm_steps": round(sum(arm[k]["llm_steps"] for k in paired) / len(paired), 1) if paired else 0.0}

    injections = [with_bank[k]["injection"] for k in paired]
    audit = {"episodes": len(paired),
             "injected": sum(1 for inj in injections if inj),
             "dropped": sum(inj["dropped"] for inj in injections if inj),
             "parent_overlap": sum(inj["parent_overlap"] for inj in injections if inj),
             "skills_per_episode": sorted({inj["skills"] for inj in injections if inj}),
             "mistakes_per_episode": sorted({inj["mistakes"] for inj in injections if inj}),
             "no_bank_arm_injected": sum(1 for k in paired if no_bank[k]["injection"])}
    missing = {"with_bank": [seed_of(*k) for k in sorted(no_bank) if k not in with_bank or not with_bank[k]["finished"]],
               "no_bank": [seed_of(*k) for k in sorted(with_bank) if k not in no_bank or not no_bank[k]["finished"]]}
    return {"paired_episodes": len(paired), "seeds": seeds,
            "with_bank": {"successes": wins_bank, **arm_stats(with_bank)},
            "no_bank": {"successes": wins_none, **arm_stats(no_bank)},
            "delta": wins_bank - wins_none,
            "only_with_bank": only_bank, "only_no_bank": only_none,
            "mcnemar_p": mcnemar_exact(len(only_bank), len(only_none)),
            "per_task": per_task, "audit": audit, "missing": missing}


def format_report(result: dict) -> str:
    n = result["paired_episodes"]
    if not n:
        return "No episode was finished by both arms.\n" + json.dumps(result["missing"], indent=2)
    wb, nb = result["with_bank"], result["no_bank"]
    lines = [f"Paired episodes: {n}",
             f"  with bank : {wb['successes']}/{n} = {wb['successes'] / n:.1%}   cost ${wb['cost_usd']:.2f}, "
             f"mean LLM steps {wb['mean_llm_steps']}",
             f"  no bank   : {nb['successes']}/{n} = {nb['successes'] / n:.1%}   cost ${nb['cost_usd']:.2f}, "
             f"mean LLM steps {nb['mean_llm_steps']}",
             f"  delta     : {result['delta']:+d} episodes ({result['delta'] / n:+.1%})",
             f"  discordant: {len(result['only_with_bank'])} solved only with the bank, "
             f"{len(result['only_no_bank'])} only without; exact McNemar p = {result['mcnemar_p']:.3f}",
             "", f"{'task':>4} {'eps':>4} {'bank':>5} {'none':>5}  instruction"]
    for task, row in result["per_task"].items():
        lines.append(f"{task:>4} {row['episodes']:>4} {row['with_bank']:>5} {row['no_bank']:>5}  {row['description']}")
    audit = result["audit"]
    lines += ["", f"Injection audit (with-bank arm): {audit['injected']}/{audit['episodes']} episodes received the bank; "
              f"{audit['dropped']} entries dropped for the budget; {audit['parent_overlap']} parent-overlap entries; "
              f"skills per episode {audit['skills_per_episode']}, mistakes {audit['mistakes_per_episode']}; "
              f"no-bank arm injections: {audit['no_bank_arm_injected']}"]
    for arm, seeds in result["missing"].items():
        if seeds:
            lines.append(f"Not finished in the {arm.replace('_', '-')} arm ({len(seeds)}): {' '.join(seeds)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-bank", required=True, help="Results directory of the arm that ran with PIGEY_SKILL_BANK")
    parser.add_argument("--no-bank", required=True, help="Results directory of the control arm")
    parser.add_argument("--suite", default="libero_goal_task")
    parser.add_argument("--json", default=None, help="Also write the full result as JSON here")
    args = parser.parse_args(argv)
    result = compare(load_arm(args.with_bank, args.suite), load_arm(args.no_bank, args.suite))
    result["with_bank_dir"], result["no_bank_dir"] = str(args.with_bank), str(args.no_bank)
    print(format_report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
