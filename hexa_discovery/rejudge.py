"""Re-score stored episodes with an INDEPENDENT judge model.

Why this exists: `--cli-model` forces every CLI call through one model, so it
silently overrides `--judge-model` and the actor ends up grading its own
explanations. The DiscoverPhysics paper judges with `claude-opus-4-6` and names
"Claude-on-Claude favoritism" as an unresolved concern even in their weaker
setup (their judge is a different model from every agent under test). Ours is
the same model, which is worse.

`expl_score` is half the reward and, on the 8 FIT_WORLDS, effectively the whole
pass criterion — a wrong functional form still fits trajectories to ~1e-5 — so a
biased judge would not be a detail.

This re-runs the judge ONLY. Episodes are not replayed: the agent's stored
explanation is re-scored against the same ground truth and rubric, so the
comparison isolates the judge. Originals are never mutated; results land in a
separate JSON so the provenance of the first scoring survives.

The paper's pass threshold is 0.9 explanation score; `scripts/run_benchmark.py`
hardcodes 0.75 and our config inherited that. Both are reported, because the two
disagree and any claim should survive the stricter one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexa_discovery import _paths

_paths.bootstrap()

from hexa_discovery.config import PASS_ERR_THRESHOLD, WORLD_ENGINE
from hexa_discovery.reward import blended_reward

PAPER_SCORE_THRESHOLD = 0.90
CODE_SCORE_THRESHOLD = 0.75


def rejudge_file(path: Path, judge_model: str, noise_std: float = 0.075) -> dict:
    from scienceagent.worlds import get_world
    from scienceagent.evaluator import ExplanationJudge

    d = json.loads(path.read_text())
    world, seed = d["world"], d["seed"]
    wd = get_world(world, engine=WORLD_ENGINE.get(world, "nbody"),
                   noise_std=noise_std, noise_seed=seed)
    res = ExplanationJudge(judge_model=judge_model).score(
        agent_explanation=d.get("explanation"),
        optimal_explanation=wd.get("optimal_explanation", ""),
        rubric=wd.get("explanation_rubric", ""),
        verbose=False,
    )
    new = res.get("score")
    new = 0.0 if new is None else float(new)
    nm = d.get("norm_mse")
    return {
        "file": str(path), "world": world, "seed": seed,
        "arm": path.parts[-3],
        "norm_mse": nm,
        "expl_self": d.get("expl_score"),
        "expl_independent": new,
        "reward_self": d.get("reward"),
        "reward_independent": blended_reward(new, nm),
        "pass_self_075": bool(d.get("success")),
        "pass_indep_075": bool(nm is not None and nm < PASS_ERR_THRESHOLD
                               and new >= CODE_SCORE_THRESHOLD),
        "pass_indep_090": bool(nm is not None and nm < PASS_ERR_THRESHOLD
                               and new >= PAPER_SCORE_THRESHOLD),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Re-judge stored episodes with an independent judge")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--judge-model", default="claude-opus-4-6",
                   help="The paper uses claude-opus-4-6. Must differ from the actor.")
    p.add_argument("--glob", default="transfer/*/*/discovery/trajectory_seed*.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from hexa_discovery.cli_client import patch_llm_client
    # No model_override: the judge's own model string must be honoured, which is
    # precisely what the run-time override defeats.
    patch_llm_client()

    run = Path(args.run_dir)
    files = sorted(run.glob(args.glob))
    print(f"re-judging {len(files)} episode(s) with {args.judge_model}\n")

    rows = []
    for f in files:
        r = rejudge_file(f, args.judge_model)
        rows.append(r)
        print(f"  {r['world']:7s} {r['arm']:9s} seed {r['seed']}: "
              f"expl {r['expl_self']} -> {r['expl_independent']}   "
              f"reward {r['reward_self']:+.3f} -> {r['reward_independent']:+.3f}")

    out = Path(args.out or run / f"rejudge_{args.judge_model}.json")
    out.write_text(json.dumps({"judge_model": args.judge_model, "rows": rows}, indent=2))
    print(f"\nwrote {out}")

    # Per-(world, arm) summary at both thresholds.
    print(f"\n{'world':8s} {'arm':10s} {'expl self':>10s} {'expl indep':>11s} "
          f"{'rew self':>9s} {'rew indep':>10s} {'p@.75':>6s} {'p@.90':>6s}")
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["world"], r["arm"]), []).append(r)
    for (w, arm), rs in sorted(groups.items()):
        n = len(rs)
        print(f"{w:8s} {arm:10s} "
              f"{sum(x['expl_self'] or 0 for x in rs)/n:10.3f} "
              f"{sum(x['expl_independent'] for x in rs)/n:11.3f} "
              f"{sum(x['reward_self'] for x in rs)/n:+9.3f} "
              f"{sum(x['reward_independent'] for x in rs)/n:+10.3f} "
              f"{sum(x['pass_indep_075'] for x in rs):3d}/{n:<2d} "
              f"{sum(x['pass_indep_090'] for x in rs):3d}/{n:<2d}")


if __name__ == "__main__":
    main()
