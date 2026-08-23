"""Turn a dynamic-workflow solve into HeXA baseline artifacts.

Reads the probe interaction log written by `scripts/interphyre_probe.py` plus a
JSON file describing the run (direct one-shot proposal, per-hypothesis results,
committed placement), re-verifies both the direct proposal and the committed
placement against the simulator, and writes the same
`trajectory_seed<N>_raw<model>.json` / `summary_<model>_direct.json` pair the
existing direct baseline emits — so `react_agent/plot_eval.py` and the rest of
the analysis path work unchanged.

    python scripts/record_dw_run.py --run-json <dw_run.json> --eval-dir <dir>
"""

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "interphyre"))

import cv2  # noqa: E402

from react_agent.tools import InterphyreToolkit  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-json", required=True, help="Workflow result JSON")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--model", default="claude", help="Logging label for filenames")
    p.add_argument("--actor-model", default="claude-opus-5")
    a = p.parse_args()

    with open(a.run_json) as f:
        run = json.load(f)

    level, seed = run["level"], int(run["seed"])
    os.makedirs(a.eval_dir, exist_ok=True)

    toolkit = InterphyreToolkit(level_name=level, seed=seed, is_oss=False)
    scene = toolkit.dispatch_tool("get_level_state", {})
    cv2.imwrite(os.path.join(a.eval_dir, f"seed{seed}_raw{a.model}_initial.png"),
                toolkit.get_image())

    def verify(pl):
        if not pl:
            return None, "(no placement)", False
        x, y, r = float(pl["x"]), float(pl["y"]), float(pl["radius"])
        obs = toolkit.dispatch_tool("simulate_action", {"x": x, "y": y, "radius": r})
        return (x, y, r), obs, "SUCCESS!" in obs

    d_act, d_obs, d_success = verify(run.get("direct_baseline_proposal"))
    c_act, c_obs, c_success = verify(run.get("committed"))

    cv2.imwrite(os.path.join(a.eval_dir, f"seed{seed}_raw{a.model}_final.png"),
                toolkit.get_image())
    toolkit.close()

    # Interaction history from the probe log (the "experiments" the DW actually ran).
    probe_log = os.path.join(a.eval_dir, "probe_log.jsonl")
    probes = []
    if os.path.exists(probe_log):
        with open(probe_log) as f:
            for line in f:
                line = line.strip()
                if line:
                    probes.append(json.loads(line))
    sims = [q for q in probes if q["tool"] in ("simulate_action", "simulate_partial")]

    trajectory = [
        {"thought": "(dynamic workflow: scripted peek)",
         "action": "get_level_state({})",
         "observation": scene},
    ]
    if d_act:
        dp = run["direct_baseline_proposal"]
        trajectory.append({
            "thought": f"Thought: {dp['thought']}",
            "action": f'direct_one_shot({{"x": {d_act[0]}, "y": {d_act[1]}, "radius": {d_act[2]}}})',
            "observation": d_obs,
        })
    for h in run.get("hypotheses", []):
        trajectory.append({
            "thought": (f"[hypothesis:{h['lens']}] {h['hypothesis']}\n"
                        f"Evidence: {h['evidence']}\nVerdict: {h['verdict']}"),
            "action": (f'hypothesis_best({{"x": {h["best_x"]}, "y": {h["best_y"]}, '
                       f'"radius": {h["best_radius"]}}})'),
            "observation": (f"success={h['best_success']} reward={h.get('best_reward')} "
                            f"probes={h['probes_run']}"),
        })
    if c_act:
        trajectory.append({
            "thought": f"Thought: {run['committed']['thought']}",
            "action": f'finish({{"x": {c_act[0]}, "y": {c_act[1]}, "radius": {c_act[2]}}})',
            "observation": c_obs,
        })

    traj = {
        "seed": seed,
        "success": c_success,
        "action": list(c_act) if c_act else None,
        "iterations": len(trajectory),
        "elapsed_time": run.get("elapsed_time"),
        "method": "dynamic_workflow_hypothesis",
        "actor_model": a.actor_model,
        "direct_one_shot": {
            "action": list(d_act) if d_act else None,
            "success": d_success,
        },
        "env_interactions": {
            "total_probe_calls": len(probes),
            "simulate_calls": len(sims),
            "distinct_placements": len({
                (q["args"].get("x"), q["args"].get("y"), q["args"].get("radius"))
                for q in sims
            }),
        },
        "branches_succeeded": run.get("branches_succeeded"),
        "trajectory": trajectory,
    }
    tp = os.path.join(a.eval_dir, f"trajectory_seed{seed}_raw{a.model}.json")
    with open(tp, "w") as f:
        json.dump(traj, f, indent=2)

    summary = {
        "model": a.model,
        "actor_model": a.actor_model,
        "level": level,
        "baseline": "direct_answer",
        "method": "dynamic_workflow_hypothesis",
        "variant": "cot",
        "seeds": [seed],
        "total_seeds": 1,
        "successes": int(c_success),
        "success_rate": float(c_success),
        "parse_failures": 0,
        "direct_one_shot_successes": int(d_success),
        "results": [{
            "seed": seed,
            "success": c_success,
            "action": list(c_act) if c_act else None,
            "elapsed_time": run.get("elapsed_time"),
        }],
    }
    sp = os.path.join(a.eval_dir, f"summary_{a.model}_direct.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"level={level} seed={seed}")
    print(f"  direct one-shot : action={d_act} success={d_success}")
    print(f"  DW committed    : action={c_act} success={c_success}")
    print(f"  env interactions: {len(probes)} probe calls, {len(sims)} rollouts, "
          f"{traj['env_interactions']['distinct_placements']} distinct placements")
    print(f"  wrote {tp}")
    print(f"  wrote {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
