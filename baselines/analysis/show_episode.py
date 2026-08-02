"""Pretty-print one episode: what was injected, then what the agent did.

    python -m baselines.analysis.show_episode baselines/data/memrl_down_to_earth_qwen7b 9
    python -m baselines.analysis.show_episode <run_dir> 9 --full-block --max-obs 2000

Reads the per-seed trajectory JSON plus that seed's line in episodes.jsonl, so
you see the retrieval decision (which memories, their similarity and Q, and why
nothing was injected when nothing was) alongside the resulting behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def find_trajectory(run_dir: Path, seed: int) -> Optional[Path]:
    hits = sorted(run_dir.glob(f"*/trajectory_seed{seed}_*.json"))
    return hits[0] if hits else None


def episode_row(run_dir: Path, seed: int) -> Optional[dict]:
    path = run_dir / "episodes.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("seed") == seed:
            return row
    return None


def clip(text: str, limit: int) -> str:
    text = str(text or "")
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f" ...[+{len(text) - limit} chars]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Show one episode in detail")
    ap.add_argument("run_dir")
    ap.add_argument("seed", type=int)
    ap.add_argument("--max-obs", type=int, default=400, help="chars of each observation")
    ap.add_argument("--max-thought", type=int, default=300)
    ap.add_argument("--steps", type=int, default=0, help="show only the last N steps")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traj_path = find_trajectory(run_dir, args.seed)
    if traj_path is None:
        raise SystemExit(f"No trajectory for seed {args.seed} under {run_dir}")
    result = json.loads(traj_path.read_text())
    row = episode_row(run_dir, args.seed)

    print("=" * 78)
    print(f"{run_dir.name}   seed {args.seed}")
    print("=" * 78)
    print(f"outcome          : {'SUCCESS' if result.get('success') else 'FAILURE'}")
    print(f"iterations       : {result.get('iterations')}   "
          f"simulator calls: {result.get('simulator_calls')}")
    print(f"final action     : {result.get('action')}")
    print(f"injected tokens  : {result.get('added_context_tokens')}")
    print(f"final observation: {clip(result.get('final_observation'), 200)}")

    diag = result.get("retrieval_diagnostics") or {}
    if diag:
        print("\n--- retrieval ---")
        for key in ("mode", "phase_a", "simmax", "delta", "pool_size", "num_candidates"):
            if key in diag:
                print(f"  {key:16}: {diag[key]}")
        for chosen in diag.get("selected", []):
            print(f"  chose seed {chosen['seed']:>3} ({chosen['kind']:<7}) "
                  f"sim {chosen['similarity']:+.3f}  Q {chosen['q']:+.3f}  "
                  f"score {chosen['score']:+.3f}")
        if "similarities" in diag:
            print(f"  similarities    : {diag['similarities']}")

    # Episodic RAG has no utilities, so its rows carry an empty update record.
    if row and (row.get("utility_updates") or {}).get("updates"):
        upd = row["utility_updates"]
        reward = upd.get("reward")
        label = f"{reward:+.1f}" if isinstance(reward, (int, float)) else "n/a"
        print(f"\n--- utility update (reward {label}) ---")
        for u in upd.get("updates", []):
            print(f"  {u['memory_id']}: Q {u['old_q']:+.3f} -> {u['new_q']:+.3f} "
                  f"(visits {u['visits']})")

    steps = result.get("trajectory") or []
    shown = steps[-args.steps:] if args.steps else steps
    offset = len(steps) - len(shown)
    print(f"\n--- trajectory ({len(steps)} steps"
          f"{f', showing last {len(shown)}' if offset else ''}) ---")
    for i, step in enumerate(shown, offset + 1):
        if isinstance(step, list):
            thought, action, obs = (list(step) + ["", "", ""])[:3]
        else:
            thought, action, obs = (step.get("thought"), step.get("action"),
                                    step.get("observation"))
        print(f"\n[{i}] {clip(thought, args.max_thought)}")
        print(f"    ACTION: {action}")
        print(f"    OBS   : {clip(obs, args.max_obs)}")


if __name__ == "__main__":
    main()
