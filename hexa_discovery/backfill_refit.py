"""Recover `norm_mse_own` for trajectories written before it was stored.

Re-scores each stored law WITHOUT the evaluator's parameter refit -- simulation
only, no LLM calls and no re-run of any episode. The gap against the recorded
(post-refit) `norm_mse` is what tells the teacher that declaring an
already-measured constant as fittable is what cost the episode.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

from hexa_discovery import _paths
_paths.bootstrap()

from hexa_discovery.config import FIT_WORLDS, WORLD_ENGINE, WORLD_VARS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", required=True)
    ap.add_argument("--root", default="hexa_discovery/data/offline/sonnet")
    args = ap.parse_args()

    from scienceagent.worlds import get_world
    from hexa_discovery.discovery_runner import _make_evaluator

    done = skipped = 0
    for w in args.worlds:
        if w not in FIT_WORLDS:
            print(f"  {w}: not a FIT_WORLD, nothing to recover"); continue
        for f in sorted(Path(f"{args.root}/{w}/discovery").glob("trajectory_seed*.json")):
            d = json.loads(f.read_text())
            if d.get("norm_mse_own") is not None or not d.get("final_law_raw"):
                skipped += 1; continue
            try:
                wd = get_world(w, engine=WORLD_ENGINE.get(w, "nbody"),
                               noise_std=d.get("noise_std", 0.075), noise_seed=d["seed"])
                r = _make_evaluator(w, wd["executor"]).evaluate(d["final_law_raw"], verbose=False)
                own = r.get("mean_pos_error", float("inf")) / WORLD_VARS[w]
            except Exception as exc:
                print(f"  {w} seed {d['seed']}: eval failed ({str(exc)[:50]})"); continue
            d["norm_mse_own"] = own
            f.write_text(json.dumps(d, indent=2, default=str))
            rec = d.get("norm_mse")
            tag = ""
            if rec and own and own > 0:
                ratio = rec / own
                tag = f"  refit {'HURT' if ratio > 2 else ('helped' if ratio < 0.5 else 'neutral')} ({ratio:.1f}x)"
            print(f"  {w} seed {d['seed']}: own={own:.3g} recorded={rec:.3g}{tag}")
            done += 1
    print(f"\nbackfilled {done}, skipped {skipped}")


if __name__ == "__main__":
    main()
