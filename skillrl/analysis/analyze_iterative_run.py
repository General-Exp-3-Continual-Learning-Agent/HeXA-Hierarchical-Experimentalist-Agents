"""Compute training + final-eval stats for an iterative-loop run.

Mirrors analyze_evolving_run.py but reads the iterative layout:
    progress.json            (per-round progress, with accumulated_successes)
    round_<N>/<level>/trajectory_seed*_skillrl.json
    skill_bank_<N>.json
    final_eval/<level>/trajectory_seed*_skillrl.json   (optional)

Usage:
    python -m skillrl.analysis.analyze_iterative_run \\
        --run-dir skillrl/data/iterative/catapult \\
        --level catapult

    # restrict training stats to specific rounds
    python -m skillrl.analysis.analyze_iterative_run \\
        --run-dir skillrl/data/iterative/catapult \\
        --level catapult \\
        --rounds 1-6

    # exclude rounds from the training aggregate
    python -m skillrl.analysis.analyze_iterative_run \\
        --run-dir skillrl/data/iterative/catapult \\
        --level catapult \\
        --exclude-rounds 7
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def parse_round_spec(spec: str) -> set[int]:
    """Parse "1-6,8" -> {1,2,3,4,5,6,8}."""
    out: set[int] = set()
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def load_progress(run_dir: Path) -> list[dict]:
    path = run_dir / "progress.json"
    if not path.exists():
        sys.exit(f"[Error] progress.json not found at {path}")
    return json.loads(path.read_text())


def load_round_trajectories(run_dir: Path, level: str, round_num: int) -> list[dict]:
    """Read per-seed trajectory files inside round_<N>/<level>/."""
    round_dir = run_dir / f"round_{round_num}" / level
    if not round_dir.exists():
        return []
    out = []
    for fp in sorted(round_dir.glob("trajectory_seed*_skillrl.json"),
                     key=lambda p: int(p.stem.split("seed")[1].split("_")[0])):
        try:
            d = json.loads(fp.read_text())
        except json.JSONDecodeError:
            continue
        out.append({
            "seed": d.get("seed"),
            "success": bool(d.get("success")),
            "iterations": d.get("iterations", 0) or 0,
            "elapsed": d.get("elapsed_time") or 0.0,
        })
    return out


def load_eval_trajectories(run_dir: Path, level: str) -> list[dict]:
    eval_dir = run_dir / "final_eval" / level
    if not eval_dir.exists():
        return []
    files = sorted(
        glob.glob(str(eval_dir / "trajectory_seed*_skillrl.json")),
        key=lambda f: int(os.path.basename(f).split("seed")[1].split("_")[0]),
    )
    out = []
    for fp in files:
        try:
            d = json.loads(Path(fp).read_text())
        except json.JSONDecodeError:
            print(f"  [Warning] Bad JSON in {fp}, skipping")
            continue
        out.append({
            "seed": d.get("seed"),
            "success": bool(d.get("success")),
            "iterations": d.get("iterations", 0) or 0,
            "elapsed": d.get("elapsed_time") or 0.0,
        })
    return out


def _fmt_seed_range(seeds: list[int]) -> str:
    if not seeds:
        return "-"
    return f"{min(seeds)}-{max(seeds)}"


def print_training_stats(progress: list[dict], include: set[int], exclude: set[int],
                         run_dir: Path, level: str) -> dict:
    print("=" * 90)
    print("ITERATIVE LOOP — per-round training stats")
    print("=" * 90)
    print(f"{'Round':<7}{'Seeds':<12}{'Succ/Tot':<10}{'Acc':<8}{'AccSucc':<10}"
          f"{'AvgIt':<8}{'AvgIt(S)':<10}{'Time(s)':<10}{'Kept':<6}")
    print("-" * 90)

    total_s = total_n = 0
    total_t = 0.0
    total_iters = 0
    total_iters_succ = 0
    total_succ_for_iters = 0
    kept_rounds = []
    training_seeds: set[int] = set()
    last_accumulated = 0

    for r in progress:
        rn = r["round"]
        kept = (not include or rn in include) and rn not in exclude
        seeds = r["seeds"]
        n = len(seeds)
        s = r["successes"]
        t = r["elapsed_seconds"]
        accumulated = r.get("accumulated_successes", 0)

        trajs = load_round_trajectories(run_dir, level, rn)
        if trajs:
            iters = [x["iterations"] for x in trajs]
            avg_it = sum(iters) / len(iters)
            succ_iters = [x["iterations"] for x in trajs if x["success"]]
            avg_it_s = (sum(succ_iters) / len(succ_iters)) if succ_iters else 0.0
            avg_it_s_str = f"{avg_it_s:.2f}" if succ_iters else "-"
        else:
            avg_it = 0.0
            avg_it_s_str = "-"

        mark = "yes" if kept else "-"
        print(f"{rn:<7}{_fmt_seed_range(seeds):<12}{s}/{n:<8}{(s/n if n else 0):<8.3f}"
              f"{accumulated:<10}{avg_it:<8.2f}{avg_it_s_str:<10}{t:<10.2f}{mark:<6}")

        last_accumulated = max(last_accumulated, accumulated)

        if kept:
            total_s += s
            total_n += n
            total_t += t
            kept_rounds.append(rn)
            training_seeds.update(seeds)
            if trajs:
                total_iters += sum(x["iterations"] for x in trajs)
                total_iters_succ += sum(x["iterations"] for x in trajs if x["success"])
                total_succ_for_iters += sum(1 for x in trajs if x["success"])

    print("-" * 90)
    if total_n:
        avg_iters = total_iters / total_n
        avg_iters_succ = (total_iters_succ / total_succ_for_iters) if total_succ_for_iters else 0.0
        print(f"Kept rounds: {kept_rounds}")
        print(
            f"TRAINING AGG: {total_s}/{total_n} = {total_s/total_n:.4f} ({total_s/total_n*100:.2f}%) "
            f"| avg {avg_iters:.2f} iters/seed, {avg_iters_succ:.2f} iters on successes "
            f"| {total_t:.1f}s ({total_t/60:.1f} min), avg {total_t/total_n:.2f}s/seed"
        )
        print(f"Final accumulated_successes (across all rounds in progress.json): {last_accumulated}")
    else:
        print("No rounds kept.")
        avg_iters = avg_iters_succ = 0.0

    return {
        "rounds": kept_rounds,
        "successes": total_s,
        "seeds_total": total_n,
        "accuracy": total_s / total_n if total_n else 0.0,
        "avg_iters": avg_iters,
        "avg_iters_on_successes": avg_iters_succ,
        "time": total_t,
        "training_seeds": training_seeds,
        "accumulated_successes": last_accumulated,
    }


def print_eval_stats(results: list[dict], training_seeds: set[int]) -> dict:
    print()
    print("=" * 72)
    print("FINAL EVAL — all trajectories in final_eval/<level>/")
    print("=" * 72)
    if not results:
        print("(no eval trajectories found)")
        return {}

    print(f"{'Seed':<6}{'Success':<10}{'Iters':<8}{'Time(s)':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['seed']:<6}{str(r['success']):<10}{r['iterations']:<8}{r['elapsed']:<10.2f}")

    n = len(results)
    succ = sum(1 for r in results if r["success"])
    avg_i = sum(r["iterations"] for r in results) / n
    avg_t = sum(r["elapsed"] for r in results) / n
    tot_t = sum(r["elapsed"] for r in results)

    print("-" * 40)
    print(
        f"EVAL AGG: {succ}/{n} = {succ/n:.4f} ({succ/n*100:.2f}%) "
        f"| avg {avg_i:.2f} iters, {avg_t:.2f}s/seed | total {tot_t:.1f}s ({tot_t/60:.1f} min)"
    )

    seen = [r for r in results if r["seed"] in training_seeds]
    unseen = [r for r in results if r["seed"] not in training_seeds]

    def _blk(tag: str, rs: list[dict]):
        if not rs:
            return
        k = sum(1 for r in rs if r["success"])
        ai = sum(r["iterations"] for r in rs) / len(rs)
        at = sum(r["elapsed"] for r in rs) / len(rs)
        seeds_rng = _fmt_seed_range([r["seed"] for r in rs])
        print(
            f"  {tag} ({seeds_rng}, n={len(rs)}): {k}/{len(rs)} = {k/len(rs):.4f} "
            f"({k/len(rs)*100:.2f}%) | avg {ai:.2f} iters, {at:.2f}s/seed"
        )

    print()
    print("  Splits:")
    _blk("Training-seen ", seen)
    _blk("Fresh (unseen)", unseen)

    return {
        "n": n,
        "successes": succ,
        "accuracy": succ / n,
        "avg_iters": avg_i,
        "avg_time": avg_t,
        "fresh_accuracy": (sum(1 for r in unseen if r["success"]) / len(unseen)) if unseen else None,
    }


def print_comparison(train: dict, ev: dict):
    if not ev or not train.get("seeds_total"):
        return
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    t_acc = train["accuracy"] * 100
    e_acc = ev["accuracy"] * 100
    print(f"Training acc (rounds {train['rounds'][0]}-{train['rounds'][-1]}, excl gaps):"
          f" {t_acc:.2f}% ({train['successes']}/{train['seeds_total']})")
    print(f"Eval acc (all final_eval seeds): {e_acc:.2f}% ({ev['successes']}/{ev['n']})")
    print(f"  Δ = {e_acc - t_acc:+.2f} pp")
    if ev.get("fresh_accuracy") is not None:
        f_acc = ev["fresh_accuracy"] * 100
        print(f"Eval acc (fresh seeds only): {f_acc:.2f}%")
        print(f"  Δ vs training = {f_acc - t_acc:+.2f} pp")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Iterative-loop output dir (contains progress.json)")
    p.add_argument("--level", required=True, type=str, help="Level name (e.g. catapult)")
    p.add_argument("--rounds", type=str, default="",
                   help='Rounds to include, e.g. "1-6". Empty = all.')
    p.add_argument("--exclude-rounds", type=str, default="",
                   help='Rounds to exclude, e.g. "7". Applied after --rounds.')
    p.add_argument("--json-out", type=Path, default=None,
                   help="Optional path to dump aggregate stats as JSON")
    args = p.parse_args()

    include = parse_round_spec(args.rounds)
    exclude = parse_round_spec(args.exclude_rounds)

    progress = load_progress(args.run_dir)
    train = print_training_stats(progress, include, exclude, args.run_dir, args.level)

    eval_results = load_eval_trajectories(args.run_dir, args.level)
    ev = print_eval_stats(eval_results, train["training_seeds"])

    print_comparison(train, ev)

    if args.json_out:
        payload = {
            "run_dir": str(args.run_dir),
            "level": args.level,
            "training": {k: v for k, v in train.items() if k != "training_seeds"},
            "eval": ev,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nAggregate stats saved to {args.json_out}")


if __name__ == "__main__":
    main()
