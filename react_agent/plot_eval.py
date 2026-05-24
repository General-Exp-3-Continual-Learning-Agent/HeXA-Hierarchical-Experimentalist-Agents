import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# example run: python plot_eval.py --eval-dir results_20260319_100000
def load_results_from_trajectories(eval_dir: str):
    """
    Load per-seed results from trajectory_*.json files in eval_dir.

    Supports both:
      - trajectory_seed{N}.json
      - trajectory_seed{N}_raw{model}.json
    """
    pattern = os.path.join(eval_dir, "trajectory_seed*.json")
    traj_paths = sorted(glob.glob(pattern))
    results = []

    for path in traj_paths:
        try:
            with open(path, "r") as f:
                traj = json.load(f)
        except Exception as e:
            print(f"[Warning] Skipping {os.path.basename(path)} (failed to load: {e})")
            continue

        seed = traj.get("seed")
        if seed is None:
            print(f"[Warning] Skipping {os.path.basename(path)} (missing 'seed' field)")
            continue

        results.append(
            {
                "seed": seed,
                "success": bool(traj.get("success", False)),
                "action": traj.get("action"),
                "iterations": traj.get("iterations", 0),
                # Some trajectory files may not store elapsed_time
                "elapsed_time": traj.get("elapsed_time"),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results of ReAct agent.")
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Directory containing trajectory_*.json and/or summary json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name used to find summary_{model}.json (optional, falls back to summary.json)",
    )
    args = parser.parse_args()

    # Optional: load summary for metadata (model, level, max_iterations)
    data = {}
    summary_file = None
    if args.model:
        candidate = os.path.join(args.eval_dir, f"summary_{args.model}.json")
        if os.path.exists(candidate):
            summary_file = candidate
        else:
            print(f"[Info] {candidate} not found, trying summary.json...")
            fallback = os.path.join(args.eval_dir, "summary.json")
            if os.path.exists(fallback):
                summary_file = fallback
    else:
        candidate = os.path.join(args.eval_dir, "summary.json")
        if os.path.exists(candidate):
            summary_file = candidate

    if summary_file:
        with open(summary_file, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load summary file {summary_file}: {e}")
                data = {}

    # Primary source of per-seed metrics: trajectory_*.json files
    results = load_results_from_trajectories(args.eval_dir)
    if not results:
        # Fallback to legacy summary.json structure if no trajectories found
        results = data.get("results", [])
        if not results:
            print(
                "No trajectory_*.json files or per-seed results found. "
                "Make sure eval-dir contains trajectory_seed*.json files or a summary.json with 'results'."
            )
            return

    # Optionally ignore runs that failed early due to infrastructure issues
    # (e.g., Claude CLI error) rather than exhausting the full ReAct budget.
    # If max_iterations is known, we only count failures where iterations
    # reached the max; failures with fewer iterations are treated as "ignored".
    # max_iters_limit = data.get("max_iterations")
    # if max_iters_limit is not None:
    #     original_total = len(results)
    #     filtered_results = []
    #     ignored_seeds = []
    #     for r in results:
    #         if r["success"] or r.get("iterations", 0) >= max_iters_limit:
    #             filtered_results.append(r)
    #         else:
    #             ignored_seeds.append(r["seed"])
    #     if ignored_seeds:
    #         print(
    #             f"[Info] Ignoring {len(ignored_seeds)} run(s) that failed before "
    #             f"max_iterations={max_iters_limit}: seeds {sorted(ignored_seeds)}"
    #         )
    #     results = filtered_results

    seeds = [str(r["seed"]) for r in results]
    iterations = [r["iterations"] for r in results]
    successes = [r["success"] for r in results]
    # Some runs may not record elapsed_time (None). Treat missing/None as 0.0 for aggregation/plots.
    raw_times = [r.get("elapsed_time") for r in results]
    times = [t if isinstance(t, (int, float)) else 0.0 for t in raw_times]

    # Calculate overall stats
    num_total = len(results)
    num_success = sum(successes)
    num_fail = num_total - num_success
    success_rate = (num_success / num_total) * 100 if num_total > 0 else 0
    avg_iters = np.mean(iterations)
    avg_time = np.mean(times)
    
    # --- 1) Print Markdown Summary Table to Console ---
    print(f"\n# Evaluation Summary: {data.get('model', 'Unknown')}")
    print(f"**Level:** {data.get('level', 'down_to_earth')} | **Success Rate:** {success_rate:.1f}% ({num_success}/{num_total}) | **Avg Iterations:** {avg_iters:.1f} | **Avg Time:** {avg_time:.1f}s\n")
    
    print("| Seed | Outcome | Iterations | Time (s) | Final Action |")
    print("|------|---------|------------|----------|--------------|")
    for r, t in zip(results, times):
        status = "✅ Success" if r["success"] else "❌ Failure"
        action = str(r["action"]).ljust(20) if r["action"] else "None".ljust(20)
        time_str = f"{t:.1f}"
        print(f"| {str(r['seed']).ljust(4)} | {status.ljust(9)} | {str(r['iterations']).ljust(10)} | {time_str.ljust(8)} | {action} |")
    print("\n")

    # --- 2) Generate 2x2 Dashboard ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"ReAct Agent Evaluation Dashboard\nModel: {data.get('model', 'Unknown')} | Success Rate: {success_rate:.1f}%", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # A) Success Rate (Donut Chart)
    ax1 = plt.subplot(2, 2, 1)
    if num_success == 0:
        sizes, labels, colors = [num_fail], [f"Failure ({num_fail})"], ['#d62728']
    elif num_fail == 0:
        sizes, labels, colors = [num_success], [f"Success ({num_success})"], ['#2ca02c']
    else:
        sizes, labels, colors = [num_success, num_fail], [f"Success ({num_success})", f"Failure ({num_fail})"], ['#2ca02c', '#d62728']

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'), 
                                       textprops=dict(size=14, weight="bold"))
    ax1.set_title("Overall Success Rate", fontsize=14, fontweight='bold')

    # B) Iterations per Seed (Bar Chart)
    ax2 = plt.subplot(2, 2, 2)
    bar_colors = ['#2ca02c' if s else '#d62728' for s in successes]
    bars = ax2.bar(seeds, iterations, color=bar_colors, alpha=0.8)
    ax2.set_xlabel("Seed", fontsize=12)
    ax2.set_ylabel("Iterations", fontsize=12)
    ax2.set_title("Iterations taken per Seed", fontsize=14, fontweight='bold')
    ax2.axhline(y=avg_iters, color='b', linestyle='--', label=f'Mean ({avg_iters:.1f})')
    ax2.legend()
    # Annotate bars
    for bar in bars:
        ax2.annotate(f'{bar.get_height()}', 
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # C) Time Taken per Seed (Bar Chart)
    ax3 = plt.subplot(2, 2, 3)
    bars_time = ax3.bar(seeds, times, color='#1f77b4', alpha=0.8)
    ax3.set_xlabel("Seed", fontsize=12)
    ax3.set_ylabel("Time (seconds)", fontsize=12)
    ax3.set_title("Computation Time per Seed", fontsize=14, fontweight='bold')
    ax3.axhline(y=avg_time, color='orange', linestyle='--', label=f'Mean Time ({avg_time:.1f}s)')
    ax3.legend()
    for bar in bars_time:
        ax3.annotate(f'{bar.get_height():.1f}s', 
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # D) Iteration Distribution (Boxplot with Stripplot overlay)
    ax4 = plt.subplot(2, 2, 4)
    succ_iters = [it for it, s in zip(iterations, successes) if s]
    fail_iters = [it for it, s in zip(iterations, successes) if not s]
    
    data_to_plot = []
    labels_bp = []
    colors_bp = []
    
    if succ_iters:
        data_to_plot.append(succ_iters)
        labels_bp.append('Success')
        colors_bp.append('#2ca02c')
    if fail_iters:
        data_to_plot.append(fail_iters)
        labels_bp.append('Failure')
        colors_bp.append('#d62728')
        
    bp = ax4.boxplot(data_to_plot, patch_artist=True)
    ax4.set_xticklabels(labels_bp, fontsize=12)
    
    # Fill colors for boxes
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    # Overlay points for each run
    for i, data_series in enumerate(data_to_plot):
        y = data_series
        # Add random jitter on x-axis
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax4.plot(x, y, 'ko', alpha=0.5)
        
    ax4.set_ylabel("Iterations", fontsize=12)
    ax4.set_title("Iteration Distribution by Outcome", fontsize=14, fontweight='bold')
    if data.get('max_iterations'):
        ax4.axhline(y=data['max_iterations'], color='black', linestyle=':', label='Max Iterations Limit')
        ax4.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle
    
    plot_path = os.path.join(args.eval_dir, "evaluation_dashboard.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
