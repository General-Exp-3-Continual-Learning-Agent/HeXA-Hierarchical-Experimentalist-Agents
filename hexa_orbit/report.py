"""Aggregation for Maze runs, in ORBIT's Figure 3 shape plus what D-plan adds.

Every headline metric is reported three ways: pooled, and split by whether the
eval instance's layout also occurs in the training pool. That split is not
optional and not a post-hoc slice -- it is the control that keeps a
cross-instance-transfer claim honest when 19.1% of eval instances share a layout
with the train pool (DECISIONS.md, the section after D1; RECON §10). It is
computed here rather than in the plotting code so the baseline and the HExA arm
cannot drift in how they define it.

Two reporting rules worth stating because they are easy to get wrong:

* **Errored instances are excluded from rates and counted separately.** Folding a
  transport failure in as a failure would depress a success rate for a reason
  that has nothing to do with the agent.
* **Diagnostics come before rates.** ``unparsed_rate`` and ``truncated_rate``
  gate whether a success rate means anything at all: a run where a third of turns
  produced no parseable action is measuring the response cap, not the policy.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Sequence

from .adapter import EPISODES_PER_INSTANCE
from .actor import InstanceRun


def _rate(values: Sequence[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else 0.0


def episode_success_rates(runs: Sequence[InstanceRun]) -> dict[str, float]:
    """Success rate at episode 1..T -- ORBIT's Figure 3 / Table 2 shape.

    Under the hard T=3 cap every completed instance has exactly three episodes,
    so no ``-1`` sentinel filtering is needed (contrast ORBIT's ``get_metrics``,
    RECON §5). A short list here means an instance errored mid-run, and those are
    excluded upstream.
    """
    out: dict[str, float] = {}
    for index in range(EPISODES_PER_INSTANCE):
        present = [r.episode_successes[index] for r in runs if len(r.episode_successes) > index]
        out[f"episode_{index + 1}"] = _rate(present)
    return out


def success_count_distribution(runs: Sequence[InstanceRun]) -> dict[int, int]:
    """Histogram over R(M;tau) in {0,1,2,3}, ORBIT's own instance reward."""
    counts = {k: 0 for k in range(EPISODES_PER_INSTANCE + 1)}
    for run in runs:
        counts[run.success_count] = counts.get(run.success_count, 0) + 1
    return counts


def arm_summary(runs: Sequence[InstanceRun]) -> dict[str, Any]:
    """Metrics for one set of runs (an arm, or one side of the leakage split)."""
    if not runs:
        return {"n": 0}

    total_steps = sum(len(r.steps) for r in runs)
    solved_excess = [
        length - r.shortest_path_len
        for r in runs
        for length, success in zip(r.episode_lengths, r.episode_successes)
        if success
    ]
    return {
        "n": len(runs),
        # The headline: ORBIT's episode-indexed curve. Episode 1 is the claim.
        "episode_success_rate": episode_success_rates(runs),
        "any_success_rate": _rate([r.success_count > 0 for r in runs]),
        "mean_success_count": mean(r.success_count for r in runs),
        "success_count_distribution": success_count_distribution(runs),
        # The failure mode ORBIT's single-episode RL baseline falls into.
        "no_adaptation_rate": _rate(
            [not r.adapted_across_episodes for r in runs if r.success_count == 0]
        ),
        "n_never_solved": sum(1 for r in runs if r.success_count == 0),
        # Efficiency, against the known-map optimum (see oracle caveats).
        "mean_excess_steps_on_success": mean(solved_excess) if solved_excess else None,
        "mean_steps_per_instance": total_steps / len(runs),
        # Cost.
        "mean_added_tokens": mean(r.added_tokens for r in runs),
        # Diagnostics -- read these before the rates above.
        "unparsed_rate": sum(r.n_unparsed for r in runs) / total_steps if total_steps else 0.0,
        "truncated_rate": sum(r.n_truncated for r in runs) / total_steps if total_steps else 0.0,
    }


def aggregate(
    runs: Iterable[InstanceRun], *, errored: Sequence[InstanceRun] = ()
) -> dict[str, Any]:
    """Pooled metrics plus the mandatory leaked/clean split."""
    completed = [r for r in runs if r.error is None]
    leaked = [r for r in completed if r.leaked]
    clean = [r for r in completed if r.leaked is False]
    unknown = [r for r in completed if r.leaked is None]

    report: dict[str, Any] = {
        "pooled": arm_summary(completed),
        "clean": arm_summary(clean),
        "leaked": arm_summary(leaked),
        "n_errored": len(list(errored)),
        "errors": [
            {"instance_id": r.instance_id, "error": r.error} for r in errored
        ][:20],
        "token_counter": completed[0].token_counter if completed else None,
    }
    if unknown:
        report["n_leakage_unknown"] = len(unknown)
    if leaked and clean:
        # The number that says whether leakage is inflating the headline.
        report["episode_1_leakage_gap"] = (
            report["leaked"]["episode_success_rate"]["episode_1"]
            - report["clean"]["episode_success_rate"]["episode_1"]
        )
    return report


def format_summary(report: dict[str, Any], *, title: str = "Maze") -> str:
    """Terminal-readable digest. Diagnostics first, deliberately."""
    lines = [f"=== {title} ===", ""]
    pooled = report["pooled"]
    if not pooled.get("n"):
        return "\n".join(lines + ["no completed instances"])

    lines.append(
        f"unparsed turns {pooled['unparsed_rate']:.1%} | "
        f"truncated turns {pooled['truncated_rate']:.1%} | "
        f"errored instances {report['n_errored']}"
    )
    if pooled["unparsed_rate"] > 0.05:
        lines.append(
            "  WARNING: >5% of turns produced no parseable action -- the response "
            "cap or prompt is suspect; success rates below are not yet meaningful."
        )
    lines.append("")
    header = f"{'arm':<8}{'n':>5}{'ep1':>9}{'ep2':>9}{'ep3':>9}{'any':>9}{'mean R':>9}"
    lines.append(header)
    for arm in ("pooled", "clean", "leaked"):
        summary = report[arm]
        if not summary.get("n"):
            continue
        rates = summary["episode_success_rate"]
        lines.append(
            f"{arm:<8}{summary['n']:>5}"
            f"{rates['episode_1']:>9.1%}{rates['episode_2']:>9.1%}"
            f"{rates['episode_3']:>9.1%}{summary['any_success_rate']:>9.1%}"
            f"{summary['mean_success_count']:>9.2f}"
        )
    if "episode_1_leakage_gap" in report:
        lines += [
            "",
            f"episode-1 leaked-minus-clean gap: {report['episode_1_leakage_gap']:+.1%}"
            "  (large positive = headline inflated by memorised layouts)",
        ]
    lines += [
        "",
        f"R distribution (0/1/2/3): {pooled['success_count_distribution']}",
        f"never solved: {pooled['n_never_solved']}"
        f"  of which no adaptation across own episodes: {pooled['no_adaptation_rate']:.1%}",
        f"mean excess steps on success: {pooled['mean_excess_steps_on_success']}",
        f"mean added tokens/instance: {pooled['mean_added_tokens']:.0f}"
        f"  ({report['token_counter']})",
    ]
    return "\n".join(lines)
