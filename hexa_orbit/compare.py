"""Paired arm comparison: McNemar on discordant pairs (DESIGN_PHASE3_MAZE §14).

The primary endpoint is **episode-1 success, arm B minus arm A, on `clean` eval
instances** -- one pre-registered number. Everything else this module computes is
secondary and reported without significance claims, because §13's nine arms times
four metrics times three episode indices times two leakage strata is more than
enough surface to find a significant-looking result by accident.

**Pairing is the thing to get wrong.** McNemar assumes paired observations, and
`hexa_core/WEAKNESSES.md:300` records this project already shipping a version
where "records stored seeds as though the arms were paired" when they were not.
Two arms are paired here only if they ran the *same instances* with the *same
per-instance sampling seed* -- at temperature 0.6 the same arm twice gives
different runs. :func:`pair_runs` therefore refuses to compare arms whose instance
sets differ, rather than silently intersecting them, and :func:`compare_arms`
surfaces the shared-seed question as something the caller must have arranged
(``scripts/run_orbit_maze_hexa.py:paired_seed``).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .actor import InstanceRun
from .trajectory import read_runs


class Unpaired(ValueError):
    """The two arms did not run the same instances, so the test does not apply."""


# ── the tests ────────────────────────────────────────────────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts.

    ``b`` = arm A failed and arm B succeeded; ``c`` = the reverse. Concordant
    pairs carry no information about the difference and are excluded by
    construction -- which is why power tracks ``b + c`` rather than *n*.

    Exact rather than the chi-square approximation: discordant counts here are
    expected in the tens, where the continuity-corrected chi-square is unreliable.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval. Falls back to a normal approximation.

    Reported alongside every rate because a rate without an interval invites
    reading a 0.10-vs-0.14 difference on 81 instances as a finding.
    """
    if n == 0:
        return (0.0, 0.0)
    try:
        from scipy.stats import beta  # noqa: PLC0415

        lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, n - successes + 1)
        upper = 1.0 if successes == n else beta.ppf(1 - alpha / 2, successes + 1, n - successes)
        return (float(lower), float(upper))
    except ImportError:  # pragma: no cover - scipy is present in both envs
        rate = successes / n
        half = 1.96 * (rate * (1 - rate) / n) ** 0.5
        return (max(0.0, rate - half), min(1.0, rate + half))


def wilcoxon_signed_rank(paired: Sequence[tuple[float, float]]) -> Optional[float]:
    """Two-sided p for paired differences, or ``None`` if scipy is unavailable.

    Secondary only. It uses more information than the episode-1 binary, but a gain
    concentrated in episodes 2-3 is exactly what ORBIT's own within-instance
    mechanism can produce, so it does not discriminate the claim under test.
    """
    differences = [after - before for before, after in paired if after != before]
    if len(differences) < 6:
        return None
    try:
        from scipy.stats import wilcoxon  # noqa: PLC0415

        return float(wilcoxon(differences).pvalue)
    except ImportError:  # pragma: no cover
        return None


# ── pairing ──────────────────────────────────────────────────────────────

@dataclass
class Pair:
    """One instance, both arms."""

    instance_id: str
    leaked: Optional[bool]
    a: InstanceRun
    b: InstanceRun

    def episode_success(self, index: int) -> tuple[bool, bool]:
        def at(run: InstanceRun) -> bool:
            return (
                run.episode_successes[index]
                if len(run.episode_successes) > index
                else False
            )

        return at(self.a), at(self.b)


def pair_runs(
    arm_a: Sequence[InstanceRun], arm_b: Sequence[InstanceRun], *, strict: bool = True
) -> list[Pair]:
    """Pair two arms by instance id. Raises :class:`Unpaired` on a mismatch.

    Errored runs are dropped from *both* arms when either side failed: a pair is
    only informative if both arms actually produced a trajectory, and keeping the
    surviving half would break the pairing McNemar assumes.

    ``strict=False`` intersects instead of raising -- for inspecting a partial run,
    never for a reported number.
    """
    by_id_a = {r.instance_id: r for r in arm_a if r.error is None}
    by_id_b = {r.instance_id: r for r in arm_b if r.error is None}
    only_a, only_b = set(by_id_a) - set(by_id_b), set(by_id_b) - set(by_id_a)

    if strict and (only_a or only_b):
        raise Unpaired(
            f"arms ran different instances: {len(only_a)} only in A, "
            f"{len(only_b)} only in B. McNemar assumes pairing -- rerun both arms "
            f"on the same --eval-size, or pass strict=False to inspect."
        )

    shared = [i for i in by_id_a if i in by_id_b]
    return [
        Pair(
            instance_id=i,
            leaked=by_id_a[i].leaked if by_id_a[i].leaked is not None else by_id_b[i].leaked,
            a=by_id_a[i],
            b=by_id_b[i],
        )
        for i in shared
    ]


# ── the comparison ───────────────────────────────────────────────────────

def _episode_test(pairs: Sequence[Pair], index: int) -> dict[str, Any]:
    n = len(pairs)
    a_wins = b_wins = a_count = b_count = 0
    for pair in pairs:
        succeeded_a, succeeded_b = pair.episode_success(index)
        a_count += succeeded_a
        b_count += succeeded_b
        if succeeded_b and not succeeded_a:
            b_wins += 1
        elif succeeded_a and not succeeded_b:
            a_wins += 1
    return {
        "n": n,
        "rate_a": a_count / n if n else 0.0,
        "rate_b": b_count / n if n else 0.0,
        "ci_a": clopper_pearson(a_count, n),
        "ci_b": clopper_pearson(b_count, n),
        "difference": (b_count - a_count) / n if n else 0.0,
        "b_only_wins": b_wins,          # McNemar's b: A failed, B succeeded
        "a_only_wins": a_wins,          # McNemar's c
        "discordant": a_wins + b_wins,
        "p_exact": mcnemar_exact(b_wins, a_wins),
    }


def _leakage_gaps(
    clean: Sequence[Pair], leaked: Sequence[Pair]
) -> dict[str, Optional[float]]:
    """Leaked-minus-clean episode-1 rates, or ``None`` if a stratum is empty."""
    if not clean or not leaked:
        return {"leakage_gap_a": None, "leakage_gap_b": None,
                "note": "one stratum is empty; the gap is undefined"}
    clean_test, leaked_test = _episode_test(clean, 0), _episode_test(leaked, 0)
    return {
        "leakage_gap_a": leaked_test["rate_a"] - clean_test["rate_a"],
        "leakage_gap_b": leaked_test["rate_b"] - clean_test["rate_b"],
    }


def compare_arms(
    arm_a: Sequence[InstanceRun],
    arm_b: Sequence[InstanceRun],
    *,
    name_a: str = "no-bank",
    name_b: str = "frozen-bank",
    strict: bool = True,
) -> dict[str, Any]:
    """Full paired comparison, primary endpoint first."""
    pairs = pair_runs(arm_a, arm_b, strict=strict)
    clean = [p for p in pairs if p.leaked is False]
    leaked = [p for p in pairs if p.leaked]

    def diagnostics(runs: Sequence[InstanceRun]) -> dict[str, Any]:
        turns = sum(len(r.steps) for r in runs)
        return {
            "unparsed_rate": sum(r.n_unparsed for r in runs) / turns if turns else 0.0,
            "truncated_rate": sum(r.n_truncated for r in runs) / turns if turns else 0.0,
            "turns": turns,
        }

    return {
        "arms": {"a": name_a, "b": name_b},
        "n_paired": len(pairs),
        "n_clean": len(clean),
        "n_leaked": len(leaked),
        # THE pre-registered number.
        "primary": {
            "endpoint": "episode-1 success, clean instances, paired",
            **_episode_test(clean, 0),
        },
        "secondary": {
            "episode_1_pooled": _episode_test(pairs, 0),
            "episode_2_clean": _episode_test(clean, 1),
            "episode_3_clean": _episode_test(clean, 2),
            "episode_1_leaked": _episode_test(leaked, 0),
            "success_count_wilcoxon_p": wilcoxon_signed_rank(
                [(p.a.success_count, p.b.success_count) for p in clean]
            ),
            "mean_success_count_a": (
                sum(p.a.success_count for p in clean) / len(clean) if clean else 0.0
            ),
            "mean_success_count_b": (
                sum(p.b.success_count for p in clean) / len(clean) if clean else 0.0
            ),
        },
        # Descriptive only -- see §14. Strata differ in n and difficulty.
        # `None` when either stratum is empty: an empty stratum has rate 0.0, so
        # subtracting would report a confident-looking -100% gap that is really
        # "there were no leaked instances". That reads as alarming in a report.
        "descriptive": _leakage_gaps(clean, leaked),
        "diagnostics": {
            name_a: diagnostics([p.a for p in pairs]),
            name_b: diagnostics([p.b for p in pairs]),
        },
        "added_tokens": {
            name_a: sum(p.a.added_tokens for p in pairs) / len(pairs) if pairs else 0,
            name_b: sum(p.b.added_tokens for p in pairs) / len(pairs) if pairs else 0,
        },
    }


def load_arm(path: str | Path) -> list[InstanceRun]:
    """Read an arm's runs from a run directory or a ``runs.jsonl`` directly."""
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("**/runs.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"no runs.jsonl under {path}")
        runs: list[InstanceRun] = []
        for candidate in candidates:
            runs.extend(read_runs(candidate))
        return runs
    return read_runs(path)


def format_comparison(result: Mapping[str, Any]) -> str:
    """Terminal report. Diagnostics first, primary endpoint second, rest after."""
    name_a, name_b = result["arms"]["a"], result["arms"]["b"]
    lines = [f"=== {name_b}  vs  {name_a} ===", ""]

    diag = result["diagnostics"]
    lines.append(
        f"{'diagnostics':<16}{name_a:>16}{name_b:>16}\n"
        f"{'unparsed':<16}{diag[name_a]['unparsed_rate']:>15.1%}"
        f"{diag[name_b]['unparsed_rate']:>16.1%}\n"
        f"{'truncated':<16}{diag[name_a]['truncated_rate']:>15.1%}"
        f"{diag[name_b]['truncated_rate']:>16.1%}"
    )
    worst = max(diag[name_a]["unparsed_rate"], diag[name_b]["unparsed_rate"])
    if worst > 0.05:
        lines.append(
            "  WARNING: >5% of turns produced no parseable action. The arms may "
            "differ in how often the model emitted an action at all, which makes "
            "the success-rate difference below uninterpretable."
        )

    primary = result["primary"]
    lines += [
        "",
        f"PRIMARY ENDPOINT -- {primary['endpoint']}  (n={primary['n']})",
        f"  {name_a:<14} {primary['rate_a']:.1%}  "
        f"[{primary['ci_a'][0]:.1%}, {primary['ci_a'][1]:.1%}]",
        f"  {name_b:<14} {primary['rate_b']:.1%}  "
        f"[{primary['ci_b'][0]:.1%}, {primary['ci_b'][1]:.1%}]",
        f"  difference     {primary['difference']:+.1%}",
        f"  discordant     b={primary['b_only_wins']} (only {name_b} solved), "
        f"c={primary['a_only_wins']} (only {name_a} solved)",
        f"  McNemar exact  p = {primary['p_exact']:.4g}",
    ]
    if primary["discordant"] < 10:
        lines.append(
            f"  NOTE: only {primary['discordant']} discordant pairs. Significance "
            f"needs roughly b >= 10; this is underpowered either way, and a null "
            f"here is not evidence of no effect."
        )

    secondary = result["secondary"]
    lines += ["", "secondary (no significance claimed)"]
    for key in ("episode_1_pooled", "episode_2_clean", "episode_3_clean", "episode_1_leaked"):
        test = secondary[key]
        if test["n"]:
            lines.append(
                f"  {key:<20} n={test['n']:<4} {test['rate_a']:>6.1%} -> "
                f"{test['rate_b']:>6.1%}  ({test['difference']:+.1%}, "
                f"b={test['b_only_wins']} c={test['a_only_wins']}, p={test['p_exact']:.3g})"
            )
    lines.append(
        f"  {'mean R (clean)':<20} {secondary['mean_success_count_a']:.2f} -> "
        f"{secondary['mean_success_count_b']:.2f}"
        + (
            f"  (Wilcoxon p={secondary['success_count_wilcoxon_p']:.3g})"
            if secondary["success_count_wilcoxon_p"] is not None
            else "  (too few discordant pairs for Wilcoxon)"
        )
    )
    descriptive = result["descriptive"]
    lines += ["", "descriptive only -- not tested (strata differ in n and difficulty)"]
    if descriptive.get("leakage_gap_a") is None:
        lines.append(
            f"  leaked-minus-clean episode 1: n/a ({descriptive.get('note', 'undefined')})"
        )
    else:
        lines.append(
            f"  leaked-minus-clean episode 1: {name_a} {descriptive['leakage_gap_a']:+.1%}, "
            f"{name_b} {descriptive['leakage_gap_b']:+.1%}"
        )
    lines.append(
        f"  added tokens/instance: {name_a} {result['added_tokens'][name_a]:.0f}, "
        f"{name_b} {result['added_tokens'][name_b]:.0f}"
    )
    return "\n".join(lines)
