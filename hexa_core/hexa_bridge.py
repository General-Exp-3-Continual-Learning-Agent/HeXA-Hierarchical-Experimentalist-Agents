"""Swap HeXA's InterPhyre-coupled parts for CORE-compatible ones, by rebinding.

HeXA's method — contrastive batch distillation, hierarchical bank, retrieve,
inject, evolve — is task-general; only a handful of module-level names in
``skillrl/`` assume a Box2D world with an ``[x, y, r]`` action and an
iteration-count reward. This module rebinds exactly those names, using the idiom
the repo already established in
``skillrl/distillation/teacher_prompts_catapult.py::patch`` and
``teacher_prompts_contrastive_only.py::patch``.

Why rebinding is safe: every consumer resolves these names from its own module
globals *at call time*. ``distill.distill_contrastive`` looks up
``CONTRASTIVE_DISTILLATION_PROMPT`` in ``distill``'s globals when it runs;
``teacher_prompts.format_trajectories_block`` looks up both
``compute_trajectory_reward`` (its sort key) and ``format_trajectory_for_teacher``
in ``teacher_prompts``' globals when it runs, even though ``distill`` and
``evolving_distill`` imported ``format_trajectories_block`` itself by value. So
assigning to the module attribute is enough, and no file under ``skillrl/`` is
edited.

:func:`patch` rebinds:

===========================================================  ============================================
``distill.CONTRASTIVE_DISTILLATION_PROMPT``                  neutral variant (``distill_contrastive``)
``distill.COMMON_MISTAKES_PROMPT``                           neutral variant (``distill_mistakes``)
``evolving_distill.SKILL_BANK_EVOLUTION_PROMPT``             neutral variant (``evolve_skill_bank``)
``teacher_prompts.compute_trajectory_reward``                binary reward (M2)
``distill.compute_trajectory_reward``                        binary reward (M2)
``teacher_prompts.format_trajectory_for_teacher``            CORE formatter (final answer, not ``x=,y=,r=``)
``config.LEVEL_DESCRIPTIONS``                                mutated in place with the CORE task descriptions
``distill.load_trajectories`` / ``evolving_distill.…``       only if ``load_trajectories_fn`` is passed
===========================================================  ============================================

The reward is rebound in **two** modules because it is read in three places:
``teacher_prompts.format_trajectory_for_teacher`` and
``format_trajectories_block``'s sort key resolve it in ``teacher_prompts``, while
``distill._compute_initial_confidence`` and ``distill_contrastive``'s logging
resolve it in ``distill``.

``LEVEL_DESCRIPTIONS`` is *mutated*, not rebound: it is a plain dict imported by
reference into six modules, so one ``dict`` update reaches all of them where six
rebinds would be needed. :func:`patch` refuses to overwrite a pre-existing key,
and :func:`unpatch` pops only the keys it added.

Two coupled things are deliberately **not** patched:

* ``retriever.RELATED_LEVELS`` — ``retrieve_skills`` does
  ``RELATED_LEVELS.get(level_name, [])`` and no CORE task name is a key, so
  cross-task transfer is already inert. Mutating a shared global to express
  "nothing" would be worse than leaving it alone.
* ``SkillBank.format_skills_as_prompt`` — a *method*, which the patch idiom
  cannot reach. Its hardcoded "PHYSICS" labels are relabelled after rendering by
  :func:`retrieve_within_token_budget`, the same fix
  ``discovery_runner.py:97`` already uses.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from hexa_core.prompts_core_tasks import (
    CORE_TASK_DESCRIPTIONS,
    COMMON_MISTAKES_PROMPT_CORE,
    CONTRASTIVE_DISTILLATION_PROMPT_CORE,
    SKILL_BANK_EVOLUTION_PROMPT_CORE,
    evolution_prompt,
)
from hexa_core.tokens import count_tokens, counter_name
from skillrl.core.retriever import retrieve_skills
from skillrl.core.skill_bank import Mistake, Skill, SkillBank

logger = logging.getLogger(__name__)

#: Rendered instead of ``=== LEARNED PHYSICS SKILLS ===``.
SKILLS_BLOCK_HEADING = "=== LEARNED PROBLEM-SOLVING STRATEGIES ==="

_OBSERVATION_MAX_CHARS = 1200

#: The reasoning trace is what the evolver is supposed to learn from, so this cap is
#: much wider than the observation one -- but it has to exist. A Hanoi ``Thought``
#: renders to ~6.6k chars, so 12 uncapped trajectories would be ~80k chars of prompt
#: against an input budget of roughly 32k tokens (``max_model_len`` 65536 minus the
#: teacher's 32768 output cap). At 2400 a CORE attempt costs ~3.6k chars including its
#: observation, so 12 fit in ~43k chars / ~11k tokens with room for the bank.
_THOUGHT_MAX_CHARS = 2400


# ── M2: binary reward ──────────────────────────────────────────────────

def binary_trajectory_reward(trajectory: dict) -> float:
    """+1.0 if the verifier accepted the answer, -1.0 if it rejected it.

    All three CORE tasks in scope are binary-scored
    (``uses_verifier_score_for_reward()`` is False, so CORE's own
    ``_reward_from_metrics`` returns ``float(bool(is_correct))``), and a CORE
    attempt has no iteration count for HeXA's shaped reward to read. Reading only
    ``success`` keeps this correct for any trajectory dict, not just ours.

    The +/-1 range is deliberate: ``distill._compute_initial_confidence`` maps
    mean reward from [-1, 1] onto [0.1, 1.0], so an all-correct source set gives
    a new skill confidence 1.0 and an all-incorrect one gives 0.1.
    """
    return 1.0 if trajectory.get("success") else -1.0


# ── M1: baseline-relative confidence ───────────────────────────────────

def attempt_utility(trajectory: dict) -> float:
    """One attempt's baseline-relative utility, ``float(correct) - b_q``.

    Recomputed from ``success`` and ``baseline`` rather than read from the dict's
    ``utility`` key, so this cannot drift from :attr:`hexa_core.trajectory.Attempt.
    utility`; a test pins the two equal.

    Deliberately **not** ``reward - b_q``. ``reward`` is +/-1 under M2 and ``b_q`` is
    a rate in [0, 1], so that difference spans [-2, 1] and everything past -1 clamps
    against the [-1, 1] -> [0.1, 1.0] map -- pushing every failure on a mid-difficulty
    problem to the confidence floor. ``float(correct)`` is CORE's own convention
    (``_reward_from_metrics`` returns ``float(bool(is_correct))``) and lands in
    [-1, 1] exactly.

    Falls back to the binary reward when ``baseline`` is absent, which is precisely
    HeXA's pre-M1 behaviour rather than a silent ``b_q = 0``.
    """
    baseline = trajectory.get("baseline")
    if baseline is None:
        return binary_trajectory_reward(trajectory)
    return float(bool(trajectory.get("success"))) - float(baseline)


def compute_initial_confidence_core(
    source_seeds: list,
    all_trajs: list[dict],
    fallback_confidence: float = 0.5,
) -> float:
    """M1 confidence: pool **every** attempt on the source problems, score by utility.

    Replaces ``distill._compute_initial_confidence``, which is correct for InterPhyre
    and wrong here for two independent reasons:

    1. **It reads the binary outcome, so base rates masquerade as skill quality.**
       A Matchstick skill (b_q ~ 0.68) outscores a Hanoi skill (b_q ~ 0.18) for
       reasons that have nothing to do with either skill.
    2. **It keeps one trajectory per seed.** Under InterPhyre a seed is attempted
       once, so seed -> trajectory is 1:1. Under CORE a problem is attempted `k`
       times -- 180 at n=10, 380 at n=5 -- and ``seed_to_traj`` keeps the *first*
       occurrence. Since ``all_trajs = successes + failures``, that first occurrence
       is a success whenever the problem was ever solved, so confidence collapses to
       {1.0, 0.1} and discards k-1 attempts of evidence.

    Together those made confidence answer "was this problem ever solved?" rather than
    "how much did this skill help?" -- and confidence is the retrieval ranking key,
    the pruning signal, and an input to admission. Measured before this fix: a skill
    from a problem solved 3/10 and one from a problem solved 1/10 both scored 1.0.

    Pooling all attempts makes the mean utility exactly ``observed_rate - b_q``, so a
    skill that merely reproduces the baseline scores 0.5 (neutral), one that beats it
    scores above, and one that underperforms scores below.
    """
    if not source_seeds or not all_trajs:
        return fallback_confidence

    by_seed: dict[Any, list[dict]] = {}
    for trajectory in all_trajs:
        by_seed.setdefault(trajectory.get("seed", -1), []).append(trajectory)

    scores = [
        attempt_utility(trajectory)
        for seed in source_seeds
        for trajectory in by_seed.get(seed, ())
    ]
    if not scores:
        # No named source matched -- fall back to the level-wide average, as the
        # original does, rather than to a constant.
        scores = [attempt_utility(t) for t in all_trajs]

    average = sum(scores) / len(scores)
    return max(0.1, min(1.0, (average + 1.0) / 2.0))


# ── Balanced contrast for the evolver ──────────────────────────────────

#: How the `max_trajs` slots shown to the evolver are chosen.
#:
#: ``"top5"``    HeXA's own selection: sort every trajectory by reward, take the top N.
#: ``"balanced"`` a global quota -- N_success highest-reward successes plus N_fail
#:               lowest-reward failures.
#: ``"paired"``  same-problem pairs first (see :func:`_select_paired`), then backfill.
TRAJECTORY_SELECTIONS = ("top5", "balanced", "paired")

#: Default split for ``"balanced"`` and for `"paired"`'s backfill, out of 5 slots.
DEFAULT_SUCCESS_SLOTS = 3

#: The success share the default slot count encodes (3 of 5), used to scale it when
#: `max_trajs` changes. Without this the absolute 3 would mean 3 successes and **9
#: failures** at ``max_trajs=12`` -- inverting the block rather than widening it.
#: Successes stay the majority on purpose: the evolver is asked what separates solving
#: from failing, which needs enough of the solving side to generalise over.
DEFAULT_SUCCESS_FRACTION = DEFAULT_SUCCESS_SLOTS / 5


def default_success_slots(max_trajs: int) -> int:
    """How many of ``max_trajs`` slots go to successes, at least one of each class."""
    slots = round(max_trajs * DEFAULT_SUCCESS_FRACTION)
    return max(1, min(max_trajs - 1, slots)) if max_trajs > 1 else max_trajs

#: ``"paired"``, not HeXA's ``"top5"``, because ``"top5"`` provably sends the evolver no
#: failures on these tasks. Successes always outrank failures on reward -- under M2
#: (+1 vs -1), under M1 (``1-b_q >= 0 >= -b_q``), and in InterPhyre too -- so any round
#: with at least `max_trajs` successes fills every slot with a success. Measured: all 10
#: rounds of all 3 Tower of Hanoi runs, and all rounds of all 3 Matchstick runs (base
#: rate 0.74). Every *mistake* record in those banks was written without the evolver
#: observing one failure, which makes the method's stated contrast mechanism inoperative.
#: ``"top5"`` remains available as the ablation arm.
DEFAULT_TRAJECTORY_SELECTION = "paired"

#: 12, against HeXA's 5. At 600 training calls the evolver had ever seen 50 of 600
#: trajectories -- 8% of the evidence the run paid for. This is bounded by the teacher's
#: *input* context, not by anything methodological, and 12 fits: see
#: :data:`_THOUGHT_MAX_CHARS`, which is what makes it fit. Raising this without the
#: thought cap overflows the context and the evolution call fails to parse -- the exact
#: silent failure that froze the bank in an earlier campaign.
DEFAULT_MAX_TRAJS = 12

#: Warn above this. 12 trajectories measure ~9.3k tokens on Hanoi and ~8.2k on
#: Matchstick against real round data, so 12k is "larger than anything measured" rather
#: than a hard limit. Capacity is not the constraint -- 12.5k of input against a 65536
#: context leaves ~20k spare even with a full 32768-token teacher response -- but the
#: teacher's reasoning and its bank JSON share that output budget, and a longer block
#: means longer reasoning. That is how the bank froze at max_tokens=8192.
_BLOCK_TOKENS_WARN = 12000


def _split(trajectories: list[dict]) -> tuple[list[dict], list[dict]]:
    successes = [t for t in trajectories if t.get("success")]
    return successes, [t for t in trajectories if not t.get("success")]


def _select_top(trajectories: list[dict], max_trajs: int, reward) -> list[dict]:
    """HeXA's own selection, unchanged. The baseline arm."""
    return sorted(trajectories, key=reward, reverse=True)[:max_trajs]


def _select_balanced(
    trajectories: list[dict], max_trajs: int, reward, success_slots: int
) -> list[dict]:
    """A global quota: best successes plus the most damning failures.

    "Most damning" is deliberate. Under M1 a failure scores ``-b_q``, so sorting
    failures *ascending* surfaces failures on problems the model usually solves --
    something went wrong that normally does not. Sorting them descending would surface
    failures on problems it fails routinely, which is the least informative end.
    """
    successes, failures = _split(trajectories)
    fail_slots = max_trajs - success_slots
    chosen = (
        sorted(successes, key=reward, reverse=True)[:success_slots]
        + sorted(failures, key=reward)[:fail_slots]
    )
    # Backfill from whichever class had too few -- a round with no failures still
    # sends `max_trajs` successes rather than a short block.
    if len(chosen) < max_trajs:
        used = {id(t) for t in chosen}
        rest = [t for t in trajectories if id(t) not in used]
        chosen += sorted(rest, key=reward, reverse=True)[: max_trajs - len(chosen)]
    return chosen


def _select_paired(
    trajectories: list[dict], max_trajs: int, reward, success_slots: int
) -> list[dict]:
    """Same-problem success/failure pairs first, kept adjacent, then backfill.

    `PHASE3_DESIGN.md` §1a: "Same-problem contrast, where available, is the cleaner of
    the two: it holds problem difficulty fixed, so the distilled difference is about
    *method* rather than about which problem happened to be easier."

    A global quota can hand the evolver three successes on problem A and two failures
    on problem B, and the "difference" it extracts may be nothing more than A being
    easier than B. A success and a failure on the *same* problem cannot be explained
    that way.

    The pairs exist in practice: pilot diagnostic 3 reported "2/2 problems had both
    outcomes" in every round of every run, on both tasks. Only the selector never
    asked for them.

    Pairs are emitted adjacently and ordered by the reward of their success, so the
    most informative contrast leads. Within a pair: the best success, and the
    *lowest*-reward failure (see :func:`_select_balanced` for why).
    """
    by_problem: dict = {}
    for trajectory in trajectories:
        by_problem.setdefault(trajectory.get("seed"), []).append(trajectory)

    pairs = []
    for group in by_problem.values():
        successes, failures = _split(group)
        if successes and failures:
            best = max(successes, key=reward)
            worst = min(failures, key=reward)
            pairs.append((reward(best), best, worst))
    pairs.sort(key=lambda item: item[0], reverse=True)

    chosen: list[dict] = []
    for _, success, failure in pairs:
        if len(chosen) + 2 > max_trajs:
            break
        chosen.extend((success, failure))

    # Backfill the remainder with the balanced rule over what is left, so a round with
    # no mixed-outcome problem degrades to `balanced` rather than to nothing.
    #
    # The backfill's success quota is what is LEFT of `success_slots` after the pairs,
    # not the whole quota again. Passing the whole quota overshoots by however many
    # successes the pairs already placed: at max_trajs=12 with 2 problems per round the
    # pairs fill 4 slots (2S/2F), so asking for 7 more successes out of 8 remaining sent
    # 9S/3F instead of 7S/5F. It happened to be exact at max_trajs=5 -- 2 pairs then one
    # backfill slot capped to 1 success -- which is why raising the width exposed it.
    if len(chosen) < max_trajs:
        used = {id(t) for t in chosen}
        rest = [t for t in trajectories if id(t) not in used]
        remaining = max_trajs - len(chosen)
        already_success = sum(1 for t in chosen if t.get("success"))
        backfill_slots = max(0, min(success_slots - already_success, remaining))
        chosen += _select_balanced(rest, remaining, reward, backfill_slots)
    return chosen


def make_trajectory_formatter(
    selection: str = DEFAULT_TRAJECTORY_SELECTION,
    success_slots: int | None = None,
    max_trajs: int = DEFAULT_MAX_TRAJS,
):
    """Build the ``format_trajectories_block`` replacement for one selection mode.

    Whichever mode is chosen, the returned function **reports what it sent**. That
    matters because `evolve_skill_bank`'s own log line prints the trajectories
    *available* ("Evolving skill bank from 118 successes, 82 failures") while it sends
    five -- and successes always outrank failures, in InterPhyre (successes 1.0..0.25,
    failures -0.5/-0.75) as much as here. Under ``"top5"`` every round of every Hanoi
    run sent the evolver **zero failures**, and nothing in the logs said so.

    ``max_trajs`` **overrides the value the caller passes**, deliberately. See
    :data:`DEFAULT_MAX_TRAJS`. ``success_slots`` defaults to a fixed *share* of
    ``max_trajs`` rather than a fixed count, so widening the block widens both classes.
    """
    if selection not in TRAJECTORY_SELECTIONS:
        raise ValueError(
            f"unknown trajectory selection {selection!r}; "
            f"expected one of {', '.join(TRAJECTORY_SELECTIONS)}"
        )
    if max_trajs < 1:
        raise ValueError(f"max_trajs must be >= 1, got {max_trajs}")
    configured_max_trajs = max_trajs
    # Scaled from `max_trajs`, not defaulted to a constant: see
    # :data:`DEFAULT_SUCCESS_FRACTION`.
    if success_slots is None:
        success_slots = default_success_slots(max_trajs)
    elif not 1 <= success_slots <= max_trajs:
        raise ValueError(
            f"success_slots must be in 1..max_trajs ({max_trajs}), got {success_slots}"
        )

    def format_trajectories_block(
        trajectories: list[dict], max_trajs: int = 5
    ) -> str:
        import skillrl.distillation.teacher_prompts as _tp

        # The call-site argument is ignored on purpose: `evolving_distill.py:69`
        # hardcodes `max_trajs=5`, and that file is shared `skillrl` code this project
        # does not modify. Overriding here is the only seam. Recorded in the per-round
        # log line below so a run's actual width is never inferred from the default.
        max_trajs = configured_max_trajs

        reward = _tp.compute_trajectory_reward
        if selection == "top5":
            chosen = _select_top(trajectories, max_trajs, reward)
        elif selection == "balanced":
            chosen = _select_balanced(trajectories, max_trajs, reward, success_slots)
        else:
            chosen = _select_paired(trajectories, max_trajs, reward, success_slots)

        n_success = sum(1 for t in chosen if t.get("success"))
        available_s, available_f = (len(x) for x in _split(trajectories))
        paired_problems = len(
            {t.get("seed") for t in chosen if t.get("success")}
            & {t.get("seed") for t in chosen if not t.get("success")}
        )

        # Print the rewards actually sent. This is the only artifact that
        # distinguishes a graded M1 reward from binary after the fact: a run whose
        # rewards are all exactly +/-1.00 saw the binary reward, whichever flags its
        # header claimed. Three completed campaigns cannot be attributed because this
        # line did not exist.
        rewards = [reward(t) for t in chosen]
        graded = any(abs(abs(r) - 1.0) > 1e-9 for r in rewards)
        print(
            f"  Evolver RECEIVED {len(chosen)}/{max_trajs} trajectories "
            f"[{selection}]: {n_success} success / {len(chosen) - n_success} failure "
            f"(of {available_s} / {available_f} available; "
            f"{paired_problems} same-problem pair(s))"
            + ("   <-- no failures sent; mistakes will be unobserved"
               if n_success == len(chosen) else "")
        )
        print(
            f"    rewards sent: {[round(r, 2) for r in rewards]}  "
            + ("(graded / baseline-relative)" if graded
               else "(BINARY -- b_q is not reaching the evolver)")
        )
        block = "\n\n".join(_tp.format_trajectory_for_teacher(t) for t in chosen)
        # Make the size visible rather than discovering it as an unparseable evolution
        # response. The evolver reads this block *plus* the whole current bank (~2.2k
        # tokens at max_skills=10) and must emit a new bank as JSON inside
        # TEACHER_MAX_TOKENS, sharing that budget with its own high-effort reasoning.
        # Measured at max_trajs=12: ~9.3k tokens on Hanoi, ~8.2k on Matchstick.
        approx_tokens = len(block) // 4
        print(f"    block size: {len(block)} chars (~{approx_tokens} tokens)")
        if approx_tokens > _BLOCK_TOKENS_WARN:
            print(
                f"    [WARNING] the trajectory block is ~{approx_tokens} tokens, over "
                f"the ~{_BLOCK_TOKENS_WARN} this was sized for. The teacher must still "
                f"fit the bank and emit a new one; watch for '[Teacher TRUNCATED]' and "
                f"a round that fails to parse. Lower --max-trajs if either appears."
            )
        return block

    return format_trajectories_block


# ── Neutral trajectory formatter ───────────────────────────────────────

def _elide_middle(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` chars by removing the **middle**, not the tail.

    Head-truncation is wrong for a reasoning trace: the two ends carry the signal the
    evolver needs. The head is where the solver sets up its representation -- the thing
    a skill is supposed to capture -- and the tail is where it derives the answer and so
    where a wrong answer's error usually surfaces. A tail cut would keep only the setup
    and discard every mistake, which is the opposite of what this is for.
    """
    if len(text) <= limit:
        return text
    keep = limit // 2
    dropped = len(text) - 2 * keep
    return f"{text[:keep]}\n    ... [{dropped} chars elided from the middle] ...\n{text[-keep:]}"


def format_trajectory_for_teacher_core(trajectory: dict, max_steps: int = 10) -> str:
    """Render one CORE attempt for the teacher.

    Differs from the stock formatter in three ways: the header names the problem
    and the task instead of a seed and an iteration count; the observation window
    is wide enough for a verdict plus a 63-move Hanoi answer; and the final line
    prints the parsed answer verbatim rather than indexing it as ``[x, y, r]``.

    The reward is resolved through ``teacher_prompts.compute_trajectory_reward``
    at call time, so this function is correct whichever reward is currently bound
    and the order of the two rebindings inside :func:`patch` does not matter.
    """
    import skillrl.distillation.teacher_prompts as _tp

    reward = _tp.compute_trajectory_reward(trajectory)
    verdict = "CORRECT" if trajectory.get("success") else "INCORRECT"
    header = (
        f"--- Problem {trajectory.get('seed', '?')} | "
        f"Task: {trajectory.get('task_name', '?')} | "
        f"Verdict: {verdict} | Reward: {reward:+.2f}"
    )
    if trajectory.get("truncated"):
        header += " | truncated at the token limit"
    lines = [header + " ---"]

    steps = trajectory.get("trajectory", [])
    if len(steps) > max_steps:
        lines.append(f"  (showing last {max_steps} of {len(steps)} steps)\n")
        steps = steps[-max_steps:]

    for i, step in enumerate(steps, 1):
        if isinstance(step, dict):
            thought = step.get("thought", "")
            action = step.get("action", "")
            obs = step.get("observation", "")
        elif isinstance(step, (list, tuple)) and len(step) >= 3:
            thought, action, obs = step[0], step[1], step[2]
        else:
            continue
        obs = str(obs)
        if len(obs) > _OBSERVATION_MAX_CHARS:
            obs = obs[:_OBSERVATION_MAX_CHARS] + "... [truncated]"
        lines.append(f"  Step {i}:")
        lines.append(f"    Thought: {_elide_middle(str(thought), _THOUGHT_MAX_CHARS)}")
        lines.append(f"    Action: {action}")
        lines.append(f"    Observation: {obs}")
        lines.append("")

    final_answer = trajectory.get("action")
    if final_answer:
        lines.append(f"  Final answer: {final_answer}")

    return "\n".join(lines)


# ── patch / unpatch ────────────────────────────────────────────────────

class _Missing:
    """Sentinel for an attribute that did not exist before patching."""


_MISSING = _Missing()

_PatchState = dict[str, Any]

_ORIGINALS: _PatchState | None = None


def is_patched() -> bool:
    return _ORIGINALS is not None


def patch(
    *,
    reward_fn: Callable[[dict], float] = binary_trajectory_reward,
    confidence_fn: Callable[..., float] = compute_initial_confidence_core,
    trajectory_selection: str = DEFAULT_TRAJECTORY_SELECTION,
    max_trajs: int = DEFAULT_MAX_TRAJS,
    task_descriptions: Mapping[str, str] = CORE_TASK_DESCRIPTIONS,
    load_trajectories_fn: Callable[..., Any] | None = None,
    truncation_aware_evolution: bool = False,
) -> None:
    """Install the CORE-compatible prompts, reward and formatter into ``skillrl``.

    Call before any ``distill_*`` / ``evolve_skill_bank`` invocation. Idempotent:
    re-applying is harmless, and the snapshot taken by the *first* call is the one
    :func:`unpatch` restores, so calling ``patch`` again with different arguments
    (say a utility-based ``reward_fn`` for M1) re-applies without losing the true
    originals.

    Parameters
    ----------
    reward_fn
        Replacement for ``compute_trajectory_reward``. This is M2 (binary), and it
        stays binary: it feeds the *teacher-facing* reward line and the sort that
        picks which five trajectories the teacher sees. M1 arrives through
        ``confidence_fn`` instead, so the two remain independently ablatable --
        folding M1 into ``reward_fn`` would also change what the teacher reads,
        confounding the two modifications.
    trajectory_selection
        Which ``max_trajs`` trajectories the evolver is shown: ``"paired"`` (the
        default -- same-problem contrast pairs first), ``"balanced"`` (a global
        success/failure quota) or ``"top5"`` (HeXA's own sort-by-reward, which sends no
        failures at all on these tasks -- see :data:`DEFAULT_TRAJECTORY_SELECTION`).
        See :func:`make_trajectory_formatter`.
    max_trajs
        How many trajectories the evolver is shown per round, overriding the 5 that
        ``evolving_distill`` hardcodes. See :data:`DEFAULT_MAX_TRAJS`.
    confidence_fn
        Replacement for ``distill._compute_initial_confidence``. Defaults to M1's
        :func:`compute_initial_confidence_core`. Pass the stock
        ``distill._compute_initial_confidence`` to run the pre-M1 arm.
    task_descriptions
        Merged into ``config.LEVEL_DESCRIPTIONS``. Raises :class:`ValueError`
        rather than overwrite a key some other level already owns.
    load_trajectories_fn
        Optional replacement for ``load_trajectories`` in both consumers, e.g.
        :func:`hexa_core.trajectory.load_attempt_trajectories`. Left alone when
        ``None`` so this module stays useful to callers that pass trajectory
        lists directly.
    truncation_aware_evolution
        Adds :data:`~hexa_core.prompts_core_tasks.TRUNCATION_AWARE_GUIDANCE` to the
        evolution prompt: tells the evolver to count truncated attempts and, when they
        dominate the failures, to bank a search-shortening skill rather than another
        verification check. **Off by default**, and deliberately so -- every banked
        result (ZebraLogic +0.247, Hanoi +0.257, both n=3) was produced without it, and
        with it off the prompt is byte-identical to what produced them. Turning it on is
        an ablation arm, not a bug fix. Motivation and measurements: `WEAKNESSES.md` §11.
    """
    global _ORIGINALS

    import skillrl.core.config as _config
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving_distill
    import skillrl.distillation.skills_only_distill as _skills_only
    import skillrl.distillation.teacher_prompts as _tp

    _selector = make_trajectory_formatter(trajectory_selection, max_trajs=max_trajs)

    state: _PatchState = (
        _ORIGINALS
        if _ORIGINALS is not None
        else {"bindings": [], "description_keys": []}
    )
    added_keys: list[str] = state["description_keys"]

    # Validate before mutating anything, so a rejected patch leaves no residue.
    clashes = sorted(
        key
        for key in task_descriptions
        if key in _config.LEVEL_DESCRIPTIONS and key not in added_keys
    )
    if clashes:
        raise ValueError(
            "refusing to overwrite existing LEVEL_DESCRIPTIONS entries: "
            f"{clashes}. Rename the CORE task keys or clear those levels first."
        )

    bindings: list[tuple[Any, str, Any]] = [
        (_distill, "CONTRASTIVE_DISTILLATION_PROMPT", CONTRASTIVE_DISTILLATION_PROMPT_CORE),
        (_distill, "COMMON_MISTAKES_PROMPT", COMMON_MISTAKES_PROMPT_CORE),
        (_evolving_distill, "SKILL_BANK_EVOLUTION_PROMPT",
         evolution_prompt(truncation_aware=truncation_aware_evolution)),
        (_tp, "compute_trajectory_reward", reward_fn),
        (_distill, "compute_trajectory_reward", reward_fn),
        (_tp, "format_trajectory_for_teacher", format_trajectory_for_teacher_core),
        # M1. Rebound in all three modules, not just `_distill`: `evolving_distill`
        # and `skills_only_distill` do `from .distill import _compute_initial_
        # confidence`, so they hold their own reference and a single rebinding would
        # miss `evolve_skill_bank` -- the one call site that actually runs here.
        (_distill, "_compute_initial_confidence", confidence_fn),
        (_evolving_distill, "_compute_initial_confidence", confidence_fn),
        (_skills_only, "_compute_initial_confidence", confidence_fn),
        # Reporting only -- same selection, now visible. Rebound in `_evolving_distill`
        # too because it imports the name directly and holds its own reference.
        # Which trajectories reach the evolver. Default "top5" is HeXA's own
        # selection, kept as the default so existing results stay reproducible; the
        # formatter only adds reporting. Rebound in `_evolving_distill` as well because
        # it imports the name directly and holds its own reference.
        (_evolving_distill, "format_trajectories_block", _selector),
        (_tp, "format_trajectories_block", _selector),
    ]
    if load_trajectories_fn is not None:
        bindings.append((_distill, "load_trajectories", load_trajectories_fn))
        bindings.append((_evolving_distill, "load_trajectories", load_trajectories_fn))

    snapshotted = {(module.__name__, attr) for module, attr, _ in state["bindings"]}
    for module, attr, value in bindings:
        if (module.__name__, attr) not in snapshotted:
            state["bindings"].append((module, attr, getattr(module, attr, _MISSING)))
            snapshotted.add((module.__name__, attr))
        setattr(module, attr, value)

    for key, description in task_descriptions.items():
        _config.LEVEL_DESCRIPTIONS[key] = description
        if key not in added_keys:
            added_keys.append(key)

    _ORIGINALS = state


def unpatch() -> None:
    """Restore every name :func:`patch` rebound. No-op when not patched."""
    global _ORIGINALS

    if _ORIGINALS is None:
        return

    import skillrl.core.config as _config

    for module, attr, original in reversed(_ORIGINALS["bindings"]):
        if original is _MISSING:
            delattr(module, attr)
        else:
            setattr(module, attr, original)
    for key in _ORIGINALS["description_keys"]:
        _config.LEVEL_DESCRIPTIONS.pop(key, None)

    _ORIGINALS = None


# ── M7: token-budgeted retrieval ───────────────────────────────────────

class RetrievedBlock(NamedTuple):
    """A rendered guidelines block together with exactly what went into it.

    The two ``dropped_*`` counts exist so that a budget-bound retrieval is visible
    in the *return value* and not only in this module's docstring. Without them a
    caller cannot tell a block that carried everything the bank had from one that
    silently lost half of it to the token budget, and a log line saying "injected 2
    skills" reads as coverage when it is truncation.
    """

    block_text: str
    skills: list[Skill]
    mistakes: list[Mistake]
    tokens: int
    dropped_skills: int = 0
    dropped_mistakes: int = 0

    @property
    def budget_bound(self) -> bool:
        """True iff at least one retrieved candidate did not fit. Log this."""
        return bool(self.dropped_skills or self.dropped_mistakes)


def _warn_if_starved(
    result: RetrievedBlock, task_name: str, budget_tokens: int, overhead_tokens: int
) -> RetrievedBlock:
    """Log loudly when a non-empty bank was budgeted down to nothing at all.

    A total drop is almost always a misconfigured budget, and it is the one
    outcome that cannot be spotted downstream: an empty block makes
    ``build_prompt(block)`` byte-identical to ``build_prompt(None)``, so a
    memory-augmented run degrades into a no-memory baseline and reports "no gain
    from HeXA" with every log line looking healthy.

    Warn rather than raise: a raise would kill a long run over a recoverable
    config error. Silent for ``budget_tokens <= 0``, which is an unambiguous
    "inject nothing" instruction — that is how the no-memory ablation arm is
    expressed, and warning on it would emit one line per problem per run.
    """
    if budget_tokens <= 0 or result.block_text or not result.budget_bound:
        return result
    logger.warning(
        "retrieve_within_token_budget(%r): budget %d tokens (overhead %d) fits none "
        "of the %d skill(s) + %d mistake(s) retrieved, so NOTHING is injected and "
        "this run is a no-memory baseline in disguise. Raise budget_tokens, lower "
        "max_specific, or pass skip_oversized=True. Token counter: %s.",
        task_name,
        budget_tokens,
        overhead_tokens,
        result.dropped_skills,
        result.dropped_mistakes,
        counter_name(),
    )
    return result


def _relabel(block: str, task_name: str) -> str:
    """Retitle the physics-specific headings ``format_skills_as_prompt`` hardcodes."""
    block = block.replace("=== LEARNED PHYSICS SKILLS ===", SKILLS_BLOCK_HEADING)
    auto_header = f"## {task_name.replace('_', ' ').title()}-Specific Skills"
    return block.replace(auto_header, f"## Task-Specific Strategies ({task_name})")


def retrieve_within_token_budget(
    bank: SkillBank | None,
    task_name: str,
    budget_tokens: int = 1000,
    *,
    max_general: int = 0,
    max_specific: int = 8,
    max_mistakes: int = 4,
    overhead_tokens: int = 0,
    skip_oversized: bool = False,
    relabel: bool = True,
) -> RetrievedBlock:
    """Retrieve skills+mistakes for *task_name*, capped at a rendered token budget.

    Wraps the existing pieces rather than reimplementing them:
    ``retriever.retrieve_skills`` chooses and orders the candidates (that
    confidence ordering *is* the retriever's statement of priority, and it is
    preserved), and ``SkillBank.format_skills_as_prompt`` does the rendering. This
    function only decides where to stop, by re-rendering after each candidate and
    counting tokens, then returns the **last accepted render** together with the
    lists that produced it — so ``block_text`` can never disagree with ``skills``
    and ``mistakes``.

    The default 1000 tokens tracks CORE's reported 0.92k of added task-external
    context. ``overhead_tokens`` lets a caller reserve room for the wording CORE's
    own guidelines slot wraps the block in, since the metric of record is added
    tokens on the *full* prompt.

    Three caveats worth logging next to any budget decision:

    * Skills are considered before mistakes, so a bank of long skills can consume
      the whole budget before a single mistake is reached. ``skip_oversized=True``
      (keep going past a candidate that does not fit) or a smaller
      ``max_specific`` are the levers. ``RetrievedBlock.budget_bound`` and the
      ``dropped_*`` counts report when this fired.
    * ``max_specific`` / ``max_mistakes`` are a top-N cut applied *before* the
      budget walk, so candidates the bank held but the retriever never offered are
      **not** counted in ``dropped_*``. Compare against the bank size if total
      coverage is the question.
    * ``count_tokens`` falls back to a heuristic when ``tiktoken`` is absent, so
      record :func:`hexa_core.tokens.counter_name` alongside the count.
    """
    empty = RetrievedBlock("", [], [], 0)
    if bank is None:
        return empty

    candidate_skills = retrieve_skills(
        bank, task_name, max_general=max_general, max_specific=max_specific
    )
    candidate_mistakes = bank.get_mistakes_for_level(task_name)[:max_mistakes]
    if budget_tokens <= 0 or (not candidate_skills and not candidate_mistakes):
        # A non-positive budget with a non-empty bank IS a total drop; say so
        # rather than returning the same value as "the bank had nothing".
        return _warn_if_starved(
            RetrievedBlock("", [], [], 0, len(candidate_skills), len(candidate_mistakes)),
            task_name,
            budget_tokens,
            overhead_tokens,
        )

    def render(skills: list[Skill], mistakes: list[Mistake]) -> tuple[str, int]:
        block = bank.format_skills_as_prompt(
            task_name, skills=skills, mistakes=mistakes
        )
        if relabel:
            # Relabel BEFORE counting: the replacement headings change the count.
            block = _relabel(block, task_name)
        return block, count_tokens(block)

    accepted_skills: list[Skill] = []
    accepted_mistakes: list[Mistake] = []
    best = empty

    candidates: list[tuple[str, Any]] = [("skill", s) for s in candidate_skills]
    candidates += [("mistake", m) for m in candidate_mistakes]

    for index, (kind, item) in enumerate(candidates):
        trial_skills = accepted_skills + [item] if kind == "skill" else accepted_skills
        trial_mistakes = (
            accepted_mistakes + [item] if kind == "mistake" else accepted_mistakes
        )
        block, tokens = render(trial_skills, trial_mistakes)
        if tokens + overhead_tokens <= budget_tokens:
            accepted_skills, accepted_mistakes = trial_skills, trial_mistakes
            best = RetrievedBlock(
                block, list(accepted_skills), list(accepted_mistakes), tokens
            )
        elif not skip_oversized:
            # Everything from here on is dropped, including every mistake: skills
            # are considered first, so one fat skill starves the mistakes.
            remaining = candidates[index:]
            return _warn_if_starved(
                best._replace(
                    dropped_skills=sum(1 for k, _ in remaining if k == "skill"),
                    dropped_mistakes=sum(1 for k, _ in remaining if k == "mistake"),
                ),
                task_name,
                budget_tokens,
                overhead_tokens,
            )

    return _warn_if_starved(
        best._replace(
            dropped_skills=len(candidate_skills) - len(accepted_skills),
            dropped_mistakes=len(candidate_mistakes) - len(accepted_mistakes),
        ),
        task_name,
        budget_tokens,
        overhead_tokens,
    )
