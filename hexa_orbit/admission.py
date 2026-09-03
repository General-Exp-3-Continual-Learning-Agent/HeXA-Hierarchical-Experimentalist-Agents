"""M5 admission gating for Maze skills, at single-episode granularity.

``hexa_core.admission.admit_skills`` cannot be reused directly: it is built around
CORE's single-turn ``Problem`` and a ``generate(prompt) -> str`` callable, and a
maze trial is a multi-turn episode. Its *result* types are reused
(:class:`~hexa_core.admission.AdmissionResult`, ``accept_rate``) so the reporting
is identical across the two arms of the project.

Per DESIGN_PHASE3_MAZE §6, a trial is **one episode**, not a whole instance: ≤9
calls instead of ≤27, and episode-1 performance is what the headline claim is
about anyway.

**The open problem, stated plainly.** A binary admission criterion needs a base
rate to be meaningful. If unaided episode-1 success is low -- which is what a
9-turn budget against a 7-8 move path with purely local observations suggests --
then 3 trials scored on success alone return 0 for almost every candidate, and the
gate rejects everything including the good skills. That is not a conservative gate,
it is a broken one.

Two responses, and this module implements both:

* :func:`progress_score` -- a **graded** trial score. A failed episode still
  reveals whether the agent got closer to the goal, so progress carries
  information where binary success carries none. Default scorer.
* ``enabled=False`` **by default.** The criterion cannot be calibrated before the
  Phase 2 baseline reports the real episode-1 rate, so the pilot runs with the gate
  off and the candidates recorded. Turning it on is a flag, and the results still
  say ``reason="gate-disabled"`` so the two arms are distinguishable in the logs
  rather than only in the config -- ``hexa_core``'s convention.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Optional, Sequence

from hexa_core.admission import AdmissionResult, accept_rate  # noqa: F401  (re-export)

from .oracle import optimal_actions
from .pools import MazeInstanceSpec

#: Trials per candidate skill. Three single-episode trials is ~27 calls per
#: candidate -- the same cost as one whole-instance trial would have been.
DEFAULT_TRIALS = 3

#: Fraction of the shortest path a trial must cover to count as a pass, when
#: scoring on progress rather than success. 1.0 means "reached the goal".
DEFAULT_PROGRESS_THRESHOLD = 1.0

#: How far a candidate must beat the no-skill control to be admitted. Small but
#: nonzero: with three trials the paired means are noisy, and admitting on a tie
#: would let a skill that does nothing occupy a slot and cost tokens on every turn
#: of every future episode. Expressed in the same [0,1] units as `progress_score`.
DEFAULT_ADMISSION_MARGIN = 0.05


def _solo_bank(skill: Any) -> Any:
    """A bank containing exactly one skill, so a trial measures that skill alone."""
    from skillrl.core.skill_bank import SkillBank  # noqa: PLC0415

    bank = SkillBank()
    bank.level_skills = {"maze": [skill]}
    bank.level_mistakes = {}
    return bank


def progress_score(instance: Any, reached_goal: bool) -> float:
    """Fraction of the shortest path the agent closed, in ``[0, 1]``.

    ``1.0`` iff the goal was reached. Otherwise the *reduction* in known-map
    distance-to-goal from the start cell, which is well defined on a failed episode
    and is why this exists: binary success on a low base rate cannot separate a
    useful skill from a useless one, and progress can.

    Uses the oracle's BFS on the true map, so this is a *measurement* of the
    trial, never a signal the agent sees.
    """
    if reached_goal:
        return 1.0
    env = instance._env
    if env.map is None or env.init_position is None:
        return 0.0
    start_distance = len(optimal_actions(env.map, env.init_position))
    if start_distance == 0:
        return 1.0
    try:
        remaining = len(optimal_actions(env.map, env.current_position))
    except ValueError:  # unreachable from where it ended up -- no progress credited
        return 0.0
    return max(0.0, (start_distance - remaining) / start_distance)


def run_instance_trial(
    block: Optional[str],
    spec: MazeInstanceSpec,
    task: Any,
    actor_factory: Callable[..., Any],
) -> float:
    """Score one *whole instance* -- all three episodes -- under *block*.

    ``1.0`` if any episode reached the goal, else the best progress any episode
    made. ``block=None`` runs the same instance with no bank at all, which is the
    control this gate compares against.

    **Why a whole instance and not the single episode `run_trial` uses.** A trial
    on ``new_episode()`` has no earlier episode, so a cross-episode skill -- "stop
    retracing your episode-1 route once you reach a junction it already failed at"
    -- cannot express itself and scores like a null skill. That is the wrong gate
    for an episode-3 endpoint: it would reject precisely the skills the measured
    driver points at (instances later solved depart from their own episode-1 route
    at turn 1.6 against 2.6 for those never solved). Costs ~27 calls instead of ~9.
    """
    instance = task.instance(spec)
    actor = actor_factory(block, instance)
    run = actor.run_instance(instance, leaked=False)
    if any(e.success for e in instance.episodes):
        return 1.0
    return float(getattr(run, "best_progress", 0.0) or 0.0)


def run_trial(
    skill: Any,
    spec: MazeInstanceSpec,
    task: Any,
    actor_factory: Callable[..., Any],
    *,
    render_block: Callable[[Any], str],
) -> tuple[bool, float]:
    """One single-episode trial of one skill. Returns ``(reached_goal, score)``.

    The episode is driven through :meth:`MazeInstance.new_episode`, which is the
    plan's Phase 1 ``Episode`` shape and deliberately bypasses the three-episode
    bookkeeping -- a trial must not accumulate cross-episode context, or it would
    measure in-context learning rather than the skill.
    """
    instance = task.instance(spec)
    episode = instance.new_episode()
    actor = actor_factory(render_block(_solo_bank(skill)), instance)

    observation = episode.obs_0
    reached_goal = False
    messages = [{"role": "system", "content": actor.system_prompt}]
    for _ in range(episode.max_steps):
        messages.append({"role": "user", "content": observation})
        generation = actor.client.generate_structured(messages=list(messages))
        if generation is None:
            break
        from .actor import boxed, extract_last_boxed  # noqa: PLC0415

        action = boxed(extract_last_boxed(generation.raw_text))
        messages.append({"role": "assistant", "content": generation.final_text or action})
        observation, reward, done = episode.step(action)
        if done:
            reached_goal = reward > 0
            break

    return reached_goal, progress_score(instance, reached_goal)


def _run_parallel(thunks: Sequence[Callable[[], Any]], concurrency: int) -> list[Any]:
    """Run *thunks* concurrently, preserving order.

    Trials are independent -- different skills, different instances, no shared
    state -- but `admit_skills` ran them one call at a time while `loop.play_batch`
    used a 12-way pool for the same work. Measured cost of that asymmetry on v7:
    ~950 sequential calls for one round's gate, about 2h15m, against ~35 minutes
    for the round itself. Ten rounds would have exceeded the job's 6h walltime
    around round 2, with no eval.
    """
    if concurrency <= 1 or len(thunks) <= 1:
        return [t() for t in thunks]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(thunks))) as pool:
        return list(pool.map(lambda t: t(), thunks))


def control_mean_score(
    trial_specs: Sequence[MazeInstanceSpec],
    task: Any,
    actor_factory: Callable[..., Any],
    *,
    trials: int = DEFAULT_TRIALS,
    budget: Any = None,
    concurrency: int = 1,
) -> float:
    """Mean whole-instance score on *trial_specs* with **no bank injected**.

    Computed once per round and shared by every candidate that round, so the gate
    costs one extra set of trials rather than one per skill. Pairing matters more
    than freshness here: the comparison must be against the same instances, since
    instance difficulty varies far more than the effect being measured.
    """
    chosen = list(trial_specs[:trials])
    if not chosen:
        return 0.0
    if budget is not None:
        for _ in chosen:
            budget.charge_admission(1)
    scores = _run_parallel(
        [lambda spec=spec: run_instance_trial(None, spec, task, actor_factory)
         for spec in chosen],
        concurrency,
    )
    return sum(scores) / len(scores)


def admit_skills(
    skills: Iterable[Any],
    trial_specs: Sequence[MazeInstanceSpec],
    task: Any,
    actor_factory: Callable[..., Any],
    *,
    render_block: Callable[[Any], str],
    trials: int = DEFAULT_TRIALS,
    enabled: bool = False,
    threshold: float = DEFAULT_PROGRESS_THRESHOLD,
    max_tests: Optional[int] = None,
    budget: Any = None,
    control_mean: Optional[float] = None,
    margin: float = DEFAULT_ADMISSION_MARGIN,
    whole_instance: bool = False,
    concurrency: int = 1,
) -> tuple[list[Any], list[AdmissionResult]]:
    """Gate candidate skills. Returns ``(admitted, all_results)``.

    ``enabled=False`` (the default, see the module docstring) admits everything and
    records ``reason="gate-disabled"``, so an ungated run is visible in the logs.

    ``trial_specs`` must come from the **training** pool and from outside the
    current round: a trial drawn from the same instances a skill was distilled from
    is a replay and passes trivially. ``hexa_core/seeds.py``'s namespace discipline
    is the precedent.

    ``max_tests`` caps how many candidates are judged, and overflow candidates are
    **rejected** rather than waved through -- the conservative direction, since it
    can only cost a skill that might have been kept and never manufactures one that
    was not earned. Recorded as ``reason="admission-budget-capped"``.
    """
    candidates = list(skills)
    results: list[AdmissionResult] = []
    admitted: list[Any] = []
    judged = 0

    for skill in candidates:
        skill_id = getattr(skill, "skill_id", "<unknown>")

        if not enabled:
            results.append(AdmissionResult(
                skill_id=skill_id, problem_id="", admitted=True, trials=0,
                successes=0, reason="gate-disabled", injected=False,
            ))
            admitted.append(skill)
            continue

        if max_tests is not None and judged >= max_tests:
            results.append(AdmissionResult(
                skill_id=skill_id, problem_id="", admitted=False, trials=0,
                successes=0, reason="admission-budget-capped", injected=False,
            ))
            continue

        if not trial_specs:
            results.append(AdmissionResult(
                skill_id=skill_id, problem_id="", admitted=False, trials=0,
                successes=0, reason="no-trial-instances", injected=False,
            ))
            continue

        judged += 1
        chosen = list(trial_specs[:trials])
        if budget is not None:
            for _ in chosen:
                budget.charge_admission(1)

        block = render_block(_solo_bank(skill)) if whole_instance else None

        def _one(spec, _skill=skill, _block=block):
            if whole_instance:
                score = run_instance_trial(_block, spec, task, actor_factory)
                return score >= 1.0, score
            return run_trial(
                _skill, spec, task, actor_factory, render_block=render_block
            )

        outcomes = _run_parallel(
            [lambda spec=spec: _one(spec) for spec in chosen], concurrency
        )
        scores = [score for _, score in outcomes]
        passes = sum(
            int(score >= threshold if threshold < 1.0 else reached)
            for reached, score in outcomes
        )

        mean_score = sum(scores) / len(scores) if scores else 0.0
        # Admit on a *paired* comparison against the same instances run with no
        # bank, not on an absolute pass count. `passes > 0` at a ~28% base rate
        # over three trials admits roughly two thirds of candidates on luck alone,
        # which is not a gate. The control is computed once per round and shared
        # across candidates, so it costs one set of trials rather than one per
        # skill. `margin` keeps a skill that merely ties from displacing nothing:
        # a tie is not evidence of use.
        if control_mean is None:
            keep = passes > 0
        else:
            keep = mean_score >= control_mean + margin
        results.append(AdmissionResult(
            skill_id=skill_id,
            problem_id=chosen[0].instance_id if chosen else "",
            admitted=keep,
            trials=len(chosen),
            successes=passes,
            reason=(
                f"mean={mean_score:.2f}"
                + (f" vs control {control_mean:.2f}" if control_mean is not None else "")
                + ("" if keep else " -- not above control")
            ),
            injected=True,
        ))
        if keep:
            admitted.append(skill)

    return admitted, results
