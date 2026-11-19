"""Cross-world teacher prompts for HeXA on DiscoverPhysics (Axis B).

The bank is a single CROSS-WORLD meta-bank of *experimentation strategies*, not
per-world answers.  These prompts therefore:
  - describe the discovery meta-task (design experiments -> propose a law,
    scored by trajectory error + explanation), not ball placement;
  - state the reward scale used here (blend of explanation quality and
    normalized trajectory error, in [-0.5, 1.0]);
  - explicitly demand WORLD-AGNOSTIC methods and forbid recording any fitted
    constant value (anti-leakage).

`{level_description}` is filled by the loop with the CURRENT world's answer-free
factual block (see config.WORLD_FACTS), so the teacher knows what setup produced
the new trajectories without being told the answer.

`patch()` swaps these into distill / evolving_distill and replaces the
trajectory formatter with a DiscoverPhysics-aware one (no [x,y,r] assumption,
graded-reward header, generous observation truncation).
"""

from __future__ import annotations

import os

from typing import Optional

from hexa_discovery.traj_adapter import redact_law_forms

# ── Shared framing ─────────────────────────────────────────────────────
_ENV = """\
**Environment**: an LLM agent does open-ended physics discovery. Each episode it is
dropped into a simulated world with an unknown governing law. It designs experiments
(placing particles with chosen positions / velocities / scalar properties and choosing
measurement times), observes NOISY particle trajectories, may fit candidate-law
parameters mid-run, and finally submits a governing law as executable Python plus a
short prose explanation. The submission is scored by (a) trajectory error against
held-out tests and (b) an explanation-judge score.

**Reward** per trajectory is a blend of explanation quality and normalized trajectory
error, in [-0.5, 1.0] (higher is better): +1.0 = a near-perfect law with a correct
explanation; ~0 = right at the pass line; negative = the law does not reproduce the
trajectories. Weight your analysis by reward.

**These skills are CROSS-WORLD**: they will be injected into the agent on *other*,
unseen worlds. Extract transferable EXPERIMENTATION and REASONING methods — how to
probe, what to vary, how to read the data, how/when to fit, how to rule out a
candidate law — NOT the answer to any single world. Do NOT write down specific fitted
constant values (exponents, coupling strengths, screening lengths, periods): those do
not transfer and leak answers. Phrase skills as procedures a scientist would reuse on a
brand-new world."""



#: Generic hypothesis-proposal instruction, appended to both distillation prompts.
#:
#: WHY: the evolver's only signal about explanation quality is a scalar
#: `expl_score` (the judge's critique is oracle and excluded, finding 5). From
#: "good fit, low score" it cannot tell over-claiming from under-claiming, and
#: with no passing episode to contrast it has no evidence either way. Measured
#: on three circle runs: the one whose evolver spontaneously proposed a
#: mechanism at round 1 reached expl 0.887 with 11/15 explanations naming it;
#: the two whose evolvers instead wrote epistemic caution ("treat the gate as
#: still unsolved") landed at 0.573 and 0.533, cut the actor's experiment
#: rounds from 4.3 to ~2.4, and produced 0/15 explanations naming any
#: mechanism. The difference was the evolver volunteering a hypothesis, which
#: nothing in the prompt asked for.
#:
#: This makes that behaviour deterministic. It is deliberately WORLD-AGNOSTIC:
#: it names no law, no functional form and no world. The evolver proposes
#: candidates from its own physics knowledge and instructs the agent to TEST
#: them -- a hypothesis to falsify, not an answer to copy. Writing anything
#: world-specific here would be fitting the prompt to a rubric we have already
#: seen, which is the thing this must not become.
#: Opt-in. The block below changes what the evolver writes, so a run using it is
#: NOT a replication of any arm produced without it. Default OFF so the faithful
#: original behaviour is what you get unless you ask for the change.
_HYPOTHESIS_ON = os.environ.get("HEXA_DISCOVERY_HYPOTHESIS_PROMPT", "0") == "1"

_HYPOTHESIS_BLOCK_TEXT = """
ADDITIONALLY, when the episodes show low trajectory error but low explanation
scores: the experiments are working and the explanation is describing the data
without identifying a MECHANISM. For at least one of your skills in that case,
reason from the measurements yourself, propose one or more CANDIDATE MECHANISMS
or functional families consistent with what was observed, and phrase the skill
as an instruction to TEST those candidates and state which the data supports and
which it rules out. Propose hypotheses to be falsified, never assert an answer;
name the class of model to test and the experiment that would discriminate it,
with no fitted constant values. This is in addition to the skills you extract
above, not a replacement for them.
"""

_HYPOTHESIS_BLOCK = _HYPOTHESIS_BLOCK_TEXT if _HYPOTHESIS_ON else ""


CONTRASTIVE_DISTILLATION_PROMPT = ("""\
You are an expert experimental physicist distilling agent behavior into concise,
transferable discovery skills.

""" + _ENV + """

**Current world (setup only — the governing law is hidden even from you):**
{level_description}

Below are SUCCESSFUL and FAILED discovery episodes on this world. Each shows the
agent's reasoning (Thought), the experiment / fit / submission it took (Action), and
what the simulator returned (Observation), plus a Reward.

=== SUCCESSFUL EPISODES ({n_successes}) ===
{success_block}

=== FAILED EPISODES ({n_failures}) ===
{failure_block}

---

By CONTRASTING successes and failures, extract the KEY EXPERIMENTATION SKILLS that
distinguished discovering the law from failing. Focus on high-reward successes: what
probing / variable-isolation / data-reading / fitting move let them succeed, and what
did the failures miss?

{hypothesis_block}
For each skill provide:
- **title**: short name (3-7 words)
- **principle**: the transferable method (2-3 sentences). What to do and why it works.
  World-agnostic; no specific constant values.
- **when_to_apply**: the trigger condition (1 sentence)
- **source_seeds**: seed numbers from the episodes above this skill came from

Output a JSON array:
```json
[
  {{"title": "...", "principle": "...", "when_to_apply": "...", "source_seeds": [0, 2]}}
]
```

Extract 4-6 DISTINCT skills. Avoid redundancy. No fitted constant values.
""").replace("{hypothesis_block}", _HYPOTHESIS_BLOCK)


COMMON_MISTAKES_PROMPT = """\
You are an expert at analyzing physics-discovery failures and distilling avoidable
mistake patterns.

""" + _ENV + """

**Current world (setup only):**
{level_description}

Below are FAILED discovery episodes.

{failure_block}

---

**Part 1 — Mistakes**: identify COMMON MISTAKE PATTERNS across these failures. For each:
1. what the agent did wrong (experiment design, data interpretation, fitting, or the
   submitted law/explanation),
2. WHY (the broken belief or shortcut behind it),
3. a concrete, transferable fix.

**Part 2 — Partial insights**: even failed episodes contain good moves (a smart probe,
a correct ruling-out, a useful fit) that were then wasted. Extract 1-2 transferable
skills from those good steps.

Output a JSON object:
```json
{{
  "mistakes": [
    {{"description": "...", "why_it_happens": "...", "how_to_avoid": "..."}}
  ],
  "partial_skills": [
    {{"title": "...", "principle": "...", "when_to_apply": "...", "source_seeds": [1]}}
  ]
}}
```

Extract 3-5 mistakes and 1-2 partial skills. Group similar failures; focus on root
causes. World-agnostic methods only — no fitted constant values.
"""


SKILL_BANK_EVOLUTION_PROMPT = ("""\
You are a physics-discovery mentor evolving a CROSS-WORLD skill bank for a discovery
agent. The bank travels with the agent to unseen worlds, so every skill must be a
transferable experimentation/reasoning method — never a specific world's answer.

""" + _ENV + """

**World that produced the NEW episodes below (setup only):**
{level_description}

CURRENT SKILL BANK (from previous worlds/rounds):
{existing_skills_block}

EPISODES:
Successes: {n_successes}   Failures: {n_failures}

Episodes are marked by provenance. **[NEW — this round]** are the fresh episodes
from the world this round is about. **[REFERENCE — earlier world]** are episodes
from worlds already seen, included so you have both outcomes to contrast: a world
is usually solved on every seed or none, so a round's own episodes are often all
successes or all failures. Reference episodes are there to contrast AGAINST, not
to describe. Ground every skill in the NEW episodes, or in the contrast between
new and reference — never in a reference episode alone, and never restate a
lesson that is only about a reference world.

{new_trajectories_block}

YOUR TASK: merge the existing bank with insights from the new episodes into the
COMPLETE FINAL bank (not a diff).

RULES:
1. Output the full final bank — retained skills AND new ones.
2. Hard caps: at most {max_skills} skills and {max_mistakes} mistakes total.
3. Per skill: set "is_new" (true = extracted from the new episodes, false = retained);
   include "source_seeds" and a "confidence" in [0.1, 1.0].
4. Retained skills keep their existing confidence unless the new episodes clearly
   strengthen or weaken them.
5. New-skill confidence reflects: reward of source episodes, and how broadly the
   method should apply across DIFFERENT worlds (a method that only helped this one
   world's answer is low value — prefer general procedures).
6. REMOVE skills that are: redundant/subsumed, contradicted by the new episodes,
   world-specific answers that slipped in, or low confidence (<0.3) and unsupported by
   the new episodes. Actively DEMOTE or remove a skill the new world contradicts (e.g.
   a "assume the force is static / central" heuristic when this world's trajectories
   show otherwise) — capturing that contradiction is the point.
7. Never record fitted constant values in any skill or example.

{hypothesis_block}
OUTPUT JSON OBJECT:
{{
  "skills": [
    {{"title": "...", "principle": "...", "when_to_apply": "...",
      "example": "<optional, method-level, NO constant values>",
      "source_seeds": [<ints>], "confidence": <float>, "is_new": <bool>}}
  ],
  "mistakes": [
    {{"description": "...", "why_it_happens": "...", "how_to_avoid": "...",
      "is_new": <bool>}}
  ],
  "removed_skill_titles": ["..."],
  "reasoning": "<brief: what changed and why, esp. anything demoted/removed>"
}}

Be concise and precise. Ground every skill in the episodes.
""").replace("{hypothesis_block}", _HYPOTHESIS_BLOCK)


# ── World anonymisation ────────────────────────────────────────────────
#
# The trajectory header prints the world a batch came from, which the teacher
# needs: it must know two episodes came from DIFFERENT worlds to generalise
# across them rather than merge them. But several world names ARE the answer —
# "yukawa", "coulomb_easy", "hubble", "ether", "extra_dimensions",
# "dark_matter" each name the law the agent was supposed to discover, and the
# name reaches the teacher intact no matter how thoroughly the prose around it
# is redacted.
#
# So the header shows a stable anonymous label instead: the teacher still sees
# which episodes share a world and which do not, and gets no name to reason
# from. Labels are assigned in order of first appearance and are stable for the
# life of the process, so one run's blocks are internally consistent.

_WORLD_LABELS: dict[str, str] = {}


def world_label(world: Optional[str]) -> str:
    """Stable per-process anonymous label for a world name ("W1", "W2", ...)."""
    if not world:
        return "unknown"
    if world not in _WORLD_LABELS:
        _WORLD_LABELS[world] = f"W{len(_WORLD_LABELS) + 1}"
    return _WORLD_LABELS[world]


def reset_world_labels() -> None:
    """Clear the label registry (tests; and a fresh run in the same process)."""
    _WORLD_LABELS.clear()


# ── DiscoverPhysics-aware trajectory formatter ─────────────────────────

def _truncate_text(t: str, n: int) -> str:
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n] + "... [truncated]"


def scrub_law_forms(t: str) -> str:
    """Mask law-form names in judge critiques (they state the answer outright)."""
    return redact_law_forms(t) if t else ""


def format_trajectory_for_teacher_discovery(trajectory: dict, max_steps: int = 12) -> str:
    """Render one DiscoverPhysics trajectory for the teacher.

    Reads the graded reward via the (patched) compute_trajectory_reward, keeps a
    generous observation window, and makes no InterPhyre [x,y,r] assumption.
    """
    # Resolve reward through whatever compute_trajectory_reward is currently
    # bound in teacher_prompts (reward.patch() installs the blended version).
    import skillrl.distillation.teacher_prompts as _tp
    reward = _tp.compute_trajectory_reward(trajectory)

    # Provenance, present only when the batch was built by the replay buffer:
    # a replayed success from an earlier world must not read as fresh evidence
    # about the world this round is actually about. Absent for round-1
    # distillation, which leaves that output byte-identical.
    if "_replay" in trajectory:
        origin = ("[REFERENCE — earlier world] " if trajectory.get("_replay")
                  else "[NEW — this round] ")
    else:
        origin = ""

    lines = [
        f"--- {origin}Seed {trajectory.get('seed', '?')} | "
        f"World: {world_label(trajectory.get('world'))} | "
        f"Success: {trajectory.get('success', '?')} | "
        f"norm_mse={trajectory.get('norm_mse')} | expl_score={trajectory.get('expl_score')} | "
        f"Reward: {reward:+.2f} ---"
    ]

    # Append the judge's critique so the teacher can attribute the explanation
    # half of the reward. Without it the teacher sees a scalar and cannot learn
    # what a good explanation must contain; every rubric's top band demands the
    # operator class, the role of each controllable parameter, and a numerical
    # estimate of the coupling -- and the agent reliably supplies only the
    # empirical falloff.
    # Surface the refit penalty. `fit_parameters()` is optional, and the
    # evaluator re-optimises whatever the law declares against the agent's own
    # training data before scoring on held-out cases -- so a constant the agent
    # already measured precisely can be overfitted away. Without this line the
    # teacher sees only the post-refit number and cannot tell that declaring the
    # parameter is what cost the episode.
    own = trajectory.get("norm_mse_own")
    rec = trajectory.get("norm_mse")
    if own is not None and rec is not None and own == own and rec == rec and own > 0:
        ratio = rec / own
        if ratio > 2.0:
            lines.append(
                f"  [parameter refit] the submitted constants scored "
                f"norm_mse={own:.3g}; after the evaluator re-fitted the law's "
                f"declared free parameters against this episode's own training "
                f"trajectories it scored {rec:.3g} ({ratio:.0f}x worse). "
                f"Declaring an already-measured constant as fittable is what "
                f"cost this episode.")
            lines.append("")
        elif ratio < 0.5:
            lines.append(
                f"  [parameter refit] refitting the declared free parameters "
                f"improved this episode: {own:.3g} -> {rec:.3g}.")
            lines.append("")

    # The judge's CRITIQUE is deliberately NOT shown to the teacher.
    #
    # It was surfaced here earlier, to fix "the teacher never learns why an
    # explanation failed". But the judge reads the world's ground-truth
    # `optimal_explanation` and rubric, so its critique is oracle feedback: on
    # oscillator it said the student "correctly identifies a cos(wt) with the
    # right w~pi/2, but crucially fails..." -- i.e. it states the hidden law.
    # Redaction masked "Laplacian" but not the trig, and the round-0 bank then
    # carried "velocity swings sinusoidally ... with a period", with 3/3 agents
    # naming the mechanism against 1/5 at baseline.
    #
    # The SCORE stays (it is legitimate evaluation signal, and reward already
    # depends on it). The critique does not: a scientist gets their result
    # judged, not a list of what they failed to discover. `expl_reasoning` is
    # still STORED on the trajectory for our own analysis -- it simply never
    # reaches the teacher.

    steps = trajectory.get("trajectory", [])
    if len(steps) > max_steps:
        lines.append(f"  (showing last {max_steps} of {len(steps)} steps)\n")
        steps = steps[-max_steps:]

    # Second line of defence for law-form redaction. The adapter redacts when a
    # trajectory is WRITTEN, but `run_batch_discovery` reuses existing files on
    # resume, so any trajectory written before law-form redaction existed (every
    # file from the 2-world smoke run, for one) would otherwise hand the teacher
    # "Yukawa-type exp(-r/L)/r" verbatim. Re-running the mask here is idempotent
    # on already-redacted text and catches the stale ones. `redacted: False`
    # (an explicit --no-redact run) is respected.
    scrub = redact_law_forms if trajectory.get("redacted", True) else (lambda x: x)

    for i, step in enumerate(steps, 1):
        if isinstance(step, dict):
            thought = step.get("thought", "")
            action = step.get("action", "")
            obs = step.get("observation", "")
        elif isinstance(step, (list, tuple)) and len(step) >= 3:
            thought, action, obs = step[0], step[1], step[2]
        else:
            continue
        thought, action, obs = scrub(thought), scrub(action), scrub(str(obs))
        if len(str(obs)) > 1400:
            obs = str(obs)[:1400] + "... [truncated]"
        lines.append(f"  Step {i}:")
        lines.append(f"    Thought: {thought}")
        lines.append(f"    Action: {action}")
        lines.append(f"    Observation: {obs}")
        lines.append("")

    return "\n".join(lines)


# ── Balanced contrast for the evolver ──────────────────────────────────

#: Share of the evolver's trajectory slots reserved for successes. Successes
#: stay the majority on purpose: the teacher is asked what separates solving
#: from failing, which needs enough of the solving side to generalise over.
SUCCESS_FRACTION = 0.6


def _split(trajectories: list[dict]) -> tuple[list[dict], list[dict]]:
    successes = [t for t in trajectories if t.get("success")]
    failures = [t for t in trajectories if not t.get("success")]
    return successes, failures


def make_evolver_block(max_trajs=None, success_fraction: float = SUCCESS_FRACTION):
    """Build a `format_trajectories_block` replacement with per-class quotas.

    `evolve_skill_bank` calls
    ``format_trajectories_block(successes + failures, max_trajs=5)`` -- one
    pooled list, sorted by reward, top 5. Round-1 distillation does the
    opposite: it formats successes and failures in two calls with separate caps
    (`max_success_trajs` / `max_failure_trajs`), so the contrast is guaranteed.
    The evolver has no such guarantee, and successes always outrank failures
    under any sane reward, so past ~5 episodes per world the failures are
    crowded out and the "contrast successes against failures" instruction is
    asking about evidence the teacher never received.

    At `--seeds-per-world 3` (the default) there are at most 3 trajectories and
    nothing is dropped either way, so this changes nothing for a default run --
    it makes larger seed counts safe, which is the point.

    `max_trajs=None` keeps the call site's value (5), so the block does not grow
    unless asked; the split is then 3 successes / 2 failures.

    Follows the precedent in `hexa_core.hexa_bridge.make_trajectory_formatter`,
    reimplemented here rather than imported so the two research arms stay
    independent and the log lines name this environment.
    """
    configured = max_trajs
    if configured is not None and configured < 1:
        raise ValueError(f"max_trajs must be >= 1, got {configured}")
    if not 0.0 < success_fraction < 1.0:
        raise ValueError(
            f"success_fraction must be in (0, 1), got {success_fraction}"
        )

    def format_trajectories_block(
        trajectories: list[dict], max_trajs: int = 5
    ) -> str:
        import skillrl.distillation.teacher_prompts as _tp

        if configured is not None:
            max_trajs = configured
        reward = _tp.compute_trajectory_reward

        successes, failures = _split(trajectories)
        success_slots = min(max(1, round(max_trajs * success_fraction)), max_trajs)
        fail_slots = max_trajs - success_slots

        chosen = (
            sorted(successes, key=reward, reverse=True)[:success_slots]
            # Ascending: the worst failures are the most instructive contrast.
            + sorted(failures, key=reward)[:fail_slots]
        )
        # Backfill from whichever class was short, so a round with no failures
        # still sends a full block rather than a truncated one.
        if len(chosen) < max_trajs:
            used = {id(t) for t in chosen}
            rest = [t for t in trajectories if id(t) not in used]
            chosen += sorted(rest, key=reward, reverse=True)[: max_trajs - len(chosen)]

        n_success = sum(1 for t in chosen if t.get("success"))
        print(
            f"  Evolver RECEIVED {len(chosen)}/{max_trajs} trajectories: "
            f"{n_success} success / {len(chosen) - n_success} failure "
            f"(of {len(successes)} / {len(failures)} available)"
            + ("   <-- no failures sent; mistakes will be unobserved"
               if chosen and n_success == len(chosen) and failures else "")
        )
        _tp.record_selection(chosen)
        return "\n\n".join(_tp.format_trajectory_for_teacher(t) for t in chosen)

    return format_trajectories_block


def format_skill_bank_for_teacher(bank, level_name: str) -> str:
    """Render the existing bank for the evolver, WITH the full mistake record.

    The stock `evolution_prompts.format_skill_bank_for_teacher` emits only a
    mistake's `description`, although its own docstring promises "mistakes with
    description and how_to_avoid". A mistake is stored as the triple
    (description, why_it_happens, how_to_avoid) -- the appendix's (delta, rho,
    alpha) -- so two thirds of every mistake record never reached the evolver.

    The consequence is specific: the evolver cannot see what corrective action a
    previous round already prescribed for a failure mode, so each round it
    re-derives one from the description alone and can contradict what the bank
    already says. Skills are rendered exactly as the stock function does; only
    the mistake block changes.
    """
    lines = []

    level_skills = bank.level_skills.get(level_name, [])
    if level_skills:
        lines.append("EXISTING LEVEL-SPECIFIC SKILLS:")
        for skill in level_skills:
            lines.append(f"  - [{skill.confidence * 100:.0f}%] {skill.title}")
            lines.append(f"      principle: {skill.principle}")
            if skill.when_to_apply:
                lines.append(f"      apply when: {skill.when_to_apply}")
            if getattr(skill, "example", ""):
                lines.append(f"      example: {skill.example}")
        lines.append("")

    level_mistakes = bank.level_mistakes.get(level_name, [])
    if level_mistakes:
        lines.append("EXISTING MISTAKES:")
        for mistake in level_mistakes:
            lines.append(f"  - {mistake.description}")
            why = getattr(mistake, "why_it_happens", "")
            how = getattr(mistake, "how_to_avoid", "")
            if why:
                lines.append(f"      why it happens: {why}")
            if how:
                lines.append(f"      how to avoid: {how}")
        lines.append("")

    if not level_skills and not level_mistakes:
        lines.append("(No existing skills or mistakes yet)")

    return "\n".join(lines)


def patch(evolve_max_trajs: int | None = None) -> None:
    """Swap discovery prompts + formatter into HeXA's distill / evolving_distill.

    Call AFTER reward.patch() so the formatter picks up the blended reward.

    `evolve_max_trajs` must match the contrast batch size the replay buffer
    builds. `evolve_skill_bank` hardcodes `max_trajs=5` at the call site, so
    without this a 6-trajectory batch is silently truncated back to 5 — and the
    trajectory dropped is whichever the reward ordering ranks last, which is
    exactly the replayed minority-class episode the batch was assembled to
    include.
    """
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving_distill
    import skillrl.distillation.teacher_prompts as _tp

    _distill.CONTRASTIVE_DISTILLATION_PROMPT = CONTRASTIVE_DISTILLATION_PROMPT
    _distill.COMMON_MISTAKES_PROMPT = COMMON_MISTAKES_PROMPT
    _evolving_distill.SKILL_BANK_EVOLUTION_PROMPT = SKILL_BANK_EVOLUTION_PROMPT
    # Imported by value at evolving_distill:27, so rebinding the module
    # global is what takes effect at its call site (line 70).
    if os.environ.get("HEXA_DISCOVERY_FULL_MISTAKES", "0") == "1":
        _evolving_distill.format_skill_bank_for_teacher = format_skill_bank_for_teacher

    # format_trajectories_block (imported by value into distill/evolving_distill)
    # calls the module-global format_trajectory_for_teacher at call time, so
    # overriding it here is picked up everywhere.
    _tp.format_trajectory_for_teacher = format_trajectory_for_teacher_discovery

    # Per-class quotas for the evolver only. `distill` holds its own by-value
    # binding and already formats successes and failures in separate capped
    # calls, so it is deliberately left alone.
    _evolving_distill.format_trajectories_block = make_evolver_block(
        max_trajs=evolve_max_trajs)
