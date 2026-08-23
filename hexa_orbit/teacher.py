"""Teacher-facing rendering and prompts for Maze instances, plus the patch wiring.

Two jobs.

**1. Replace a formatter that would emit silent garbage.** ``skillrl``'s stock
``format_trajectory_for_teacher`` renders the action field as
``x=<char0>, y=<char1>, r=<char2>`` -- it assumes InterPhyre's ``[x, y, r]``
triple and produces nonsense on anything else. This is a live hazard, not an
absence: ``hexa_core`` hit exactly it on CORE and fixed it through the same seam.

The render is built around what a maze trajectory's information actually *is*:
the coordinate sequence and what it revealed about walls. Twenty-seven turns of
chain-of-thought would swamp the teacher's context and bury that, so
:func:`format_trajectory_for_teacher` emits a per-episode turn table --
``at`` / ``chose`` / ``result`` -- and reasoning is elided by default.

**2. Replace prompts written for single-turn attempts.** ``hexa_core``'s CORE
prompts describe "one prompt, one response, a verifier's verdict, once". A Maze
instance is three interactive episodes against a fixed hidden map, so the setting
paragraph, the reward explanation and the anti-memorisation clause all have to be
different. The output JSON contracts are kept byte-compatible with what
``distill``/``evolving_distill`` parse, since those parsers are shared code this
project does not modify.

The anti-memorisation clause is doing more work here than on CORE. The specific
degenerate outcome on this env is a "skill" that is really a remembered route,
which would show as transfer on the 19.1% of eval instances sharing a layout with
the training pool and nothing on the rest (RECON §10). The clause names that
failure explicitly, and :mod:`hexa_orbit.audit` checks for it independently rather
than trusting the prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

from .reward import reward_bin, trajectory_reward

logger = logging.getLogger(__name__)

#: Facts and scoring rules only -- no strategy. The same restraint
#: ``hexa_core.prompts_core_tasks`` applies: any strategy the teacher banks should
#: be one it read out of the trajectories, not out of the task description. In
#: particular this does NOT mention that the goal sits at the farthest reachable
#: cell, that shortest paths are 7-8, or that the grid is 6-7 per side -- those
#: are exactly the transferable regularities we want to see whether the evolver
#: can *discover* (DESIGN_PHASE3_MAZE §9), so stating them here would invalidate the
#: test.
MAZE_TASK_DESCRIPTION = (
    "Navigate a grid maze from a fixed start cell to a fixed goal cell. The agent sees "
    "only its own coordinates and what lies in the four adjacent cells (path, wall, or "
    "goal); it never sees the grid, the goal's location, or how far away the goal is. "
    "Each turn it emits one of up/down/left/right. Moving into a wall or emitting an "
    "unrecognised action consumes the turn and leaves the agent where it was. An episode "
    "ends when the agent reaches the goal (success) or exhausts its turn budget "
    "(failure). Each instance is played for three episodes against the same hidden maze, "
    "with the agent starting from the same cell each time and keeping the full transcript "
    "of earlier episodes in context."
)

_ORBIT_SETTING = """\
**Setting**: an interactive, partially observed navigation task played over \
repeated attempts. Each *instance* is one hidden maze played for three *episodes*. \
The maze, the start cell and the goal are identical in all three episodes; only the \
agent's knowledge changes, because it keeps the transcript of earlier episodes in \
context. So episode 1 is a cold start on an unknown map, and episodes 2 and 3 are \
attempts by an agent that has already probed it.

This means there are two distinct things a trajectory can teach:
- how to explore an unknown maze efficiently on a cold start, and
- how to convert what episode 1 revealed into a better episode 2 or 3.

The second is only visible because you are shown all three episodes at once. Look \
for it: an agent that re-walked a corridor episode 1 already proved was a dead end \
learned nothing, whatever its final outcome.

**Reward** is printed in each instance header and grades *when* the maze was first \
solved, not how many episodes succeeded:
- near +1.00 -- solved on episode 1, at or near the shortest possible number of \
moves. The cold-start competence that matters most.
- +0.70 to +1.00 -- solved on episode 1 with some wasted moves.
- +0.60 -- first solved on episode 2; +0.30 -- first solved on episode 3. \
Cross-episode learning worked, slowly.
- -0.50 -- never solved, but later episodes tried something genuinely different.
- -1.00 -- never solved, and every episode repeated the same move sequence. The \
agent failed to learn from its own experience. This is the worst outcome and the \
one most worth explaining.

So instances are **not** equally informative. Weigh by the reward, not by whether \
the maze was eventually solved."""

_ANTI_ROUTE_MEMORISATION = """\
Write every skill as guidance for a FRESH, unseen maze: a reusable method for \
exploring, for recording what has been learned, or for deciding when to abandon a \
direction. The bank is injected into the prompts of *other* instances, with \
different layouts, so a remembered route is worse than useless there.

Concretely, NEVER write a skill that contains:
- a sequence of specific moves ("go left, left, then up"),
- specific coordinates ("the goal is near (5, 2)", "avoid (3, 3)"),
- any claim about where the goal is in a particular maze.

A skill that would stop working if the walls were rearranged is not a skill."""

_SOURCE_SEEDS = """\
- **source_seeds**: the instance identifiers this skill was primarily derived \
from. Copy them verbatim (they are strings) from the `=== Instance <id>` headers; \
do not invent, renumber or abbreviate them. Include only instances whose behaviour \
directly demonstrates or motivates the skill."""


CONTRASTIVE_DISTILLATION_PROMPT_MAZE = """\
You are an expert analyst of agent behaviour in interactive environments, distilling it into concise, actionable skills.

""" + _ORBIT_SETTING + """

**Task: {level_name}**
{level_description}

Below are SOLVED and UNSOLVED instances. Each shows three episodes on one hidden maze, as a table of turns: the cell the agent stood in, the direction it chose, and what happened.

=== SOLVED INSTANCES ({n_successes}) ===
{success_block}

=== UNSOLVED INSTANCES ({n_failures}) ===
{failure_block}

---

Your task: by CONTRASTING the solved and unsolved instances, extract the KEY NAVIGATION AND EXPLORATION SKILLS that separate solving these mazes from failing them. Consider both dimensions: what did solved instances do differently on their *first* episode, and how did they use what episode 1 revealed? Let the trajectories, not your prior expectations about mazes, determine what works.

""" + _ANTI_ROUTE_MEMORISATION + """

For each skill, provide:
- **title**: Short name (3-7 words)
- **principle**: The insight (2-3 sentences). What was done, and why does it work?
- **when_to_apply**: Specific trigger condition (1 sentence)
""" + _SOURCE_SEEDS + """

Output a JSON array:
[
  {{
    "title": "<short name>",
    "principle": "<2-3 sentences>",
    "when_to_apply": "<condition>",
    "example": "<a generic pattern seen in the trajectories -- NOT a move list and NOT coordinates>",
    "source_seeds": ["<instance id>", "<instance id>"]
  }}
]
"""


COMMON_MISTAKES_PROMPT_MAZE = """\
You are an expert analyst of agent failures in interactive environments.

""" + _ORBIT_SETTING + """

**Task: {level_name}**
{level_description}

Below are UNSOLVED instances -- three episodes each, none of which reached the goal.

{failure_block}

---

Your task, in two parts.

1. **Mistakes.** Identify the recurring errors that cost these agents the maze. Pay particular attention to instances rewarded -1.00: those repeated the same move sequence across all three episodes, so the mistake is not just a bad route but a failure to use their own prior episodes. Distinguish that from agents that explored differently each time and still failed.

2. **Partial skills.** Some unsolved instances still contain good behaviour -- a sound probing pattern, a correct inference about a wall, a sensible decision to reverse out of a dead end. Extract those separately; they are worth banking even though the instance failed.

""" + _ANTI_ROUTE_MEMORISATION + """

Output a JSON object:
{{
  "mistakes": [
    {{
      "description": "<what the mistake is, 1 sentence>",
      "why_it_happens": "<why agents make it, 1 sentence>",
      "how_to_avoid": "<concrete actionable fix, 1-2 sentences>"
    }}
  ],
  "partial_skills": [
    {{
      "title": "<short name>",
      "principle": "<2-3 sentences>",
      "when_to_apply": "<condition>",
      "example": "<generic pattern -- NOT a move list, NOT coordinates>",
      "source_seeds": ["<instance id>"]
    }}
  ]
}}
"""


SKILL_BANK_EVOLUTION_PROMPT_MAZE = """\
You are curating a skill bank for an agent solving interactive maze navigation.

""" + _ORBIT_SETTING + """

**Task: {level_name}**
{level_description}

=== CURRENT SKILL BANK ===
{existing_skills_block}

=== NEW EVIDENCE: {n_successes} solved, {n_failures} unsolved instances ===
{new_trajectories_block}

---

Revise the bank against this new evidence. Rules:

1. **Keep** skills the new instances still support, preserving their confidence.
2. **Update** a skill whose wording is right but whose trigger condition or reasoning the new evidence sharpens.
3. **Add** a skill only for a pattern genuinely not covered by an existing entry. The bank holds at most {max_skills} skills and at most {max_mistakes} mistakes, so an addition beyond that budget must displace something: name in "reasoning" the existing skill it displaces and why the evidence favours the newcomer. **If no existing skill is weaker than the candidate, add nothing.** Replacing a skill that is working with one that merely sounds better makes the bank worse.
4. **Remove** skills that are:
   - Redundant or subsumed by another skill.
   - Contradicted by the new instances.
   - Route-specific rather than transferable (see below) -- these are the most important to remove, and the easiest to miss.
   - Low confidence (< 0.3) AND not observed in the new instances.
5. Do not remove mistakes unless the new instances show they are no longer common.

""" + _ANTI_ROUTE_MEMORISATION + """

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<generic pattern -- NOT a move list, NOT coordinates>",
      "source_seeds": ["<instance id>", "<instance id>"],
      "confidence": <float in [0.1, 1.0]>,
      "is_new": <true|false>
    }}
  ],
  "mistakes": [
    {{
      "description": "<what the mistake is>",
      "why_it_happens": "<why agents make this mistake>",
      "how_to_avoid": "<actionable fix>",
      "is_new": <true|false>
    }}
  ],
  "removed_skill_titles": ["<title of removed skill 1>", ...],
  "reasoning": "<brief explanation of key changes: what was removed, what was added, why>"
}}

Be concise but precise. Focus on transferable exploration strategy grounded in the trajectories.
"""


# ── rendering ────────────────────────────────────────────────────────────

def _cell(position: Optional[Sequence[int]]) -> str:
    if not position:
        return "?"
    return f"({position[0]},{position[1]})"


def render_episodes(
    trajectory: Mapping[str, Any],
    *,
    max_turns_per_episode: int = 9,
    include_reasoning: bool = False,
) -> list[str]:
    """The per-episode turn tables. Split out so tests can read them directly."""
    entries = trajectory.get("trajectory") or []
    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for entry in entries:
        by_episode.setdefault(int(entry.get("episode", 1)), []).append(entry)

    successes = trajectory.get("episode_successes") or []
    lengths = trajectory.get("episode_lengths") or []
    optimum = trajectory.get("shortest_path_len")

    lines: list[str] = []
    for episode in sorted(by_episode):
        solved = successes[episode - 1] if episode <= len(successes) else None
        steps = lengths[episode - 1] if episode <= len(lengths) else len(by_episode[episode])
        if solved:
            verdict = f"SOLVED in {steps} turns"
            if optimum:
                verdict += f" (shortest possible: {optimum})"
        else:
            verdict = f"FAILED ({steps} turns used)"
        lines.append(f"--- Episode {episode}: {verdict}")
        lines.append("  turn  at        chose   result")
        turns = by_episode[episode]
        shown = turns[:max_turns_per_episode]
        for entry in shown:
            lines.append(
                f"  {entry.get('turn', '?'):>4}  {_cell(entry.get('position')):<9} "
                f"{str(entry.get('action', '?'))[:7]:<7} {entry.get('result', '?')}"
            )
            if include_reasoning and entry.get("thought"):
                lines.append(f"        reasoning: {entry['thought'][:300]}")
        if len(turns) > len(shown):
            lines.append(f"  ... {len(turns) - len(shown)} further turns omitted")
    return lines


def format_trajectory_for_teacher(
    trajectory: Mapping[str, Any],
    max_steps: int = 9,
    *,
    include_reasoning: bool = False,
) -> str:
    """One instance as a compact three-episode report.

    ``max_steps`` is reinterpreted as **turns rendered per episode** rather than
    total turns kept from the end. The stock function keeps the last N steps of a
    single episode; here, truncating from the end would drop episode 1 entirely,
    which is the episode Design B is about. The parameter name is kept because
    ``hexa_bridge``'s selector calls this positionally.
    """
    reward = trajectory_reward(trajectory)
    first = trajectory.get("first_success_episode")
    header = (
        f"=== Instance {trajectory.get('seed', '?')} | "
        f"reward {reward:+.2f} ({reward_bin(trajectory)}) | "
        f"solved first on episode {first if first else 'never'} | "
        f"R={trajectory.get('success_count', '?')}/{trajectory.get('num_episodes', 3)}"
    )
    lines = [header]
    if trajectory.get("n_unparsed"):
        lines.append(
            f"  note: {trajectory['n_unparsed']} turn(s) produced no parseable action "
            f"and were spent doing nothing"
        )
    if not trajectory.get("adapted_across_episodes", True):
        lines.append(
            "  note: every episode repeated the same move sequence -- this agent did "
            "not use its own earlier episodes"
        )
    lines += render_episodes(
        trajectory, max_turns_per_episode=max_steps, include_reasoning=include_reasoning
    )
    return "\n".join(lines)


# ── patch wiring ─────────────────────────────────────────────────────────

#: What ``patch`` should show the evolver. "paired" puts a solved and an unsolved
#: instance adjacent, which is the contrast that teaches; 6 keeps the block inside
#: budget given each instance renders ~30 lines.
DEFAULT_TRAJECTORY_SELECTION = "paired"
DEFAULT_MAX_TRAJS = 6

_INSTALLED: dict[str, Any] = {}

#: Running tally of what ``source_seeds`` lookups did this process. Read with
#: :func:`confidence_stats`; reset with :func:`reset_confidence_stats`.
_CONFIDENCE_STATS: dict[str, int] = {
    "calls": 0,
    "citations": 0,
    "exact": 0,
    "repaired": 0,
    "unknown": 0,
    "fell_back": 0,
    "uncited": 0,
}


def confidence_stats() -> dict[str, int]:
    """What the confidence lookups did. ``fell_back`` is the one that matters.

    ``fell_back`` counts skills whose citations *all* failed to resolve, so their
    confidence is the round's mean reward rather than their own evidence. A
    nonzero value invalidates the confidence ordering for that round -- and since
    confidence is the retriever's ranking key, that changes what gets injected.
    """
    return dict(_CONFIDENCE_STATS)


def reset_confidence_stats() -> None:
    for key in _CONFIDENCE_STATS:
        _CONFIDENCE_STATS[key] = 0


def make_confidence_fn(stock: Any, *, repair: bool = True) -> Any:
    """Wrap ``_compute_initial_confidence`` so a bad citation cannot pass silently.

    The stock function looks ``source_seeds`` up in the round's trajectories and,
    when *nothing* matches, falls back to the mean reward over every trajectory in
    the round -- with no warning. Our instance ids are 16-hex strings, exactly the
    shape a model abbreviates, so that fallback is a live risk rather than a
    theoretical one (see :mod:`hexa_orbit.audit`).

    This wrapper changes behaviour in one direction only:

    * A citation that is a unique prefix or extension of a real id is **repaired**
      to that id, so a truncated hash yields the right evidence instead of the
      round average. Ambiguous matches are never guessed.
    * When at least one citation resolves, only the resolved ones are passed
      through -- identical to what the stock function computes, since it already
      filters to hits.
    * When *none* resolve, the original list is passed through unchanged, so the
      stock round-mean fallback still applies and no number silently moves. The
      event is counted and logged at WARNING instead.
    """

    def confidence(source_seeds: Any, all_trajs: Any, fallback_confidence: float = 0.5) -> float:
        from .audit import classify_citations  # local: keeps import graph acyclic

        _CONFIDENCE_STATS["calls"] += 1
        known = [t.get("seed") for t in (all_trajs or []) if t.get("seed") is not None]
        citations = classify_citations(source_seeds or [], known) if repair else []

        if not source_seeds:
            _CONFIDENCE_STATS["uncited"] += 1
            return stock(source_seeds, all_trajs, fallback_confidence=fallback_confidence)

        _CONFIDENCE_STATS["citations"] += len(citations)
        for citation in citations:
            _CONFIDENCE_STATS[citation.status] += 1

        resolved = [c.resolved for c in citations if c.resolved is not None]
        if resolved:
            repaired = [c for c in citations if c.status == "repaired"]
            if repaired:
                logger.info(
                    "repaired %d source_seeds citation(s): %s",
                    len(repaired),
                    [(c.cited, c.resolved) for c in repaired],
                )
            return stock(resolved, all_trajs, fallback_confidence=fallback_confidence)

        _CONFIDENCE_STATS["fell_back"] += 1
        logger.warning(
            "no source_seeds citation resolved (%s); confidence falls back to the "
            "round mean reward and is NOT evidence-based",
            [c.cited for c in citations],
        )
        return stock(source_seeds, all_trajs, fallback_confidence=fallback_confidence)

    return confidence


def install(
    *,
    trajectory_selection: str = DEFAULT_TRAJECTORY_SELECTION,
    max_trajs: int = DEFAULT_MAX_TRAJS,
    include_reasoning: bool = False,
    confidence_fn: Any = None,
) -> None:
    """Install the Maze reward, formatter and prompts into ``skillrl``.

    Layered on ``hexa_core.hexa_bridge.patch``, which owns the machinery (the
    reward and confidence rebindings across the three modules that each hold their
    own reference, and the trajectory selector). This function then overrides the
    formatter and the three prompts with the Maze versions -- the selector looks
    the formatter up on the module at call time, so a later override takes effect.

    Idempotent. Call before any ``distill_*`` or ``evolve_skill_bank`` invocation.
    """
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving
    import skillrl.distillation.teacher_prompts as _tp
    from hexa_core import hexa_bridge

    if confidence_fn is None:
        # The stock confidence, deliberately: M1's baseline-relative utility is
        # lower priority on this env, where the reward is a graded 5-level scale
        # with real variance rather than a coin flip (DESIGN_PHASE3_MAZE §3). Wrapped
        # so an unresolvable source_seeds citation is repaired or logged rather
        # than silently collapsing to the round's mean reward.
        stock = getattr(
            _distill._compute_initial_confidence, "__wrapped_stock__", None
        ) or _distill._compute_initial_confidence
        confidence_fn = make_confidence_fn(stock)
        confidence_fn.__wrapped_stock__ = stock  # idempotent re-install

    # `load_trajectories_fn` is not optional decoration: `evolve_skill_bank` takes a
    # *directory* and calls `load_trajectories` on it, and the stock loader globs
    # `trajectory_seed*.json`. Without this rebinding it finds nothing in a round
    # directory and every round from the second onward evolves from ZERO
    # trajectories -- silently, since `evolve_skill_bank` treats an empty round as
    # "no new evidence" rather than an error. That is the whole integration seam
    # (DESIGN_PHASE3_MAZE §12); `test_loop.test_evolver_receives_the_rounds_trajectories`
    # is what keeps it wired.
    from .trajectory import load_instance_trajectories  # noqa: PLC0415

    hexa_bridge.patch(
        reward_fn=trajectory_reward,
        confidence_fn=confidence_fn,
        trajectory_selection=trajectory_selection,
        max_trajs=max_trajs,
        task_descriptions={"maze": MAZE_TASK_DESCRIPTION},
        load_trajectories_fn=load_instance_trajectories,
    )

    def formatter(trajectory: Mapping[str, Any], max_steps: int = 9) -> str:
        return format_trajectory_for_teacher(
            trajectory, max_steps, include_reasoning=include_reasoning
        )

    bindings = [
        (_tp, "format_trajectory_for_teacher", formatter),
        (_distill, "CONTRASTIVE_DISTILLATION_PROMPT", CONTRASTIVE_DISTILLATION_PROMPT_MAZE),
        (_distill, "COMMON_MISTAKES_PROMPT", COMMON_MISTAKES_PROMPT_MAZE),
        (_evolving, "SKILL_BANK_EVOLUTION_PROMPT", SKILL_BANK_EVOLUTION_PROMPT_MAZE),
    ]
    for module, attr, value in bindings:
        setattr(module, attr, value)
    _INSTALLED["bindings"] = bindings


def is_installed() -> bool:
    return bool(_INSTALLED)
