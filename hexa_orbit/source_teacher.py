"""Teacher prompts and installation for Design C's source tasks.

:mod:`hexa_orbit.mastermind_teacher` with the level parameterised. The prompt
*structure* is deliberately not redesigned: the Mastermind v3 arm (+17.6 episode-3,
p<0.0001) got there through three prompt revisions, and the two that mattered were
generic -- "write the covering law, not the enumeration" (v1's failure) and feeding
the evolver real slow-successes (v2's). Both are structural and are kept verbatim.
What is task-specific is the level description, what counts as memorisation, and
what the reward scale means, so those are the only pieces written per task.

Source banks are never a headline (see :mod:`hexa_orbit.source_tasks`), but they
are the *input* to Phase 3's synthesis, so their generality matters more here than
it would for a within-task result: a Minesweeper skill phrased around a specific
board is not merely useless on other boards, it is a fabricated correspondence
waiting to be re-grounded into a Maze skill.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .adapter import ORBIT_SYSTEM_PROMPT_MULTI

LEVELS = ("minesweeper", "wordle")

TASK_DESCRIPTIONS: dict[str, str] = {
    "minesweeper": """\
The agent must clear a 5x5 board holding 5 hidden mines by revealing every cell
that is not a mine. A revealed cell shows how many of its up-to-8 neighbours are
mines; revealing a cell with no adjacent mines opens its neighbours too. Revealing
a mine ends the episode immediately. The FIRST reveal of every episode is played
for the agent at the board centre, so the layout is the same in all three episodes
-- every count the agent saw in an earlier episode still describes this board.

VALID ACTIONS, exhaustively: `\\boxed{reveal R C}` and `\\boxed{flag R C}`, one
action per turn. There is NO terminate, quit, stop, done, submit or unflag action.
Anything else is rejected as "you did not provide a valid guess" and ENDS THE
EPISODE. Only the LAST `\\boxed{...}` in a response is read, so a skill that tells
the agent to emit several actions at once silently discards all but the last.
Never write a skill that invokes an action outside this list.""",
    "wordle": """\
The agent must find a hidden 5-letter word. After each guess every position is
marked G (right letter, right place), Y (letter is in the word, wrong place) or X
(letter absent). Repeating an earlier guess wastes the turn and returns nothing.
THE WORD DOES NOT CHANGE BETWEEN EPISODES -- every guess and every mark from an
earlier episode still constrains it.""",
}

_ALREADY_KNOWN_TAIL = {
    "minesweeper": (
        "and the environment restates the rules -- what the numbers mean, how to "
        "phrase a reveal, that dots are unrevealed -- at the top of every episode, "
        "and redraws the current board under every single observation.\n\n"
        "**The test.** Ask of each candidate skill: does an agent that has read "
        "those already know to do this? If yes, the skill is worth nothing; it "
        "costs context on every turn and changes no decision. Restating what a "
        "number means, or saying \"avoid mines\", or \"use logic\", all fail this "
        "test.\n\n"
        "Bank only method the prompt does NOT supply: how to turn a set of counts "
        "into a decision about a specific cell, which cell to probe when several "
        "are equally unknown, and what to carry from one episode into the next and "
        "in what form."
    ),
    "wordle": (
        "and the environment restates the rules -- word length, what G/Y/X mean, "
        "how many turns remain -- at the top of every episode, and appends the "
        "full guess history under every observation.\n\n"
        "**The test.** Ask of each candidate skill: does an agent that has read "
        "those already know to do this? If yes, the skill is worth nothing; it "
        "costs context on every turn and changes no decision. Restating what G, Y "
        "or X mean, or saying \"use the feedback\", or \"pick a common word\", all "
        "fail this test.\n\n"
        "Bank only method the prompt does NOT supply: how to turn a set of marks "
        "into a constraint on the next guess, which word to pick when several "
        "remain consistent, and what to carry from one episode into the next and "
        "in what form."
    ),
}

_ANTI_MEMORISATION = {
    "minesweeper": """\
Write every skill as guidance for a DIFFERENT board. The bank is injected into the \
prompts of *other* instances, whose mine layouts are unrelated to these, so a \
remembered layout is worse than useless there.

Concretely, NEVER write a skill that contains:
- a specific cell to reveal ("reveal (0, 4) next"),
- a claim about where the mines are on these boards,
- an opening or follow-up justified by what worked on these instances rather than
  by a property that holds for any layout.

A skill that would stop working if the mines moved is not a skill.""",
    "wordle": """\
Write every skill as guidance for a DIFFERENT secret word. The bank is injected \
into the prompts of *other* instances, whose words are unrelated to these, so a \
remembered word is worse than useless there.

Concretely, NEVER write a skill that contains:
- a specific word to guess ("open with CRANE"),
- a claim about which letters are in the word,
- an opening justified by what worked on these instances rather than by a property
  that holds for any word.

A skill that would stop working if the secret word changed is not a skill.""",
}

#: One scale, three tasks. Identical wording to Mastermind's beyond the noun,
#: because `reward.reward_from_parts` is literally the same function -- the bins
#: must mean the same thing or cross-task comparison of source banks is nonsense.
_REWARD_SCALE = {
    "minesweeper": """\
**Instance reward** grades *when* the board was first cleared, not how many
episodes succeeded:
- near +1.00 -- cleared on episode 1, in few reveals. The cold-start case.
- +0.60 -- first cleared on episode 2. Cross-episode deduction worked.
- +0.30 -- first cleared on episode 3. It worked, slowly.
- negative -- never cleared. Graded by the largest fraction of the safe board
  reached, so a run that left two cells hidden scores above one that hit a mine
  early.
- -1.00 -- never cleared, and every episode repeated the same reveals. The agent
  learned nothing from counts it had already seen. This is the worst outcome and
  the one most worth explaining.""",
    "wordle": """\
**Instance reward** grades *when* the word was first found, not how many episodes
succeeded:
- near +1.00 -- solved on episode 1, in few guesses. The cold-start case.
- +0.60 -- first solved on episode 2. Cross-episode deduction worked.
- +0.30 -- first solved on episode 3. It worked, slowly.
- negative -- never solved. Graded by the most positions ever turned green, so a
  run that pinned four letters scores above one that pinned none.
- -1.00 -- never solved, and every episode repeated the same guesses. The agent
  learned nothing from its own marks. This is the worst outcome and the one most
  worth explaining.""",
}

_ACTION_NOUN = {"minesweeper": "reveal", "wordle": "guess"}
_ACTION_NOUN_PL = {"minesweeper": "reveals", "wordle": "guesses"}
_SUCCESS_VERB = {"minesweeper": "CLEARED", "wordle": "SOLVED"}
_SIGNAL = {
    "minesweeper": "the counts it uncovered",
    "wordle": "the marks it drew",
}


def _already_known(level: str) -> str:
    return (
        "\n**What the agent is already told.** Every agent receives this system "
        "prompt in every episode:\n\n    \""
        + ORBIT_SYSTEM_PROMPT_MULTI.replace("{", "{{").replace("}", "}}")
        + "\"\n\n"
        + _ALREADY_KNOWN_TAIL[level]
    )


# ── trajectory rendering ───────────────────────────────────────────────

def _render_action(action: Any) -> str:
    text = str(action or "").strip()
    return text.replace("\\boxed{", "").rstrip("}") or "(none)"


def formatter_for(level: str):
    """A ``format_trajectory_for_teacher`` bound to *level*."""
    noun_pl = _ACTION_NOUN_PL[level]
    verb = _SUCCESS_VERB[level]
    label = "revealed" if level == "minesweeper" else "guessed"

    def format_trajectory_for_teacher(
        trajectory: Mapping[str, Any],
        max_steps: int = 10,
        *,
        include_reasoning: bool = False,
    ) -> str:
        """One instance as a compact per-episode table.

        ``max_steps`` is turns rendered *per episode*, not total: truncating from
        the end would drop episode 1, where the constraints later episodes must
        respect are established.
        """
        from .reward import reward_bin, trajectory_reward  # noqa: PLC0415

        reward = trajectory_reward(trajectory)
        first = trajectory.get("first_success_episode")
        lines = [
            f"=== Instance {trajectory.get('seed', '?')} | reward {reward:+.2f} "
            f"({reward_bin(trajectory)}) | first {verb.lower()} on episode "
            f"{first if first else 'never'} | "
            f"R={trajectory.get('success_count', '?')}/"
            f"{trajectory.get('num_episodes', 3)}"
        ]
        if not trajectory.get("adapted_across_episodes", True):
            lines.append(
                f"  note: every episode repeated the same {noun_pl} -- this agent "
                f"did not use {_SIGNAL[level]}"
            )
        by_episode: dict[int, list[Mapping[str, Any]]] = {}
        for entry in trajectory.get("trajectory") or []:
            by_episode.setdefault(int(entry.get("episode", 1)), []).append(entry)

        successes = trajectory.get("episode_successes") or []
        for index in sorted(by_episode):
            solved = successes[index - 1] if index <= len(successes) else None
            turns = by_episode[index]
            lines.append(
                f"--- Episode {index}: {verb if solved else 'not ' + verb.lower()} "
                f"({len(turns)} {noun_pl})"
            )
            lines.append(f"  turn  {label:<15}  result")
            for entry in turns[:max_steps]:
                lines.append(
                    f"  {entry.get('turn', '?'):>4}  "
                    f"{_render_action(entry.get('action')):<15}  "
                    f"{entry.get('result', '?')}"
                )
                if include_reasoning and entry.get("thought"):
                    lines.append(f"        reasoning: {entry['thought'][:300]}")
            if len(turns) > max_steps:
                lines.append(f"  ... {len(turns) - max_steps} further {noun_pl} omitted")
        return "\n".join(lines)

    return format_trajectory_for_teacher


# ── prompts ────────────────────────────────────────────────────────────

_SKILL_FIELDS = """\
      "title": "<short name of skill>",
      "principle": "<2-3 sentence insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<generic pattern -- NOT a specific {instance_noun}>",
      "changes_action": "<instance id, turn N: what the agent did vs what this skill would do>",
      "source_seeds": ["<instance id>", "<instance id>"],
      "confidence": <float in [0.1, 1.0]>"""


def contrastive_prompt(level: str) -> str:
    verb = _SUCCESS_VERB[level]
    return (
        f"""\
You are analysing an agent playing {level.capitalize()} and distilling what
separates its successes from its failures into reusable skills.

**Task: {{level_name}}**
{{level_description}}

"""
        + _REWARD_SCALE[level]
        + f"""

=== {verb} ({{n_successes}} instances) ===
{{success_block}}

=== NOT {verb} ({{n_failures}} instances) ===
{{failure_block}}

---

Compare them. Each block shows every {_ACTION_NOUN[level]} and what it produced,
episode by episode, for the same hidden instance. What did the solvers do with
{_SIGNAL[level]} that the others did not?

Write the most general skill the comparison supports. If you find yourself writing
several skills that differ only in which value sets them off, they are one skill --
state the one that covers all of them. A skill that fires on one specific value is
worth recording only when no more general statement accounts for what you observed.

"""
        + _already_known(level)
        + "\n\n"
        + _ANTI_MEMORISATION[level]
        + """

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
"""
        + _SKILL_FIELDS.format(instance_noun="board" if level == "minesweeper" else "word")
        + """
    }}
  ]
}}
"""
    )


def mistakes_prompt(level: str) -> str:
    return (
        f"""\
You are analysing failed {level.capitalize()} attempts to name the mistakes behind
them.

**Task: {{level_name}}**
{{level_description}}

=== FAILED ATTEMPTS ===
{{failure_block}}

---

Name what these agents did wrong, in terms of {_SIGNAL[level]} that they had and
did not use. A mistake worth recording is one an agent could avoid on a
*different* hidden instance.

"""
        + _already_known(level)
        + """

OUTPUT JSON OBJECT:
{{
  "mistakes": [
    {{
      "description": "<what the mistake is>",
      "why_it_happens": "<why agents make it>",
      "how_to_avoid": "<actionable fix>"
    }}
  ]
}}
"""
    )


def evolution_prompt(level: str) -> str:
    noun = _ACTION_NOUN[level]
    return (
        f"""\
You are curating a skill bank for an agent playing {level.capitalize()}.

**Task: {{level_name}}**
{{level_description}}

"""
        + _REWARD_SCALE[level]
        + """

=== CURRENT SKILL BANK ===
{existing_skills_block}

=== NEW EVIDENCE: {n_successes} solved, {n_failures} unsolved instances ===
{new_trajectories_block}

---

**First: what is invariant here?** Read the block above for regularities that hold
for every hidden instance, not just these. State each as a claim, with the count
supporting it. Put these in "invariants".

**Then generalise before you write anything down.** Look at the invariants you just
listed. Several of them are usually consequences of one underlying law -- the law is
what belongs in the bank, not the consequences. For each candidate entry, ask: *is
this a special case of something more general I could state instead?* If it is,
state the general form. A bank of narrow cases fails on the case it does not list;
one law covers cases nobody enumerated, including the ones this evidence did not
happen to contain.

Concretely, before adding an entry:
  - If it fires only on one specific value, count, or position, look for the rule
    that covers all values of that quantity and add that instead.
  - If two or more entries you are considering differ only in which value triggers
    them, they are one rule. Write the one.
  - Prefer an entry that applies at every turn to several that each apply at one.

Then revise the bank against this new evidence. Rules:

1. **Keep** skills the new instances still support, preserving their confidence.
2. **Update** a skill whose wording is right but whose trigger the evidence sharpens.
3. **Add** a skill for a pattern the new instances support and no existing entry
   covers. The bank holds at most {max_skills} skills and at most {max_mistakes}
   mistakes; count the entries above before deciding which case you are in.
   - **Below the cap, add it.** A free slot is not a reward for restraint, and an
     addition there displaces nothing.
   - **At the cap**, an addition must displace something: name in "reasoning" the
     skill it displaces and why the evidence favours the newcomer.
4. **Remove** skills that are redundant, contradicted by the new instances, tied to
   a specific hidden instance, or low confidence and unobserved in the new
   instances.
   - **Consolidate.** If several entries already in the bank are special cases of a
     single law, replace all of them with one entry stating the law, and list the
     replaced titles in "removed_skill_titles". This frees slots and widens
     coverage at the same time; it is the preferred way to make room at the cap.
5. Do not remove mistakes unless the new instances show they are no longer common.
6. **Every skill must earn its slot by citing a turn.** In "changes_action", name
   one instance above and the turn at which an agent following this skill would
   have """
        + f"{noun}ed"
        + """ differently. A skill that cannot be tied to a specific turn is
   an aphorism: it will read as sound and change no decision. A general law cites a
   turn the same way a narrow one does -- citing a single turn is evidence that it
   bites, not a claim that it bites only there, so do not narrow a rule to make its
   citation feel exact.

"""
        + _already_known(level)
        + "\n\n"
        + _ANTI_MEMORISATION[level]
        + """

OUTPUT JSON OBJECT:
{{
  "invariants": ["<a regularity that held, with the count supporting it>"],
  "skills": [
    {{
"""
        + _SKILL_FIELDS.format(instance_noun="board" if level == "minesweeper" else "word")
        + """,
      "is_new": <true|false>
    }}
  ],
  "mistakes": [
    {{
      "description": "<what the mistake is>",
      "why_it_happens": "<why agents make it>",
      "how_to_avoid": "<actionable fix>",
      "is_new": <true|false>
    }}
  ],
  "removed_skill_titles": ["<title>"],
  "reasoning": "<what was removed, what was added, why>"
}}
"""
    )


# ── installation ───────────────────────────────────────────────────────

_INSTALLED: dict[str, Any] = {}


def install(level: str, max_trajs: int = 6, reward_bin_selection: bool = True) -> None:
    """Point the shared distillation code at *level*. Idempotent per level.

    Mirrors :func:`hexa_orbit.mastermind_teacher.install`, including its one
    non-obvious requirement: rebind the formatter and the trajectory block in
    *every* module that holds its own reference, because ``distill`` does a
    module-level ``from teacher_prompts import format_trajectories_block`` and
    rebinding only the source module leaves the seeding path on the stock
    selector.

    ``reward_bin_selection`` defaults to True here rather than False. The stock
    ``paired`` selector needs two runs of the same instance to pair, and an
    instance is played once per round, so on Mastermind it logged "0 same-problem
    pair(s)" every round and degraded to sorting by reward -- which is what
    starved the v2 evolver of slow-successes. Same loop, same shape, same failure.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown source level {level!r}; known: {LEVELS}")
    if _INSTALLED.get("level") == level:
        return

    import skillrl.distillation.distill as _distill  # noqa: PLC0415
    import skillrl.distillation.evolving_distill as _evolving  # noqa: PLC0415
    import skillrl.distillation.teacher_prompts as _tp  # noqa: PLC0415

    from hexa_core import hexa_bridge  # noqa: PLC0415

    from . import citations as _citations  # noqa: PLC0415
    from . import loop as _loop  # noqa: PLC0415
    from . import reward as _reward  # noqa: PLC0415
    from .source_trajectory import loader_for  # noqa: PLC0415

    _loop.set_level(level)
    _reward.set_graded_failures(True)
    _evolving._CITATION_VERIFIER = _citations.filter_fabricated

    formatter = formatter_for(level)
    hexa_bridge.patch(
        reward_fn=_reward.trajectory_reward,
        max_trajs=max_trajs,
        task_descriptions={level: TASK_DESCRIPTIONS[level]},
        load_trajectories_fn=loader_for(level),
    )

    if reward_bin_selection:
        import skillrl.distillation.evolving_distill as _ev  # noqa: PLC0415

        from .teacher import reward_bin, select_by_reward_bin  # noqa: PLC0415

        def make_block(forced, label):
            def block(trajectories: list, max_trajs: int = 6) -> str:
                limit = forced if forced is not None else max_trajs
                chosen = select_by_reward_bin(
                    trajectories, limit, _tp.compute_trajectory_reward
                )
                _tp.record_selection(chosen)
                bins: dict[str, int] = {}
                for trajectory in chosen:
                    name = reward_bin(trajectory)
                    bins[name] = bins.get(name, 0) + 1
                print(f"    {label} RECEIVED {len(chosen)}/{limit} [reward-bin]: {bins}")
                return "\n\n".join(_tp.format_trajectory_for_teacher(x) for x in chosen)

            return block

        evolve_block = make_block(max_trajs, "Evolver")
        _ev.format_trajectories_block = evolve_block
        _tp.format_trajectories_block = evolve_block
        _distill.format_trajectories_block = make_block(None, "Seeding")

    bindings = [
        (_tp, "format_trajectory_for_teacher", formatter),
        (_distill, "CONTRASTIVE_DISTILLATION_PROMPT", contrastive_prompt(level)),
        (_distill, "COMMON_MISTAKES_PROMPT", mistakes_prompt(level)),
        (_evolving, "SKILL_BANK_EVOLUTION_PROMPT", evolution_prompt(level)),
    ]
    for module, attr, value in bindings:
        setattr(module, attr, value)
    _INSTALLED.clear()
    _INSTALLED.update({"level": level, "bindings": bindings})


def installed_level() -> Optional[str]:
    return _INSTALLED.get("level")
