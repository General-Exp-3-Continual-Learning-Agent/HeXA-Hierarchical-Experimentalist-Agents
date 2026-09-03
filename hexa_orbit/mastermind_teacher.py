"""Teacher prompts for Mastermind, and the wiring that installs them.

The Maze analogue is :mod:`hexa_orbit.teacher`, and the structure is deliberately
the same: a task description, a contrastive seeding prompt, a mistakes prompt, an
evolution prompt, and an ``install`` that rebinds the shared ``skillrl``
distillation code. Three things differ, all forced by the task:

* **What a turn is.** A Maze turn is a move from a cell; a Mastermind turn is a
  guess and the pegs it drew. The renderer below prints ``guess -> B/W`` because
  that pair is the entire observable signal.
* **What memorisation looks like.** Maze's failure mode was banking a route. Here
  it is banking a *code*: "try 3 7 4 5 first" would be worthless on any other
  instance and actively misleading. :data:`_ANTI_CODE_MEMORISATION` is the
  counterpart of Maze's anti-route section.
* **What the agent already knows.** The env states the rules in full every episode,
  including what black and white pegs mean. A skill restating them is worth
  nothing, exactly as a Maze skill restating "leverage earlier episodes" was.

The measured target this arm has to reach: a single hand-written consistency rule
("guess only codes consistent with every peg result so far") is worth **+14.5
points of episode-3 accuracy** on gpt-oss-120b (0.453 vs 0.309, paired n=256,
p=0.0005). That number is what tells us whether a *distilled* bank finds what a
human could write in one sentence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from .adapter import ORBIT_SYSTEM_PROMPT_MULTI

LEVEL = "mastermind"

MASTERMIND_TASK_DESCRIPTION = """\
The agent must find a hidden code of distinct digits. After each guess it is told
how many digits were right and in the right position ("black pegs") and how many
were right but misplaced ("white pegs"). It gets a fixed number of guesses per
episode, and THE CODE DOES NOT CHANGE BETWEEN EPISODES -- every guess and every peg
result from an earlier episode still constrains it."""

_ALREADY_KNOWN = (
    "\n**What the agent is already told.** Every agent receives this system prompt "
    "in every episode:\n\n    \""
    + ORBIT_SYSTEM_PROMPT_MULTI.replace("{", "{{").replace("}", "}}")
    + "\"\n\nand the environment restates the rules of the game -- code length, "
    "digit range, what black and white pegs mean, how many turns remain -- at the "
    "top of every episode.\n\n"
    "**The test.** Ask of each candidate skill: does an agent that has read those "
    "already know to do this? If yes, the skill is worth nothing; it costs context "
    "on every turn and changes no decision. Restating the peg definitions, or "
    "saying \"use the feedback\", or \"think carefully before guessing\", all fail "
    "this test.\n\n"
    "Bank only method the prompt does NOT supply: how to *use* a peg count to rule "
    "candidates in or out, which guess to make when several remain possible, what "
    "to carry from one episode into the next and in what form."
)

_ANTI_CODE_MEMORISATION = """\
Write every skill as guidance for a DIFFERENT hidden code. The bank is injected \
into the prompts of *other* instances, whose codes are unrelated to these, so a \
remembered code is worse than useless there.

Concretely, NEVER write a skill that contains:
- a specific code or digit sequence to try ("start with 1 2 3 4"),
- a claim about which digits are in the code,
- an opening guess justified by what worked on these instances rather than by a
  property that holds for any code.

A skill that would stop working if the hidden code changed is not a skill."""

_REWARD_SCALE = """\
**Instance reward** grades *when* the code was first cracked, not how many
episodes succeeded:
- near +1.00 -- solved on episode 1, in few guesses. The cold-start case.
- +0.60 -- first solved on episode 2. Cross-episode deduction worked.
- +0.30 -- first solved on episode 3. It worked, slowly.
- negative -- never solved. Graded by the best black-peg count reached, so a run
  that pinned three of four digits scores above one that pinned none.
- -1.00 -- never solved, and every episode repeated the same guesses. The agent
  learned nothing from its own feedback. This is the worst outcome and the one
  most worth explaining."""


def _render_action(action: Any) -> str:
    """A guess, or a marker when the turn produced none.

    An unparseable reply leaves the raw text in `parsed_action`, so truncating it
    to fifteen characters renders a guess column full of sentence fragments
    ("Awaiting feedba"). The Maze bank once distilled a skill about exactly such a
    fragment, mistaking a rendering artefact for agent behaviour.
    """
    text = str(action or "").strip()
    if not text:
        return "(none)"
    cleaned = text.replace(",", " ").split()
    if cleaned and all(part.isdigit() for part in cleaned):
        return " ".join(cleaned)[:15]
    return "(unparseable)"


def format_trajectory_for_teacher(
    trajectory: Mapping[str, Any],
    max_steps: int = 9,
    *,
    include_reasoning: bool = False,
) -> str:
    """One instance as a compact per-episode guess table.

    ``max_steps`` is turns rendered *per episode*, not total: truncating from the
    end would drop episode 1, which is where the constraints that later episodes
    must respect are established.
    """
    from .reward import reward_bin, trajectory_reward  # noqa: PLC0415

    reward = trajectory_reward(trajectory)
    first = trajectory.get("first_success_episode")
    lines = [
        f"=== Instance {trajectory.get('seed', '?')} | reward {reward:+.2f} "
        f"({reward_bin(trajectory)}) | first solved on episode "
        f"{first if first else 'never'} | "
        f"R={trajectory.get('success_count', '?')}/"
        f"{trajectory.get('num_episodes', 3)}"
    ]
    if not trajectory.get("adapted_across_episodes", True):
        lines.append(
            "  note: every episode repeated the same guesses -- this agent did not "
            "use its own feedback"
        )
    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for entry in trajectory.get("trajectory") or []:
        by_episode.setdefault(int(entry.get("episode", 1)), []).append(entry)

    successes = trajectory.get("episode_successes") or []
    for index in sorted(by_episode):
        solved = successes[index - 1] if index <= len(successes) else None
        turns = by_episode[index]
        lines.append(
            f"--- Episode {index}: {'SOLVED' if solved else 'not solved'} "
            f"({len(turns)} guesses)"
        )
        lines.append("  turn  guessed          result")
        for entry in turns[:max_steps]:
            lines.append(
                f"  {entry.get('turn', '?'):>4}  "
                f"{_render_action(entry.get('action')):<15}  "
                f"{entry.get('result', '?')}"
            )
            if include_reasoning and entry.get("thought"):
                lines.append(f"        reasoning: {entry['thought'][:300]}")
        if len(turns) > max_steps:
            lines.append(f"  ... {len(turns) - max_steps} further guesses omitted")
    return "\n".join(lines)


CONTRASTIVE_DISTILLATION_PROMPT_MASTERMIND = """\
You are analysing an agent playing Mastermind and distilling what separates its
successes from its failures into reusable skills.

**Task: {level_name}**
{level_description}

""" + _REWARD_SCALE + """

=== SOLVED ({n_successes} instances) ===
{success_block}

=== NOT SOLVED ({n_failures} instances) ===
{failure_block}

---

Compare them. Each block shows every guess and the pegs it drew, episode by
episode, for the same hidden code. What did the solvers do with the feedback they
had that the others did not?

Write the most general skill the comparison supports. If you find yourself writing
several skills that differ only in which feedback value sets them off, they are one
skill -- state the one that covers all of them. A skill that fires on one specific
value is worth recording only when no more general statement accounts for what you
observed.

""" + _ALREADY_KNOWN + """

""" + _ANTI_CODE_MEMORISATION + """

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<generic pattern -- NOT a specific code>",
      "changes_action": "<instance id, turn N: what the agent guessed vs what this skill would guess>",
      "source_seeds": ["<instance id>", "<instance id>"],
      "confidence": <float in [0.1, 1.0]>
    }}
  ]
}}
"""

COMMON_MISTAKES_PROMPT_MASTERMIND = """\
You are analysing failed Mastermind attempts to name the mistakes behind them.

**Task: {level_name}**
{level_description}

=== FAILED ATTEMPTS ===
{failure_block}

---

Name what these agents did wrong, in terms of the peg feedback they had and did
not use. A mistake worth recording is one an agent could avoid on a *different*
code.

""" + _ALREADY_KNOWN + """

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

SKILL_BANK_EVOLUTION_PROMPT_MASTERMIND = """\
You are curating a skill bank for an agent playing Mastermind.

**Task: {level_name}**
{level_description}

""" + _REWARD_SCALE + """

=== CURRENT SKILL BANK ===
{existing_skills_block}

=== NEW EVIDENCE: {n_successes} solved, {n_failures} unsolved instances ===
{new_trajectories_block}

---

**First: what is invariant here?** Read the block above for regularities that hold
for every code, not just these. State each as a claim, with the count supporting it.
Put these in "invariants".

**Then generalise before you write anything down.** Look at the invariants you just
listed. Several of them are usually consequences of one underlying law -- the law is
what belongs in the bank, not the consequences. For each candidate entry, ask: *is
this a special case of something more general I could state instead?* If it is,
state the general form. A bank of narrow cases fails on the case it does not list;
one law covers cases nobody enumerated, including the ones this evidence did not
happen to contain.

Concretely, before adding an entry:
  - If it fires only on one specific feedback value, one count, or one position,
    look for the rule that covers all values of that quantity and add that instead.
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
   a specific code, or low confidence and unobserved in the new instances.
   - **Consolidate.** If several entries already in the bank are special cases of a
     single law, replace all of them with one entry stating the law, and list the
     replaced titles in "removed_skill_titles". This frees slots and widens
     coverage at the same time; it is the preferred way to make room at the cap.
5. Do not remove mistakes unless the new instances show they are no longer common.
6. **Every skill must earn its slot by citing a turn.** In "changes_action", name
   one instance above and the turn at which an agent following this skill would
   have guessed differently. A skill that cannot be tied to a specific turn is an
   aphorism: it will read as sound and change no decision. A general law cites a
   turn the same way a narrow one does -- citing a single turn is evidence that it
   bites, not a claim that it bites only there, so do not narrow a rule to make its
   citation feel exact.

""" + _ALREADY_KNOWN + """

""" + _ANTI_CODE_MEMORISATION + """

OUTPUT JSON OBJECT:
{{
  "invariants": ["<a regularity that held, with the count supporting it>"],
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<generic pattern -- NOT a specific code>",
      "changes_action": "<instance id, turn N: what the agent guessed vs what this skill would guess>",
      "source_seeds": ["<instance id>", "<instance id>"],
      "confidence": <float in [0.1, 1.0]>,
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

_INSTALLED: dict[str, Any] = {}


def install(max_trajs: int = 6, reward_bin_selection: bool = False) -> None:
    """Point the shared distillation code at Mastermind. Idempotent.

    Mirrors :func:`hexa_orbit.teacher.install`. Rebinds the formatter and the three
    prompts in *every* module that holds its own reference -- ``distill`` does a
    module-level ``from teacher_prompts import format_trajectories_block``, so
    rebinding only the source module leaves the seeding path on the stock
    selector, which cost a Maze run six cross-episode learners in its seeding batch.
    """
    if _INSTALLED:
        return
    import skillrl.distillation.distill as _distill  # noqa: PLC0415
    import skillrl.distillation.evolving_distill as _evolving  # noqa: PLC0415
    import skillrl.distillation.teacher_prompts as _tp  # noqa: PLC0415

    from hexa_core import hexa_bridge  # noqa: PLC0415

    from . import citations as _citations  # noqa: PLC0415
    from . import loop as _loop  # noqa: PLC0415
    from . import reward as _reward  # noqa: PLC0415
    from .mastermind_trajectory import load_instance_trajectories  # noqa: PLC0415

    _loop.set_level(LEVEL)
    _reward.set_graded_failures(True)
    _evolving._CITATION_VERIFIER = _citations.filter_fabricated

    hexa_bridge.patch(
        reward_fn=_reward.trajectory_reward,
        max_trajs=max_trajs,
        task_descriptions={LEVEL: MASTERMIND_TASK_DESCRIPTION},
        load_trajectories_fn=load_instance_trajectories,
    )

    if reward_bin_selection:
        # Mastermind was left on the stock `paired` selector, which cannot work
        # here: `paired` looks for the same instance id among successes and
        # failures, and an instance is played exactly once, so it logs
        # "0 same-problem pair(s)" every round and degrades to sorting by reward.
        # Measured over rounds 0-2 of v2 that sent an identical profile every time
        # -- [1.0, 1.0, 1.0, 1.0, -0.8, -0.8] -- four episode-1 solves and two total
        # failures. Round 2 held 13 instances that failed episode 1 and then solved
        # it, the cross-episode adaptation Design B exists to distil, and sent none
        # of them. Bound in every module holding its own reference, as `patch` does.
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
                print(f"    {label} RECEIVED {len(chosen)}/{limit} "
                      f"[reward-bin]: {bins}")
                return "\n\n".join(
                    _tp.format_trajectory_for_teacher(x) for x in chosen
                )

            return block

        # No cross-instance aggregate: `hexa_orbit.observed` computes Maze
        # quantities (revisited cells, wall hits) that do not exist here, and the
        # one run that carried an aggregate regressed 10 points below baseline.
        evolve_block = make_block(max_trajs, "Evolver")
        _ev.format_trajectories_block = evolve_block
        _tp.format_trajectories_block = evolve_block
        _distill.format_trajectories_block = make_block(None, "Seeding")

    bindings = [
        (_tp, "format_trajectory_for_teacher", format_trajectory_for_teacher),
        (_distill, "CONTRASTIVE_DISTILLATION_PROMPT",
         CONTRASTIVE_DISTILLATION_PROMPT_MASTERMIND),
        (_distill, "COMMON_MISTAKES_PROMPT", COMMON_MISTAKES_PROMPT_MASTERMIND),
        (_evolving, "SKILL_BANK_EVOLUTION_PROMPT",
         SKILL_BANK_EVOLUTION_PROMPT_MASTERMIND),
    ]
    for module, attr, value in bindings:
        setattr(module, attr, value)
    _INSTALLED["bindings"] = bindings


def is_installed() -> bool:
    return bool(_INSTALLED)
