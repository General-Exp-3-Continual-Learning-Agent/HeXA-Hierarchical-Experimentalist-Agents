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

from .adapter import ORBIT_SYSTEM_PROMPT_MULTI
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

#: What the agent is ALREADY told, quoted to the teacher so it stops re-deriving it.
#:
#: Measured on the first run: three of five banked skills were paraphrases of
#: *"Leverage information gathered from earlier episodes to succeed faster"* -- a
#: sentence already in the actor's system prompt in **both** arms. Roughly 600-900
#: injected tokens were therefore spent restating an instruction the agent receives
#: anyway, which is a mechanical reason for a null result rather than evidence about
#: whether banks can help.
#:
#: Braces are doubled because these prompts are consumed by ``str.format`` and the
#: quoted system prompt contains ``\boxed{}``.
_ALREADY_KNOWN = (
    "\n**What the agent is already told.** Every agent in this task receives this "
    "system prompt, in every episode, whether or not a skill bank exists:\n\n    \""
    + ORBIT_SYSTEM_PROMPT_MULTI.replace("{", "{{").replace("}", "}}")
    + "\"\n\n"
    "**The test, which is not a list of banned phrases.** Ask of each candidate "
    "skill: does an agent that has read the prompt above already know to do this? "
    "If yes, the skill is worth nothing -- it costs context in every future episode "
    "and changes no decision.\n\n"
    "Applying the test honestly matters more than avoiding particular wording. A "
    "banned-phrase list is easy to satisfy by rewording, and rewording is the "
    "failure this replaces: five skills that all said \"use episode 1's "
    "exploration\" in five different ways passed a phrase check while collectively "
    "restating one sentence of the prompt.\n\n"
    "So a skill about reusing earlier episodes is admissible, but ONLY if it says "
    "*mechanically* what to extract and how to reuse it -- which observation to "
    "record, in what form, and what decision it changes on a later turn. \"Leverage "
    "prior knowledge\" fails the test. \"Record which cells reported a wall on each "
    "side, and treat a cell with three walls as closed rather than re-entering it\" "
    "passes: the prompt does not say it, and it changes a specific move.\n\n"
    "Bank only what the prompt does NOT say -- concrete method the agent could not "
    "derive from the instruction alone: how to choose between two open directions, "
    "what to record from an observation and how to reuse it, when to abandon a "
    "direction, how to distinguish an unexplored cell from a revisited one."
)


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

""" + _ALREADY_KNOWN + """

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

""" + _ALREADY_KNOWN + """

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

**First, before revising anything: what is invariant here?**

The block above pools what the agents observed across every instance of this round.
Read it for regularities that hold in *every* maze rather than in one -- these are
what transfer, and they are the only thing worth banking. State each as a claim
about the environment, not as advice: "X is always true" or "X was true in N of N
instances", with the count that supports it. If a distribution above is tight, say
so; if it is spread, say that instead. Put these in "invariants".

A regularity you can name from the pooled counts is worth more than any number of
observations from a single transcript, because a single transcript cannot tell you
whether what you saw was the maze or the agent.

Revise the bank against this new evidence. Rules:

1. **Keep** skills the new instances still support. Keeping is not freezing: reassess
   the confidence against this round (rule 7), and leave the wording alone unless the
   new evidence actually sharpens it (rule 8).
2. **Update** a skill whose wording is right but whose trigger condition or reasoning the new evidence sharpens.
3. **Add** a skill for a pattern the new instances support and no existing entry covers. The bank holds at most {max_skills} skills and at most {max_mistakes} mistakes; count the entries above before deciding which case you are in.
   - **Below the cap, add it.** A free slot is not a reward for restraint, and an addition there displaces nothing, so the only question is whether the pattern is real and transferable -- not whether the existing skills are good. A round that ends with fewer entries than it started and nothing added has thrown away that round's evidence.
   - **At the cap**, an addition must displace something: name in "reasoning" the skill it displaces and why the evidence favours the newcomer. If nothing existing is weaker than the candidate, leave the bank as it is -- replacing a skill that is working with one that merely sounds better makes the bank worse.
4. **Remove** skills that are:
   - Redundant or subsumed by another skill.
   - Contradicted by the new instances.
   - Route-specific rather than transferable (see below) -- these are the most important to remove, and the easiest to miss.
   - Low confidence (< 0.3) AND not observed in the new instances.
5. Do not remove mistakes unless the new instances show they are no longer common.
6. **Every skill you keep or add must earn its slot by citing a turn.** In
   "changes_action", name one instance above and the turn number at which an agent
   following this skill would have chosen differently from what it actually chose.
   A skill that cannot be tied to a specific turn in a specific transcript is an
   aphorism: it will read as sound, cost context in every future episode, and change
   no decision. If you cannot find the turn, drop the skill rather than reword it.
7. **Confidence moves in both directions.** The whole bank above was in the prompt of
   every instance this round. Raise a skill only where an agent acted on it and solved;
   where an agent could have acted on it and the instance went unsolved, lower it by at
   least 0.1; otherwise leave it. Confidence is the retriever's ranking key, so a number
   that only ever rises makes a skill permanent however little it does. Put every
   movement in "confidence_changes".
8. **Keeping a skill means keeping its mechanism**: which observation to record, in what
   form, and which later decision it changes, plus any quantity the evidence established
   ("a cell reporting three walls"). Copy those through verbatim, and change a quantity
   only when an instance this round shows a different one. A retained principle
   compressed into a restatement of its own title is an aphorism carrying the old
   confidence, and no single round makes that loss visible.

""" + _ALREADY_KNOWN + """

""" + _ANTI_ROUTE_MEMORISATION + """

OUTPUT JSON OBJECT:
{{
  "invariants": [
    "<a regularity that held across instances, with the count supporting it>"
  ],
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<generic pattern -- NOT a move list, NOT coordinates>",
      "changes_action": "<instance id, turn N: what the agent chose vs what this skill would choose>",
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
  "confidence_changes": [
    {{"title": "<skill title>", "from": <old confidence>, "to": <new confidence>, "why": "<acted on in <instance id>, solved / in the prompt for unsolved <instance id> and changed nothing / no occasion to apply>"}}
  ],
  "reasoning": "<what was removed, what was added, why -- and for any invariant you stated but banked no skill from, name it and say why it is unactionable. If a quantity inside a retained skill changed, name the instance that justified it>"
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
    # `shortest_path_len` is deliberately NOT read here. It comes from the
    # config's path-length filter, not from anything the agent saw, so quoting it
    # to the teacher opens a privileged channel straight to
    # `audit.STRUCTURAL_REGULARITIES` #2 ("the path is always 7-8, so re-plan past
    # 8"). A skill written from that number would cite real `source_seeds` and look
    # earned while being ours. Observed turn counts carry the same information
    # legitimately -- see `hexa_orbit.observed`.

    lines: list[str] = []
    for episode in sorted(by_episode):
        solved = successes[episode - 1] if episode <= len(successes) else None
        steps = lengths[episode - 1] if episode <= len(lengths) else len(by_episode[episode])
        if solved:
            verdict = f"SOLVED in {steps} turns"
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

#: Which outcome class gets a contested slot. The order is not cosmetic: the
#: weights below can leave classes tied on the same fractional remainder, so this
#: constant decides the block's composition.
#:
#: **This ordering is endpoint-dependent, and has been set both ways.** The
#: measured facts, from the 243-instance paired evals and from an analysis of the
#: 488 instances that failed episode 1:
#:
#: * Optimising **episode 1** argues for "fast" first -- an episode-1 solve is the
#:   only direct evidence of what works on first contact. That ordering was tried
#:   (v5) and episode-1 accuracy did not move; across v3/v4/v5 rounds 1-2 produced
#:   identical episode-1 counts under three different banks, while solving
#:   different instances. Episode 1 looks close to a fixed-rate draw here.
#: * Optimising **episode 3 / solved-at-all** argues for "slow" first, and that is
#:   what is measured to work: v3 ran slow-heavy and is the only run with a
#:   positive effect (ep3 +6.2, solved-at-all +9.5 at p=0.013, concentrated in the
#:   clean instances). Among instances that failed episode 1, the ones later solved
#:   diverge from their own episode-1 path at turn 1.6 rather than 2.6 and cover
#:   4.9 new cells rather than 2.35 -- a cross-episode signal that lives entirely
#:   in the slow successes.
#:
#: "slow" first because episode 3 is the endpoint being optimised. Revisit this
#: constant, not silently, if that changes.
_PRIORITY = ("slow", "failed", "fast")

#: Slots per outcome class, before renormalising over the classes present.
#:
#: Back to v3's 0.4/0.4/0.2 -- 3 slow / 2 failed / 1 fast at the six slots of
#: DESIGN_PHASE3_MAZE §12. The equal thirds tried in v5 bought two fast successes
#: for the first-contact contrast at the cost of a slow success, and the
#: first-contact contrast did not move episode 1. On this env failures are the
#: majority class (45-70% of a round), so they stay equal to "slow": the
#: cross-episode story is only legible against instances that never got there.
_CLASS_WEIGHTS = {"slow": 0.4, "failed": 0.4, "fast": 0.2}


def _apportion(
    total: int, pools: dict[str, list], weights: dict[str, float]
) -> dict[str, int]:
    """Split *total* slots across the non-empty *pools* in proportion to *weights*.

    Largest-remainder apportionment, so the shares always sum to exactly *total*
    (a plain ``round`` per class does not, which is why the previous fixed-thirds
    version needed a ``max(1, total - a - b)`` correction term). Empty pools are
    dropped before renormalising, and every surviving pool gets at least one slot
    while slots remain. Shares are not capped at pool size here; the caller's
    redistribution step handles a pool that cannot fill its share.
    """
    present = [k for k in _PRIORITY if pools.get(k)] + [
        k for k in pools if k not in _PRIORITY and pools[k]
    ]
    if not present or total <= 0:
        return dict.fromkeys(pools, 0)

    scale = sum(weights[k] for k in present)
    ideal = {k: total * weights[k] / scale for k in present}
    share = {k: int(ideal[k]) for k in present}

    # Floor first, then hand out what floor discarded to the largest fractions.
    # Ties are common (6 slots at 0.4/0.4/0.2 leaves two classes on .4), and they
    # are broken by `_PRIORITY` order, not dict order -- see that constant for why
    # a tie must never go to "fast".
    leftover = total - sum(share.values())
    for k in sorted(present, key=lambda k: (-(ideal[k] - share[k]), present.index(k))):
        if leftover <= 0:
            break
        share[k] += 1
        leftover -= 1

    # A present class with a zero share would be silently unrepresented, which
    # defeats the point of bucketing at all -- take the slot from the largest.
    for k in present:
        if share[k] == 0:
            donor = max(present, key=lambda j: share[j])
            if share[donor] > 1:
                share[donor] -= 1
                share[k] = 1

    return {k: share.get(k, 0) for k in pools}


def select_by_reward_bin(
    trajectories: list, max_trajs: int, reward
) -> list:
    """Quota-per-outcome-class selection, prioritising cross-episode learners.

    **Why the stock selectors fail on this env.** ``"paired"`` looks for the same
    problem id among both successes and failures. In CORE a problem is attempted
    repeatedly so pairs exist; here an instance appears exactly once and is either
    solved or not, so no pair can ever form -- measured live as
    ``0 same-problem pair(s)`` -- and it silently degrades to ``"balanced"``.

    ``"balanced"`` then sorts successes by reward *descending*. Our reward ranks a
    fast success (+1.00) above a slow one (+0.60/+0.30), so the top slots are always
    episode-1 solves. Measured live: every success sent was ``+1.00``. The class the
    evolver therefore **never sees** is ``[False, True, True]`` -- failed episode 1,
    then solved -- which is exactly the cross-episode adaptation Design B exists to
    distill. The selector was excluding the evidence for the mechanism under test.

    This selector buckets by outcome class and fills a quota from each, so a
    "solved on retry" instance is guaranteed a slot whenever one exists:

    * **slow success** -- first solved on episode 2 or 3. The mechanism, demonstrated.
      Gets the largest share, because it is the only class that shows adaptation
      working.
    * **fast success** -- solved episode 1. Shows competent cold-start play.
    * **failure** -- never solved. Shows what does not work.

    Unfilled quota is redistributed rather than wasted, so a round with no slow
    successes still sends ``max_trajs`` trajectories.

    **Quotas are apportioned over the classes actually present**, not over all
    three. This matters because the round-0 seeding path
    (``distill.contrastive_distill``) splits the batch itself and calls this twice
    -- once with successes only, once with failures only. Under fixed thirds a
    successes-only call would reserve a slot for a class that cannot appear and
    hand it back to the fast solves via redistribution, which is the bias this
    function exists to remove. Renormalising over the non-empty pools gives
    successes-only 3 slow / 2 fast out of 5 instead of 2 slow / 3 fast.
    """
    slow, fast, failed = [], [], []
    for trajectory in trajectories:
        first = trajectory.get("first_success_episode")
        if first is None:
            failed.append(trajectory)
        elif first == 1:
            fast.append(trajectory)
        else:
            slow.append(trajectory)

    # Sorted so the most informative member of each class leads: the slowest slow
    # success (most adaptation to explain), the most efficient fast success, and the
    # failure that got furthest before failing.
    slow.sort(key=reward)
    fast.sort(key=reward, reverse=True)
    failed.sort(key=reward, reverse=True)

    pools = {"slow": slow, "fast": fast, "failed": failed}
    want = _apportion(max_trajs, pools, _CLASS_WEIGHTS)
    chosen: list = []
    for key in ("slow", "fast", "failed"):
        chosen.extend(pools[key][: want[key]])

    # Redistribute whatever the quotas could not fill -- to the classes that
    # still have members, in `_PRIORITY` order. Sorting the leftovers by reward
    # descending (the obvious thing, and what this used to do) hands every spare
    # slot to a +1.00 fast success by construction, because that is what "highest
    # reward" means here. Round 1 of v2 lost a slot that way: "slow" had a quota
    # of 3 and only 2 members, and the spare went to a third fast success rather
    # than to any of the 14 unused failures.
    if len(chosen) < max_trajs:
        used = {id(x) for x in chosen}
        for key in _PRIORITY:
            for candidate in pools[key]:
                if len(chosen) >= max_trajs:
                    break
                if id(candidate) not in used:
                    chosen.append(candidate)
                    used.add(id(candidate))
    return chosen[:max_trajs]


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
    reward_bin_selection: bool = False,
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

    # The three fixes ship together because they are one change: the evolver never
    # saw a cross-episode learner (selection), could not rank 46% of the data
    # (failure reward), and spent its output restating the system prompt (prompt).
    # Enabling one without the others leaves the mechanism half-fed.
    from .reward import set_graded_failures  # noqa: PLC0415

    set_graded_failures(reward_bin_selection)

    if reward_bin_selection:
        # Override hexa_bridge's selector with the outcome-class one. Bound in
        # *every* module that holds its own reference, exactly as `patch` does.
        # `_distill` is easy to miss and was: it does a module-level `from
        # teacher_prompts import format_trajectories_block`, so rebinding only
        # `_tp` and `_ev` left the round-0 seeding path on the stock
        # reward-descending selector. Measured cost of that omission: a seeding
        # batch holding six cross-episode learners sent the teacher one.
        import skillrl.distillation.evolving_distill as _ev

        def make_block(forced: int | None, label: str, aggregate: bool = False):
            """A block function selecting by outcome class.

            ``forced`` pins the trajectory count; ``None`` honours the caller's
            argument. The two paths need different rules and must not share one
            closure. ``evolving_distill`` hardcodes ``max_trajs=5`` at its call
            site, so the evolution binding has to *override* it to reach the 6 of
            DESIGN_PHASE3_MAZE §12 -- taking the caller's number there would have
            quietly cut the evolution block to 5. The seeding path in ``distill``
            already asks for exactly what its prompt budgets (5 successes, then 5
            failures, in two calls), so there the caller is right.
            """

            def block(trajectories: list, max_trajs: int = DEFAULT_MAX_TRAJS) -> str:
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
                rendered = "\n\n".join(
                    _tp.format_trajectory_for_teacher(x) for x in chosen
                )
                if not aggregate:
                    return rendered
                # Prepend the cross-instance aggregate over the *whole* round, not
                # over the `limit` transcripts selected from it. An invariant of the
                # generator is not visible in six case studies -- see
                # `hexa_orbit.observed` for why that is the shape of the problem and
                # for the privilege boundary the aggregate respects.
                from hexa_orbit import observed as _obs  # noqa: PLC0415

                return (
                    # Pooled, matching the v3 block exactly. The contrastive
                    # split (`observed.format_contrastive_for_teacher`) is kept in
                    # the module but not wired: v6 carried it plus rules 7-8 and
                    # came in at ep3 0.408 against v3's 0.594 and a no-bank
                    # baseline of 0.506 -- a 10-point regression below baseline on
                    # n=98. The teacher demonstrably *read* the contrast (it stated
                    # "agents re-enter cells 2.2 times per episode" as an
                    # invariant) and still banked nothing from it, while the extra
                    # instructions grew the prompt from ~8.2k to ~11.5k characters.
                    # Re-wire only against a run that isolates it.
                    _obs.format_for_teacher(trajectories)
                    + "\n\n=== INDIVIDUAL TRANSCRIPTS ===\n\n"
                    + rendered
                )

            return block

        max_trajs_override = [max_trajs]
        evolve_block = make_block(max_trajs_override[0], "Evolver", aggregate=True)
        _ev.format_trajectories_block = evolve_block
        _tp.format_trajectories_block = evolve_block
        _distill.format_trajectories_block = make_block(None, "Seeding")

    # Enforce rule 6 rather than trusting it: a skill whose cited turn contradicts
    # the transcript is dropped. See `hexa_orbit.citations` for the 58% figure.
    from hexa_orbit import citations as _citations  # noqa: PLC0415

    _evolving._CITATION_VERIFIER = _citations.filter_fabricated

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
