"""Within-seed skill refinement at inference time.

Static inference injects a frozen bank K* once per episode. Within-seed
experimentation (Algorithm "Inference with a Learned Skill Bank", mode
within-seed) instead lets the evolver refine a SEED-LOCAL copy of the bank
after every turn of a single evaluation episode:

    K_{j,t+1} = pi_ev(K_{j,t}, h_{t+1}),    K_{j,0} = K*

The evolver sees only this seed's interaction history so far -- never other
evaluation seeds -- and the refined bank is discarded when the episode ends,
so no information passes between evaluation seeds and K* is never modified.

Design (v3), from two catapult pilots:
  * v1 (unguarded): the bank turned into a per-turn coordinate queue the actor
    executed verbatim, and the K* skills were churned out within ~10 turns.
  * v2 (K* removal guarded): deletion was blocked, but the evolver drove the
    confidence of exactly the K* skills behind the static solve down to 0.1
    after a few failures caused by an unrelated parameter, and promoted a
    mechanism it inferred from predict_first_contact normals alone to 0.8.
So in v3 K* is READ-ONLY within a seed: its text, confidence and membership
never change, and the evolver may only attach one short caveat per K* entry
(replaced, never accumulated). Everything learned about this instance lives
in at most max_new_skills / max_new_mistakes seed-local entries. Refinement is
triggered only by turns with a simulated OUTCOME (SIMULATION_TOOLS), not by
predictions; predictions still appear in the experiment log.

v4/v5 (actor side): the learned-skills block the actor sees never changes within
an episode (same text and numbering as static injection). Everything learned in
the episode goes in a separate, code-built notes block (format_episode_notes):
how to use it, coverage, closest result to the goal, anchoring / stall warnings,
a near-miss CLOSE line that replaces the stall warning, K* caveats by skill
number, and episode findings E1/E2.
"""

from __future__ import annotations

import ast
import copy
import re
from typing import NamedTuple

from skillrl.core.config import LEVEL_DESCRIPTIONS, TEACHER_MODEL
from skillrl.core.skill_bank import Mistake, Skill, SkillBank
from skillrl.distillation.distill import _parse_json_object, call_teacher


# Turns whose observation is a simulated outcome; these trigger refinement
# (the runner's refine_on="sim" gate). trace_green_ball runs the same physics
# (capped, stops once green is at rest) and reports Success -- leaving it out
# meant a catapult actor that probes with it got only 2 refinements in 25 turns.
SIMULATION_TOOLS = ("simulate_action", "simulate_with_trace", "simulate_partial", "trace_green_ball")
# Everything shown in the experiment log: outcomes plus predictions.
EXPERIMENT_TOOLS = SIMULATION_TOOLS + ("predict_first_contact",)

CAVEAT_MARK = "\n   Caveat (this episode): "
MAX_CAVEAT_CHARS = 200

# Seed-local skills rest on one episode; keep them from outranking K* on
# self-assigned confidence (v3 pilots: "direction is geometry-determined" at
# 0.92 pulled the actor off the lever entirely).
SEED_CONF_MAX = 0.7
SEED_CONF_MAX_WEAK = 0.5   # fewer than SEED_STRONG_SUPPORT supporting simulations
SEED_STRONG_SUPPORT = 3

# Episode-notes heuristics (computed in code from the actor's own simulations).
STALL_WINDOW = 3          # last N simulations ...
STALL_X_SPAN = 0.3        # ... within this x span ...
STALL_R_SPAN = 0.3        # ... and radius span, all failed -> stall warning
ANCHOR_MIN_SIMS = 4       # with at least this many failed simulations ...
ANCHOR_X_SPAN = 0.3       # ... and this share of them inside one x band of
ANCHOR_SHARE = 0.75       #     this width -> anchoring warning
NEAR_MISS_DIST = 1.25     # green ended within this distance of the goal object
INSENSITIVE_WINDOW = 3    # last N simulations with different parameters ...
INSENSITIVE_TOL = 0.05    # ... whose green end positions agree within this -> insensitive
RADIUS_RANGE = (0.1, 2.0)

# Titles of this shape are action queues or scratchpads, not skills.
_QUEUE_TITLE = re.compile(
    r"next action|next step|immediate|final action|try x\s*=|"
    r"map of|tried placements|exhausted",
    re.IGNORECASE,
)


WITHIN_SEED_REFINEMENT_PROMPT = """\
You are a physics teacher helping an agent WHILE it solves one puzzle instance.
The agent is mid-episode; what you return is shown to it before its next turn.
The agent decides its own actions -- your job is to add what THIS episode's
experiments have taught, not to choose its next move.

LEVEL: {level_name}
LEVEL DESCRIPTION: {level_description}
SUCCESS CONDITION: only actually achieving the level's goal counts. In a failed
simulation, how far an object moved or where it came to rest is NOT progress
unless it actually reached or nearly reached the goal object.

LEARNED SKILLS (K*, distilled from many other instances -- READ-ONLY; you cannot
change, re-weight or remove them):
{kstar_block}

EPISODE-SPECIFIC ENTRIES (yours; at most {max_new_skills} skills and {max_new_mistakes} mistake):
{seed_block}

EXPERIMENT LOG (computed from the episode, authoritative; "predicted" rows are
pre-simulation estimates, not outcomes):
{experiment_log}

THIS EPISODE SO FAR (seed {seed}, {n_turns} of at most {max_iterations} turns used):
{history_block}

YOUR TASK: return the episode-specific entries and any K* caveats.

RULES:
1. K* IS EVIDENCE FROM MANY INSTANCES. One episode's failures rarely refute it.
   When an experiment that followed a K* skill failed, first ask which OTHER
   parameter of that experiment (radius, height, position) differed from what
   the K* skills recommend -- that parameter is the more likely cause.
2. CAVEATS: you may attach one caveat (<= 30 words) to a K* entry, only when a
   SIMULATED outcome in this episode bears on it. It is replaced each time, so
   restate it fully. A caveat qualifies; it must not tell the agent to ignore
   the skill.
3. EPISODE-SPECIFIC SKILLS state a mechanism this episode revealed that K* does
   not cover, with the condition under which it applies. NEVER write the exact
   next placement to try, a queue of placements, or a list of what has been
   tried -- the experiment log records that and the agent chooses its actions.
4. ONE RESULT IS NOT A TREND, AND A PREDICTION IS NOT AN OUTCOME. Base a
   mechanism on at least two consistent SIMULATED outcomes. Never base one on
   predict_first_contact output alone. A placement that ended close to the goal
   ("dist" in the log) is evidence to refine around, not a rule to generalise.
5. PUSH EXPLORATION ACROSS PARAMETERS. If several failures share the same value
   of some parameter (see "values tried"), say which unvaried parameter's
   mechanism (e.g. radius sets mass and impact energy) is most likely to matter,
   and record small nudges inside a failing region as the mistake.
6. Mistakes must name a concrete corrective action and must not be about action
   validity only (out of bounds, overlap).
7. Never contradict K* in an episode-specific entry without two simulated
   outcomes that show it, and cite their turns.
8. Returning nothing new is correct when the latest turn taught nothing.

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "id": "<existing episode-specific id, or \\"new\\">",
      "title": "<short name>",
      "principle": "<2-3 sentence mechanism>",
      "when_to_apply": "<condition for applicability>",
      "supporting_turns": [<simulated turn numbers>],
      "confidence": <float in [0.1, 1.0]>
    }}
  ],
  "mistakes": [
    {{
      "id": "<existing episode-specific id, or \\"new\\">",
      "description": "<what the mistake is>",
      "why_it_happens": "<why agents make this mistake>",
      "how_to_avoid": "<actionable fix>"
    }}
  ],
  "kstar_caveats": [{{"id": "<K* id>", "caveat": "<<= 30 words>", "turns": [<turn numbers>]}}],
  "reasoning": "<one or two sentences citing turn numbers>"
}}
"""


# ── Rendering ──────────────────────────────────────────────────────────

def _step_parts(step):
    if isinstance(step, dict):
        return step.get("thought", ""), step.get("action", "") or "", step.get("observation", "") or ""
    return step[0], step[1] or "", step[2] or ""


def _parse_action(action: str) -> tuple[str, dict]:
    """'simulate_action({'x': 0.0, ...})' -> ('simulate_action', {...})."""
    m = re.match(r"\s*([a-z_]+)\((.*)\)\s*$", action, re.DOTALL)
    if not m:
        return action, {}
    try:
        args = ast.literal_eval(m.group(2)) if m.group(2).strip() else {}
    except (ValueError, SyntaxError):
        args = {}
    return m.group(1), args if isinstance(args, dict) else {}


def _outcome(tool: str, obs: str) -> str:
    """One-line factual summary of an experimental observation."""
    if obs.startswith("INVALID"):
        return "INVALID: " + obs.splitlines()[0][:120]
    if obs.startswith("REFUSED"):
        return "REFUSED (inside a ruled-out region; not simulated)"
    if tool == "predict_first_contact":
        fields = dict(re.findall(r"^(first_contact_object|approach_speed|contact_normal[^:]*):\s*(.+)$", obs, re.M))
        return "predicted " + ", ".join(f"{k.split(' ')[0]}={v.strip()}" for k, v in fields.items())
    head = "SUCCESS" if "SUCCESS" in obs or "Success: True" in obs else "FAILURE"
    positions = re.findall(r"(green_ball|blue_ball|blue_basket): pos=\(([^)]*)\)", obs)
    tail = ", ".join(f"{name} ended at ({pos})" for name, pos in positions[:2])
    trace = re.search(r"green_ball: start=\(([^)]*)\) end=\(([^)]*)\).*?peak_y=(\S+)", obs)
    if trace and not tail:  # trace_green_ball format
        tail = f"green_ball ended at ({trace.group(2)}), peak_y={trace.group(3)}"
    return f"{head}" + (f"; {tail}" if tail else "")


class Scene(NamedTuple):
    """What the outcome helpers need to know about this seed's scene.
    goal: the goal object's initial (x, y); goal_name: its object name in the
    simulation output (blue_ball, blue_basket); green: green's initial (x, y)."""
    goal: tuple | None
    goal_name: str = "blue_ball"
    green: tuple | None = None
    validate: object = None   # the tool's own placement check: (x, y, r) -> error str | None


def _as_scene(goal_pos) -> Scene:
    """Accepts a Scene or a legacy (x, y) goal position (catapult callers)."""
    if isinstance(goal_pos, Scene):
        return goal_pos
    return Scene(goal=goal_pos)


MOVE_TOL = 0.15   # green displacement below this counts as "did not move"


def _green_motion(obs: str, goal_pos) -> str | None:
    """'did not move' / 'moved left by 0.83 (dy -2.10)' from green's start to its final position."""
    sc = _as_scene(goal_pos)
    end = _green_end(obs)
    if end is None or sc.green is None:
        return None
    dx, dy = end[0] - sc.green[0], end[1] - sc.green[1]
    if (dx * dx + dy * dy) ** 0.5 < MOVE_TOL:
        return "did not move"
    side = "left" if dx < -MOVE_TOL else "right" if dx > MOVE_TOL else "straight down" if dy < 0 else "up"
    return f"moved {side} (dx {dx:+.2f}, dy {dy:+.2f})"


def _behaviour(obs: str, goal_pos) -> str | None:
    """Coarse class of green's motion, used by the evolver trigger."""
    m = _green_motion(obs, goal_pos)
    if m is None:
        return None
    if m == "did not move":
        return "still"
    side = m.split(" (")[0].replace("moved ", "")
    end, sc = _green_end(obs), _as_scene(goal_pos)
    fell = end[1] - sc.green[1] < -1.0
    return f"{side}{'+fell' if fell else ''}"


def _green_end(obs: str):
    """Final green position from a simulate_action / trace_green_ball observation."""
    m = re.search(r"green_ball: pos=\(([^,]+),\s*([^)]+)\)", obs) or \
        re.search(r"green_ball: start=\([^)]*\) end=\(([^,]+),\s*([^)]+)\)", obs)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _goal_distance(obs: str, goal_pos) -> float | None:
    """Distance from green's final position to the goal object. Uses the goal's
    final position when the observation reports it (it can move), else goal_pos."""
    g = _green_end(obs)
    if g is None or goal_pos is None:
        return None
    sc = _as_scene(goal_pos)
    if sc.goal is None:
        return None
    m = re.search(sc.goal_name + r": pos=\(([^,]+),\s*([^)]+)\)", obs)
    bx, by = (float(m.group(1)), float(m.group(2))) if m else sc.goal
    return ((g[0] - bx) ** 2 + (g[1] - by) ** 2) ** 0.5


def format_experiment_log(trajectory: list, goal_pos=None) -> str:
    """Deterministic table of every experiment so far plus the values tried
    per parameter -- the factual record the bank never has to keep."""
    rows, values = [], {}
    for turn, step in enumerate(trajectory, 1):
        _, action, obs = _step_parts(step)
        tool, args = _parse_action(action)
        if tool not in EXPERIMENT_TOOLS:
            continue
        if tool in SIMULATION_TOOLS:
            for k in ("x", "y", "radius"):
                if k in args:
                    values.setdefault(k, set()).add(args[k])
        arg_str = ", ".join(f"{k}={args[k]}" for k in ("x", "y", "radius") if k in args)
        d = _goal_distance(obs, goal_pos) if tool in SIMULATION_TOOLS else None
        dist = f", dist to goal {d:.2f}" if d is not None else ""
        mv = _green_motion(obs, goal_pos) if tool in SIMULATION_TOOLS else None
        dist += f", green {mv}" if mv else ""
        rows.append(f"  Turn {turn}: {tool}({arg_str}) -> {_outcome(tool, obs)}{dist}")
    if not rows:
        return "  (no experiments yet)"
    vals = "; ".join(f"{k} in {sorted(v)}" for k, v in values.items()) or "(none simulated yet)"
    n_fail = sum("-> FAILURE" in r for r in rows)
    return "\n".join(rows + ["", f"  Values tried in simulations so far: {vals}",
                             f"  Failed simulations so far: {n_fail}"])


def format_episode_history(trajectory: list, max_obs_chars: int = 800) -> str:
    """Render the in-progress (thought, action, observation) steps, numbered by turn."""
    lines = []
    for i, step in enumerate(trajectory, 1):
        thought, action, obs = _step_parts(step)
        if len(obs) > max_obs_chars:
            obs = obs[:max_obs_chars] + "... [truncated]"
        lines.append(f"  Turn {i}:")
        lines.append(f"    Thought: {thought}")
        lines.append(f"    Action: {action}")
        lines.append(f"    Observation: {obs}")
        lines.append("")
    return "\n".join(lines) if lines else "  (no turns yet)"


def _format_entries(skills: list[Skill], mistakes: list[Mistake]) -> str:
    lines = []
    for s in skills:
        lines.append(f"  [{s.skill_id}] [conf {s.confidence:.2f}] {s.title}")
        lines.append(f"      principle: {s.principle}")
        if s.when_to_apply:
            lines.append(f"      apply when: {s.when_to_apply}")
        if s.example:
            lines.append(f"      example: {s.example}")
    for m in mistakes:
        lines.append(f"  [{m.mistake_id}] MISTAKE: {m.description}")
        if m.why_it_happens:
            lines.append(f"      why it happens: {m.why_it_happens}")
        if m.how_to_avoid:
            lines.append(f"      how to avoid:   {m.how_to_avoid}")
    return "\n".join(lines) if lines else "  (none)"


# ── Refinement step ────────────────────────────────────────────────────

def refine_bank_within_seed(
    level_name: str,
    bank: SkillBank,
    seed: int,
    trajectory: list,
    max_iterations: int,
    kstar: SkillBank,
    max_new_skills: int = 2,
    max_new_mistakes: int = 1,
    teacher_model: str = TEACHER_MODEL,
    goal_pos=None,
) -> tuple[SkillBank, dict]:
    """One within-seed refinement step: K_{j,t+1} = pi_ev(K_{j,t}, h_{t+1}).

    kstar is the frozen bank the episode started from. Its entries for this
    level are always rebuilt verbatim from kstar (plus an optional caveat), so
    the evolver can only change the seed-local entries. Returns (new_bank,
    info). On a teacher or parse failure the input bank is returned unchanged
    (as a copy) and info["ok"] is False.
    """
    turn = len(trajectory)
    k_skills = kstar.level_skills.get(level_name, [])
    k_mistakes = kstar.level_mistakes.get(level_name, [])
    k_ids = {s.skill_id for s in k_skills} | {m.mistake_id for m in k_mistakes}
    seed_skills = [s for s in bank.level_skills.get(level_name, []) if s.skill_id not in k_ids]
    seed_mistakes = [m for m in bank.level_mistakes.get(level_name, []) if m.mistake_id not in k_ids]
    # Show K* with this turn's caveats so the evolver can restate or drop them.
    shown_k_skills = [s for s in bank.level_skills.get(level_name, []) if s.skill_id in k_ids]
    shown_k_mistakes = [m for m in bank.level_mistakes.get(level_name, []) if m.mistake_id in k_ids]

    prompt = WITHIN_SEED_REFINEMENT_PROMPT.format(
        level_name=level_name,
        level_description=LEVEL_DESCRIPTIONS.get(level_name, "No description available."),
        kstar_block=_format_entries(shown_k_skills, shown_k_mistakes),
        seed_block=_format_entries(seed_skills, seed_mistakes),
        experiment_log=format_experiment_log(trajectory, goal_pos),
        seed=seed,
        n_turns=turn,
        max_iterations=max_iterations,
        history_block=format_episode_history(trajectory),
        max_new_skills=max_new_skills,
        max_new_mistakes=max_new_mistakes,
    )

    parsed = _parse_json_object(call_teacher(prompt, model=teacher_model))
    if not parsed or not isinstance(parsed.get("skills", []), list):
        return copy.deepcopy(bank), {"turn": turn, "ok": False}

    sim_turns = {
        t for t, step in enumerate(trajectory, 1)
        if _parse_action(_step_parts(step)[1])[0] in SIMULATION_TOOLS
    }
    events = {"rejected_queue": [], "rejected_over_cap": [], "rejected_unsupported": [], "caveats": {}}

    # Caveats need a cited simulated outcome; capped in length; replace any previous one.
    for c in parsed.get("kstar_caveats", []) or []:
        if not isinstance(c, dict) or c.get("id") not in k_ids:
            continue
        text = " ".join(str(c.get("caveat", "")).split())[:MAX_CAVEAT_CHARS]
        cited = {t for t in c.get("turns", []) if isinstance(t, int)} & sim_turns
        if text and cited:
            events["caveats"][c["id"]] = text

    # Everything outside this level's K* entries is copied through unchanged.
    new_bank = SkillBank()
    new_bank.general_skills = copy.deepcopy(bank.general_skills)
    new_bank.common_mistakes = copy.deepcopy(bank.common_mistakes)
    for lvl, skills in bank.level_skills.items():
        if lvl != level_name:
            new_bank.level_skills[lvl] = copy.deepcopy(skills)
    for lvl, mistakes in bank.level_mistakes.items():
        if lvl != level_name:
            new_bank.level_mistakes[lvl] = copy.deepcopy(mistakes)

    # K*: verbatim from the frozen bank, plus this turn's caveat if any.
    for s in k_skills:
        s = copy.deepcopy(s)
        if s.skill_id in events["caveats"]:
            s.when_to_apply += CAVEAT_MARK + events["caveats"][s.skill_id]
        new_bank.add_skill(s)
    for m in k_mistakes:
        m = copy.deepcopy(m)
        if m.mistake_id in events["caveats"]:
            m.how_to_avoid += CAVEAT_MARK + events["caveats"][m.mistake_id]
        new_bank.add_mistake(m)

    # Seed-local skills: existing ones keep their id; all need simulated support.
    old_seed = {s.skill_id: s for s in seed_skills}
    n_seed = 0
    for i, raw in enumerate(parsed.get("skills", []) or []):
        if not isinstance(raw, dict) or raw.get("id") in k_ids:
            continue
        title = raw.get("title", "")
        if _QUEUE_TITLE.search(title):
            events["rejected_queue"].append(title)
            continue
        support = {t for t in raw.get("supporting_turns", []) if isinstance(t, int)} & sim_turns
        if not support:
            events["rejected_unsupported"].append(title)
            continue
        if n_seed >= max_new_skills:
            events["rejected_over_cap"].append(title)
            continue
        n_seed += 1
        conf_cap = SEED_CONF_MAX if len(support) >= SEED_STRONG_SUPPORT else SEED_CONF_MAX_WEAK
        sid = raw.get("id") if raw.get("id") in old_seed else f"{level_name[:3]}_ws{seed}_t{turn}_{i:03d}"
        new_bank.add_skill(Skill(
            skill_id=sid, title=title, principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""), example=raw.get("example", ""),
            source_level=level_name, source_type="within_seed",
            source_seeds=[seed], generation=-1,  # -1 = seed-local, never persisted
            confidence=max(0.1, min(conf_cap, float(raw.get("confidence", 0.5)))),
        ))

    old_seed_m = {m.mistake_id for m in seed_mistakes}
    n_seed_m = 0
    for i, raw in enumerate(parsed.get("mistakes", []) or []):
        if not isinstance(raw, dict) or raw.get("id") in k_ids:
            continue
        if n_seed_m >= max_new_mistakes:
            events["rejected_over_cap"].append(raw.get("description", "")[:60])
            continue
        n_seed_m += 1
        mid = raw.get("id") if raw.get("id") in old_seed_m else f"{level_name[:3]}_ws{seed}_t{turn}_err{i:03d}"
        new_bank.add_mistake(Mistake(
            mistake_id=mid,
            description=raw.get("description", ""),
            why_it_happens=raw.get("why_it_happens", ""),
            how_to_avoid=raw.get("how_to_avoid", ""),
            source_level=level_name, source_seeds=[seed], generation=-1,
        ))

    level_skills = new_bank.level_skills.get(level_name, [])
    return new_bank, {
        "turn": turn,
        "ok": True,
        "n_skills": len(level_skills),
        "n_mistakes": len(new_bank.level_mistakes.get(level_name, [])),
        "n_seed_skills": sum(1 for s in level_skills if s.skill_id not in k_ids),
        "reasoning": parsed.get("reasoning", ""),
        **events,
    }


# ── Free mode: no guards ────────────────────────────────────────────────
#
# --ws-free: the evolver may rewrite, re-weight or delete ANY entry of the
# seed-local bank (including the K* entries it started from) and add as many
# seed-specific skills and mistakes as it wants, with no support, cap or queue
# checks. K* itself (the file on disk) is still never modified, and the
# seed-local bank is still discarded at the end of the episode. The actor sees
# the refined bank as its skills block every turn.

WITHIN_SEED_FREE_PROMPT = """\
You are a physics teacher helping an agent WHILE it solves one puzzle instance.
The agent is mid-episode; the skill bank you return replaces the skills it sees
before its next turn.

LEVEL: {level_name}
LEVEL DESCRIPTION: {level_description}

CURRENT SKILL BANK FOR THIS EPISODE (started from skills learned on other
instances):
{bank_block}

TRANSFERRED ENTRIES (text is fixed; you may change their confidence or delete
them, but not reword them): {frozen_ids}

EXPERIMENT LOG (computed from the episode, authoritative; "predicted" rows are
pre-simulation estimates, not outcomes):
{experiment_log}

THIS EPISODE SO FAR (seed {seed}, {n_turns} of at most {max_iterations} turns used):
{history_block}

YOUR TASK: update the skill bank so it helps the agent solve THIS instance.
You have complete freedom:
- REFINE existing entries to fit what this episode has shown: rewrite any part
  of your episode-specific entries; for transferred entries, change only the
  confidence.
- ADD new skills and mistakes specific to this instance's situation: at most
  {max_new_skills} episode-specific skills and {max_new_mistakes} episode-specific mistakes in the bank
  at any time (it has {n_seed_skills} and {n_seed_mistakes} now). To record something new when
  full, rewrite or delete one of your existing episode-specific entries.
- DELETE entries that this episode has shown to be wrong or unhelpful here.
Entries you do not mention stay as they are.

RULES:
- INVALID PLACEMENTS: when the INVALID-PLACEMENT DIAGNOSIS below reports a
  streak, write (or rewrite) ONE episode-specific skill titled "Escape invalid
  placements". It must name the parameter the agent has kept fixed and give a
  concrete valid value or range for it. Take the numbers from the "Valid
  placements" lines of the diagnosis (they are computed exactly; do not compute
  your own) and choose the option that best fits the skills (form: "<parameter> was fixed at <value>
  in every invalid attempt; set it to <range> instead, because <reason from the
  positions>"). Give it confidence 1.0.
- If the agent cites a skill while repeating invalid placements, lower that
  skill's confidence below 0.5 so your fix is shown first.
- NO NEAR-COPIES: before adding a skill, check your episode-specific skills; if
  one already says the same thing, rewrite that one instead of adding another.

INVALID-PLACEMENT DIAGNOSIS (computed from the episode):
{invalid_block}
{regions_block}
OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "id": "<existing id to rewrite it, or \\"new\\">",
      "delete": <true to remove this existing entry, else false>,
      "title": "<short name>",
      "principle": "<mechanism or guidance>",
      "when_to_apply": "<condition>",
      "example": "<optional>",
      "confidence": <float in [0.1, 1.0]>
    }}
  ],
  "mistakes": [
    {{
      "id": "<existing id to rewrite it, or \\"new\\">",
      "delete": <true to remove this existing entry, else false>,
      "description": "<what the mistake is>",
      "why_it_happens": "<why>",
      "how_to_avoid": "<actionable fix>"
    }}
  ],
  "reasoning": "<one or two sentences citing turn numbers>"{regions_field}
}}
"""

REGIONS_BLOCK = """
RULED-OUT REGIONS: you may also rule out rectangles of the placement space
(x and y of the red ball) that this episode's simulations have shown to fail.
Write each region as "x": [low, high], "y": [low, high] with plain numbers
(the values in the output example below are placeholders, not advice).
The tools will REFUSE any simulation placed inside an active region, so the
agent must try elsewhere. Each region must contain at least {min_fail} FAILED
simulated placements from this episode (cite their turns), be at most
{max_span} wide in x and in y, and you may keep at most {max_regions}. The list you
return REPLACES the current one (return [] to clear; omit the key to keep it).
Current regions:
{current_regions}
"""
REGIONS_FIELD = """,
  "ruled_out": [{{"x": [1.0, 1.5], "y": [0.0, 0.8], "turns": [6, 8], "reason": "green not pushed"}}]"""
REGION_MIN_FAIL = 2
REGION_MAX_SPAN = 2.0
REGION_MAX = 3


def _in_region(x, y, r) -> bool:
    return r["x_min"] <= x <= r["x_max"] and (y is None or r["y_min"] <= y <= r["y_max"])


def format_regions(regions: list) -> str:
    if not regions:
        return "  (none)"
    return "\n".join(f"  R{i}: x in [{r['x_min']:.2f}, {r['x_max']:.2f}], y in [{r['y_min']:.2f}, "
                     f"{r['y_max']:.2f}] -- {r.get('reason', '')} (failed at turns {r['turns']})"
                     for i, r in enumerate(regions, 1))


def _validate_regions(raw_list, trajectory, goal_pos):
    """Keep evolver-proposed regions that contain >= REGION_MIN_FAIL cited failed
    simulations and are at most REGION_MAX_SPAN wide. Returns (kept, rejected)."""
    failed = {t: a for t, a, o, _ in _simulations(trajectory, goal_pos) if o.startswith("FAILURE")}
    kept, rejected = [], []
    for raw in raw_list or []:
        try:
            if "x" in raw:   # {"x": [lo, hi], "y": [lo, hi]}
                r = {"x_min": float(raw["x"][0]), "x_max": float(raw["x"][1]),
                     "y_min": float(raw["y"][0]), "y_max": float(raw["y"][1])}
            else:            # {"x_min": .., "x_max": .., "y_min": .., "y_max": ..}
                r = {k: float(raw[k]) for k in ("x_min", "x_max", "y_min", "y_max")}
        except (KeyError, IndexError, TypeError, ValueError):
            rejected.append("malformed"); continue
        if r["x_min"] > r["x_max"] or r["y_min"] > r["y_max"]:
            rejected.append("inverted bounds"); continue
        if r["x_max"] - r["x_min"] > REGION_MAX_SPAN or r["y_max"] - r["y_min"] > REGION_MAX_SPAN:
            rejected.append("too wide"); continue
        turns = sorted({t for t in raw.get("turns", []) if isinstance(t, int) and t in failed
                        and _in_region(float(failed[t]["x"]), float(failed[t]["y"]) if "y" in failed[t] else None, r)})
        if len(turns) < REGION_MIN_FAIL:
            rejected.append(f"only {len(turns)} cited failures inside"); continue
        if len(kept) >= REGION_MAX:
            rejected.append("over cap"); continue
        r["turns"] = turns
        r["reason"] = " ".join(str(raw.get("reason", "")).split())[:120]
        kept.append(r)
    return kept, rejected


def invalid_diagnosis(trajectory: list, goal_pos=None, min_streak: int = 2) -> str:
    """Code-computed summary of the current run of INVALID placements: which
    parameters stayed fixed, the ranges of the varied ones, and the error types.
    WSEV2 stuck episodes kept y at green's height and moved only x, alternating
    between overlapping green and a wall; the evolver's vague "place red above
    green" never got the actor to change y."""
    streak = []
    for turn in range(len(trajectory), 0, -1):
        _, action, obs = _step_parts(trajectory[turn - 1])
        tool, args = _parse_action(action)
        if tool not in SIMULATION_TOOLS:
            continue          # inspection turns don't break a streak
        if not obs.startswith("INVALID"):
            break
        streak.append((turn, args, obs))
    if len(streak) < min_streak:
        return "  (no invalid streak)"
    streak.reverse()
    errs = {}
    for _, _, obs in streak:
        for line in obs.splitlines()[1:]:
            line = line.strip(" -")
            if not line or line.startswith("The simulation"):
                continue
            key = re.sub(r"[-+]?\d+\.?\d*", "#", line.split(":")[0] if "Overlaps" in line else line)
            errs[key] = errs.get(key, 0) + 1
    sc = _as_scene(goal_pos)
    lines = [f"  The last {len(streak)} placements (turns {streak[0][0]}-{streak[-1][0]}) were all INVALID."]
    for k in ("x", "y", "radius"):
        vals = [float(a[k]) for _, a, _ in streak if isinstance(a.get(k), (int, float))]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        if hi - lo <= 0.05:
            note = ""
            if sc.green is not None and k in ("x", "y"):
                g = sc.green[0 if k == "x" else 1]
                if abs(lo - g) <= 0.05:
                    note = f" (= green ball's {k})"
            lines.append(f"  {k} was FIXED at {lo:g} in every invalid attempt{note}.")
        else:
            lines.append(f"  {k} varied over [{lo:g}, {hi:g}].")
    lines.append("  Errors: " + "; ".join(f"{k} x{n}" for k, n in sorted(errs.items(), key=lambda kv: -kv[1])))
    last = streak[-1][1]
    if sc.validate is not None and all(isinstance(last.get(k), (int, float)) for k in ("x", "y", "radius")):
        try:
            lines += valid_ranges(sc.validate, float(last["x"]), float(last["y"]), float(last["radius"]),
                                  green=sc.green)
        except Exception:  # never let the diagnosis break a refinement step
            pass
    return "\n".join(lines)


def _intervals(grid, ok, step):
    out, start = [], None
    for v, good in zip(grid, ok):
        if good and start is None:
            start = v
        if not good and start is not None:
            out.append((start, v - step)); start = None
    if start is not None:
        out.append((start, grid[-1]))
    return out


def valid_ranges(validate, x: float, y: float, r: float, step: float = 0.05,
                 green: tuple | None = None) -> list[str]:
    """Valid placements around the last attempt, computed with the tool's own
    placement check (pure geometry: walls, ball overlaps, platform). No
    simulation and no knowledge of what solves the level."""
    n = int(round(10 / step))
    grid = [round(-5 + i * step, 3) for i in range(n + 1)]
    fmt = lambda iv: ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in iv) or "NONE"
    lines = [f"  Valid placements with radius {r:g} (computed with the tool's placement check):",
             f"    keeping y = {y:g}: x in {fmt(_intervals(grid, [validate(v, y, r) is None for v in grid], step))}",
             f"    keeping x = {x:g}: y in {fmt(_intervals(grid, [validate(x, v, r) is None for v in grid], step))}"]
    if green is not None:
        gx = round(green[0], 2)
        lines.append(f"    at the green ball's x = {gx:g}: y in "
                     f"{fmt(_intervals(grid, [validate(gx, v, r) is None for v in grid], step))}")
    best = None
    for gx in grid[::2]:
        for gy in grid[::2]:
            d = (gx - x) ** 2 + (gy - y) ** 2
            if (best is None or d < best[0]) and validate(gx, gy, r) is None:
                best = (d, gx, gy)
    if best:
        lines.append(f"    nearest valid placement to the last attempt: (x={best[1]:.2f}, y={best[2]:.2f})")
    return lines


def _parse_reply(text: str) -> dict:
    """_parse_json_object, then one lenient retry for common 7B slips
    (// comments, trailing commas, Python literals)."""
    parsed = _parse_json_object(text or "")
    if parsed:
        return parsed
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    body = re.sub(r"//[^\n]*", "", m.group(0))
    body = re.sub(r",\s*([}\]])", r"\1", body)
    for cand in (body, body.replace("True", "true").replace("False", "false").replace("None", "null")):
        try:
            import json as _json
            out = _json.loads(cand)
            return out if isinstance(out, dict) else {}
        except ValueError:
            pass
    try:
        out = ast.literal_eval(body.replace("true", "True").replace("false", "False").replace("null", "None"))
        return out if isinstance(out, dict) else {}
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        # TypeError: a reply like {"x": {{...}}} parses as a set of dicts.
        return {}


def refine_bank_within_seed_free(
    level_name: str,
    bank: SkillBank,
    seed: int,
    trajectory: list,
    max_iterations: int,
    teacher_model: str = TEACHER_MODEL,
    goal_pos=None,
    max_new_skills: int = 3,
    max_new_mistakes: int = 1,
    frozen_ids: frozenset = frozenset(),
    regions: list | None = None,
    require_numbers: bool = False,
) -> tuple[SkillBank, dict]:
    """One within-seed refinement step with a free evolver (see WITHIN_SEED_FREE_PROMPT):
    any entry may be rewritten or deleted; the only limit is the number of
    episode-specific entries in the bank at once (new ones over the cap are dropped)
    and the TEXT of entries in frozen_ids (the transferred K* entries) is
    read-only: their confidence can change and they can be deleted.
    regions (a list, or None to disable): current ruled-out regions; the
    evolver's validated replacement list is returned in info["ruled_out"]
    (absent when the evolver did not return the key). (Smoke test
    job 65333580: with K* text editable, the 7B evolver's first call flipped the
    FIP push-direction rule.)
    On a teacher or parse failure the input bank is returned unchanged (as a
    copy) and info["ok"] is False."""
    turn = len(trajectory)
    cur_skills = bank.level_skills.get(level_name, [])
    cur_mistakes = bank.level_mistakes.get(level_name, [])
    prompt = WITHIN_SEED_FREE_PROMPT.format(
        level_name=level_name,
        level_description=LEVEL_DESCRIPTIONS.get(level_name, "No description available."),
        bank_block=_format_entries(cur_skills, cur_mistakes),
        experiment_log=format_experiment_log(trajectory, goal_pos),
        seed=seed,
        n_turns=turn,
        max_iterations=max_iterations,
        history_block=format_episode_history(trajectory),
        max_new_skills=max_new_skills,
        max_new_mistakes=max_new_mistakes,
        n_seed_skills=sum(1 for s in cur_skills if s.generation == -1),
        n_seed_mistakes=sum(1 for m in cur_mistakes if m.generation == -1),
        frozen_ids=", ".join(sorted(i for i in frozen_ids
                                    if i in {x.skill_id for x in cur_skills} | {x.mistake_id for x in cur_mistakes}))
                   or "(none left)",
        invalid_block=invalid_diagnosis(trajectory, goal_pos),
        regions_block="" if regions is None else REGIONS_BLOCK.format(
            min_fail=REGION_MIN_FAIL, max_span=REGION_MAX_SPAN, max_regions=REGION_MAX,
            current_regions=format_regions(regions)),
        regions_field="" if regions is None else REGIONS_FIELD,
    )
    reply = call_teacher(prompt, model=teacher_model)
    parsed = _parse_reply(reply)
    numbers_retry = None
    if require_numbers and parsed:
        # During an invalid streak the update must quote numbers from the
        # computed "Valid placements" lines (WSEV3 Two Body: most updates were a
        # number-free "place red above green"). One retry with feedback.
        valid_nums = _valid_line_numbers(invalid_diagnosis(trajectory, goal_pos))
        if valid_nums and not _quotes_numbers(parsed, valid_nums):
            retry = call_teacher(prompt + NUMBERS_FEEDBACK.format(previous=(reply or "")[-2000:]),
                                 model=teacher_model)
            parsed_retry = _parse_reply(retry)
            numbers_retry = bool(parsed_retry) and _quotes_numbers(parsed_retry, valid_nums)
            if numbers_retry:
                parsed, reply = parsed_retry, retry
    if not parsed or not isinstance(parsed.get("skills", []), list):
        # keep the raw reply so failures can be diagnosed
        return copy.deepcopy(bank), {"turn": turn, "ok": False, "raw_reply": (reply or "")[-1500:]}

    new_bank = copy.deepcopy(bank)
    skills = {s.skill_id: s for s in new_bank.level_skills.get(level_name, [])}
    mistakes = {m.mistake_id: m for m in new_bank.level_mistakes.get(level_name, [])}
    events = {"edited": [], "added": [], "deleted": [], "rejected_over_cap": [],
              "frozen_text_ignored": []}

    def _conf(raw, default):
        try:
            return max(0.1, min(1.0, float(raw.get("confidence", default))))
        except (TypeError, ValueError):
            return default

    # Apply deletions first, so a delete + add in the same reply stays under the cap.
    for key, pool in (("skills", skills), ("mistakes", mistakes)):
        for raw in parsed.get(key, []) or []:
            if isinstance(raw, dict) and raw.get("delete") is True and raw.get("id") in pool:
                del pool[raw["id"]]; events["deleted"].append(raw["id"])

    def _n_seed(pool):
        return sum(1 for e in pool.values() if e.generation == -1)

    for i, raw in enumerate(parsed.get("skills", []) or []):
        if not isinstance(raw, dict) or raw.get("delete") is True:
            continue
        sid = raw.get("id")
        if sid in skills:
            s = skills[sid]
            changed = False
            for f in ("title", "principle", "when_to_apply", "example"):
                if isinstance(raw.get(f), str) and raw[f].strip() and raw[f] != getattr(s, f):
                    if sid in frozen_ids:
                        events["frozen_text_ignored"].append(f"{sid}.{f}")
                        continue
                    setattr(s, f, raw[f]); changed = True
            new_conf = _conf(raw, s.confidence)
            if new_conf != s.confidence:
                s.confidence = new_conf; changed = True
            if changed:
                events["edited"].append(sid)
        elif raw.get("title") or raw.get("principle"):
            if _n_seed(skills) >= max_new_skills:
                events["rejected_over_cap"].append(raw.get("title", "")[:60]); continue
            nid = f"{level_name[:3]}_ws{seed}_t{turn}_{i:03d}"
            skills[nid] = Skill(
                skill_id=nid, title=raw.get("title", ""), principle=raw.get("principle", ""),
                when_to_apply=raw.get("when_to_apply", ""), example=raw.get("example", ""),
                source_level=level_name, source_type="within_seed",
                source_seeds=[seed], generation=-1, confidence=_conf(raw, 0.5),
            )
            events["added"].append(nid)
    for i, raw in enumerate(parsed.get("mistakes", []) or []):
        if not isinstance(raw, dict) or raw.get("delete") is True:
            continue
        mid = raw.get("id")
        if mid in mistakes:
            m = mistakes[mid]
            if mid in frozen_ids:
                continue
            changed = False
            for f in ("description", "why_it_happens", "how_to_avoid"):
                if isinstance(raw.get(f), str) and raw[f].strip() and raw[f] != getattr(m, f):
                    setattr(m, f, raw[f]); changed = True
            if changed:
                events["edited"].append(mid)
        elif raw.get("description"):
            if _n_seed(mistakes) >= max_new_mistakes:
                events["rejected_over_cap"].append(raw.get("description", "")[:60]); continue
            nid = f"{level_name[:3]}_ws{seed}_t{turn}_err{i:03d}"
            mistakes[nid] = Mistake(
                mistake_id=nid, description=raw.get("description", ""),
                why_it_happens=raw.get("why_it_happens", ""),
                how_to_avoid=raw.get("how_to_avoid", ""),
                source_level=level_name, source_seeds=[seed], generation=-1,
            )
            events["added"].append(nid)

    if regions is not None and isinstance(parsed.get("ruled_out"), list):
        kept, rejected = _validate_regions(parsed["ruled_out"], trajectory, goal_pos)
        events["ruled_out"] = kept
        events["regions_rejected"] = rejected

    new_bank.level_skills[level_name] = list(skills.values())
    # Episode-specific mistakes first: mistake injection takes the first N in order.
    new_bank.level_mistakes[level_name] = sorted(mistakes.values(), key=lambda m: m.generation != -1)
    return new_bank, {
        "turn": turn,
        "ok": True,
        "free": True,
        "n_skills": len(skills),
        "n_mistakes": len(mistakes),
        "n_seed_skills": sum(1 for s in skills.values() if s.source_type == "within_seed"),
        "reasoning": parsed.get("reasoning", ""),
        "numbers_retry": numbers_retry,
        **events,
    }


NUMBERS_FEEDBACK = """

YOUR PREVIOUS REPLY:
{previous}

PROBLEM: the agent is in an invalid-placement streak, but no skill in your reply
gives a concrete range taken from the "Valid placements" lines of the
diagnosis. Return the full JSON again. Your "Escape invalid placements" skill
must quote one of those ranges with its numbers exactly as written there.
"""


def _valid_line_numbers(diag: str) -> set[str]:
    """Numbers (as written, 2 decimals) on the diagnosis's valid-placement lines."""
    nums = set()
    for line in diag.splitlines():
        if line.strip().startswith(("keeping", "at the green", "nearest valid")):
            nums.update(re.findall(r"-?\d+\.\d{2}", line.split(":", 1)[1] if ":" in line else line))
    return nums


def _quotes_numbers(parsed: dict, valid_nums: set[str]) -> bool:
    """True if any returned skill's text contains a number from the valid lines."""
    for raw in parsed.get("skills", []) or []:
        if not isinstance(raw, dict) or raw.get("delete") is True:
            continue
        text = " ".join(str(raw.get(k, "")) for k in ("title", "principle", "when_to_apply", "example"))
        found = set(re.findall(r"-?\d+\.\d+", text))
        found |= {f"{float(x):.2f}" for x in found}
        if found & valid_nums:
            return True
    return False


# ── When to call the evolver (refine_on="event") ───────────────────────
#
# Decided in code after every turn from the simulation output alone. 15-seed
# WSFREE3 run: 54% of calls followed an INVALID placement, 13% of simulations
# repeated the previous one, and only ~12% of calls changed seed-skill text.

TRIG_REPEAT_TOL = 0.05     # placement within this of an earlier one (x, y, r) = repeat
TRIG_BEST_GAIN = 0.2       # new closest-to-goal by at least this
TRIG_REGION = 0.5          # failures within this in x and y count as one region
TRIG_HEARTBEAT = 3         # call at least every N new valid simulations
TRIG_INVALID_FIRST = 2     # call on the 2nd INVALID placement in a row ...
TRIG_INVALID_EVERY = 3     # ... and every 3rd one after that while the streak lasts
# (WSEV run: skipping invalid turns left 7/15 FIP episodes stuck in overlap
#  loops; with calls after invalid turns (WSFREE3) the evolver's "place red
#  above green" skills got the actor out within a few turns.)


def new_trigger_state() -> dict:
    return {"best": None, "behaviours": set(), "valid": [], "since_call": 0, "invalid_run": 0}


def evolver_trigger(trajectory: list, goal_pos, state: dict) -> str | None:
    """Reason to call the evolver after the latest turn, or None. Updates state."""
    _, action, obs = _step_parts(trajectory[-1])
    tool, a = _parse_action(action)
    if tool in SIMULATION_TOOLS and obs.startswith("INVALID"):
        state["invalid_run"] += 1
        n = state["invalid_run"]
        if n == TRIG_INVALID_FIRST or (n > TRIG_INVALID_FIRST and (n - TRIG_INVALID_FIRST) % TRIG_INVALID_EVERY == 0):
            return "invalid_streak"
        return None
    if tool not in SIMULATION_TOOLS or obs.startswith("REFUSED") or "x" not in a:
        return None
    if _green_end(obs) is None:
        return None
    state["invalid_run"] = 0
    try:
        x, y, r = float(a["x"]), float(a.get("y", 0.0)), float(a.get("radius", 0.0))
    except (TypeError, ValueError):
        return None
    if any(abs(x - px) <= TRIG_REPEAT_TOL and abs(y - py) <= TRIG_REPEAT_TOL and abs(r - pr) <= TRIG_REPEAT_TOL
           for px, py, pr, _ in state["valid"]):
        return None
    failed = not ("SUCCESS" in obs or "Success: True" in obs)
    d = _goal_distance(obs, goal_pos)
    beh = _behaviour(obs, goal_pos)
    near_fail = sum(1 for px, py, _, pf in state["valid"]
                    if pf and abs(x - px) <= TRIG_REGION and abs(y - py) <= TRIG_REGION)
    state["since_call"] += 1
    reason = None
    if d is not None and (state["best"] is None or d < state["best"] - TRIG_BEST_GAIN):
        reason = "new_best"
    elif beh is not None and beh not in state["behaviours"]:
        reason = "new_behaviour"
    elif failed and near_fail == 1:
        reason = "repeat_failure_region"
    elif state["since_call"] >= TRIG_HEARTBEAT:
        reason = "heartbeat"
    if d is not None and (state["best"] is None or d < state["best"]):
        state["best"] = d
    if beh is not None:
        state["behaviours"].add(beh)
    state["valid"].append((x, y, r, failed))
    if reason:
        state["since_call"] = 0
    return reason


# ── Actor-facing episode notes ─────────────────────────────────────────

NOTES_GUIDANCE = (
    "How to use this block: it is recomputed after every simulation you run, so "
    "always read the current version rather than an earlier one. Your own "
    "simulation results in this episode outweigh the learned skills above: when a "
    "caveat below says a learned skill failed here, follow the caveat. Act on any "
    "WARNING, STALLED, CLOSE or INSENSITIVE line. Learned skills keep the same numbers all "
    "episode; episode findings are E1, E2 -- refer to skills by these labels."
)


NOTES_GUIDANCE_FREE = (
    "How to use this block: it is recomputed after every simulation you run, so "
    "always read the current version rather than an earlier one. The learned skills "
    "above are updated during this episode from your own simulation results. Act on "
    "any WARNING, STALLED, CLOSE or INSENSITIVE line, and do not place the red ball "
    "inside a RULED OUT region."
)


def _simulations(trajectory: list, goal_pos=None) -> list[tuple[int, dict, str, float | None]]:
    """(turn, args, outcome, goal distance) for every simulated-outcome turn with x/y/radius."""
    sims = []
    for turn, step in enumerate(trajectory, 1):
        _, action, obs = _step_parts(step)
        tool, args = _parse_action(action)
        if (tool in SIMULATION_TOOLS and not obs.startswith(("INVALID", "REFUSED"))
                and "x" in args and "radius" in args):
            sims.append((turn, args, _outcome(tool, obs), _goal_distance(obs, goal_pos)))
    return sims


def _green_end_from_outcome(outcome: str):
    m = re.search(r"green_ball ended at \(([^,]+),\s*([^)]+)\)", outcome)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _span(vals) -> tuple[float, float]:
    vals = [float(v) for v in vals]
    return min(vals), max(vals)


def _densest_x_band(xs: list[float], width: float) -> tuple[float, float, int]:
    """(lo, hi, count) of the width-wide x band holding the most values."""
    best = (xs[0], xs[0], 0)
    for lo in xs:
        inside = [x for x in xs if lo <= x <= lo + width]
        if len(inside) > best[2]:
            best = (min(inside), max(inside), len(inside))
    return best


def _fmt_place(a: dict) -> str:
    return f"(x={a['x']}, y={a.get('y', '?')}, r={a['radius']})"


def _budget_line(trajectory: list, max_iterations: int | None) -> str:
    """Turns used, split into simulations / predictions / invalid / other."""
    counts = {"sim": 0, "pred": 0, "invalid": 0, "other": 0}
    for step in trajectory:
        _, action, obs = _step_parts(step)
        tool, _ = _parse_action(action)
        if obs.startswith("INVALID"):
            counts["invalid"] += 1
        elif tool in SIMULATION_TOOLS:
            counts["sim"] += 1
        elif tool == "predict_first_contact":
            counts["pred"] += 1
        else:
            counts["other"] += 1
    used = len(trajectory)
    line = (f"Turns used: {used}" + (f" of {max_iterations}" if max_iterations else "")
            + f" -- {counts['sim']} simulations, {counts['pred']} predictions, "
              f"{counts['invalid']} invalid, {counts['other']} other.")
    if counts["pred"] + counts["invalid"] > counts["sim"]:
        line += " Predictions only report the first contact; only simulations show whether a placement works."
    return line


def format_episode_notes(
    trajectory: list,
    bank: SkillBank,
    level_name: str,
    kstar_numbers: dict[str, int],
    goal_pos=None,
    max_iterations: int | None = None,
    goal_name: str = "blue ball",
    include_bank_entries: bool = True,
    regions: list | None = None,
) -> str:
    """Block appended to the actor's system prompt during within-seed mode.

    The learned-skills block above it never changes within an episode (same text,
    same numbering as static injection); everything learned in the episode lives
    here. Coverage, closest-result, anchoring, stall and near-miss lines are
    computed deterministically from the actor's own simulations; caveats and
    findings are the evolver's current seed-local output. Empty before the first
    simulation, so the turn-1 prompt is identical to static injection.

    kstar_numbers maps K* skill id -> the number it carries in the skills block.
    """
    sims = _simulations(trajectory, goal_pos)
    if not sims:
        return ""
    n_fail = sum(o.startswith("FAILURE") for _, _, o, _ in sims)
    xs = _span(a["x"] for _, a, _, _ in sims)
    ys = _span(a["y"] for _, a, _, _ in sims if "y" in a) if any("y" in a for _, a, _, _ in sims) else None
    rs = _span(a["radius"] for _, a, _, _ in sims)
    lines = ["=== THIS EPISODE SO FAR (updated after each of your simulations) ===",
             NOTES_GUIDANCE if include_bank_entries else NOTES_GUIDANCE_FREE, "",
             _budget_line(trajectory, max_iterations),
             f"Simulations: {len(sims)} ({n_fail} failed).",
             f"Simulated so far: x in [{xs[0]:.2f}, {xs[1]:.2f}]"
             + (f", y in [{ys[0]:.2f}, {ys[1]:.2f}]" if ys else "")
             + f", radius in [{rs[0]:.2f}, {rs[1]:.2f}] (valid radius range [{RADIUS_RANGE[0]}, {RADIUS_RANGE[1]}])."]

    measured = [(d, t, a) for t, a, _, d in sims if d is not None]
    closest = min(measured, key=lambda m: m[0]) if measured else None
    if closest:
        lines.append(f"Closest so far: turn {closest[1]} {_fmt_place(closest[2])} -- "
                     f"the green ball ended {closest[0]:.2f} from the {goal_name}.")

    last = sims[-STALL_WINDOW:]
    near = [(d, t, a) for t, a, _, d in last if d is not None and d <= NEAR_MISS_DIST]
    if near:
        # Refining a near-miss: don't push the actor away; name the least-varied parameter.
        region = [a for _, a, _, d in sims if d is not None and d <= NEAR_MISS_DIST]
        spans = {k: (_span(a[k] for a in region if k in a), scale)
                 # radius first: on ties, mass/impact energy is the generic lever to try
                 for k, scale in (("radius", RADIUS_RANGE[1] - RADIUS_RANGE[0]), ("x", 10.0), ("y", 10.0))
                 if any(k in a for a in region)}
        least = min(spans, key=lambda k: (spans[k][0][1] - spans[k][0][0]) / spans[k][1])
        lo, hi = spans[least][0]
        best = min(near, key=lambda m: m[0])
        lines.append(f"CLOSE: a recent simulation ended {best[0]:.2f} from the goal (turn {best[1]} "
                     f"{_fmt_place(best[2])}). Stay in this region and vary {least} -- so far only "
                     f"{least} in [{lo:.2f}, {hi:.2f}] has been tried here -- rather than moving away.")
    else:
        failed_xs = [float(a["x"]) for _, a, o, _ in sims if o.startswith("FAILURE")]
        if len(failed_xs) >= ANCHOR_MIN_SIMS:
            lo, hi, cnt = _densest_x_band(failed_xs, ANCHOR_X_SPAN)
            # Don't tell the actor to leave the band that holds its best result.
            best_in_band = closest is not None and lo <= float(closest[2]["x"]) <= hi
            if cnt >= ANCHOR_SHARE * len(sims) and not best_in_band:
                lines.append(f"WARNING: x in [{lo:.2f}, {hi:.2f}] has now failed in {cnt} of {len(sims)} "
                             "simulations. Placements outside this x range are barely explored; the "
                             "solution may lie elsewhere.")
        if len(last) == STALL_WINDOW and all(o.startswith("FAILURE") for _, _, o, _ in last):
            lx = _span(a["x"] for _, a, _, _ in last)
            lr = _span(a["radius"] for _, a, _, _ in last)
            if lx[1] - lx[0] <= STALL_X_SPAN and lr[1] - lr[0] <= STALL_R_SPAN:
                lines.append(f"STALLED: your last {STALL_WINDOW} simulations stayed within x in "
                             f"[{lx[0]:.2f}, {lx[1]:.2f}], radius in [{lr[0]:.2f}, {lr[1]:.2f}] and all "
                             "failed without getting close. Next, change x by at least 0.4 or radius "
                             "by at least 0.5 rather than nudging.")

    # Outcome stopped responding to parameter changes (seed-6 v5: six different
    # placements all ended at exactly (2.57, -2.17)).
    recent = [(a, o) for _, a, o, _ in sims[-INSENSITIVE_WINDOW:]]
    ends = [_green_end_from_outcome(o) for _, o in recent]
    if (len(recent) == INSENSITIVE_WINDOW and all(e is not None for e in ends)
            and len({(a["x"], a.get("y"), a["radius"]) for a, _ in recent}) == INSENSITIVE_WINDOW
            and max(abs(e[0] - ends[0][0]) + abs(e[1] - ends[0][1]) for e in ends) <= INSENSITIVE_TOL):
        lines.append(f"INSENSITIVE: your last {INSENSITIVE_WINDOW} simulations used different placements "
                     f"but the green ball ended at ({ends[0][0]:.2f}, {ends[0][1]:.2f}) every time. "
                     "Small changes here have no effect; change a parameter by a much larger step.")

    if regions:
        lines.append("RULED OUT this episode (simulations inside these are refused by the tools):")
        lines.append(format_regions(regions))
    if not include_bank_entries:  # free mode: the refined bank is the skills block
        lines.append("=== END EPISODE ===")
        return "\n".join(lines)

    caveats = []
    for s in bank.level_skills.get(level_name, []):
        if s.skill_id in kstar_numbers and CAVEAT_MARK in s.when_to_apply:
            caveats.append(f"- Skill {kstar_numbers[s.skill_id]} [{s.title}]: "
                           f"{s.when_to_apply.split(CAVEAT_MARK, 1)[1]}")
    for m in bank.level_mistakes.get(level_name, []):
        if CAVEAT_MARK in m.how_to_avoid:
            caveats.append(f"- Mistake [{m.description[:60]}]: {m.how_to_avoid.split(CAVEAT_MARK, 1)[1]}")
    if caveats:
        lines.append("Caveats on the learned skills, from this episode:")
        lines.extend(caveats)

    findings = [s for s in bank.level_skills.get(level_name, []) if s.source_type == "within_seed"]
    if findings:
        lines.append("Episode findings:")
        for i, s in enumerate(findings, 1):
            lines.append(f"- E{i} [{s.title}] {s.principle}")
            if s.when_to_apply:
                lines.append(f"     Apply when: {s.when_to_apply}")
    seed_mistakes = [m for m in bank.level_mistakes.get(level_name, []) if m.generation == -1]
    if seed_mistakes:
        lines.append("Episode mistake to avoid:")
        lines.extend(f"- {m.description} How to avoid: {m.how_to_avoid}" for m in seed_mistakes)
    lines.append("=== END EPISODE ===")
    return "\n".join(lines)
