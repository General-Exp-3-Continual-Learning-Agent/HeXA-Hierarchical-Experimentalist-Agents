"""Phase 2: Wrap ReactAgent with skill-injected prompts.

Uses the same monkey-patching pattern as experiments/experiment_runner.py
(lines 126-180) to inject skills into the system prompt without modifying
any existing files.

Supports ablation levels (0-6) from experiments/prompt_builder.py to control
how much built-in hint content the base prompt contains.  The key use-case is
ablation_level=6 ("no hints") so that the agent relies entirely on learned
skills rather than hand-written guidance.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Import the *module* so we can monkey-patch its top-level names
import react_agent.react_agent as agent_module
from react_agent.react_agent import ReactAgent
from react_agent.tools import InterphyreToolkit
from react_agent.level_prompts import build_system_prompt, build_initial_user_message

# Ablation support — imported lazily to avoid hard dependency when not used
try:
    from experiments.prompt_builder import (
        build_system_prompt_ablation,
        build_initial_user_message_ablation,
    )
    from experiments.feedback_reducer import AblationToolkitWrapper
    _HAS_ABLATION = True
except ImportError:
    _HAS_ABLATION = False

from skillrl.core.config import TEACHER_MODEL
from skillrl.core.skill_bank import Skill, SkillBank
from skillrl.core.retriever import retrieve_skills
from skillrl.distillation.within_seed_evolve import (
    SIMULATION_TOOLS,
    Scene,
    _in_region,
    _parse_action,
    evolver_trigger,
    format_episode_notes,
    format_regions,
    new_trigger_state,
    refine_bank_within_seed,
    refine_bank_within_seed_free,
)


WS_ANNOUNCE_NOTE = (
    "SKILL UPDATES DURING THIS EPISODE: a teacher watches your simulations in this "
    "puzzle and updates the learned skills above while you work. Skills titled "
    "\"(this episode)\" come from YOUR OWN results in this puzzle and override the "
    "general ones when they conflict. Each update is also shown at the end of the "
    "tool result as \"SKILL UPDATE\". When you receive one, follow it in your next "
    "action and say in your Thought which skill you are following and which "
    "parameter you are changing because of it."
)


def _skill_update_text(info: dict, bank: SkillBank, level_name: str) -> str | None:
    """Tool-result addendum listing what the evolver just added or rewrote."""
    if info.get("skipped") or not info.get("ok"):
        return None
    changed = set(info.get("added", [])) | set(info.get("edited", []))
    lines = []
    for s in bank.level_skills.get(level_name, []):
        if s.skill_id in changed and s.source_type == "within_seed":
            tag = "NEW" if s.skill_id in info.get("added", []) else "REVISED"
            lines.append(f"- {tag} skill (this episode) [{s.title}]: {s.principle}"
                         + (f" Apply when: {s.when_to_apply}" if s.when_to_apply else ""))
    for m in bank.level_mistakes.get(level_name, []):
        if m.mistake_id in changed and m.generation == -1:
            lines.append(f"- Mistake to avoid (this episode): {m.description} How to avoid: {m.how_to_avoid}")
    for r in info.get("ruled_out") or []:
        lines.append(f"- Ruled out: x in [{r['x_min']:.2f}, {r['x_max']:.2f}], y in [{r['y_min']:.2f}, "
                     f"{r['y_max']:.2f}] ({r.get('reason', '')}); simulations there are refused.")
    if not lines:
        return None
    return ("SKILL UPDATE (the teacher revised your skills after this result; follow it in "
            "your next action):\n" + "\n".join(lines))


def _tag_episode_skills(bank: SkillBank, level_name: str) -> SkillBank:
    """Copy of the bank with seed-local skill titles prefixed '(this episode)'."""
    b = copy.deepcopy(bank)
    for s in b.level_skills.get(level_name, []):
        if s.source_type == "within_seed" and not s.title.startswith("(this episode)"):
            s.title = "(this episode) " + s.title
    return b


class RegionToolkit:
    """Refuses simulations placed inside a ruled-out region. The regions are
    proposed by the within-seed evolver from this episode's failed simulations
    (validated in refine_bank_within_seed_free) and shared via `regions`, which
    the hook updates in place."""

    def __init__(self, toolkit, regions: list):
        self._toolkit = toolkit
        self._regions = regions
        self.refused = 0

    def dispatch_tool(self, tool_name, args):
        if tool_name in SIMULATION_TOOLS and self._regions and isinstance(args, dict) and "x" in args:
            try:
                x = float(args["x"]); y = float(args["y"]) if "y" in args else None
            except (TypeError, ValueError):
                x = None
            if x is not None:
                for i, r in enumerate(self._regions, 1):
                    if _in_region(x, y, r):
                        self.refused += 1
                        return (f"REFUSED: ({x}, {y}) is inside ruled-out region R{i} "
                                f"(x in [{r['x_min']:.2f}, {r['x_max']:.2f}], y in [{r['y_min']:.2f}, "
                                f"{r['y_max']:.2f}]), where this episode's simulations at turns "
                                f"{r['turns']} failed. The simulation was not run. Choose a "
                                f"placement outside every ruled-out region.\n"
                                f"Active regions:\n{format_regions(self._regions)}")
        return self._toolkit.dispatch_tool(tool_name, args)

    def __getattr__(self, name):
        return getattr(self._toolkit, name)


PREDICTION_CAP_NOTE = (
    "TOOL BUDGET: predict_first_contact is limited to ONE call before each "
    "simulation (simulate_action, simulate_with_trace, simulate_partial or "
    "trace_green_ball). A second prediction without simulating in between is "
    "refused and still uses up a turn. This overrides any instruction above to "
    "always predict first: predictions do not tell you whether the puzzle is "
    "solved, so spend your turns on simulations."
)


class PredictionCapToolkit:
    """Refuses predict_first_contact once `cap` predictions have been made since
    the last simulated outcome. In the v5.1 within-seed runs predictions took
    11-20 of 25 turns and the prompt-level budget line did not curb it."""

    def __init__(self, toolkit, cap: int = 1):
        self._toolkit = toolkit
        self._cap = cap
        self._since_sim = 0
        self.refused = 0

    def dispatch_tool(self, tool_name, args):
        if tool_name == "predict_first_contact":
            if self._since_sim >= self._cap:
                self.refused += 1
                return ("REFUSED: you already called predict_first_contact since your "
                        "last simulation (limit: one prediction per simulation). "
                        "Run a simulation now, e.g. simulate_action with the "
                        "placement you just predicted.")
            self._since_sim += 1
        elif tool_name in SIMULATION_TOOLS:
            self._since_sim = 0
        return self._toolkit.dispatch_tool(tool_name, args)

    def __getattr__(self, name):
        return getattr(self._toolkit, name)


EXPLORE_NOTE = (
    "EXPLORATION RULES (enforced by the tools):\n"
    "1. Radius coverage: your first valid simulations must cover all three "
    "radius bands -- small [0.1, 0.73), medium [0.73, 1.37) and large "
    "[1.37, 2.0] -- one each, in any order, before you repeat a band. Learned "
    "skills are hypotheses from OTHER levels; this checks them before you "
    "commit. The rule lifts once all bands are covered or after {lift} "
    "simulation calls.\n"
    "2. No stalling: if your last {stall_n} valid simulations were all within "
    "{delta} of each other in x, y and radius, a simulation that is again "
    "within {delta} of all of them is refused -- make a larger change.\n"
    "3. Position diversity: if your last {pos_n} valid simulations all kept x "
    "(or y) within a {pos_span}-wide window, the next simulation must move "
    "that coordinate by at least {pos_jump} from the window's centre, whatever "
    "you change in the other parameters.\n"
    "Refused calls still use up a turn."
)

RADIUS_BANDS = ((0.1, 0.73, "small"), (0.73, 1.37, "medium"), (1.37, 2.0 + 1e-9, "large"))


class ExploreToolkit:
    """Test-time exploration, target-agnostic (uses only this episode's tool
    calls; no level knowledge beyond the shared 0.1-2.0 radius range).

    1. Radius coverage: until each radius band has one valid simulation (or
       `lift_after` simulation calls have been made), a simulation whose radius
       falls in an already-covered band is refused.
    2. Stall breaker: a simulation within `delta` (x, y and radius) of each of
       the last `stall_n` valid simulations is refused.
    3. Position diversity: if the last `pos_n` valid simulations kept x (or
       y) within a `pos_span` window, a simulation that does not move that
       coordinate at least `pos_jump` from the window centre is refused.
       Added after the v4 + explore run: the actor covered all radius bands
       but held x at the arm tip for up to 12 simulations while varying y,
       which rule 2 never caught.

    Motivation: with the v4 cross-level catapult bank the actor followed the
    bank's small-radius prior for 15-20 simulations, nudging r by 0.05, and
    never tested the rest of the action space."""

    def __init__(self, toolkit, lift_after: int = 6, stall_n: int = 3, delta: float = 0.15,
                 pos_n: int = 4, pos_span: float = 0.2, pos_jump: float = 0.5):
        self._toolkit = toolkit
        self.lift_after, self.stall_n, self.delta = lift_after, stall_n, delta
        self.pos_n, self.pos_span, self.pos_jump = pos_n, pos_span, pos_jump
        self.position_refused = 0
        self._sim_calls = 0
        self._covered: set[str] = set()
        self._valid: list[tuple[float, float, float]] = []
        self.coverage_refused = 0
        self.stall_refused = 0

    @staticmethod
    def _band(r: float) -> str:
        for lo, hi, name in RADIUS_BANDS:
            if lo <= r < hi:
                return name
        return "out_of_range"

    def _coverage_active(self) -> bool:
        return len(self._covered) < len(RADIUS_BANDS) and self._sim_calls < self.lift_after

    def dispatch_tool(self, tool_name, args):
        if tool_name not in SIMULATION_TOOLS:
            return self._toolkit.dispatch_tool(tool_name, args)
        try:
            x, y, r = (float(args.get("x", 0)), float(args.get("y", 0)),
                       float(args.get("radius", 0.5)))
        except (TypeError, ValueError):
            return self._toolkit.dispatch_tool(tool_name, args)

        band = self._band(r)
        if self._coverage_active() and band in self._covered:
            self.coverage_refused += 1
            missing = [n for _, _, n in RADIUS_BANDS if n not in self._covered]
            spans = ", ".join(f"{n} [{lo:g}, {min(hi, 2.0):g}]"
                              for lo, hi, n in RADIUS_BANDS if n in missing)
            return (f"REFUSED (exploration rule 1): radius {r:g} is in the {band} band, "
                    f"which you already tested. Test an untested band first: {spans}. "
                    f"Adjust x/y if the larger ball overlaps something.")

        last = self._valid[-self.stall_n:]
        if (len(last) == self.stall_n
                and all(abs(x - a) <= self.delta and abs(y - b) <= self.delta
                        and abs(r - c) <= self.delta for a, b, c in last)):
            self.stall_refused += 1
            return (f"REFUSED (exploration rule 2): your last {self.stall_n} simulations "
                    f"were all within {self.delta:g} of ({x:g}, {y:g}, r={r:g}) without "
                    f"solving, and so is this one. Make a larger change: move x or y by "
                    f"at least {2 * self.delta:g}, or change the radius by at least "
                    f"{2 * self.delta:g}, or try a qualitatively different placement.")

        recent = self._valid[-self.pos_n:]
        if len(recent) == self.pos_n:
            for axis, val in (("x", x), ("y", y)):
                vals = [p[0] if axis == "x" else p[1] for p in recent]
                centre = (max(vals) + min(vals)) / 2
                if max(vals) - min(vals) <= self.pos_span and abs(val - centre) < self.pos_jump:
                    self.position_refused += 1
                    return (f"REFUSED (exploration rule 3): your last {self.pos_n} simulations "
                            f"all kept {axis} near {centre:g}, and {axis}={val:g} does too. "
                            f"Move {axis} by at least {self.pos_jump:g} (to {axis} <= "
                            f"{centre - self.pos_jump:g} or {axis} >= {centre + self.pos_jump:g}).")

        self._sim_calls += 1
        obs = self._toolkit.dispatch_tool(tool_name, args)
        if not str(obs).lstrip().upper().startswith(("INVALID", "ERROR")):
            self._covered.add(band)
            self._valid.append((x, y, r))
        return obs

    def __getattr__(self, name):
        return getattr(self._toolkit, name)


DIAGNOSTIC_ADDENDUM = """
IMPORTANT — After each FAILED simulation, before trying a new placement, you MUST:
1. Review the LEARNED PHYSICS SKILLS listed in the system prompt above.
2. Identify which skill applies to the failure you just observed.
3. Explicitly state which skill you are applying and why in your next Thought.
"""

# Qwen2.5-3B-only addendum. Diagnosed against real trajectories, not assumed:
# 46-92% of Qwen3B eval episodes across every HeXA build/level checked hit the
# full iteration cap WITHOUT ever calling finish, and a meaningful minority
# (16/50 on one build) enter an exact repeating cycle between 2-4 already-tried
# (x, y, radius) values, ignoring the specific numeric correction the simulator
# gives in its INVALID/FAILURE message every single turn. This reproduces
# independent of any injected skill bank -- a plain ReAct run on the same
# model/level shows the identical pathology -- so it targets a general 3B
# execution weakness, not a HeXA-specific one, which is why it is gated on
# model_label rather than folded into DIAGNOSTIC_ADDENDUM for every actor.
SMALL_MODEL_ADDENDUM = """
IMPORTANT — you have a STRICT LIMIT on the number of attempts. Follow these
rules on every turn:
1. NEVER submit an (x, y, radius) you have already tried. If your last attempt
   was INVALID or FAILURE, the observation tells you exactly how far off you
   were (e.g. "Move at least 0.41 units further away", "Radius out of range
   [0.1, 2.0]") — apply that exact number to change x, y, or radius. Do not
   guess a different value that ignores it.
2. If two of your last three attempts were invalid or failed, STOP refining
   the same region. Try a placement on the OTHER side of the target object,
   or a substantially different radius, rather than a small nudge.
3. A submitted guess can still succeed; an unsubmitted one cannot. If you are
   unsure you have the optimal placement but have already made several
   attempts, call finish with your best current guess rather than continuing
   to explore — running out of attempts without finishing is treated as a
   failure exactly the same as a wrong guess.
"""


def format_worked_examples(trajectories: list[dict], max_examples: int = 1) -> str:
    """Format successful trajectories as worked examples for cold-start.

    Used when the skill bank is sparse to give the agent an in-context
    demonstration of effective reasoning.
    """
    if not trajectories:
        return ""

    lines = ["\n=== WORKED EXAMPLE ==="]
    for traj in trajectories[:max_examples]:
        lines.append(f"\n[Seed {traj.get('seed', '?')} — SUCCESS in {traj.get('iterations', '?')} steps]")
        steps = traj.get("trajectory", [])
        # Show only the last few decisive steps
        for step in steps[-3:]:
            thought = step.get("thought", "")
            action = step.get("action", "")
            obs = step.get("observation", "")
            if len(obs) > 300:
                obs = obs[:300] + "..."
            lines.append(f"  Thought: {thought}")
            lines.append(f"  Action: {action}")
            lines.append(f"  Observation: {obs}")
            lines.append("")
        final = traj.get("action")
        if final:
            lines.append(f"  Final: x={final[0]}, y={final[1]}, r={final[2]}")
    lines.append("=== END WORKED EXAMPLE ===\n")
    return "\n".join(lines)


def run_skill_augmented(
    model_fn,
    level_name: str,
    seed: int,
    skill_bank: SkillBank,
    max_iterations: int = 25,
    verbose: bool = False,
    temperature: float = 0.3,
    max_new_tokens: int = 800,
    is_oss: bool = False,
    max_general_skills: int = 0,
    max_specific_skills: int = 6,
    max_mistakes: Optional[int] = 4,
    worked_examples: Optional[list[dict]] = None,
    ablation_level: Optional[int] = None,
    xl_framing: bool = False,
    model_label: str = "",
    within_seed: bool = False,
    teacher_model: str = TEACHER_MODEL,
    ws_max_new_skills: int = 2,
    ws_max_new_mistakes: int = 1,
    refine_on: str = "all",
    ws_evolver: bool = True,
    ws_free: bool = False,
    ws_regions: bool = False,
    ws_announce: bool = False,
    ws_require_numbers: bool = False,
    prediction_cap: Optional[int] = None,
    explore: bool = False,
) -> dict:
    """Run ReactAgent with skill-augmented prompts.

    Parameters
    ----------
    model_fn : Callable compatible with ReactAgent (messages, temp, max_tokens) -> str
    level_name : Puzzle level name.
    seed : Random seed for level generation.
    skill_bank : The SkillBank to retrieve skills from.
    max_iterations : Max ReAct loop iterations.
    verbose : Print debug info.
    temperature : Sampling temperature.
    max_new_tokens : Max generation tokens.
    is_oss : Whether using an OSS model (affects prompt format).
    max_general_skills : Max general skills to inject.
    max_specific_skills : Max level-specific skills to inject.
    worked_examples : Optional list of successful trajectory dicts for cold-start.
    ablation_level : If set (0-6), use the ablation prompt as the base instead of
                     the full hint prompt.  Level 6 = no hints at all, so the agent
                     relies entirely on learned skills.  Requires the experiments/
                     package to be importable.
    within_seed : If True, refine a seed-local copy of the bank after each turn
                  (within-seed experimentation) instead of injecting it once.
                  skill_bank itself is never modified.
    teacher_model : Evolver used for within-seed refinement.
    ws_max_new_skills / ws_max_new_mistakes : Seed-specific entries allowed on
                  top of K*. K* entries are read-only within a seed apart from
                  one caveat each (see within_seed_evolve.py).
    refine_on : "all" refines after every tool turn (as in the paper's
                algorithm); "sim" only after simulation turns, which skips
                evolver calls on scene-inspection turns that carry no
                experimental outcome.
    ws_evolver : within_seed only. False = notes-only ablation: the episode
                notes are still recomputed after each turn, but the evolver is
                never called, so the bank stays K* (no caveats, no findings).
    prediction_cap : If set, allow at most this many predict_first_contact
                calls between simulations (PredictionCapToolkit) and tell the
                actor so in the system prompt.
    explore : Enforce test-time exploration (ExploreToolkit): radius-band
                coverage over the first simulations plus a stall breaker,
                and tell the actor the rules in the system prompt.

    Returns
    -------
    dict with keys: success, action, iterations, trajectory, final_observation,
                    seed, skills_used, elapsed_time
    """
    # 1. Get base prompts — either ablated or full
    if ablation_level is not None:
        if not _HAS_ABLATION:
            raise ImportError(
                "ablation_level requires experiments.prompt_builder and "
                "experiments.feedback_reducer — make sure the experiments/ "
                "package is on PYTHONPATH."
            )
        original_system = build_system_prompt_ablation(level_name, ablation_level)
        original_initial = build_initial_user_message_ablation(level_name, ablation_level, is_oss=is_oss)
    else:
        original_system = build_system_prompt(level_name, is_oss=is_oss)
        original_initial = build_initial_user_message(
            level_name, is_oss=is_oss, max_iterations=max_iterations
        )

    def render_system(bank: SkillBank, show_all: bool = False):
        """Retriever(K, level, M, N) -> skill block -> augmented system prompt.
        show_all (within-seed free mode): every level skill and mistake in the
        seed-local bank, no retrieval caps."""
        # 2. Retrieve relevant skills
        skills = retrieve_skills(
            bank, level_name,
            max_general=max_general_skills,
            max_specific=10**6 if show_all else max_specific_skills,
        )

        # 3. Retrieve mistakes
        mistakes = bank.get_mistakes_for_level(level_name)
        if max_mistakes is not None and not show_all:
            mistakes = mistakes[:max_mistakes]

        # 4. Format skills + mistakes block
        skills_block = bank.format_skills_as_prompt(
            level_name, skills=skills, mistakes=mistakes,
        )

        # 4b. Honest provenance label for cross-level transferred banks.
        # Single-line header swap — no extra paragraph, so prompt size is unchanged.
        if xl_framing and skills_block:
            original_header = f"## {level_name.replace('_', ' ').title()}-Specific Skills"
            skills_block = skills_block.replace(
                original_header,
                "## Cross-Level Skills (transferred from other puzzles — verify with simulation)",
            )

        # 5. Augment system prompt
        augmented_system = original_system
        if skills_block:
            augmented_system += "\n\n" + skills_block

        # Optional worked examples for cold-start
        if worked_examples:
            augmented_system += "\n" + format_worked_examples(worked_examples)
        return skills, mistakes, skills_block, augmented_system

    skills, mistakes, skills_block, augmented_system = render_system(skill_bank)
    if within_seed and ws_announce:
        # Tell the actor from turn 1 that its skills change during the episode.
        augmented_system += "\n\n" + WS_ANNOUNCE_NOTE
    if prediction_cap is not None:
        augmented_system += "\n\n" + PREDICTION_CAP_NOTE
    explore_cfg = dict(lift_after=6, stall_n=3, delta=0.15,
                       pos_n=4, pos_span=0.2, pos_jump=0.5)
    if explore:
        augmented_system += "\n\n" + EXPLORE_NOTE.format(
            lift=explore_cfg["lift_after"], stall_n=explore_cfg["stall_n"],
            delta=explore_cfg["delta"], pos_n=explore_cfg["pos_n"],
            pos_span=explore_cfg["pos_span"], pos_jump=explore_cfg["pos_jump"])

    # Within-seed experimentation: K_{j,0} = K*, then K_{j,t+1} = pi_ev(K_{j,t}, h_{t+1}).
    # The hook holds the seed-local bank; skill_bank (K*) is never touched.
    on_turn_end = None
    ws_state = {"bank": skill_bank, "log": []}
    kstar = copy.deepcopy(skill_bank)  # read-only reference for the refinement guards
    # Numbers the K* skills carry in the (never-changing) skills block, so the
    # episode notes can refer to them consistently.
    kstar_numbers = {
        s.skill_id: i
        for i, s in enumerate((s for s in skills if s.source_level != "general"), 1)
    }
    ws_state["goal"] = None  # Scene (goal / green positions), filled in once the toolkit exists
    ws_state["trig"] = new_trigger_state()
    ws_state["regions"] = []
    if within_seed:
        def on_turn_end(iteration: int, trajectory: list):
            last_action = trajectory[-1][1] or ""
            if refine_on == "sim" and not last_action.startswith(SIMULATION_TOOLS):
                return None
            # refine_on="event": code decides from the latest simulation output
            # (evolver_trigger); on other turns only the episode notes update.
            reason = evolver_trigger(trajectory, ws_state["goal"], ws_state["trig"]) \
                if refine_on == "event" else None
            call = refine_on != "event" or reason is not None
            t0 = time.perf_counter()
            if not call:
                new_bank, info = ws_state["bank"], {"ok": True, "evolver": False, "skipped": True}
            elif ws_evolver and ws_free:
                new_bank, info = refine_bank_within_seed_free(
                    level_name, ws_state["bank"], seed, trajectory,
                    max_iterations=max_iterations,
                    teacher_model=teacher_model,
                    goal_pos=ws_state["goal"],
                    max_new_skills=ws_max_new_skills,
                    max_new_mistakes=ws_max_new_mistakes,
                    frozen_ids=frozenset(
                        {x.skill_id for x in kstar.level_skills.get(level_name, [])}
                        | {x.mistake_id for x in kstar.level_mistakes.get(level_name, [])}),
                    regions=list(ws_state["regions"]) if ws_regions else None,
                    require_numbers=ws_require_numbers,
                )
                if ws_regions and info.get("ruled_out") is not None:
                    ws_state["regions"][:] = info["ruled_out"]  # shared with RegionToolkit
            elif ws_evolver:
                new_bank, info = refine_bank_within_seed(
                    level_name, ws_state["bank"], seed, trajectory,
                    max_iterations=max_iterations,
                    kstar=kstar,
                    max_new_skills=ws_max_new_skills,
                    max_new_mistakes=ws_max_new_mistakes,
                    teacher_model=teacher_model,
                    goal_pos=ws_state["goal"],
                )
            else:  # notes-only ablation: bank stays K*
                new_bank, info = ws_state["bank"], {"ok": True, "evolver": False}
            ws_state["bank"] = new_bank
            # The skills block stays exactly as injected at turn 1 (same text and
            # numbering as static); everything learned this episode goes in the notes.
            # Free mode: the actor sees the refined bank itself, re-rendered with the same
            # top-M skills (by confidence) / first-N mistakes caps as turn 1.
            new_system = (render_system(_tag_episode_skills(new_bank, level_name) if ws_announce else new_bank)[3]
                          if ws_free else augmented_system)
            if ws_announce and ws_free:  # the re-rendered prompt lacks the note
                new_system += "\n\n" + WS_ANNOUNCE_NOTE
            if ws_free and prediction_cap is not None:
                new_system += "\n\n" + PREDICTION_CAP_NOTE
            if ws_free and explore:
                new_system += "\n\n" + EXPLORE_NOTE.format(
                    lift=explore_cfg["lift_after"], stall_n=explore_cfg["stall_n"],
                    delta=explore_cfg["delta"], pos_n=explore_cfg["pos_n"],
                    pos_span=explore_cfg["pos_span"], pos_jump=explore_cfg["pos_jump"])
            notes = format_episode_notes(trajectory, new_bank, level_name, kstar_numbers,
                                         ws_state["goal"], max_iterations,
                                         goal_name=ws_state.get("goal_name", "blue ball"),
                                         include_bank_entries=not ws_free,
                                         regions=ws_state["regions"] if ws_regions else None)
            if notes:
                new_system += "\n\n" + notes
            info["iteration"] = iteration
            info["trigger"] = reason
            info["episode_notes"] = notes
            info["elapsed_time"] = time.perf_counter() - t0
            info["skills"] = [
                {"id": s.skill_id, "confidence": round(s.confidence, 3), "title": s.title}
                for s in new_bank.level_skills.get(level_name, [])
            ]
            info["skill_titles"] = [s["title"] for s in info["skills"]]
            if ws_regions:
                info["regions"] = copy.deepcopy(ws_state["regions"])
            # Full text of this level's bank after the refinement, so the episode
            # can be reconstructed exactly (titles alone hid what the actor saw).
            info["bank"] = {
                "skills": [asdict(s) for s in new_bank.level_skills.get(level_name, [])],
                "mistakes": [asdict(m) for m in new_bank.level_mistakes.get(level_name, [])],
            }
            update = _skill_update_text(info, new_bank, level_name) if ws_announce else None
            info["skill_update_shown"] = update
            ws_state["log"].append(info)
            if verbose:
                status = "ok" if info["ok"] else "FAILED (bank unchanged)"
                print(f"\n[Within-seed] turn {iteration}: refine {status}, "
                      f"{info.get('n_skills', '?')} skills / {info.get('n_mistakes', '?')} mistakes, "
                      f"{info.get('n_seed_skills', '?')} seed-local, "
                      f"caveats={len(info.get('caveats', {}))} "
                      f"rejected_queue={len(info.get('rejected_queue', []))} "
                      f"unsupported={len(info.get('rejected_unsupported', []))} "
                      f"over_cap={len(info.get('rejected_over_cap', []))} "
                      f"({info['elapsed_time']:.1f}s)")
            if update:
                return {"system": new_system, "append_to_observation": update}
            return new_system

    # 6. Augment initial user message with the diagnostic reminder, but only
    # when skills were actually injected. The addendum tells the agent to review
    # the LEARNED PHYSICS SKILLS section of the system prompt, so on an empty
    # bank (the ReAct / no-bank arm) it points at a section that does not exist.
    # Gating it here keeps the no-bank prompt identical to plain ReAct.
    augmented_initial = original_initial
    if skills_block and skills_block.strip():
        augmented_initial += "\n" + DIAGNOSTIC_ADDENDUM

    # Qwen2.5-3B-only fix for the oscillation/never-finishes failure mode (see
    # SMALL_MODEL_ADDENDUM above). Unlike DIAGNOSTIC_ADDENDUM this is NOT gated
    # on skills_block: the pathology reproduces on the no-bank ReAct arm too,
    # so it targets actor execution, not skill review. Every other model is
    # byte-for-byte unaffected -- model_label defaults to "" and no other
    # call site passes anything containing "3b".
    if "3b" in model_label.lower():
        augmented_initial += "\n" + SMALL_MODEL_ADDENDUM

    if verbose:
        print(f"\n[SkillRL] Injecting {len(skills)} skills + {len(mistakes)} mistakes for {level_name} seed={seed}")
        for s in skills:
            print(f"  - [skill] [{s.source_level}] {s.title}")
        for m in mistakes:
            print(f"  - [mistake] {m.description}")

    # 6. Monkey-patch (same pattern as experiment_runner.py lines 132-135)
    def patched_system(level: str, **kwargs) -> str:
        return augmented_system

    def patched_initial(level: str, **kwargs) -> str:
        return augmented_initial

    orig_build_system = agent_module.build_system_prompt
    orig_build_initial = agent_module.build_initial_user_message

    agent_module.build_system_prompt = patched_system
    agent_module.build_initial_user_message = patched_initial

    try:
        toolkit = InterphyreToolkit(level_name=level_name, seed=seed, is_oss=is_oss)
        # Goal object the green ball must reach: blue_ball (catapult, two_body_problem)
        # or blue_basket (falling_into_place).
        info_objs = getattr(toolkit, "_level_info", {})
        goal_name = next((n for n in ("blue_ball", "blue_basket") if n in info_objs), None)
        goal = info_objs.get(goal_name) if goal_name else None
        green = info_objs.get("green_ball")
        ws_state["goal"] = Scene(
            goal=(goal["x"], goal["y"]) if goal else None,
            goal_name=goal_name or "blue_ball",
            green=(green["x"], green["y"]) if green else None,
            validate=getattr(toolkit, "_validate_placement", None),
        )
        ws_state["goal_name"] = goal_name.replace("_", " ") if goal_name else "goal"

        # Wrap toolkit for ablation levels 5/6 (reduced feedback + blocked tools)
        if ablation_level is not None and ablation_level in (5, 6):
            toolkit = AblationToolkitWrapper(toolkit, ablation_level, level_name=level_name)
        if prediction_cap is not None:
            toolkit = PredictionCapToolkit(toolkit, cap=prediction_cap)
        if within_seed and ws_regions:
            toolkit = RegionToolkit(toolkit, ws_state["regions"])
        explorer = None
        if explore:
            toolkit = explorer = ExploreToolkit(toolkit, **explore_cfg)

        agent = ReactAgent(
            model_fn=model_fn,
            toolkit=toolkit,
            level_name=level_name,
            max_iterations=max_iterations,
            verbose=verbose,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            is_oss=is_oss,
            on_turn_end=on_turn_end,
        )

        start = time.perf_counter()
        result = agent.solve()
        elapsed = time.perf_counter() - start

        toolkit.close()

        # Enrich result with SkillRL metadata
        result["seed"] = seed
        result["skills_used"] = [s.skill_id for s in skills]
        result["skill_titles"] = [s.title for s in skills]
        result["mistakes_injected"] = [m.mistake_id for m in mistakes]
        result["elapsed_time"] = elapsed
        if ablation_level is not None:
            result["ablation_level"] = ablation_level
        if prediction_cap is not None:
            result["prediction_cap"] = prediction_cap
            result["predictions_refused"] = toolkit.refused
        if explorer is not None:
            result["explore"] = explore_cfg
            result["explore_coverage_refused"] = explorer.coverage_refused
            result["explore_stall_refused"] = explorer.stall_refused
            result["explore_position_refused"] = explorer.position_refused
            result["explore_bands_covered"] = sorted(explorer._covered)
        if within_seed:
            # Seed-local bank is discarded after the episode; logged for analysis only.
            result["inference_mode"] = "within_seed"
            result["within_seed_refine_on"] = refine_on
            result["within_seed_teacher"] = teacher_model
            result["within_seed_log"] = ws_state["log"]
            result["within_seed_evolver"] = ws_evolver
            result["within_seed_free"] = ws_free
            result["within_seed_regions"] = ws_regions
            result["within_seed_triggers"] = [e.get("trigger") for e in ws_state["log"] if not e.get("skipped")]
            result["within_seed_evolver_calls"] = (
                sum(1 for e in ws_state["log"] if not e.get("skipped")) if ws_evolver else 0)
            result["within_seed_notes_updates"] = len(ws_state["log"])
            result["within_seed_final_skill_titles"] = [
                s.title for s in ws_state["bank"].level_skills.get(level_name, [])
            ]
        return result

    finally:
        # Restore originals
        agent_module.build_system_prompt = orig_build_system
        agent_module.build_initial_user_message = orig_build_initial
