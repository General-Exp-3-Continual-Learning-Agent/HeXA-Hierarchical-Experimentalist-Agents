"""HeXA's teacher on Pigey trials: prompts, reward, selection, and the patch wiring.

Follows the port idiom of ``hexa_core.hexa_bridge`` and ``hexa_orbit.teacher``:
nothing under ``skillrl/`` is edited. :func:`install` rebinds the InterPhyre-
specific names (reward, trajectory formatter and loader, the three teacher
prompts, the teacher call) in the modules that resolve them at call time, and
keeps the output JSON contracts byte-compatible with what ``distill`` and
``evolving_distill`` parse.

Two Pigey-specific choices:

* **Level = suite.** One bank serves all tasks of ``libero_goal_task``; every
  trajectory header names its task, so the teacher can still write task-specific
  skills, and ``source_seeds`` are ``t<task>_e<episode>`` strings.
* **Same-task contrast.** HeXA pairs a success and a failure on the same seed.
  A Pigey seed is run once, so pairs are formed per *task* instead (best success
  with worst failure on the same task), then backfilled by hexa_bridge's balanced
  quota. The evolver is shown the whole round when it fits.

The teacher runs through the Claude Code CLI like the actor, reading the prompt
from stdin (prompts here exceed what belongs on a command line) and saving every
prompt and response under the run directory for audit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from hexa.trajectory import format_trajectory_for_teacher, load_trial_trajectories, trajectory_reward
from runner.claude_code import rate_limit_wait, wait_for_usage_limit

DEFAULT_SUITE = "libero_goal_task"
DEFAULT_TEACHER_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TRAJS = 10
SKILLS_BLOCK_HEADING = "=== LEARNED ROBOT MANIPULATION SKILLS ==="

# Facts and scoring rules only, no approach implied: any strategy the teacher
# banks should come out of the trajectories, not out of this description.
_TRIAL_RULES = (
    "Each task has many initial states (object placements); a trial is one task on one "
    "initial state. Success is decided only by the simulator's predicates (for example "
    "'on' requires contact and the object's centre within 3 cm horizontally of the "
    "support's centre); the orchestrator is never told the predicate and sees only camera "
    "images and tool results. Budgets: 35 LLM steps and 2000 simulator steps per trial. "
    "Every tool call consumes simulator steps (a single Grasp can take over a thousand) "
    "and the trial ends as a failure when either budget runs out."
)
_OBJECT_SCENES = (
    "LIBERO-Object: each task has its own floor scene with a basket and several packaged "
    "grocery items (alphabet soup, bbq sauce, butter, chocolate pudding, cream cheese, "
    "ketchup, milk, orange juice, salad dressing, tomato sauce). Every task is 'pick the "
    "named item and place it in the basket'. "
)
TASK_DESCRIPTIONS = {
    "libero_goal_task": (
        "LIBERO-Goal: one fixed tabletop scene with a stove (with a knob), a flat-topped "
        "cabinet with drawers, a wine rack, and movable objects (plate, bowl, wine bottle, "
        "cream cheese box, ...). Ten tasks share the scene and differ only in the goal "
        "sentence: putting one object on or in another, opening a drawer, pushing an object, "
        "turning the stove off. The sentences are LIBERO-PRO rewrites, so the wording and the "
        "goal can differ from the strings the VLA policy pi0.5 was trained on in this scene. "
        + _TRIAL_RULES
    ),
    "libero_object_swap": (
        _OBJECT_SCENES
        + "This is LIBERO-PRO's position-swap variant: the instruction is the one the VLA "
        "policy pi0.5 was trained on, but items start in each other's places, so the target's "
        "position differs from the training scenes. " + _TRIAL_RULES
    ),
    "libero_object_task": (
        _OBJECT_SCENES
        + "This is LIBERO-PRO's task variant: the scene and layout are those pi0.5 was trained "
        "on, but the instruction names a different item that is already in the scene, so it "
        "differs from the string pi0.5 learned for that scene. " + _TRIAL_RULES
    ),
    "libero_spatial_swap": (
        "LIBERO-Spatial: a tabletop with a wooden cabinet, a flat stove, a plate, a cookies box, "
        "a glazed ramekin and two identical black bowls. Every task is 'pick the black bowl "
        "<location> and place it on the plate'; the location phrase is what identifies the "
        "bowl. This is LIBERO-PRO's position-swap variant: the instruction is the one the VLA "
        "policy pi0.5 was trained on, but objects start in each other's places (for example "
        "the plate and the cookies box exchange regions). " + _TRIAL_RULES
    ),
    "libero_spatial_task": (
        "LIBERO-Spatial: a tabletop with a wooden cabinet, a flat stove, a plate, a cookies box, "
        "a glazed ramekin and two identical black bowls. Every task is 'pick the black bowl "
        "<location> and place it on the plate'; the location phrase is what identifies the "
        "bowl. This is LIBERO-PRO's task variant: the scene and layout are those the VLA policy "
        "pi0.5 was trained on, but the location phrase is rewritten to name the other bowl "
        "(one phrase is a negation, 'not between'), so it differs from the string pi0.5 learned "
        "for that scene. " + _TRIAL_RULES
    ),
    "libero_goal_swap": (
        "LIBERO-Goal: one fixed tabletop scene with a stove (with a knob), a flat-topped "
        "cabinet with drawers, a wine rack, and movable objects (plate, bowl, wine bottle, "
        "cream cheese box, ...). Ten tasks share the scene and differ only in the goal "
        "sentence: putting one object on or in another, opening a drawer, pushing an object, "
        "turning the stove on. This is LIBERO-PRO's position-swap variant: the sentences are "
        "the ones the VLA policy pi0.5 was trained on, but the movable objects start in each "
        "other's places, so their positions differ from the training scenes. " + _TRIAL_RULES
    ),
}


def task_description(level_name: str) -> str:
    return TASK_DESCRIPTIONS.get(level_name, (
        f"LIBERO suite {level_name}: a Franka arm manipulates tabletop objects to satisfy a "
        "natural-language goal, judged only by the simulator's predicates, within 35 LLM steps "
        "and 2000 simulator steps per trial."))


def actor_prompt_text(pigey_repo: str | Path | None = None) -> str:
    """Pigey's own system prompt (``V1F_PROMPT``), read from the source without importing it.

    Quoted to the teacher so it does not bank restatements of instructions the
    orchestrator already receives; hexa_orbit measured three of five skills doing
    exactly that. The task line is left out since it varies per trial.
    """
    repo = Path(pigey_repo or os.environ.get("PIGEY_REPO") or Path(__file__).resolve().parents[2] / "Pigey")
    try:
        source = (repo / "sim/agent_sim.py").read_text()
    except OSError:
        return ""
    match = re.search(r'^V1F_PROMPT = """(.*?)"""', source, re.S | re.M)
    if not match:
        return ""
    return match.group(1).split("## Task")[0].strip()


_SETTING = """\
**Setting**: an LLM orchestrator controls a robot arm in simulation through a small set of \
tools (perception, grasp, place, a learned VLA policy, homing, release). Each trial is one \
task on one initial state. Below, every trial shows, step by step, the orchestrator's \
Reasoning, the Tool it selected with its input, and the harness Result text (camera images \
are not reproduced). The header gives the task sentence, the outcome, the LLM and simulator \
steps used, and the **Reward**:
- Successes: +1.00 (solved in <=3 LLM steps), +0.75 (<=8), +0.50 (<=15), +0.25 (slower).
- Failures: -0.50 when a budget ran out (the simulator horizon or the LLM step cap), \
-0.75 when the trial ended early (an error, or the loop stopped without using either budget).
Weigh your analysis by reward: a fast success is stronger evidence than a slow one, and a \
failure that burned the whole simulator budget on one tool shows where the budget goes."""

_ANTI_MEMORISATION = """\
Write every skill as guidance for a FRESH trial: a different initial state of one of these \
tasks, or a task with the same structure. Prefer skills about how to choose tools, inputs, \
targets and verification given what the images and results show; never bank a fact that \
holds only for one initial state (a particular detection index, pixel box or object \
position), and never bank a generic restatement of the orchestrator's own instructions."""

_SOURCE_SEEDS = """\
- **source_seeds**: the trial identifiers (strings like "t05_e03") of the trials above \
that this skill was primarily derived from. Copy them verbatim from the `--- Trial` \
headers; do not invent or abbreviate them. Include only trials whose behaviour directly \
demonstrates or motivates the skill."""


def _already_known(actor_prompt: str) -> str:
    if not actor_prompt:
        return ""
    quoted = actor_prompt.replace("{", "{{").replace("}", "}}")
    return (
        "\n**What the orchestrator is already told.** Every trial runs with this system prompt, "
        "whether or not a skill bank exists:\n\n\"\"\"\n" + quoted + "\n\"\"\"\n\n"
        "Ask of each candidate skill: does an orchestrator that has read the prompt above already "
        "know to do this? If yes, the skill is worth nothing; it costs context in every future "
        "trial and changes no decision. A skill that qualifies or contradicts the prompt based on "
        "what the trials show is admissible, but it must say mechanically what to do instead.\n"
    )


def build_prompts(actor_prompt: str = "") -> dict[str, str]:
    """The three teacher prompts, with HeXA's placeholder names and JSON contracts."""
    known = _already_known(actor_prompt)
    contrastive = (
        "You are an expert analyst of robot-manipulation agent traces, distilling behaviour into "
        "concise, actionable skills.\n\n" + _SETTING + "\n\n**Suite: {level_name}**\n{level_description}\n"
        + known +
        "\nBelow are SUCCESSFUL and FAILED trials from this suite.\n\n"
        "=== SUCCESSFUL TRIALS ({n_successes}) ===\n{success_block}\n\n"
        "=== FAILED TRIALS ({n_failures}) ===\n{failure_block}\n\n---\n\n"
        "Your task: by CONTRASTING the successes and the failures, extract the KEY SKILLS that "
        "separate completing these tasks from failing them. What did the successful trials do, in "
        "which tool they chose, how they identified the target in the images, which `surface_type` "
        "or subgoal wording they used, when they re-perceived or went home, how they spent the "
        "simulator budget, that the failed trials did not? Let the trials, not your prior "
        "expectations, determine what works.\n\n" + _ANTI_MEMORISATION + "\n\n"
        "For each skill, provide:\n"
        "- **title**: Short name (3-7 words)\n"
        "- **principle**: The insight (2-3 sentences). What was done, and why does it work?\n"
        "- **when_to_apply**: Specific trigger condition (1 sentence)\n"
        "- **example**: Optional concrete tool call or wording from a trial\n" + _SOURCE_SEEDS + "\n\n"
        "Output a JSON array:\n```json\n[\n  {{\n    \"title\": \"...\",\n    \"principle\": \"...\",\n"
        "    \"when_to_apply\": \"...\",\n    \"example\": \"...\",\n"
        "    \"source_seeds\": [\"t05_e03\", \"t08_e01\"]\n  }}\n]\n```\n\n"
        "Extract 4-6 skills. Each should capture a DISTINCT insight from the success/failure "
        "contrast. Avoid redundancy.\n")
    mistakes = (
        "You are an expert at analysing robot-manipulation agent failures and distilling them into "
        "avoidable mistake patterns.\n\n" + _SETTING + "\n\n**Suite: {level_name}**\n{level_description}\n"
        + known +
        "\nBelow are FAILED trials.\n\n{failure_block}\n\n---\n\n"
        "Your task has TWO parts:\n\n"
        "**Part 1 - Mistakes**: identify the COMMON MISTAKE PATTERNS across these failures. For each:\n"
        "1. What exactly went wrong: distinguish a wrong target choice in the image, a wrong tool or "
        "tool input, a wrong subgoal sent to the VLA policy, a missing verification, and a simulator "
        "budget spent on retries.\n"
        "2. WHY the orchestrator made this error (what mistaken belief or shortcut led to it)\n"
        "3. A concrete actionable fix\n\n"
        "**Part 2 - Partial insights**: even failed trials often contain a correct step (a clean grasp "
        "before a bad placement, a correct target identification followed by the wrong surface type). "
        "Extract 1-2 skills from these good steps within bad trials. They must be genuine reusable "
        "insights, not restatements of what went wrong.\n\n" + _ANTI_MEMORISATION + "\n\n"
        "Format as a JSON object with two arrays:\n```json\n{{\n  \"mistakes\": [\n    {{\n"
        "      \"description\": \"What the mistake is (1 sentence)\",\n"
        "      \"why_it_happens\": \"The mistaken belief or reasoning error that causes it (1 sentence)\",\n"
        "      \"how_to_avoid\": \"Concrete actionable fix: what to do instead (1-2 sentences)\"\n"
        "    }}\n  ],\n  \"partial_skills\": [\n    {{\n      \"title\": \"Short name (3-7 words)\",\n"
        "      \"principle\": \"The reusable insight from the failed trial (2-3 sentences)\",\n"
        "      \"when_to_apply\": \"Specific trigger condition (1 sentence)\",\n"
        "      \"source_seeds\": [\"t05_e03\"]\n    }}\n  ]\n}}\n```\n\n"
        "Extract 3-5 mistakes and 1-2 partial skills. Group similar failures and focus on ROOT "
        "CAUSES. Only extract partial skills that are genuinely useful.\n")
    evolution = (
        "You are a teacher evolving a skill bank for an LLM orchestrator that controls a robot arm.\n\n"
        + _SETTING + "\n\nSUITE: {level_name}\nSUITE DESCRIPTION: {level_description}\n" + known +
        "\nCURRENT SKILL BANK (from previous rounds):\n{existing_skills_block}\n\n"
        "NEW TRIALS (from the latest round):\nSuccesses: {n_successes}\nFailures: {n_failures}\n\n"
        "{new_trajectories_block}\n\n"
        "YOUR TASK: evolve the skill bank by merging the existing skills with insights from the new "
        "trials.\n\nRULES:\n"
        "1. Output the COMPLETE FINAL skill bank (not a diff): retained existing skills and any new ones.\n"
        "2. Hard constraints: at most {max_skills} skills and {max_mistakes} mistakes for this suite.\n"
        "3. For each skill: \"is_new\": false if retained from the existing bank, true if extracted from "
        "the new trials; \"source_seeds\" listing the trial identifiers (strings like \"t05_e03\") where "
        "it was observed; \"confidence\": a float in [0.1, 1.0].\n"
        "4. CONFIDENCE MOVES IN BOTH DIRECTIONS. If the orchestrator FOLLOWED a retained skill and the "
        "trial SUCCEEDED, you may raise its confidence slightly. If it FOLLOWED the skill and the trial "
        "FAILED, you MUST lower it by at least 0.1. Not applicable this round: leave it unchanged. Never "
        "raise the confidence of a skill that appeared in a failing trial.\n"
        "5. New skills: confidence from the success rate among source trials, how many tasks and "
        "initial states they generalise over, and how actionable the principle is. A skill drawn only "
        "from failures starts at confidence <= 0.4.\n"
        "6. No duplicates. If new trials confirm an existing skill, keep the existing one.\n"
        "7. PRESERVE VALIDATED SPECIFICS. Concrete tool inputs, surface types, subgoal wordings and "
        "ordering rules inside a retained skill are evidence; copy them through exactly unless a new "
        "trial in this round shows a different choice working better, and then cite that trial.\n"
        "8. Remove skills that are redundant, contradicted by the new trials, too specific to one "
        "initial state, low confidence (< 0.3) and unobserved this round, or present for several failed "
        "trials while contributing no successes since being added.\n"
        "9. Do NOT bank skills that rationalise failure or cannot be acted on (\"this task may be "
        "unsolvable\", \"results vary per layout\"). Every skill must tell the orchestrator a SPECIFIC "
        "THING TO DO.\n"
        "10. Do NOT bank skills that only forbid acting. The orchestrator has bounded budgets; prefer "
        "skills that say what to try next over skills that gate.\n"
        "11. MISTAKES ARE HELD TO THE SAME BAR. \"how_to_avoid\" must name a specific thing to do, not "
        "refer to another skill by name. Drop a mistake that no longer appears in the new trials, and "
        "prefer few sharp mistakes over a full set.\n"
        "12. MISTAKES MUST BE CONTRASTIVE. For every mistake, identify what the successful trials did "
        "differently at the same decision point and state THAT in \"how_to_avoid\". If no contrasting "
        "success exists, drop the mistake.\n"
        "13. THE BANK MUST BE SELF-CONSISTENT: resolve any two entries that give opposing guidance on "
        "the same decision, keeping the one the trials in this round support.\n\n"
        "OUTPUT JSON OBJECT:\n{{\n  \"skills\": [\n    {{\n      \"title\": \"<short name>\",\n"
        "      \"principle\": \"<2-3 sentence insight>\",\n      \"when_to_apply\": \"<condition>\",\n"
        "      \"example\": \"<optional concrete tool call or wording>\",\n"
        "      \"source_seeds\": [\"<trial id>\", ...],\n      \"confidence\": <float in [0.1, 1.0]>,\n"
        "      \"is_new\": <true|false>\n    }}\n  ],\n  \"mistakes\": [\n    {{\n"
        "      \"description\": \"<what the mistake is>\",\n      \"why_it_happens\": \"<why>\",\n"
        "      \"how_to_avoid\": \"<actionable fix>\",\n      \"is_new\": <true|false>\n    }}\n  ],\n"
        "  \"removed_skill_titles\": [\"<title>\", ...],\n"
        "  \"confidence_changes\": [{{\"title\": \"<skill>\", \"from\": <old>, \"to\": <new>, "
        "\"why\": \"<followed in failing trial N / confirmed by trial M>\"}}],\n"
        "  \"reasoning\": \"<brief explanation of what was removed, added and why>\"\n}}\n\n"
        "Be concise but precise.\n")
    return {"contrastive": contrastive, "mistakes": mistakes, "evolution": evolution}


# ── Same-task contrast selection ─────────────────────────────────────────

def _round_robin(trajectories: list[dict], count: int, reward, best_first: bool) -> list[dict]:
    """Up to ``count`` trajectories, one task at a time, so no task monopolises the block.

    Rewards tie a lot here (every failure that ran out of simulator steps scores
    -0.5), and a plain sort would then fill every failure slot from task 0.
    """
    by_task: dict = {}
    for trajectory in sorted(trajectories, key=reward, reverse=best_first):
        by_task.setdefault(trajectory.get("task_id"), []).append(trajectory)
    queues = [by_task[task] for task in sorted(by_task, key=lambda task: reward(by_task[task][0]), reverse=best_first)]
    chosen: list[dict] = []
    while queues and len(chosen) < count:
        for queue in list(queues):
            if len(chosen) >= count:
                break
            chosen.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return chosen


def select_task_paired(trajectories: list[dict], max_trajs: int, reward, success_slots: int | None = None) -> list[dict]:
    """Best success with worst failure on the same task first, then a task-spread balanced backfill.

    The backfill keeps hexa_bridge's success share (3 of 5) but draws each class
    round-robin across tasks; a class with too few members yields its slots to the
    other, so a block never comes up short while trials remain.
    """
    from hexa_core import hexa_bridge

    if success_slots is None:
        success_slots = hexa_bridge.default_success_slots(max_trajs)
    by_task: dict = {}
    for trajectory in trajectories:
        by_task.setdefault(trajectory.get("task_id"), []).append(trajectory)
    pairs = []
    for group in by_task.values():
        successes, failures = hexa_bridge._split(group)
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
    if len(chosen) < max_trajs:
        used = {id(t) for t in chosen}
        successes, failures = hexa_bridge._split([t for t in trajectories if id(t) not in used])
        remaining = max_trajs - len(chosen)
        want_success = max(0, min(success_slots - sum(1 for t in chosen if t.get("success")), remaining))
        picked_success = _round_robin(successes, want_success, reward, best_first=True)
        picked_failure = _round_robin(failures, remaining - len(picked_success), reward, best_first=False)
        if len(picked_success) + len(picked_failure) < remaining:
            # One class ran short: let the other fill what is left.
            extra = remaining - len(picked_success) - len(picked_failure)
            leftovers = [t for t in successes if t not in picked_success]
            picked_success += _round_robin(leftovers, extra, reward, best_first=True)
        chosen += picked_success + picked_failure
    return chosen


def _make_block(forced: int | None, label: str):
    """A ``format_trajectories_block`` replacement; ``forced`` pins the evolver's width.

    ``evolving_distill`` hardcodes ``max_trajs=5`` at its call site, so the evolution
    binding overrides it; ``distill`` asks for what its prompt budgets (successes and
    failures in separate calls), so there the caller's number is honoured.
    """
    def block(trajectories: list[dict], max_trajs: int = 5) -> str:
        import skillrl.distillation.teacher_prompts as _tp

        limit = forced if forced is not None else max_trajs
        chosen = select_task_paired(trajectories, limit, _tp.compute_trajectory_reward)
        _tp.record_selection(chosen)
        n_success = sum(1 for t in chosen if t.get("success"))
        tasks = sorted({t.get("task_id") for t in chosen})
        print(f"    {label} RECEIVED {len(chosen)}/{limit} trials: {n_success} success / "
              f"{len(chosen) - n_success} failure (of {len(trajectories)} available; tasks {tasks})")
        rendered = "\n\n".join(_tp.format_trajectory_for_teacher(t) for t in chosen)
        print(f"    block size: {len(rendered)} chars (~{len(rendered) // 4} tokens)")
        return rendered
    return block


# ── Teacher call through the Claude Code CLI ─────────────────────────────

CONTEXT_TO_IGNORE = ("# Context to ignore\nYou are analysing robot-agent trajectories, not working as a "
                     "coding assistant. Ignore any CLAUDE.md or cluster rules, environment details, user "
                     "account information, and dates supplied by the CLI; none of it applies here.\n\n")


class ClaudeTeacher:
    """``call_teacher`` replacement: stdin prompt, JSON result, every call logged to ``log_dir``."""

    def __init__(self, log_dir: str | Path, *, executable: str = "claude", timeout: int = 900, stock=None):
        self.log_dir = Path(log_dir)
        self.executable = shutil.which(executable) or executable
        self.timeout = timeout
        self.stock = stock
        self.calls: list[dict] = []

    def _next_index(self) -> int:
        """One past the highest ``teacher_NN_prompt.txt`` already in ``log_dir``.

        Numbering from the files, not from ``self.calls``: hexa_loop clears ``calls``
        after every round, and a resumed run or a later final evolution reuses the
        same directory, so a call counter would overwrite earlier prompts.
        """
        taken = [int(m.group(1)) for p in self.log_dir.glob("teacher_*_prompt.txt")
                 if (m := re.fullmatch(r"teacher_(\d+)_prompt\.txt", p.name))]
        return max(taken, default=0) + 1

    def __call__(self, prompt: str, model: str = DEFAULT_TEACHER_MODEL) -> str:
        if not (model.startswith("claude-") or model in ("sonnet", "opus", "haiku", "fable")):
            if self.stock is None:
                raise ValueError(f"Teacher model {model!r} is not a Claude model and no fallback is bound")
            return self.stock(prompt, model=model)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        index = self._next_index()
        (self.log_dir / f"teacher_{index:02d}_prompt.txt").write_text(prompt)
        command = [self.executable, "-p", "--model", model, "--max-turns", "1", "--tools", "",
                   "--no-session-persistence", "--output-format", "json",
                   "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                   "--settings", '{"disableAllHooks":true}']
        env = os.environ.copy()
        if env.get("ANTHROPIC_API_KEY") == "pigey-claude-code-unused":
            env.pop("ANTHROPIC_API_KEY")
        env.pop("CLAUDECODE", None)
        env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
        started = time.monotonic()
        record = {"index": index, "model": model, "prompt_chars": len(prompt)}
        waited, attempt = 0.0, 0
        while True:
            with tempfile.TemporaryDirectory(prefix="teacher-", dir=self.log_dir, ignore_cleanup_errors=True) as work:
                try:
                    completed = subprocess.run(command, input=CONTEXT_TO_IGNORE + prompt, text=True,
                                               capture_output=True, cwd=work, env=env, timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    record.update(error=f"timeout after {self.timeout}s",
                                  duration_s=round(time.monotonic() - started, 1))
                    self.calls.append(record)
                    print(f"[Teacher Error] {record['error']}")
                    return ""
            # A usage limit is waited out (as for the actor) rather than costing the round its evolution.
            if not completed.returncode or rate_limit_wait(completed.stdout + completed.stderr, 0) is None:
                break
            wait = wait_for_usage_limit(completed.stdout + completed.stderr, attempt, waited, "teacher")
            if wait is None:
                break
            record.setdefault("rate_limit_waits", []).append(round(wait))
            waited += wait
            attempt += 1
        record["duration_s"] = round(time.monotonic() - started, 1)
        if completed.returncode:
            record["error"] = f"exit={completed.returncode}: {(completed.stderr or completed.stdout)[-1000:]}"
            self.calls.append(record)
            print(f"[Teacher Error] {record['error'][:300]}")
            return ""
        try:
            output = json.loads(completed.stdout)
        except json.JSONDecodeError:
            record["error"] = "non-JSON CLI output"
            self.calls.append(record)
            print(f"[Teacher Error] {record['error']}: {completed.stdout[:300]}")
            return ""
        text = (output.get("result") or "").strip()
        for key in ("total_cost_usd", "usage", "num_turns", "duration_ms", "is_error", "subtype"):
            if key in output:
                record[key] = output[key]
        self.calls.append(record)
        (self.log_dir / f"teacher_{index:02d}_response.txt").write_text(text)
        if output.get("is_error") or output.get("subtype") != "success":
            print(f"[Teacher Error] CLI result {output.get('subtype')}: {text[:300]}")
            return ""
        return text


# ── Injection rendering ──────────────────────────────────────────────────

#: Fits a full bank (10 skills, 5 mistakes; measured about 2500 tokens). The first
#: loop run used 1200, which dropped every mistake and the newest skills from every
#: trial; this default is shared by run_claude.py and hexa_loop.py so the summary a
#: round prints matches what its trials received.
DEFAULT_INJECTION_BUDGET = 4000


def render_block(bank, level_name: str, *, budget_tokens: int = DEFAULT_INJECTION_BUDGET, max_specific: int = 10,
                 max_mistakes: int = 5):
    """The skills block for the orchestrator's system prompt, within a token budget.

    Wraps ``hexa_bridge.retrieve_within_token_budget`` (confidence-ordered retrieval,
    ``SkillBank.format_skills_as_prompt`` rendering) with this suite's heading. The
    returned ``RetrievedBlock`` reports what was dropped for budget, so a trial record
    can say exactly what the orchestrator saw.
    """
    from hexa_core import hexa_bridge

    hexa_bridge.SKILLS_BLOCK_HEADING = SKILLS_BLOCK_HEADING
    return hexa_bridge.retrieve_within_token_budget(bank, level_name, budget_tokens, max_general=0,
                                                    max_specific=max_specific, max_mistakes=max_mistakes)


# ── install ──────────────────────────────────────────────────────────────

_STOCK: dict = {}
_INSTALLED: dict = {}


def install(*, level_name: str = DEFAULT_SUITE, teacher_log_dir: str | Path | None = None,
            max_trajs: int = DEFAULT_MAX_TRAJS, executable: str = "claude", actor_prompt: str | None = None,
            teacher_timeout: int = 900) -> ClaudeTeacher | None:
    """Install the Pigey reward, formatter, loader, prompts and teacher into ``skillrl``.

    Layered on ``hexa_core.hexa_bridge.patch``, which owns the reward/confidence
    rebinding across the modules that each hold their own reference. HeXA's stock
    confidence is kept: it keys trajectories by seed, and a Pigey seed is one trial,
    exactly the 1:1 case it was written for. Idempotent; call before any
    ``run_distillation`` or ``evolve_skill_bank``.
    """
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving
    import skillrl.distillation.teacher_prompts as _tp
    from hexa_core import hexa_bridge

    if not _STOCK:
        _STOCK.update(confidence=_distill._compute_initial_confidence, call_teacher=_distill.call_teacher)
    # patch() refuses to overwrite a level some other port owns, but accepts a key
    # it added itself, so re-installing for the same suite is harmless.
    hexa_bridge.patch(reward_fn=trajectory_reward, confidence_fn=_STOCK["confidence"],
                      trajectory_selection="balanced", max_trajs=max_trajs,
                      task_descriptions={level_name: task_description(level_name)},
                      load_trajectories_fn=load_trial_trajectories)
    hexa_bridge.SKILLS_BLOCK_HEADING = SKILLS_BLOCK_HEADING

    prompts = build_prompts(actor_prompt_text() if actor_prompt is None else actor_prompt)
    teacher = None
    if teacher_log_dir is not None:
        teacher = ClaudeTeacher(teacher_log_dir, executable=executable, timeout=teacher_timeout,
                                stock=_STOCK["call_teacher"])
    bindings = [
        (_tp, "format_trajectory_for_teacher", format_trajectory_for_teacher),
        (_tp, "format_trajectories_block", _make_block(None, "Seeding")),
        (_distill, "format_trajectories_block", _make_block(None, "Seeding")),
        (_evolving, "format_trajectories_block", _make_block(max_trajs, "Evolver")),
        (_distill, "CONTRASTIVE_DISTILLATION_PROMPT", prompts["contrastive"]),
        (_distill, "COMMON_MISTAKES_PROMPT", prompts["mistakes"]),
        (_evolving, "SKILL_BANK_EVOLUTION_PROMPT", prompts["evolution"]),
    ]
    if teacher is not None:
        # Both modules hold their own reference to call_teacher.
        bindings += [(_distill, "call_teacher", teacher), (_evolving, "call_teacher", teacher)]
    for module, attr, value in bindings:
        setattr(module, attr, value)
    _INSTALLED.update(level=level_name, teacher=teacher)
    return teacher


def is_installed() -> bool:
    return bool(_INSTALLED)
