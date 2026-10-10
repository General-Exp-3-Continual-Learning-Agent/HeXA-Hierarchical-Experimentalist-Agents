"""Reflexion (Shinn et al., 2023) on ORBIT's Maze and Mastermind.

The method is verbal reinforcement: attempt the task, fail, write a
self-reflection about *why*, retry the task with that reflection in context and
nothing else from the failed attempt. The reflection is the memory, and it is
the only thing that crosses a trial boundary.

That last clause is what makes the mapping onto ORBIT non-obvious. ORBIT already
retries: ``MazeActor.run_instance`` plays three episodes on one maze (or one
code) and **never resets** ``self.messages``, so episode 3 reads episodes 1 and
2 verbatim. Simply adding a reflection between episodes would therefore explain
a trajectory the model is already holding -- Reflexion's own "trajectory-only
memory" ablation, not the method.

Three variants are implemented, and they differ only in where the boundary is
and whether the chat survives it:

``episode-reset`` -- **the reported arm.** The trial is one episode. At each
  episode boundary the agent reflects on the episode that just ended and the
  chat is truncated to the system message, so the lesson is genuinely the only
  thing that crosses. 3 episodes = 3 Reflexion trials.

``episode-keep`` -- its ablation. Same reflections, same boundaries, but the
  transcript is left in place. Isolates how much of any effect is the *analysis*
  rather than the memory, since here the trajectory was never lost. This is the
  condition that is nearly identical to the no-bank baseline.

``trials`` -- the instance-level reading: play all three episodes, reflect on
  the whole trial, replay the instance from a clean chat. Kept because its
  trial 1 injects nothing and is therefore a no-bank replicate produced inside
  the same run.

Common to all three:

* **No build pass.** Reflexion has no cross-instance memory -- the paper resets
  it per task -- so unlike Episodic RAG and MemRL there is nothing to construct
  before evaluating, and `docs/MEMORY_BASELINES_ORBIT.md`'s build-batching
  discussion does not apply.
* The injection point is ``MazeActor(guidelines=...)``, the same seam the skill
  bank and the two memory baselines use, so a reflection costs tokens in exactly
  the place a skill costs them and the arms are comparable at equal added
  context.

``episode-reset`` removes something the no-bank baseline has (the cross-episode
transcript), so a loss against baseline is ambiguous on its own -- reflection
underperforming and lost transcript are not separable. ``episode-keep`` is what
separates them, which is why the two are meant to be read together.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from hexa_orbit.actor import MazeActor

from hexa_orbit.actor import InstanceRun

# ── what the reflector is allowed to see ────────────────────────────────────
#
# The reflection is written into the next trial's system prompt, so anything the
# reflector reads, the agent effectively reads. Everything below is built from
# `InstanceRun.steps` (observations the environment emitted and actions the agent
# chose) and `rollout.format_verifier_feedback` (per-episode solved/failed and
# turn counts). Both are agent-observable.
#
# Deliberately NOT passed, mirroring `rollout._PRIVILEGED_ON_MAZE`:
#   spec.shortest_path_len  the config's path-length filter, not anything seen
#   best_progress (maze)    computed by the oracle on the known map
#   reward (maze)           penalises steps against the known-map optimum
# A reflection quoting any of those would win on a privileged channel, and it
# would do so invisibly -- the text reads like a lesson either way.

_MAZE_SYSTEM = """You are analysing a failed attempt at a grid maze navigation task.

The agent sees only its own (x, y) position and whether each of the four neighbouring cells is a wall, a path, or the goal. It cannot see the map. It got three episodes on the SAME maze and failed to solve it reliably.

Write a short reflection (at most 5 sentences) that the agent will read before a fresh attempt at this SAME maze, with no memory of the attempt you are analysing. Cover:
(a) what search or movement strategy it actually followed;
(b) the specific reason that failed -- cite concrete coordinates, the cells it revisited, the dead ends it re-entered, or the turns it wasted;
(c) one concrete different strategy for the next attempt, naming coordinates or directions where you can.

Hard rules:
- The agent will NOT have the failed transcript. Anything it needs must be in your reflection -- name the coordinates that mattered.
- Do NOT invent map knowledge the observations did not show.
- Do NOT repeat lessons already listed as prior reflections.
- Output ONLY the reflection text. No preamble, no headings, no markdown."""

_MASTERMIND_SYSTEM = """You are analysing a failed attempt at Mastermind.

The secret is a 3-digit code using digits 1-6 with no repeats. After each guess the agent is told how many digits are correct and in the right place (black) and how many are correct but misplaced (white). It got three episodes on the SAME secret code, three guesses each, and failed to crack it reliably.

Write a short reflection (at most 5 sentences) that the agent will read before a fresh attempt at this SAME code, with no memory of the attempt you are analysing. Cover:
(a) which guesses it made and what the feedback ruled in or out;
(b) where its deduction went wrong -- a guess inconsistent with earlier feedback, a wasted probe, digits never tested;
(c) a concrete opening guess and follow-up policy for the next attempt.

Hard rules:
- The agent will NOT have the failed transcript. Carry the useful constraints forward explicitly -- state which digits and positions the feedback already eliminated.
- Do NOT state or guess the secret code as though you know it; you do not. Reason only from the feedback shown.
- Do NOT repeat lessons already listed as prior reflections.
- Output ONLY the reflection text. No preamble, no headings, no markdown."""

REFLECTION_SYSTEM = {"maze": _MAZE_SYSTEM, "mastermind": _MASTERMIND_SYSTEM}

#: Budget for a reflection call. The prompt asks for <=5 sentences, so the
#: *answer* needs a few hundred tokens -- but on a hybrid thinking model the
#: budget spans reasoning + answer, and Qwen3 expands its thinking to fill
#: whatever it is given: measured 3,991 reasoning chars at 2048 and 8,755 at
#: 8192 on the same prompt. At 2048 a long transcript therefore spends the whole
#: budget thinking and the answer channel comes back EMPTY -- 3 of 4 instances in
#: the first Mastermind pilot, each silently degrading to the no-bank condition.
#: 8192 leaves headroom; `reflect` retries once above it rather than trusting it.
DEFAULT_REFLECTION_MAX_TOKENS = 8192


def format_trial(run: InstanceRun, verifier_feedback: str, *, max_obs_chars: int = 400) -> str:
    """Render one failed trial for the reflector: episodes, turns, verdict.

    Grouped by episode rather than flat, because "it solved it on episode 3
    after wasting two" and "it never solved it" are different lessons and a flat
    step list hides the difference. Observations are truncated head+tail; a maze
    observation is short but a Mastermind one accumulates guess history.
    """
    lines: list[str] = []
    current: Optional[int] = None
    for step in run.steps:
        if step.episode_index != current:
            current = step.episode_index
            lines.append(f"\n=== Episode {current + 1} ===")
        obs = step.observation or ""
        if len(obs) > max_obs_chars:
            head = max_obs_chars // 2
            obs = obs[:head] + " ...[truncated]... " + obs[-(max_obs_chars - head):]
        lines.append(f"turn {step.episode_step}: {obs.strip()}")
        lines.append(f"  -> action: {step.env_action}")
        if step.episode_done:
            lines.append("  (episode ended)")
    lines.append("")
    lines.append(verifier_feedback)
    return "\n".join(lines)


def reflect(
    client: Any,
    task_name: str,
    run: InstanceRun,
    verifier_feedback: str,
    past_reflections: Sequence[str] = (),
    *,
    max_new_tokens: int = DEFAULT_REFLECTION_MAX_TOKENS,
) -> str:
    """One reflection on one failed trial. ``""`` means "no lesson".

    Returns empty rather than raising on every failure path -- a dead reflection
    call must degrade the instance to the no-bank condition, not kill the arm.
    An empty reflection is recorded and counted; `n_empty_reflection` in the
    summary is the number of trial-2 instances that were really no-bank, the same
    diagnostic `retrieval.n_empty_block` serves for the memory baselines.
    """
    if not run.steps:
        return ""
    system = REFLECTION_SYSTEM[task_name]

    past_block = ""
    if past_reflections:
        past_block = "\n\nPrior reflections (do not repeat these):\n" + "\n".join(
            f"{i}. {r.strip()}" for i, r in enumerate(past_reflections, 1)
        )

    user = (
        "Outcome: FAILURE.\n"
        f"\nWhat happened:\n{format_trial(run, verifier_feedback)}"
        f"{past_block}"
        "\n\nWrite the reflection now. At most 5 sentences."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Two attempts, the second with a doubled budget. An empty answer channel on
    # a thinking model almost always means the reasoning ran to the cap and the
    # response was cut before the answer began -- recoverable, and worth one
    # retry, because the alternative is an instance that quietly reverts to
    # no-bank. Anything still empty after that is recorded as empty.
    for attempt, budget in enumerate((max_new_tokens, max_new_tokens * 2)):
        try:
            generation = client.generate_structured(
                messages=messages, max_new_tokens=budget
            )
        except Exception as exc:  # noqa: BLE001 -- see docstring
            print(f"[reflexion] reflection call failed: {type(exc).__name__}: {exc}",
                  flush=True)
            return ""
        if generation is None:
            print("[reflexion] reflection call returned no answer", flush=True)
            return ""
        # `final_text`, not `raw_text`: with Qwen3's reasoning parser the latter
        # carries the whole thinking channel, which would land in the next
        # trial's system prompt -- hundreds of tokens of discarded deliberation
        # the agent would read as instruction. A truncated `raw_text` is worse
        # still: it is thinking with no conclusion attached.
        text = (generation.final_text or "").strip()
        if text:
            return text
        reasoned = len(generation.reasoning_text or "")
        print(f"[reflexion] empty answer channel at budget {budget} "
              f"(finish={generation.finish_reason}, {reasoned} reasoning chars)"
              + ("; retrying" if attempt == 0 else "; giving up"), flush=True)
    return ""


def reflexion_block(reflections: Sequence[str]) -> Optional[str]:
    """The system-prompt block for a retry, or ``None`` for an untouched prompt.

    ``None`` rather than ``""`` so trial 1 -- and any trial whose reflections all
    came back empty -- is byte-identical to the no-bank baseline: `MazeActor`
    appends nothing when `guidelines` is falsy, and that identity is what makes
    trial 1 usable as the control.
    """
    kept = [r.strip() for r in reflections if r and r.strip()]
    if not kept:
        return None
    lines = [
        "## Lessons from your previous attempts at THIS task",
        "You have attempted this exact task before and failed. You cannot see those",
        "attempts, but you wrote down what went wrong:",
        "",
    ]
    lines += [f"{i}. {r}" for i, r in enumerate(kept, 1)]
    lines += [
        "",
        "Use these lessons to try a different approach. Do not repeat the strategy "
        "that already failed.",
    ]
    return "\n".join(lines)


# ── episode-level variants ──────────────────────────────────────────────────


class EpisodeReflexionActor(MazeActor):
    """Reflects between ORBIT episodes, inside a single instance.

    Hooks :meth:`MazeActor.on_episode_boundary`, which runs after an episode
    ends and before the next episode's system message is built. At that point
    this actor:

    1. reflects on the episode that just finished, if it earned a reflection;
    2. writes the accumulated lessons into ``self.guidelines[next_episode]`` --
       ``block_for_episode`` is already a per-episode lookup, so no new
       injection machinery is needed;
    3. truncates the chat to the system message when ``reset_context`` is set.

    Step 3 is the difference between the method and its ablation. With the
    reset the lesson is the only thing crossing the boundary, which is
    Reflexion. Without it the agent keeps ORBIT's cumulative transcript and the
    reflection only adds analysis.

    ``guidelines`` starts empty, so episode 1's prompt is byte-identical to the
    no-bank baseline's and ``added_tokens`` is 0 until the first lesson lands.

    Note on the record: with ``reset_context`` the returned
    ``InstanceRun.messages`` holds only the final episode's chat, because that
    is genuinely all the model had. Nothing is lost -- ``steps`` carries every
    observation, action and response from all three episodes.
    """

    def __init__(
        self,
        config,
        *,
        reflector,
        task_name: str,
        reset_context: bool = True,
        reflection_max_tokens: int = DEFAULT_REFLECTION_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        # A dict, not None: `block_for_episode` treats a mapping as a
        # per-episode lookup, and an empty one yields no block for episode 0.
        super().__init__(config, guidelines={}, **kwargs)
        self._reflector = reflector
        self._task_name = task_name
        self._reset_context = reset_context
        self._reflection_max_tokens = reflection_max_tokens
        #: One entry per boundary that earned a reflection, for the run record.
        self.reflection_log: list[dict] = []

    def _earned_reflection(self, instance, finished_episode: int) -> bool:
        """An episode is binary, so this is just "did it fail"."""
        try:
            return not instance.episodes[finished_episode].success
        except (IndexError, AttributeError):
            return False

    def on_episode_boundary(self, instance, finished_episode, steps) -> None:
        next_episode = finished_episode + 1
        lessons = [e["reflection"] for e in self.reflection_log if e["reflection"]]

        if self._earned_reflection(instance, finished_episode):
            episode_steps = [s for s in steps if s.episode_index == finished_episode]
            text = ""
            if episode_steps:
                text = reflect(
                    self._reflector,
                    self._task_name,
                    _EpisodeView(episode_steps),
                    _episode_verdict(instance, finished_episode, episode_steps),
                    lessons,
                    max_new_tokens=self._reflection_max_tokens,
                )
            self.reflection_log.append({
                "after_episode": finished_episode,
                "reflection": text,
                "empty": not text,
            })
            if text:
                lessons = lessons + [text]

        block = reflexion_block(lessons)
        if block:
            # `block_for_episode` falls back to the largest key at or below the
            # episode, so writing only at the boundary is enough -- the lesson
            # persists into later episodes without being rewritten.
            self.guidelines[next_episode] = block

        if self._reset_context:
            # Everything after the system message goes. `run_instance` rebuilds
            # slot 0 immediately after this returns and then appends the new
            # episode's first observation, so the next episode opens on
            # [system(with lessons), user(observation)] and nothing else.
            del self.messages[1:]


class _EpisodeView:
    """Minimal stand-in exposing the one attribute :func:`format_trial` reads."""

    def __init__(self, steps):
        self.steps = steps


def _episode_verdict(instance, episode_index: int, episode_steps) -> str:
    """Agent-observable verdict for one episode.

    Deliberately not `rollout.format_verifier_feedback`, which summarises a
    whole instance. Solved/failed and a turn count are all this needs, and the
    agent saw both. `EpisodeRecord.progress` is computed from the known map and
    never appears here.
    """
    try:
        solved = bool(instance.episodes[episode_index].success)
    except (IndexError, AttributeError):
        solved = False
    return (f"Verdict: episode {episode_index + 1} "
            f"{'SOLVED' if solved else 'FAILED'} after {len(episode_steps)} turns.")
