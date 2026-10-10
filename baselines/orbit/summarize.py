"""MemRL's two memory-building prompts, written for ORBIT.

MemRL builds a memory by calling an LLM once per finished episode:

  "script"      proceduralization of a *successful* trajectory into a handful of
                high-level steps (`memrl/providers/llm.py:generate_script`,
                temperature 0.0);
  "reflection"  post-mortem of a *failed* one
                (`memrl/service/updater.py:_generate_reflection`, temp 0.3).

`baselines/memory/summarize.py` carries the InterPhyre wording of that pair.
These are the same two calls at the same two temperatures with the task
description swapped -- the method is the call structure, not the prose.

Two things are said explicitly in both prompts because getting them wrong is the
failure mode that actually shows up:

* **The unit is a three-episode instance, not one attempt.**  The whole reason
  ORBIT is a multi-episode benchmark is that an agent is supposed to carry what
  episode 1 discovered into episode 2.  A summary that treats the transcript as
  a single flat attempt throws away the only thing worth remembering.
* **A memory must not contain the answer.**  The route through *that* maze and
  *that* secret code are useless on the next instance and actively misleading:
  the retrieved block is read by an agent facing a different hidden task.  The
  prompt asks for the search behaviour instead.

The second point is a live risk rather than a hypothetical.  A small actor
summarising its own transcripts will happily write down "the code was 2 4 6" as
though it were a lesson, and on a task where retrieval is degenerate (Mastermind
-- see `intent.py`) that memory is then injected into instances it cannot
possibly apply to.  Check `memory_store.json` after a run.
"""

from __future__ import annotations

from typing import Callable

from baselines.memory.summarize import (
    REFLECTION_TEMPERATURE,
    SCRIPT_TEMPERATURE,
    make_model_fn_summarizer,
)

_COMMON = """The agent played ONE instance of this task for THREE episodes. Every episode resets the environment but the hidden task stays the same, and the agent keeps the whole conversation in context, so it is expected to use what earlier episodes revealed.

The hidden part of the task is different on every instance. Anything specific to THIS instance's hidden solution is useless to a future reader and must not appear in your output."""

MAZE_SCRIPT = f"""You are analyzing a SUCCESSFUL attempt at a hidden-maze navigation task.

{_COMMON}

The agent sees only its own coordinates and whether each of the four neighbouring cells is wall, path or goal. It never sees the map.

Produce a concise, high-level script capturing the search behaviour that worked. It must be:
1. Generic across mazes: about how to explore, backtrack and use what earlier episodes revealed, never about this maze's specific route.
2. Specific enough to act on: name the decision rule and the cue that triggers it.
3. At most 5 high-level steps.
4. Explicit about what changed between episodes, if anything did.

Do NOT state the route, the goal coordinates, or any specific sequence of moves.

Output ONLY the numbered script. No preamble, no headers, no markdown fences."""

MAZE_REFLECTION = f"""You are analyzing a FAILED attempt at a hidden-maze navigation task.

{_COMMON}

The agent sees only its own coordinates and whether each of the four neighbouring cells is wall, path or goal. It never sees the map.

This attempt failed. Analyze what went wrong and what a future attempt should do differently. Cover:
1. The search behaviour that wasted turns — revisiting cells, walking into walls, oscillating between two cells.
2. Whether episodes 2 and 3 changed anything after episode 1 failed, and if not, what they should have carried over.
3. What to avoid next time, as a rule that applies to any maze.

Do NOT state this maze's route or goal position.

Keep it under 8 sentences. Output ONLY the reflection text. No preamble, no headers, no markdown fences."""

MASTERMIND_SCRIPT = f"""You are analyzing a SUCCESSFUL attempt at Mastermind (a hidden 3-digit code, digits 1-6, no duplicates, 3 guesses per episode).

{_COMMON}

After each guess the agent is told how many digits are correct and in position (black pegs) and how many are correct but misplaced (white pegs).

Produce a concise, high-level script capturing the guessing strategy that worked. It must be:
1. Generic across codes: about how to choose an opening guess and how to use peg feedback to narrow the candidates, never about this instance's code.
2. Specific enough to act on: state the inference rule, not a vague intention.
3. At most 5 high-level steps.
4. Explicit about what the agent carried from earlier episodes, if anything.

Do NOT state the secret code or the winning guess.

Output ONLY the numbered script. No preamble, no headers, no markdown fences."""

MASTERMIND_REFLECTION = f"""You are analyzing a FAILED attempt at Mastermind (a hidden 3-digit code, digits 1-6, no duplicates, 3 guesses per episode).

{_COMMON}

After each guess the agent is told how many digits are correct and in position (black pegs) and how many are correct but misplaced (white pegs).

This attempt failed. Analyze what went wrong and what a future attempt should do differently. Cover:
1. Whether any guess was inconsistent with the peg feedback the agent had already received — that is, a code its own evidence had already ruled out.
2. Whether episodes 2 and 3 repeated guesses from episode 1 instead of continuing the search.
3. What to avoid next time, as a rule that applies to any code.

Do NOT state this instance's secret code.

Keep it under 8 sentences. Output ONLY the reflection text. No preamble, no headers, no markdown fences."""

#: Generation cap for a memory-building call.
#:
#: 1024 was the InterPhyre default and it is wrong for a thinking model, which
#: is what makes this constant worth a comment. Measured on the first Mastermind
#: MemRL run (Qwen3-8B, cap 1024): **44 of 80 stored memories were raw,
#: mid-sentence chain of thought** ("Okay, let's try to figure out the strategy
#: the agent used here...") and only 9 carried the numbered script the prompt
#: asks for. Qwen3 spends its budget in the reasoning channel first, so at 1024
#: it frequently never reaches the answer channel at all. The run scored
#: episode-3 0.05 against a 0.225-0.400 no-bank baseline -- it was injecting ~5k
#: tokens of the model's own unfinished deliberation as "memory".
DEFAULT_SUMMARIZER_MAX_TOKENS = 4096


SYSTEMS: dict[str, dict[str, str]] = {
    "maze": {"script": MAZE_SCRIPT, "reflection": MAZE_REFLECTION},
    "mastermind": {
        "script": MASTERMIND_SCRIPT,
        "reflection": MASTERMIND_REFLECTION,
    },
}


#: A summary that starts like this is the model thinking, not answering.
#: Cheap prefix test rather than a classifier: what actually landed in the store
#: was "Okay, let's try to figure out...", and the point is to notice at all.
_DELIBERATION_PREFIXES = (
    "okay", "let me", "let's", "alright", "hmm", "so,", "first, i", "we need to",
    "i need to", "the user",
)


def looks_like_deliberation(text: str) -> bool:
    stripped = str(text or "").strip().lower()
    return stripped.startswith(_DELIBERATION_PREFIXES)


class SummaryQuality:
    """Counts what the summarizer actually returned, so a run can refuse itself.

    MemRL stores whatever the summarizer says.  If those calls degrade -- empty
    answers, or raw chain of thought -- the run still completes, still writes a
    full store, and still reports a success rate; it is simply measuring
    "inject 5k tokens of noise" under MemRL's name.  That happened (see
    :data:`DEFAULT_SUMMARIZER_MAX_TOKENS`) and was only caught by reading the
    store afterwards.  This makes it loud at the point of writing instead.
    """

    def __init__(self, warn_fraction: float = 0.2):
        self.warn_fraction = warn_fraction
        self.total = 0
        self.empty = 0
        self.deliberation = 0
        self._warned = False

    def observe(self, text: str) -> None:
        self.total += 1
        if not str(text or "").strip():
            self.empty += 1
        elif looks_like_deliberation(text):
            self.deliberation += 1
        self._maybe_warn()

    @property
    def bad_fraction(self) -> float:
        return (self.empty + self.deliberation) / self.total if self.total else 0.0

    def _maybe_warn(self) -> None:
        # Warn once, and only with enough calls for the fraction to mean
        # something -- the first two summaries being odd is not a signal.
        if self._warned or self.total < 10 or self.bad_fraction <= self.warn_fraction:
            return
        self._warned = True
        print(
            f"\n  [!] SUMMARIZER IS DEGRADED: {self.empty} empty and "
            f"{self.deliberation} raw-chain-of-thought out of {self.total} "
            f"memory-building calls\n      ({self.bad_fraction:.0%}). MemRL is "
            f"storing these and injecting them. A thinking model needs a larger "
            f"--summarizer-max-tokens\n      (default "
            f"{DEFAULT_SUMMARIZER_MAX_TOKENS}) to reach its answer channel. "
            f"Stop the run rather than reporting it.\n",
            flush=True,
        )

    def to_dict(self) -> dict:
        return {
            "calls": self.total,
            "empty": self.empty,
            "deliberation": self.deliberation,
            "bad_fraction": round(self.bad_fraction, 4),
        }


def systems_for(task: str) -> dict[str, str]:
    try:
        return SYSTEMS[task]
    except KeyError:
        raise ValueError(f"No summarizer prompts for task {task!r}") from None


def make_summarizer(
    model_fn,
    task: str,
    *,
    max_new_tokens: int = DEFAULT_SUMMARIZER_MAX_TOKENS,
    verbose: bool = False,
    quality: "SummaryQuality | None" = None,
) -> Callable[[str, str, str], str]:
    """MemRL's summarizer over an ORBIT actor, at the paper's temperatures.

    The actor model is reused for the memory-building calls, which is the
    self-improvement setting these arms are compared under: HeXA's evolver is
    the actor model too, so a stronger summariser here would make the baseline
    win or lose for a reason that is not the memory mechanism.
    """
    inner = make_model_fn_summarizer(
        model_fn,
        max_new_tokens=max_new_tokens,
        script_temperature=SCRIPT_TEMPERATURE,
        reflection_temperature=REFLECTION_TEMPERATURE,
        verbose=verbose,
        systems=systems_for(task),
    )
    quality = quality or SummaryQuality()

    def summarize(kind: str, intent: str, trajectory_text: str) -> str:
        text = inner(kind, intent, trajectory_text)
        quality.observe(text)
        return text

    summarize.quality = quality  # type: ignore[attr-defined]
    return summarize


def make_vllm_model_fn(config, *, client=None) -> Callable:
    """`model_fn(messages, temperature, max_new_tokens) -> str` over vLLM.

    `baselines/` gets its summarizer backend from
    `skillrl.runner.run_skill_agent._load_model`, which loads HuggingFace
    weights into the process.  ORBIT actors are served, not loaded, so the
    summarizer talks to the same endpoint the actor does.

    **Returns the answer channel only, and never falls back to the reasoning
    channel.** That fallback is the bug described in
    :data:`DEFAULT_SUMMARIZER_MAX_TOKENS`: `raw_text` is `reasoning + final`, so
    substituting it when `final_text` is empty silently converts "the model
    never answered" into "here is its scratchpad, stored as a memory and
    injected into every future instance". An empty string is the honest return,
    and `make_model_fn_summarizer` already prints a warning for it.

    `reasoning_effort` is forced off for the same reason -- these calls want a
    short piece of text, not deliberation -- but that is a request the server
    may ignore, so it is not relied on.
    """
    def model_fn(messages, temperature: float, max_new_tokens: int):
        nonlocal client
        if client is None:
            from hexa_core.llm.vllm_client import VLLMClient  # noqa: PLC0415

            client = VLLMClient(
                model=config.model,
                base_url=config.base_url,
                temperature=temperature,
                top_p=config.top_p,
                max_tokens=max_new_tokens,
                reasoning_effort=None,
                raise_on_exhausted=False,
            )
        generation = client.generate_structured(
            messages=list(messages),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        if generation is None:
            return ""
        return generation.final_text or ""

    return model_fn
