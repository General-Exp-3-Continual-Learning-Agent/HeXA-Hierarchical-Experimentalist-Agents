"""The Maze actor: a cumulative-chat, ``\\boxed{}``-parsing multi-turn agent.

Neither existing HExA actor fits ORBIT (RECON §7): ``ReactAgent`` is bound to
InterPhyre's toolkit and ``(x, y, r)`` coordinate parsing, and ``hexa_core``'s
path is single-turn. So this is the third actor, and it is written to be a
faithful re-implementation of *ORBIT's* agent (``agents/gem_text_agent.py``,
92 lines) against :class:`hexa_core.llm.VLLMClient` instead of rLLM's engine --
per D7, actor and evolver are both ``openai/gpt-oss-120b`` served by
``scripts/serve_gpt_oss_vllm.sh``.

Three behaviours are copied from ORBIT deliberately, because Phase 2's "same
protocol as ORBIT, different model" claim rests on them:

* **Cumulative chat.** One message list per *instance*, spanning all three
  episodes -- every observation and every response stays in context
  (``gem_text_agent.py:60-88``). This *is* ORBIT's within-instance memory
  mechanism; the no-bank baseline must keep it, and only the cross-instance bank
  is what HExA adds on top.
* **Last-boxed extraction, then re-wrap.** :func:`extract_last_boxed` takes the
  *last* ``\\boxed{...}`` and falls back to the whole response, then the value is
  re-wrapped as ``\\boxed{value}`` before the env sees it
  (``gem_text_agent.py:20-25, 89-92``). The re-wrap matters: ``MazeEnvAdapter``
  parses with ``re.search`` (first match) and ``[^}]+``, so on a fallback the
  env's own parse can behave oddly -- that oddity is ORBIT's, and reproducing it
  is the point.
* **No reflection prompt.** ``enable_reflection`` is off in ORBIT's eval path
  (RECON §3), so nothing is injected between episodes.

Two judgement calls ORBIT's code does not settle, because it never ran a
reasoning model with a separate analysis channel. Both are recorded per run:

* **What goes into the transcript.** ORBIT stores the whole completion. For
  gpt-oss the completion arrives split into a reasoning channel and a final
  channel, and ``hexa_core``'s ``Generation.raw_text`` concatenates them (CORE's
  *scoring* convention, not a chat convention). Feeding reasoning back would
  balloon context across 27 turns and is not how a reasoning model is normally
  driven, so the assistant message is :attr:`Generation.final_text`. When the
  model spends its whole budget on reasoning and the final channel is empty, the
  re-wrapped action is stored instead, so the transcript never contains an empty
  assistant turn.
* **What gets parsed.** ``raw_text`` -- the most permissive reading of ORBIT's
  "last boxed in the response", and it recovers an action the model emitted in
  the analysis channel rather than throwing the turn away.

Bank injection (Phase 3) appends to the **system prompt** and nowhere else, so
per-turn observations stay byte-identical to the baseline's and every added token
is attributable to one measurable block (M7).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from hexa_core import tokens
from hexa_core.llm.base import Generation

from .adapter import ORBIT_SYSTEM_PROMPT_MULTI, MazeInstance

#: Transcribed from ``agents/gem_text_agent.py:18``. Note ``[^}]+``: it stops at
#: the first ``}``, so a nested brace truncates the capture. ORBIT's behaviour.
BOXED_PATTERN = re.compile(r"\\boxed{([^}]+)}")

#: D7 default. The served name from ``scripts/serve_gpt_oss_vllm.sh:105``.
DEFAULT_MODEL = "openai/gpt-oss-120b"

#: ORBIT caps responses hard and tells the model so ("Overlong responses will be
#: penalized", RECON §3). CORE's paper config allows 32768, which on a 9-turn
#: maze episode is both wasteful and a protocol difference. This is the ORBIT-
#: matching default; override via :attr:`ActorConfig.max_response_tokens`.
DEFAULT_MAX_RESPONSE_TOKENS = 2048


def extract_last_boxed(text: str) -> str:
    """Last ``\\boxed{...}`` content, else the whole stripped text.

    Literal transcription of ``gem_text_agent.py:20-25``.
    """
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return text.strip()
    return matches[-1].group(1).strip()


def boxed(value: str) -> str:
    """ORBIT's re-wrap before handing the action to the env (:89-92)."""
    return f"\\boxed{{{value}}}"


class ContextExhausted(RuntimeError):
    """The cumulative transcript outgrew the model's context window.

    Under ``transcript_mode="raw"`` this is an expected outcome, not a bug: ORBIT
    reports the same condition and handles it by truncating the trajectory and
    assigning zero reward (§4.2). Raised as its own type so the runner can record it
    as a *result* -- remaining episodes counted as failures -- rather than discarding
    the instance the way a transport failure is discarded.
    """


class GenerationFailed(RuntimeError):
    """The client exhausted its retries and returned no generation.

    Raised rather than silently substituting a no-op action: a forced invalid
    action would consume a turn and quietly depress the success rate, which is
    exactly the kind of degradation a baseline number must not absorb. The runner
    catches this per instance and excludes it, reporting the count.
    """


@dataclass
class ActorConfig:
    """Sampling and endpoint configuration for one run.

    Defaults match ORBIT's eval sampling where the two configs differ: they agree
    on ``temperature=0.6`` and ``top_p=0.95`` (RECON §3 vs CORE's paper config),
    and differ only on the response cap and reasoning effort.
    """

    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    temperature: float = 0.6
    top_p: float = 0.95
    max_response_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS
    reasoning_effort: Optional[str] = "low"
    #: Per-call sampling seed. vLLM honours it, which makes a single instance
    #: replayable; it does not make the *pool* reproducible under concurrency
    #: (D5 -- ORBIT's env seeding is global).
    seed: Optional[int] = None

    #: What goes into the transcript as the assistant turn.
    #:
    #: ``"final"`` stores only the final channel -- for Qwen3 that is roughly
    #: ``"\n\n\boxed{down}"``, ~15 tokens. ``"raw"`` stores ``reasoning + final``,
    #: which is what ORBIT's own agent does (``agents/gem_text_agent.py`` appends the
    #: whole ``response``) and is ~800 tokens per turn with thinking enabled.
    #:
    #: This is not a cosmetic choice. Over 27 turns ``"raw"`` accumulates ~22k tokens
    #: against a 32k context, which is why ORBIT §4.2 needs the sentence
    #: *"trajectories that exceed this limit are truncated and assigned zero reward"*
    #: -- observations alone would never approach 32k. ``"final"`` keeps the context
    #: near 4k and can never truncate, so it hands the model a clean history where
    #: ORBIT hands it one filled with its own discarded reasoning.
    #:
    #: Our stock Qwen3-8B scored 0.53 episode-3 under ``"final"`` against ORBIT's
    #: reported 0.22 for the same base weights (Table 2, derived). ``"raw"`` exists to
    #: test whether that gap is this setting.
    transcript_mode: str = "final"

    #: Total completion tokens allowed across a whole instance, or ``None`` for no
    #: trajectory-level cap.
    #:
    #: ORBIT sets ``data.max_response_length=31744`` -- a **trajectory** budget, not
    #: a per-call one (``scripts/train_multi_task_multi_episode.sh``), and §4.2 says
    #: *"trajectories that exceed this limit are truncated and assigned zero
    #: reward."* Our ``max_response_tokens`` is per call and so can never bind: at
    #: ~800 completion tokens/turn over ≤27 turns we reach ~22k and stop.
    #:
    #: This exists to test whether that difference explains why our stock Qwen3-8B
    #: scores 0.49 episode-3 (clean, n=206) against ORBIT's reported 0.22 for the
    #: same base weights. Set it to 31744 to impose their budget.
    #:
    #: Note the *other* half of their truncation pressure is NOT transplantable:
    #: ``trajectory_timeout=600`` binds only under their 512-concurrent-on-4-GPU
    #: contention, where per-trajectory latency is far higher than our measured
    #: ~227 s. Copying the number would not copy the constraint.
    trajectory_token_budget: Optional[int] = None

    #: How to score episodes an instance never reached.
    #:
    #: ``"exclude"`` (ours) leaves them out: an episode that never ran is not
    #: evidence about the agent. ``"orbit"`` scores them **0**, reproducing
    #: ``MultiEpisodeEnv.get_metrics``'s early-termination branch
    #: (``envs/multi_episode_env.py:383-396``), which writes ``0`` for a missing
    #: episode where its *normal* branch writes ``-1.0`` -- a sentinel the
    #: aggregator filters. So the same missing episode is excluded on one path and
    #: counted as a failure on the other, and a trajectory cut short during episode
    #: 2 reports ``episode_3/success_rate = 0``.
    truncation_scoring: str = "exclude"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_response_tokens": self.max_response_tokens,
            "reasoning_effort": self.reasoning_effort,
            "seed": self.seed,
            "transcript_mode": self.transcript_mode,
            "trajectory_token_budget": self.trajectory_token_budget,
            "truncation_scoring": self.truncation_scoring,
        }


@dataclass
class StepRecord:
    """One turn: what the model said, what the env did with it."""

    episode_index: int
    episode_step: int
    total_step: int
    observation: str
    raw_text: str
    reasoning_text: str
    final_text: str
    parsed_action: str
    env_action: str
    reward: float
    finish_reason: Optional[str]
    episode_done: bool
    usage: Optional[dict[str, Any]] = None

    @property
    def truncated(self) -> bool:
        """Generation stopped at the token limit rather than finishing.

        Distinct from a wrong answer and worth separating: a truncated turn
        usually yields no parseable action, so the env consumes the turn on
        "Invalid action" and the failure is a budget artefact, not a reasoning
        one.
        """
        return self.finish_reason == "length"

    def to_dict(self) -> dict[str, Any]:
        """Serialise. ``raw_text`` is deliberately absent -- see :meth:`from_dict`."""
        return {
            "episode_index": self.episode_index,
            "episode_step": self.episode_step,
            "total_step": self.total_step,
            "observation": self.observation,
            "reasoning_text": self.reasoning_text,
            "final_text": self.final_text,
            "parsed_action": self.parsed_action,
            "env_action": self.env_action,
            "reward": self.reward,
            "finish_reason": self.finish_reason,
            "episode_done": self.episode_done,
            "truncated": self.truncated,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRecord":
        """Rehydrate, reconstructing ``raw_text`` exactly rather than storing it.

        ``Generation.from_parts`` builds ``raw_text`` as
        ``"".join(p for p in (reasoning_text, final_text) if p)``
        (``hexa_core/llm/base.py``), so the concatenation is lossless and storing it
        would duplicate the two largest fields in the record. A file written by an
        older schema that *does* carry ``raw_text`` is honoured as-is.
        """
        raw = data.get("raw_text")
        if raw is None:
            raw = "".join(
                part for part in (data.get("reasoning_text"), data.get("final_text")) if part
            )
        return cls(
            episode_index=int(data["episode_index"]),
            episode_step=int(data["episode_step"]),
            total_step=int(data["total_step"]),
            observation=data.get("observation", ""),
            raw_text=raw,
            reasoning_text=data.get("reasoning_text", ""),
            final_text=data.get("final_text", ""),
            parsed_action=data.get("parsed_action", ""),
            env_action=data.get("env_action", ""),
            reward=float(data.get("reward", 0.0)),
            finish_reason=data.get("finish_reason"),
            episode_done=bool(data.get("episode_done", False)),
            usage=data.get("usage"),
        )


@dataclass
class InstanceRun:
    """One instance played to completion: transcript, metrics, diagnostics."""

    instance_id: str
    dataset_seed: int
    resolved_seed: int
    shortest_path_len: int
    leaked: Optional[bool]
    steps: list[StepRecord]
    messages: list[dict[str, str]]
    metrics: dict[str, Any]
    episode_successes: list[bool]
    episode_lengths: list[int]
    success_count: int
    added_tokens: int
    token_counter: str
    guidelines: Optional[str] = None
    error: Optional[str] = None
    #: True when the instance ended because the transcript outgrew the context.
    #: ORBIT's equivalent condition zeroes the trajectory reward; recorded rather
    #: than hidden so a run's episode counts can be read honestly.
    context_exhausted: bool = False
    #: True when the instance stopped on ``trajectory_token_budget``.
    budget_exhausted: bool = False
    #: Total completion tokens spent across the instance.
    completion_tokens: int = 0
    #: ``"usage"`` if the server reported token counts, ``"estimated"`` otherwise.
    token_accounting: str = "usage"
    #: Which convention ``episode_successes_scored`` follows.
    truncation_scoring: str = "exclude"
    #: Best per-episode :attr:`~hexa_orbit.adapter.EpisodeRecord.progress` over the
    #: instance. Feeds the graded failure reward.
    best_progress: float = 0.0
    #: Per-episode progress, so the evolver can see whether episode 2 got further
    #: than episode 1 -- adaptation that a bin-only reward cannot express.
    episode_progress: list[float] = field(default_factory=list)

    @property
    def truncated_early(self) -> bool:
        return self.context_exhausted or self.budget_exhausted

    @property
    def episode_successes_scored(self) -> list[bool]:
        """Per-episode successes under the configured truncation convention.

        Under ``"orbit"``, episodes an instance never reached are appended as
        ``False`` up to the three-episode cap, which is what
        ``MultiEpisodeEnv.get_metrics`` does on its early-termination branch. Under
        ``"exclude"`` the list is returned as-is and a short list stays short, so
        `report.episode_success_rates` omits it rather than counting it against the
        agent.
        """
        if self.truncation_scoring != "orbit":
            return list(self.episode_successes)
        padded = list(self.episode_successes)
        while len(padded) < 3:
            padded.append(False)
        return padded

    @property
    def n_truncated(self) -> int:
        return sum(1 for s in self.steps if s.truncated)

    @property
    def n_unparsed(self) -> int:
        """Turns where no ``\\boxed{}`` was found, so the fallback path ran.

        A high count means the response cap or the prompt is wrong, not that the
        model cannot solve mazes -- check this before reading any success rate.
        """
        return sum(1 for s in self.steps if not BOXED_PATTERN.search(s.raw_text))

    @property
    def adapted_across_episodes(self) -> bool:
        """Whether later episodes visited anything episode 1 did not.

        The discriminator for the reward mapping's bottom two bins: "exploratory
        failure" versus "repeated the same failed strategy across all 3 episodes"
        (the plan's reward table). Computed from action sequences rather than
        positions so it is meaningful even when every episode fails immediately.
        """
        by_episode: dict[int, list[str]] = {}
        for step in self.steps:
            by_episode.setdefault(step.episode_index, []).append(step.parsed_action)
        sequences = [tuple(v) for v in by_episode.values()]
        return len(set(sequences)) > 1

    @property
    def guidelines_hash(self) -> Optional[str]:
        """Short digest of the injected bank block, or ``None`` for the no-bank arm.

        The block itself is ~1000 tokens and identical across every instance in a
        round, so storing it per instance would bloat ``runs.jsonl`` for no gain --
        the round writes it once to ``injection.json``. The hash is what makes a
        run auditable after the fact ("which bank produced these numbers?"), and it
        is cheap.
        """
        if not self.guidelines:
            return None
        # A per-episode injection is a mapping, which has no `.encode`. Hash the
        # blocks in key order so the digest is stable across runs and still
        # distinguishes two banks that differ only in one episode's block.
        if isinstance(self.guidelines, dict):
            payload = "\n\n".join(
                f"[episode {k}]\n{self.guidelines[k]}"
                for k in sorted(self.guidelines)
            )
        else:
            payload = self.guidelines
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_transcript: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "instance_id": self.instance_id,
            "dataset_seed": self.dataset_seed,
            "resolved_seed": self.resolved_seed,
            "shortest_path_len": self.shortest_path_len,
            "leaked": self.leaked,
            "metrics": self.metrics,
            "episode_successes": self.episode_successes,
            "episode_lengths": self.episode_lengths,
            "success_count": self.success_count,
            "added_tokens": self.added_tokens,
            "token_counter": self.token_counter,
            "n_truncated": self.n_truncated,
            "n_unparsed": self.n_unparsed,
            "adapted_across_episodes": self.adapted_across_episodes,
            "guidelines_hash": self.guidelines_hash,
            "context_exhausted": self.context_exhausted,
            "budget_exhausted": self.budget_exhausted,
            "completion_tokens": self.completion_tokens,
            "token_accounting": self.token_accounting,
            "truncation_scoring": self.truncation_scoring,
            "best_progress": self.best_progress,
            "episode_progress": self.episode_progress,
            "error": self.error,
        }
        if include_transcript:
            out["steps"] = [s.to_dict() for s in self.steps]
            out["messages"] = self.messages
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstanceRun":
        """Rehydrate a run written by :meth:`to_dict`.

        ``guidelines`` comes back as ``None``: only its hash is stored (see
        :attr:`guidelines_hash`), and nothing downstream of a completed run reads
        the block -- the evolver reads trajectories, not the injected prompt. The
        derived properties that *are* read (``n_unparsed``, ``n_truncated``,
        ``adapted_across_episodes``) are recomputed from the rehydrated steps
        rather than taken from the file, so a stale stored value cannot survive a
        round trip.
        """
        return cls(
            instance_id=data["instance_id"],
            dataset_seed=int(data["dataset_seed"]),
            resolved_seed=int(data["resolved_seed"]),
            shortest_path_len=int(data["shortest_path_len"]),
            leaked=data.get("leaked"),
            steps=[StepRecord.from_dict(s) for s in data.get("steps", [])],
            messages=list(data.get("messages", [])),
            metrics=dict(data.get("metrics", {})),
            episode_successes=list(data.get("episode_successes", [])),
            episode_lengths=list(data.get("episode_lengths", [])),
            success_count=int(data.get("success_count", 0)),
            added_tokens=int(data.get("added_tokens", 0)),
            token_counter=data.get("token_counter", ""),
            guidelines=None,
            error=data.get("error"),
            context_exhausted=bool(data.get("context_exhausted", False)),
            budget_exhausted=bool(data.get("budget_exhausted", False)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            token_accounting=data.get("token_accounting", "usage"),
            truncation_scoring=data.get("truncation_scoring", "exclude"),
            best_progress=float(data.get("best_progress", 0.0)),
            episode_progress=list(data.get("episode_progress", [])),
        )


class MazeActor:
    """Drives :class:`~hexa_orbit.adapter.MazeInstance` with an LLM.

    One actor can play many instances; chat state is per-instance and reset at
    the start of :meth:`run_instance`, mirroring ``GEMTextAgent.reset``.
    """

    def __init__(
        self,
        config: Optional[ActorConfig] = None,
        *,
        client: Any = None,
        system_prompt: str = ORBIT_SYSTEM_PROMPT_MULTI,
        guidelines: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: sampling/endpoint settings. Ignored when *client* is given.
            client: any object with ``generate_structured(messages=..., ...)``.
                Injected in tests with a scripted stub; defaults to a real
                :class:`~hexa_core.llm.vllm_client.VLLMClient`.
            system_prompt: ORBIT's multi-episode prompt by default, verbatim.
            guidelines: the retrieved skill/mistake block (Phase 3). ``None`` is
                the no-bank baseline and leaves the system prompt untouched, so
                the baseline prompt is byte-identical to ORBIT's.

                May also be a mapping ``{episode_index: block}`` to inject a
                different block per episode. That exists because a bank is not
                uniformly applicable across episodes: 3 of the 5 skills and 1 of
                the 2 mistakes in v3's final bank are phrased "based on prior
                exploration data" and are unusable in episode 1, so roughly 60% of
                ~700 injected tokens described data the agent did not yet have --
                during the one episode the whole claim is about. Measured ep1
                effect across three runs with a single static block: -4.1, -1.6,
                -2.9 points against the no-bank baseline, while multi-episode
                success rose. Keys are 0-based; a missing key injects nothing for
                that episode, and the largest key present covers any episode beyond
                it.
        """
        self.config = config or ActorConfig()
        self.base_system_prompt = system_prompt
        self.guidelines = guidelines
        self._client = client
        self.messages: list[dict[str, str]] = []

    @property
    def client(self) -> Any:
        """Lazily built so importing this module needs no server and no openai."""
        if self._client is None:
            from hexa_core.llm.vllm_client import VLLMClient  # noqa: PLC0415

            self._client = VLLMClient(
                model=self.config.model,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_response_tokens,
                reasoning_effort=self.config.reasoning_effort,
                seed=self.config.seed,
                raise_on_exhausted=False,
            )
        return self._client

    # ── prompt assembly ───────────────────────────────────────────────────

    def block_for_episode(self, episode_index: int) -> Optional[str]:
        """The block to inject for *episode_index*, or ``None``.

        A plain string is episode-independent -- the original behaviour, and what
        every completed run used. A mapping is looked up by episode, falling back
        to the largest key at or below it so ``{0: explore, 1: reuse}`` covers
        episode 2 without repeating itself.
        """
        if self.guidelines is None or isinstance(self.guidelines, str):
            return self.guidelines
        keys = [k for k in self.guidelines if int(k) <= int(episode_index)]
        return self.guidelines[max(keys)] if keys else None

    def system_prompt_for(self, episode_index: int) -> str:
        """ORBIT's prompt, plus this episode's block when one is injected.

        Appending keeps the baseline byte-identical to ORBIT's and confines every
        added token to one attributable place (M7).
        """
        block = self.block_for_episode(episode_index)
        if not block:
            return self.base_system_prompt
        return f"{self.base_system_prompt}\n\n{block}"

    @property
    def system_prompt(self) -> str:
        """The episode-0 prompt. Kept for callers that predate per-episode blocks."""
        return self.system_prompt_for(0)

    @property
    def added_tokens(self) -> int:
        """Tokens the largest injected block adds -- the worst case, not a mean.

        With a per-episode mapping the cost differs by episode; reporting the
        maximum keeps this an upper bound on the budget the injection consumes.
        """
        if isinstance(self.guidelines, dict):
            prompts = [self.system_prompt_for(k) for k in self.guidelines]
            return max(
                (tokens.added_tokens(self.base_system_prompt, p) for p in prompts),
                default=0,
            )
        return tokens.added_tokens(self.base_system_prompt, self.system_prompt)

    # ── one turn ──────────────────────────────────────────────────────────

    def _completion_tokens(self, generation: Generation) -> int:
        """Completion tokens for one call, from usage when the server reports it.

        Falls back to a token estimate of the raw text. The fallback matters: a
        budget enforced on an estimate is not ORBIT's budget, so
        :attr:`InstanceRun.token_accounting` records which path was used.
        """
        usage = generation.usage or {}
        for key in ("completion_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return tokens.count_tokens(generation.raw_text)

    def _generate(self) -> Generation:
        try:
            generation = self.client.generate_structured(
                messages=list(self.messages),
                max_new_tokens=self.config.max_response_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - classified, then re-raised
            # vLLM rejects an over-long prompt with a deterministic 400 rather than a
            # transient error, so the client raises instead of retrying. Duck-typed on
            # the message because the exception class depends on the openai version.
            text = str(exc).lower()
            if "context" in text and ("length" in text or "window" in text):
                raise ContextExhausted(str(exc)) from exc
            if "maximum context" in text or "longer than the maximum" in text:
                raise ContextExhausted(str(exc)) from exc
            raise
        if generation is None:
            raise GenerationFailed(
                f"{self.config.model} returned no generation after retries"
            )
        return generation

    # ── one instance ──────────────────────────────────────────────────────

    def run_instance(
        self, instance: MazeInstance, *, leaked: Optional[bool] = None
    ) -> InstanceRun:
        """Play one instance to completion and return the full record.

        Raises :class:`GenerationFailed` if the client gives up; the caller
        decides whether to drop the instance or abort the run.
        """
        self.messages = [{"role": "system", "content": self.system_prompt_for(0)}]
        steps: list[StepRecord] = []
        injected_episode = 0

        observation = instance.reset()
        context_exhausted = False
        budget_exhausted = False
        spent_tokens = 0
        exact_usage = True
        while not instance.done:
            # Re-point the system message when the episode advances, so a
            # per-episode block reaches the model. Rewriting slot 0 rather than
            # appending keeps exactly one system turn in the transcript -- ORBIT's
            # shape -- and the model sees the whole conversation re-sent each call
            # anyway, so the new block governs from here on. A plain-string
            # `guidelines` makes this a no-op: the content is identical every time.
            if instance._episode_index != injected_episode:
                injected_episode = instance._episode_index
                self.messages[0] = {
                    "role": "system",
                    "content": self.system_prompt_for(injected_episode),
                }
            self.messages.append({"role": "user", "content": observation})
            try:
                generation = self._generate()
            except ContextExhausted:
                # ORBIT's own failure mode under "raw" transcripts. Stop the instance
                # and let the episodes recorded so far stand; the unplayed ones are
                # absent rather than counted as failures, and `context_exhausted`
                # says why the count is short.
                context_exhausted = True
                break

            parsed = extract_last_boxed(generation.raw_text)
            env_action = boxed(parsed)
            # See ActorConfig.transcript_mode. "raw" reproduces ORBIT's own agent,
            # which appends the whole response; "final" keeps only the answer channel.
            # Either way an empty result falls back to the action, so the transcript
            # never contains an empty assistant turn.
            if self.config.transcript_mode == "raw":
                spoken = generation.raw_text or env_action
            else:
                spoken = generation.final_text or env_action
            self.messages.append({"role": "assistant", "content": spoken})

            spent_tokens += self._completion_tokens(generation)
            if not (generation.usage or {}).get("completion_tokens"):
                exact_usage = False

            episode_index = instance._episode_index
            episode_step = instance._episode_step + 1
            next_observation, reward, _, info = instance.step(env_action)
            steps.append(
                StepRecord(
                    episode_index=episode_index,
                    episode_step=episode_step,
                    total_step=info["total_step"],
                    observation=observation,
                    raw_text=generation.raw_text,
                    reasoning_text=generation.reasoning_text,
                    final_text=generation.final_text,
                    parsed_action=parsed,
                    env_action=env_action,
                    reward=reward,
                    finish_reason=generation.finish_reason,
                    episode_done=bool(info["episode_done"]),
                    usage=generation.usage,
                )
            )
            observation = next_observation

            budget = self.config.trajectory_token_budget
            if budget is not None and spent_tokens >= budget:
                # ORBIT's condition: the trajectory outgrew its response budget. Stop
                # here; episodes never reached are scored by `truncation_scoring`.
                budget_exhausted = True
                break

        # The final observation is never answered, but ORBIT's agent still sees
        # it (`update_from_env` is called on the terminal step), so the
        # transcript the evolver reads must include it.
        self.messages.append({"role": "user", "content": observation})

        return InstanceRun(
            instance_id=instance.id,
            dataset_seed=instance.spec.dataset_seed,
            resolved_seed=instance.spec.resolved_seed,
            shortest_path_len=instance.spec.shortest_path_len,
            leaked=leaked,
            steps=steps,
            messages=list(self.messages),
            metrics=instance.metrics(),
            episode_successes=[e.success for e in instance.episodes],
            episode_lengths=[e.steps for e in instance.episodes],
            success_count=instance.success_count,
            added_tokens=self.added_tokens,
            token_counter=tokens.counter_name(),
            guidelines=self.guidelines,
            context_exhausted=context_exhausted,
            budget_exhausted=budget_exhausted,
            completion_tokens=spent_tokens,
            token_accounting="usage" if exact_usage else "estimated",
            truncation_scoring=self.config.truncation_scoring,
            best_progress=max((e.progress for e in instance.episodes), default=0.0),
            episode_progress=[e.progress for e in instance.episodes],
        )
