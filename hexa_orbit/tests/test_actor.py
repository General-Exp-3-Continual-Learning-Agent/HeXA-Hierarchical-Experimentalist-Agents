"""Actor tests, driven by a scripted stub client -- no vLLM server required.

Covers everything except real generation: ORBIT's parsing rules, the cumulative
chat layout, the reasoning-channel judgement calls, failure handling, and the
diagnostics the Phase 2 report reads before it reads any success rate.
"""

from __future__ import annotations

import pytest
from hexa_core.llm.base import Generation

from hexa_orbit import actor, oracle
from hexa_orbit.actor import (
    BOXED_PATTERN,
    ActorConfig,
    GenerationFailed,
    MazeActor,
    boxed,
    extract_last_boxed,
)
from hexa_orbit.adapter import EPISODES_PER_INSTANCE, ORBIT_SYSTEM_PROMPT_MULTI, MazeTask


class StubClient:
    """Returns queued generations; records the messages it was handed."""

    def __init__(self, script: list[Generation] | None = None, default: Generation | None = None):
        self.script = list(script or [])
        self.default = default
        self.seen: list[list[dict[str, str]]] = []

    def generate_structured(self, *, messages, max_new_tokens=None, **_):
        self.seen.append([dict(m) for m in messages])
        if self.script:
            return self.script.pop(0)
        return self.default


def gen(final: str, *, reasoning: str = "", finish: str | None = "stop") -> Generation:
    return Generation.from_parts(reasoning, final, finish_reason=finish)


@pytest.fixture(scope="module")
def task() -> MazeTask:
    return MazeTask()


@pytest.fixture(scope="module")
def spec(task: MazeTask):
    return task.eval_specs(1)[0]


def optimal_script(task: MazeTask, spec) -> list[Generation]:
    """One generation per turn of three optimal episodes."""
    probe = task.instance(spec)
    probe.reset()
    actions = oracle.instance_optimal_actions(probe)
    return [gen(f"I will move {a}. {boxed(a)}") for a in actions * EPISODES_PER_INSTANCE]


# ── ORBIT's parsing rules ────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"go \boxed{up}", "up"),
        (r"\boxed{up} then \boxed{down}", "down"),  # last, not first
        ("  bare text  ", "bare text"),             # fallback, stripped
        (r"\boxed{ left }", "left"),                # inner whitespace stripped
    ],
)
def test_extract_last_boxed(text: str, expected: str):
    assert extract_last_boxed(text) == expected


def test_nested_brace_truncates_like_orbit():
    """``[^}]+`` stops at the first ``}``. ORBIT's behaviour, reproduced on purpose."""
    assert extract_last_boxed(r"\boxed{up{x}}") == "up{x"


# ── the happy path ───────────────────────────────────────────────────────

def test_optimal_play_solves_all_three_episodes(task: MazeTask, spec):
    client = StubClient(optimal_script(task, spec))
    actor = MazeActor(client=client)
    run = actor.run_instance(task.instance(spec), leaked=False)

    assert run.success_count == EPISODES_PER_INSTANCE
    assert run.episode_successes == [True] * EPISODES_PER_INSTANCE
    assert run.episode_lengths == [spec.shortest_path_len] * EPISODES_PER_INSTANCE
    assert run.metrics["episode/success_rate"] == 1.0
    assert run.n_truncated == 0
    assert run.n_unparsed == 0
    assert run.instance_id == spec.instance_id
    assert run.leaked is False


def test_chat_is_cumulative_and_well_formed(task: MazeTask, spec):
    """One message list per instance, spanning all three episodes.

    ORBIT's within-instance memory mechanism. Layout must be
    system, (user, assistant) x n, user -- the trailing user message being the
    terminal observation the agent sees but never answers.
    """
    client = StubClient(optimal_script(task, spec))
    actor = MazeActor(client=client)
    run = actor.run_instance(task.instance(spec))

    n = len(run.steps)
    assert len(run.messages) == 2 * n + 2
    assert run.messages[0] == {"role": "system", "content": ORBIT_SYSTEM_PROMPT_MULTI}
    assert [m["role"] for m in run.messages[1:-1]] == ["user", "assistant"] * n
    assert run.messages[-1]["role"] == "user"

    # Cumulative: each call sees every prior message, never a truncated window.
    assert [len(seen) for seen in client.seen] == [2 * i + 2 for i in range(n)]
    assert client.seen[-1][: len(client.seen[0])] == client.seen[0]

    # Episode 2's first observation carries the boundary concatenation, so the
    # transcript spans episodes rather than restarting.
    boundaries = [s for s in run.steps if s.episode_done]
    assert len(boundaries) == EPISODES_PER_INSTANCE
    assert "[Episode 2] New episode begins." in run.steps[spec.shortest_path_len].observation


# ── the reasoning-channel judgement calls ────────────────────────────────

def test_assistant_message_is_the_final_channel(task: MazeTask, spec):
    """Reasoning is parsed but never fed back, or context would balloon."""
    client = StubClient(default=gen(f"moving {boxed('up')}", reasoning="LONG ANALYSIS"))
    actor = MazeActor(client=client)
    run = actor.run_instance(task.instance(spec))

    assistants = [m["content"] for m in run.messages if m["role"] == "assistant"]
    assert all(a == f"moving {boxed('up')}" for a in assistants)
    assert all("LONG ANALYSIS" not in a for a in assistants)
    # ...but it is retained in the record, for the evolver and for auditing.
    assert all(s.reasoning_text == "LONG ANALYSIS" for s in run.steps)


def test_action_recovered_from_reasoning_when_final_is_empty(task: MazeTask, spec):
    """A budget-exhausted turn still yields an action instead of a wasted turn."""
    client = StubClient(default=gen("", reasoning=f"thinking {boxed('up')}", finish="length"))
    actor = MazeActor(client=client)
    run = actor.run_instance(task.instance(spec))

    assert all(s.parsed_action == "up" for s in run.steps)
    assert run.n_truncated == len(run.steps)
    # No empty assistant turns: the action stands in for the missing final text.
    assert all(m["content"] for m in run.messages)
    assert {m["content"] for m in run.messages if m["role"] == "assistant"} == {boxed("up")}


def test_unparseable_response_counts_and_still_advances(task: MazeTask, spec):
    """No boxed action -> fallback path -> env consumes the turn as invalid."""
    client = StubClient(default=gen("I have no idea what to do"))
    actor = MazeActor(client=client)
    run = actor.run_instance(task.instance(spec))

    assert run.n_unparsed == len(run.steps)
    assert run.success_count == 0
    assert run.episode_lengths == [9, 9, 9]  # every turn burned on invalid actions
    assert not BOXED_PATTERN.search(run.steps[0].raw_text)


# ── failure handling ─────────────────────────────────────────────────────

def test_generation_failure_raises_rather_than_degrading(task: MazeTask, spec):
    """A silent no-op action would depress the baseline invisibly."""
    actor = MazeActor(client=StubClient(default=None))
    with pytest.raises(GenerationFailed):
        actor.run_instance(task.instance(spec))


# ── diagnostics the Phase 2 report depends on ────────────────────────────

def test_adapted_across_episodes_detects_repetition(task: MazeTask, spec):
    """The plan's bottom two reward bins hinge on this discriminator."""
    identical = MazeActor(client=StubClient(default=gen(boxed("up"))))
    repeated = identical.run_instance(task.instance(spec))
    assert repeated.adapted_across_episodes is False

    varied_script = [gen(boxed(a)) for a in ["up"] * 9 + ["down"] * 9 + ["left"] * 9]
    varied = MazeActor(client=StubClient(varied_script))
    assert varied.run_instance(task.instance(spec)).adapted_across_episodes is True


def test_baseline_prompt_is_byte_identical_and_costs_nothing(task: MazeTask, spec):
    """No-bank baseline must add zero tokens and touch nothing (M7 accounting)."""
    actor = MazeActor(client=StubClient(default=gen(boxed("up"))))
    assert actor.system_prompt == ORBIT_SYSTEM_PROMPT_MULTI
    run = actor.run_instance(task.instance(spec))
    assert run.added_tokens == 0
    assert run.guidelines is None


def test_bank_injection_is_confined_to_the_system_prompt(task: MazeTask, spec):
    """Observations stay byte-identical; all added tokens live in one block."""
    block = "LEARNED SKILLS\n- Map the corridor before committing."
    plain = MazeActor(client=StubClient(default=gen(boxed("up"))))
    banked = MazeActor(client=StubClient(default=gen(boxed("up"))), guidelines=block)

    plain_run = plain.run_instance(task.instance(spec))
    banked_run = banked.run_instance(task.instance(spec))

    assert banked_run.added_tokens > 0
    assert banked_run.guidelines == block
    assert block in banked_run.messages[0]["content"]
    assert banked_run.messages[0]["content"].startswith(ORBIT_SYSTEM_PROMPT_MULTI)
    # Only the system message differs.
    assert [s.observation for s in banked_run.steps] == [
        s.observation for s in plain_run.steps
    ]


def test_config_defaults_match_orbit_sampling():
    config = ActorConfig()
    assert (config.temperature, config.top_p) == (0.6, 0.95)
    assert config.model == "openai/gpt-oss-120b"
    assert config.max_response_tokens == 2048


# ── transcript mode (ORBIT-faithful vs compact) ───────────────────────────

def test_raw_transcript_mode_stores_the_reasoning(task: MazeTask, spec):
    """ORBIT's agent appends the whole response; "raw" reproduces that.

    Not cosmetic: at ~800 reasoning tokens per turn this accumulates ~22k over 27
    turns against a 32k context, which is the condition ORBIT §4.2 handles by
    truncating the trajectory and zeroing its reward.
    """
    long_reasoning = "DELIBERATION " * 50
    client = StubClient(default=gen(oracle.boxed("up"), reasoning=long_reasoning))
    actor = MazeActor(ActorConfig(transcript_mode="raw"), client=client)
    run = actor.run_instance(task.instance(spec))

    assistants = [m["content"] for m in run.messages if m["role"] == "assistant"]
    assert all("DELIBERATION" in a for a in assistants)
    # And the compact mode must not.
    compact = MazeActor(
        ActorConfig(transcript_mode="final"),
        client=StubClient(default=gen(oracle.boxed("up"), reasoning=long_reasoning)),
    ).run_instance(task.instance(spec))
    assert all(
        "DELIBERATION" not in m["content"]
        for m in compact.messages
        if m["role"] == "assistant"
    )
    # Same actions either way -- only the transcript differs.
    assert [s.parsed_action for s in run.steps] == [s.parsed_action for s in compact.steps]


def test_raw_mode_still_falls_back_when_the_model_says_nothing(task: MazeTask, spec):
    client = StubClient(default=gen("", reasoning=""))
    actor = MazeActor(ActorConfig(transcript_mode="raw"), client=client)
    run = actor.run_instance(task.instance(spec))
    assert all(m["content"] for m in run.messages)


def test_context_exhaustion_is_a_result_not_a_lost_instance(task: MazeTask, spec):
    """A context refusal must end the instance, not discard it.

    ORBIT hits the same wall and handles it by truncating the trajectory. Treating
    it like a transport failure would throw away a real measurement; treating it
    like a normal finish would silently report a 1-episode instance as complete.
    """
    class OverflowAfter:
        def __init__(self, n): self.n, self.calls = n, 0

        def generate_structured(self, **_):
            self.calls += 1
            if self.calls > self.n:
                raise RuntimeError(
                    "This model's maximum context length is 32768 tokens, however "
                    "you requested 33000 tokens."
                )
            return gen(oracle.boxed("up"))

    actor = MazeActor(ActorConfig(transcript_mode="raw"), client=OverflowAfter(5))
    run = actor.run_instance(task.instance(spec))

    assert run.context_exhausted is True
    assert run.error is None            # a result, not a failure
    assert len(run.steps) == 5
    assert len(run.episode_successes) < EPISODES_PER_INSTANCE
    assert run.to_dict()["context_exhausted"] is True
    from hexa_orbit.actor import InstanceRun
    assert InstanceRun.from_dict(run.to_dict()).context_exhausted is True


def test_a_real_transport_failure_still_raises(task: MazeTask, spec):
    """Only context errors are reclassified; everything else must stay loud."""
    class Broken:
        def generate_structured(self, **_):
            raise RuntimeError("connection reset by peer")

    with pytest.raises(RuntimeError, match="connection reset"):
        MazeActor(client=Broken()).run_instance(task.instance(spec))


# ── ORBIT's trajectory budget and truncation scoring ─────────────────────

def test_trajectory_token_budget_stops_the_instance(task: MazeTask, spec):
    """ORBIT's budget is per trajectory, so it can cut an instance short."""
    client = StubClient(default=gen(oracle.boxed("up"), reasoning="x " * 200))
    actor = MazeActor(
        ActorConfig(transcript_mode="raw", trajectory_token_budget=600), client=client
    )
    run = actor.run_instance(task.instance(spec))

    assert run.budget_exhausted is True
    assert run.truncated_early is True
    assert run.completion_tokens >= 600
    assert len(run.episode_successes) < EPISODES_PER_INSTANCE
    assert run.error is None          # a measurement, not a failure


def test_no_budget_means_no_early_stop(task: MazeTask, spec):
    client = StubClient(default=gen(oracle.boxed("up"), reasoning="x " * 200))
    run = MazeActor(ActorConfig(), client=client).run_instance(task.instance(spec))
    assert run.budget_exhausted is False
    assert len(run.episode_successes) == EPISODES_PER_INSTANCE


def test_orbit_scoring_counts_unreached_episodes_as_failures(task: MazeTask, spec):
    """The asymmetry that likely explains the gap to their Table 2 number.

    `MultiEpisodeEnv.get_metrics` writes 0 for a missing episode on its
    early-termination branch and -1.0 (filtered) on its normal branch. So a
    trajectory cut short in episode 2 reports episode_3 as a failure.
    """
    client = StubClient(default=gen(oracle.boxed("up"), reasoning="x " * 200))
    ours = MazeActor(
        ActorConfig(transcript_mode="raw", trajectory_token_budget=600,
                    truncation_scoring="exclude"), client=client,
    ).run_instance(task.instance(spec))
    theirs = MazeActor(
        ActorConfig(transcript_mode="raw", trajectory_token_budget=600,
                    truncation_scoring="orbit"),
        client=StubClient(default=gen(oracle.boxed("up"), reasoning="x " * 200)),
    ).run_instance(task.instance(spec))

    assert len(ours.episode_successes_scored) < EPISODES_PER_INSTANCE
    assert len(theirs.episode_successes_scored) == EPISODES_PER_INSTANCE
    assert theirs.episode_successes_scored[-1] is False

    from hexa_orbit import report
    # Under ORBIT's convention the unreached episode drags the rate down; under ours
    # it is simply absent from the denominator.
    assert report.episode_success_rates([theirs])["episode_3"] == 0.0
    assert report.episode_success_rates([ours])["episode_3"] == 0.0  # no data -> 0/0


def test_token_accounting_records_whether_usage_was_reported(task: MazeTask, spec):
    """A budget enforced on an estimate is not ORBIT's budget; say which it was."""
    from hexa_core.llm.base import Generation

    class NoUsage:
        def generate_structured(self, **_):
            return Generation.from_parts("", oracle.boxed("up"), finish_reason="stop")

    run = MazeActor(ActorConfig(), client=NoUsage()).run_instance(task.instance(spec))
    assert run.token_accounting == "estimated"
    assert run.completion_tokens > 0


def test_actor_config_manifest_includes_the_new_fields():
    """The arm A summary recorded transcript_mode as null -- to_dict had drifted."""
    d = ActorConfig(transcript_mode="raw", trajectory_token_budget=31744,
                    truncation_scoring="orbit").to_dict()
    assert d["transcript_mode"] == "raw"
    assert d["trajectory_token_budget"] == 31744
    assert d["truncation_scoring"] == "orbit"


def test_a_runaway_turn_cannot_kill_the_instance():
    """The transcript copy is capped; see `_MAX_SPOKEN_CHARS`.

    A turn truncated mid-reasoning has no `\\boxed{}`, so `extract_last_boxed`
    returns the whole response and it used to land in `self.messages` whole --
    blowing the next prompt past the context window and ending the instance.
    """
    runaway = "Let me reconsider the constraints. " * 800
    capped = actor._cap_spoken(runaway)
    assert len(capped) < actor._MAX_SPOKEN_CHARS + 100
    assert "unparsed" in capped
    assert capped.startswith("Let me reconsider the constraints.")


def test_a_normal_turn_is_written_back_untouched():
    for text in (r"\boxed{CRANE}", "CRANE", "reveal 2 3", ""):
        assert actor._cap_spoken(text) == text
