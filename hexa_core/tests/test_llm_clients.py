"""LLM client checks: scored-string parity, request body, effort transport, retries.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_llm_clients

No pytest, no network, no GPU, and no ``openai``: a fake transport is injected by
replacing ``VLLMClient._make_client``, so every request body the client would send
is inspectable and every server behaviour (400s, 500s, empty completions,
truncation, a missing reasoning parser) is reproducible.
"""

from __future__ import annotations

import inspect
import logging
import sys
from types import SimpleNamespace
from typing import Any

from hexa_core.llm import (
    DEFAULT_REASONING_EFFORT,
    PAPER_MAX_TOKENS,
    REQUIRED_MANIFEST_KEYS,
    Generation,
    LLMClient,
    ProbeFailed,
    ReasoningEffortUnsupported,
    VLLMClient,
    make_client,
)
from hexa_core.llm.base import LLMClientError
from hexa_core.llm.vllm_client import (
    TRANSPORT_CHAT_TEMPLATE,
    TRANSPORT_TOP_LEVEL,
)


# --------------------------------------------------------------------- fakes


class FakeBadRequest(Exception):
    """Stands in for ``openai.BadRequestError``: classified by name and status."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400


class FakeServerError(Exception):
    def __init__(self, message: str = "upstream connect error") -> None:
        super().__init__(message)
        self.status_code = 503


def stream_chunks(
    reasoning: str,
    content: str,
    *,
    finish_reason: str | None = "stop",
    usage: dict[str, Any] | None = None,
    omit_reasoning_field: bool = False,
) -> list[Any]:
    """Chunks shaped like vLLM's, one character per chunk to exercise joining."""
    chunks: list[Any] = []
    for char in reasoning:
        delta = SimpleNamespace(content=None) if omit_reasoning_field else SimpleNamespace(
            reasoning_content=char, content=None
        )
        chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None))
    for char in content:
        delta = SimpleNamespace(reasoning_content=None, content=char)
        chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None))
    chunks.append(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning_content=None, content=None),
                    finish_reason=finish_reason,
                )
            ],
            usage=None,
        )
    )
    # vLLM's usage chunk carries an empty choices list -- the reason usage has to
    # be read before CORE's `if not chunk.choices: continue`.
    chunks.append(SimpleNamespace(choices=[], usage=usage))
    return chunks


def non_stream_response(
    reasoning: str,
    content: str,
    *,
    finish_reason: str | None = "stop",
    usage: dict[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(reasoning_content=reasoning, content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


class FakeTransport:
    """Records every request body and replays scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params: Any) -> Any:
        self.requests.append(params)
        if not self.responses:
            raise AssertionError("fake transport ran out of scripted responses")
        outcome = self.responses.pop(0)
        if callable(outcome):
            outcome = outcome(params)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def last(self) -> dict[str, Any]:
        return self.requests[-1]


def wire(client: VLLMClient, responses: list[Any]) -> FakeTransport:
    """Point the client at a fake transport and make backoff free."""
    transport = FakeTransport(responses)
    client._make_client = lambda: transport  # type: ignore[method-assign]
    client._sleep = lambda seconds: None  # type: ignore[method-assign]
    return transport


def new_client(**kwargs: Any) -> VLLMClient:
    kwargs.setdefault("model", "gpt-oss-120b")
    return VLLMClient(**kwargs)


# --------------------------------------------------------------------- checks


def check_concatenation_parity() -> None:
    """raw_text is reasoning + final, byte-exact, no separator, both modes."""
    reasoning, final = "We need 8 - 1 = 7.\n", "The answer is 7."

    client = new_client()
    wire(client, [stream_chunks(reasoning, final)])
    streamed = client.generate_structured(prompt="p")
    assert streamed is not None
    assert streamed.raw_text == reasoning + final, repr(streamed.raw_text)
    assert streamed.reasoning_text == reasoning
    assert streamed.final_text == final

    client = new_client(stream=False)
    wire(client, [non_stream_response(reasoning, final)])
    direct = client.generate_structured(prompt="p")
    assert direct is not None
    assert direct.raw_text == streamed.raw_text
    assert direct.reasoning_text == reasoning and direct.final_text == final

    # `generate` returns exactly the scored string.
    client = new_client()
    wire(client, [stream_chunks(reasoning, final)])
    assert client.generate("p") == reasoning + final
    print("[ok] raw_text == reasoning_content + content, no separator (stream + non-stream)")


def check_from_parts_matrix() -> None:
    """from_parts agrees with a literal reimplementation of nvidia_client.py:143-144."""
    for reasoning in ("", "R"):
        for final in ("", "F"):
            parts = [p for p in (reasoning, final) if p]
            assert Generation.from_parts(reasoning, final).raw_text == "".join(parts)
    assert Generation.from_parts("", "").is_empty
    assert Generation.from_parts("r", "f", "length").truncated
    assert not Generation.from_parts("r", "f", "stop").truncated
    usage = Generation.from_parts("r", "f", "stop", SimpleNamespace(total_tokens=5)).usage
    assert usage == {"total_tokens": 5}, usage
    print("[ok] Generation.from_parts matches CORE's concatenation across the empty matrix")


def check_signature_parity() -> None:
    """Parameter names AND order must match CORE's, so positional calls survive.

    Transcribed from `nvidia_client.py:70-81` and `153-163`. CORE's driver calls by
    keyword (`run_core.py:335-362`), but a mismatch here would also silently change
    which value a positional call lands on.
    """
    structured = ("self", "prompt", "max_new_tokens", "temperature", "stream", "seed",
                  "top_p", "system_prompt", "reasoning_effort", "messages")
    plain = structured[:-1]
    for cls in (VLLMClient,):
        assert tuple(inspect.signature(cls.generate_structured).parameters) == structured, cls
        assert tuple(inspect.signature(cls.generate).parameters) == plain, cls

    # Positional calls land where CORE's would.
    client = new_client()
    transport = wire(client, [non_stream_response("r", "f")])
    generation = client.generate_structured("p", 128, 0.9, False)
    body = transport.last
    assert (body["max_tokens"], body["temperature"], body["stream"]) == (128, 0.9, False)
    assert generation is not None and generation.raw_text == "rf"
    print("[ok] generate/generate_structured signatures match NvidiaClient's, order included")


def check_request_body() -> None:
    client = new_client(seed=None)
    transport = wire(client, [stream_chunks("r", "f")])
    client.generate_structured(prompt="hello", system_prompt="sys")
    body = transport.last
    assert body["max_tokens"] == PAPER_MAX_TOKENS == 32768, body["max_tokens"]
    assert body["temperature"] == 0.6 and body["top_p"] == 0.95
    assert body["reasoning_effort"] == "high", body
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert "seed" not in body, "seed must be omitted, not sent as None"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    transport = wire(client, [stream_chunks("r", "f")])
    client.generate_structured(prompt="hello", temperature=0.8, top_p=0.5, seed=17, max_new_tokens=64)
    body = transport.last
    assert (body["temperature"], body["top_p"], body["seed"], body["max_tokens"]) == (0.8, 0.5, 17, 64)

    # top_p omitted entirely when the client is configured without one.
    bare = new_client(top_p=None)
    transport = wire(bare, [stream_chunks("r", "f")])
    bare.generate_structured(prompt="hello")
    assert "top_p" not in transport.last

    # reasoning_effort=None disables the field rather than sending null.
    off = new_client(reasoning_effort=None)
    transport = wire(off, [stream_chunks("r", "f")])
    off.generate_structured(prompt="hello")
    assert "reasoning_effort" not in transport.last and "extra_body" not in transport.last

    try:
        client.generate_structured()
    except ValueError as exc:
        assert str(exc) == "Either prompt or messages must be provided"
    else:
        raise AssertionError("missing prompt/messages was not rejected")
    print("[ok] request body: 32768 budget, temp/top_p/seed passthrough, unset keys omitted")


def check_effort_transport_fallback() -> None:
    client = new_client()
    transport = wire(
        client,
        [
            FakeBadRequest("Unrecognized request argument supplied: reasoning_effort"),
            stream_chunks("r", "f"),
            stream_chunks("r2", "f2"),
        ],
    )
    generation = client.generate_structured(prompt="p")
    assert generation is not None and generation.raw_text == "rf"
    assert len(transport.requests) == 2, transport.requests
    retried = transport.requests[1]
    assert "reasoning_effort" not in retried
    assert retried["extra_body"]["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert client._effort_transport == TRANSPORT_CHAT_TEMPLATE

    # Latched: the next call goes straight to the working transport.
    client.generate_structured(prompt="p2")
    assert "reasoning_effort" not in transport.requests[2]
    assert transport.requests[2]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"
    assert client.manifest()["reasoning_effort_transport"] == TRANSPORT_CHAT_TEMPLATE

    both = new_client()
    wire(
        both,
        [
            FakeBadRequest("unknown field: reasoning_effort"),
            FakeBadRequest("chat_template_kwargs is not permitted"),
        ],
    )
    try:
        both.generate_structured(prompt="p")
    except ReasoningEffortUnsupported as exc:
        assert "reasoning-parser" in str(exc) or "vLLM version" in str(exc)
    else:
        raise AssertionError("both transports rejected but no ReasoningEffortUnsupported")

    # A 400 that has nothing to do with effort must not flip the transport.
    unrelated = new_client()
    wire(unrelated, [FakeBadRequest("messages: at least one message is required")])
    try:
        unrelated.generate_structured(prompt="p")
    except FakeBadRequest:
        pass
    else:
        raise AssertionError("unrelated 400 was swallowed")
    assert unrelated._effort_transport == TRANSPORT_TOP_LEVEL

    # stream_options is metadata: dropped and retried, never fatal.
    usage_fussy = new_client()
    transport = wire(
        usage_fussy,
        [FakeBadRequest("stream_options is not supported"), stream_chunks("r", "f")],
    )
    assert usage_fussy.generate_structured(prompt="p") is not None
    assert usage_fussy.include_usage is False
    assert "stream_options" not in transport.requests[1]
    print("[ok] reasoning_effort falls back to chat_template_kwargs, latches, and fails loudly")


def check_truncation_and_usage() -> None:
    client = new_client()
    usage = {"completion_tokens": 32768, "completion_tokens_details": {"reasoning_tokens": 32700}}
    wire(client, [stream_chunks("long reasoning", "", finish_reason="length", usage=usage)])
    generation = client.generate_structured(prompt="63 move hanoi")
    assert generation is not None
    assert generation.finish_reason == "length" and generation.truncated
    assert generation.usage == usage, "usage chunk (empty choices) was dropped"
    assert client.stats["truncated"] == 1
    assert generation.to_dict()["truncated"] is True
    print("[ok] finish_reason='length' surfaces as Generation.truncated, with usage")


def check_retries_and_none_vs_empty() -> None:
    client = new_client(max_retries=3)
    wire(client, [FakeServerError(), FakeServerError(), stream_chunks("r", "f")])
    assert client.generate_structured(prompt="p") is not None
    assert client.stats["retries"] == 2 and client.stats["failures"] == 0

    exhausted = new_client(max_retries=2)
    wire(exhausted, [FakeServerError(), FakeServerError()])
    assert exhausted.generate_structured(prompt="p") is None, "exhausted retries must yield None"
    assert exhausted.stats["failures"] == 1

    # generate() propagates that None instead of raising AttributeError on
    # None.raw_text, which is what generate_answer's `if out is None` wants.
    propagates = new_client(max_retries=1)
    wire(propagates, [FakeServerError()])
    assert propagates.generate("p") is None

    # A server that answers with nothing is NOT a failure: returning None there
    # would make CORE's wrapper retry a request that succeeded.
    empty = new_client()
    wire(empty, [stream_chunks("", "", finish_reason="stop")])
    generation = empty.generate_structured(prompt="p")
    assert generation is not None, "empty completion must not be reported as None"
    assert generation.is_empty and generation.finish_reason == "stop"
    assert empty.stats["empty_generations"] == 1 and empty.stats["failures"] == 0

    # Non-transient errors are raised immediately rather than retried 6 times.
    fatal = new_client(max_retries=6)
    transport = wire(fatal, [FakeBadRequest("context length exceeded")])
    try:
        fatal.generate_structured(prompt="p")
    except FakeBadRequest:
        pass
    else:
        raise AssertionError("deterministic 400 was retried instead of raised")
    assert len(transport.requests) == 1

    raiser = new_client(max_retries=1, raise_on_exhausted=True)
    wire(raiser, [FakeServerError()])
    try:
        raiser.generate_structured(prompt="p")
    except RuntimeError:
        pass
    else:
        raise AssertionError("raise_on_exhausted did not raise")
    print("[ok] transient errors retry; None means failure, empty Generation means silence")


def check_probe() -> None:
    good = [
        stream_chunks("thinking hard about this", "56"),          # parser check
        FakeBadRequest("invalid value for reasoning_effort"),      # garbage rejected
        stream_chunks("short", "56"),                              # low
        stream_chunks("much much much longer thinking", "56"),     # high
    ]
    client = new_client()
    wire(client, list(good))
    evidence = client.probe()
    assert evidence["reasoning_parser"] is True
    assert evidence["garbage_value_rejected"] is True
    assert evidence["effort_verified"] == "yes", evidence
    assert evidence["differential"]["ratio"] >= 1.5
    assert client.manifest()["probe"]["effort_verified"] == "yes"

    # No reasoning_content at all: the --reasoning-parser mistake. Hard failure.
    blind = new_client()
    wire(blind, [stream_chunks("ignored", "56", omit_reasoning_field=True)])
    try:
        blind.probe()
    except ProbeFailed as exc:
        assert "--reasoning-parser openai_gptoss" in str(exc)
    else:
        raise AssertionError("missing reasoning_content did not fail the probe")

    # Both signals negative: garbage accepted AND effort makes no difference.
    ignoring = new_client()
    wire(
        ignoring,
        [
            stream_chunks("thinking", "56"),
            non_stream_response("", "OK"),
            stream_chunks("same length here", "56"),
            stream_chunks("same length here", "56"),
        ],
    )
    try:
        ignoring.probe()
    except ProbeFailed as exc:
        assert "ignoring reasoning_effort" in str(exc)
    else:
        raise AssertionError("a server ignoring reasoning_effort passed the probe")

    # Exactly one negative: warn, record evidence, keep going.
    partial = new_client()
    wire(
        partial,
        [
            stream_chunks("thinking", "56"),
            non_stream_response("", "OK"),                          # garbage accepted
            stream_chunks("short", "56"),
            stream_chunks("much much much longer thinking", "56"),  # but effort works
        ],
    )
    evidence = partial.probe()
    assert evidence["effort_verified"] == "partial", evidence
    assert evidence["ignored_signals"] == ["garbage_value_accepted"]

    strict = new_client()
    wire(
        strict,
        [
            stream_chunks("thinking", "56"),
            non_stream_response("", "OK"),
            stream_chunks("short", "56"),
            stream_chunks("much much much longer thinking", "56"),
        ],
    )
    try:
        strict.probe(strict=True)
    except ProbeFailed:
        pass
    else:
        raise AssertionError("strict probe accepted a partially-verified server")

    # A dead server is a probe failure with a pointed message, not a stack trace.
    dead = new_client(max_retries=1)
    wire(dead, [FakeServerError("connection refused")])
    try:
        dead.probe()
    except ProbeFailed as exc:
        assert "returned no answer" in str(exc)
    else:
        raise AssertionError("unreachable server passed the probe")
    print("[ok] probe: parser check hard-fails; effort acceptance judged on two signals")


def check_manifests() -> None:
    client = new_client()
    manifest = client.manifest()
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    assert not missing, missing
    assert manifest["provider"] == "vllm"
    assert manifest["max_tokens"] == 32768 and manifest["reasoning_effort"] == "high"

    assert isinstance(client, LLMClient)
    try:
        make_client("bedrock")
    except ValueError as exc:
        assert "unknown provider" in str(exc)
    else:
        raise AssertionError("make_client accepted an unknown provider")
    print("[ok] the manifest carries every required provenance key")


# The NimClient section (FakeCoreClient, fake_nim, sentinel translation, the
# from_core tripwire) moved to hexa_core/_archive/ with the client itself: it
# existed to A/B our provider against CORE's, a comparison the project no longer
# makes. `Generation.from_core` is still covered by test_llm_parity.


def main() -> int:
    # Failing servers are exercised on purpose here; the client's own warnings
    # about them would otherwise look like test failures.
    logging.getLogger("hexa_core.llm").setLevel(logging.CRITICAL)
    check_concatenation_parity()
    check_from_parts_matrix()
    check_signature_parity()
    check_request_body()
    check_effort_transport_fallback()
    check_truncation_and_usage()
    check_retries_and_none_vs_empty()
    check_probe()
    check_manifests()
    print("\nall llm client checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
