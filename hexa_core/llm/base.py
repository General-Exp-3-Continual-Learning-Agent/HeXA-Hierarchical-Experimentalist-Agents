"""Provider-neutral generation record and client protocol.

The one fact this module exists to pin down is *what text gets scored*. CORE's
``NvidiaClient.generate_structured`` returns

    raw_text = reasoning_text + final_text        # concatenated, NO separator

(`core-reasoning/src/model/nvidia_client.py:143-144`) and ``generate_answer``
scores exactly that string (`core-reasoning/scripts/run_core.py:468`). The
chain-of-thought is part of the verifier's input, which is load-bearing rather
than cosmetic: Tower-of-Hanoi's extractor takes the *last* parseable move list in
the text, so including or excluding the reasoning changes which candidate wins.

That rule is written down once, in :meth:`Generation.from_parts`, and every
client in this package routes through it. :meth:`Generation.from_core` adapts
CORE's own ``NvidiaGeneration`` and re-checks the invariant, so if CORE ever
changes the concatenation the reference path fails loudly instead of quietly
diverging from ours.

``openai`` is not importable in this environment, so nothing here (or in any
module it imports) may touch it: error classification is duck-typed on status
codes and exception class names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Optional, Protocol, runtime_checkable

from hexa_core.tokens import counter_name

# Paper configuration, from CORE's shell scripts (see the shared brief). Both
# clients default to these so a run is the paper config unless something says
# otherwise, and so the numbers live in one place.
PAPER_TEMPERATURE: Final = 0.6
PAPER_TOP_P: Final = 0.95
PAPER_MAX_TOKENS: Final = 32768
PAPER_REASONING_EFFORT: Final = "high"


class _DefaultReasoningEffort:
    """Sentinel type: 'caller expressed no preference, use the client default'."""

    __slots__ = ()

    def __repr__(self) -> str:  # legible in tracebacks and request logs
        return "DEFAULT_REASONING_EFFORT"


#: Analogue of CORE's private ``_DEFAULT_REASONING_EFFORT = object()``
#: (`nvidia_client.py:8`). It is compared by identity, exactly as CORE does.
#:
#: WARNING: this is a *different object* from CORE's sentinel. Passing it into
#: ``NvidiaClient`` would take the ``elif reasoning_effort is not None`` branch
#: and put a bare object into the JSON request body. :mod:`hexa_core.llm.nim_client`
#: is responsible for translating it away.
DEFAULT_REASONING_EFFORT: Final = _DefaultReasoningEffort()


class LLMClientError(RuntimeError):
    """Base class for client-layer failures that are not transport hiccups."""


class ProbeFailed(LLMClientError):
    """Startup verification failed; running anyway would produce bad numbers."""


class ReasoningEffortUnsupported(LLMClientError):
    """The server rejected ``reasoning_effort`` through every known transport."""


class MissingDependency(LLMClientError):
    """An optional import (``openai``, CORE) is needed for this code path."""


def _safe_text(value: Any) -> str:
    """Transcribed from `nvidia_client.py:37-39`."""
    return "" if value is None else str(value)


#: Response fields that may carry the reasoning/analysis channel, newest name first.
#:
#: This is **not** cosmetic, and getting it wrong fails silently in the worst
#: possible way. CORE's ``NvidiaClient`` reads ``reasoning_content``
#: (`nvidia_client.py:137`), which is correct for NVIDIA NIM. But vLLM 0.26.0
#: renamed the *response* field to ``reasoning`` — verified in the installed
#: package: ``ChatMessage.reasoning`` (`chat_completion/protocol.py:65`) and
#: ``DeltaMessage.reasoning`` (`openai/engine/protocol.py:396`). The
#: ``reasoning_content`` → ``reasoning`` compatibility shim at
#: `chat_completion/protocol.py:503-505` runs on **inbound request** messages
#: only; there is no alias on the way out.
#:
#: So a client reading only ``reasoning_content`` against vLLM 0.26.0 gets
#: ``None``, ``raw_text`` collapses to the final channel, and the entire
#: chain-of-thought vanishes from the string the verifier parses. Nothing errors.
#: Accuracy just drops — worst on Tower of Hanoi, whose
#: ``_extract_moves_from_text`` takes the *last* parseable move list in the text,
#: so which candidate wins depends on the reasoning being present.
#:
#: Read both, newest first, and record which one fired.
REASONING_FIELDS: tuple[str, ...] = ("reasoning", "reasoning_content")


def _pick_reasoning(obj: Any) -> tuple[Optional[str], Optional[str]]:
    """Return ``(text, field_name)`` for whichever reasoning field is populated.

    ``field_name`` is what the run manifest records, so "we ran with the
    chain-of-thought included" is auditable after the fact instead of assumed.
    A field present but empty is not a match: an empty analysis channel and an
    absent one are indistinguishable downstream, and the *absent* case is the one
    that means the server is misconfigured.
    """
    for field in REASONING_FIELDS:
        value = getattr(obj, field, None)
        if value:
            return value, field
    return None, None


def _serialize_usage(usage: Any) -> Optional[dict[str, Any]]:
    """Transcribed from `nvidia_client.py:41-51` so usage normalises identically."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return {"value": str(usage)}


@dataclass
class Generation:
    """One completion. Field-for-field identical to CORE's ``NvidiaGeneration``.

    Same names, same order, same mutable-dataclass shape, so anything that reads
    ``.raw_text`` / ``.finish_reason`` off CORE's object reads ours unchanged.
    """

    raw_text: str
    reasoning_text: str
    final_text: str
    finish_reason: Optional[str]
    usage: Optional[dict[str, Any]]

    @classmethod
    def from_parts(
        cls,
        reasoning_text: str,
        final_text: str,
        finish_reason: Optional[str] = None,
        usage: Any = None,
    ) -> Generation:
        """Build a generation, applying CORE's concatenation rule.

        Literal transcription of `nvidia_client.py:143-144`::

            parts = [part for part in (reasoning_text, final_text) if part]
            raw_text = "".join(parts)

        The truthiness filter is a no-op for ``str.join``, so the rule reduces to
        ``reasoning + final`` with no separator. It is kept in transcribed form
        anyway: this is the line the whole comparison rests on, and a reader
        should be able to diff it against CORE by eye.
        """
        parts = [part for part in (reasoning_text, final_text) if part]
        return cls(
            raw_text="".join(parts),
            reasoning_text=reasoning_text,
            final_text=final_text,
            finish_reason=finish_reason,
            usage=_serialize_usage(usage),
        )

    @classmethod
    def from_core(cls, gen: Any, *, verify: bool = True) -> Generation:
        """Adapt a CORE ``NvidiaGeneration`` (duck-typed; no CORE import needed).

        ``raw_text`` is copied verbatim rather than recomputed -- this path is the
        A/B *reference*, so what CORE produced is what gets scored. ``verify``
        then asserts the copy agrees with :meth:`from_parts`, which is a drift
        tripwire on CORE's concatenation rather than a check on our own code.
        """
        reasoning_text = _safe_text(getattr(gen, "reasoning_text", None))
        final_text = _safe_text(getattr(gen, "final_text", None))
        raw_text = _safe_text(getattr(gen, "raw_text", None))
        finish_reason = getattr(gen, "finish_reason", None)
        if finish_reason is not None:
            finish_reason = str(finish_reason)

        if verify:
            expected = cls.from_parts(reasoning_text, final_text).raw_text
            if raw_text != expected:
                raise LLMClientError(
                    "CORE's raw_text no longer equals reasoning_text + final_text. "
                    "The scored-string parity rule this package is built on has "
                    "changed upstream; re-read nvidia_client.py before running "
                    f"anything.\n  len(raw)={len(raw_text)} len(expected)={len(expected)}"
                )

        return cls(
            raw_text=raw_text,
            reasoning_text=reasoning_text,
            final_text=final_text,
            finish_reason=finish_reason,
            usage=_serialize_usage(getattr(gen, "usage", None)),
        )

    @property
    def truncated(self) -> bool:
        """Hit the token ceiling. Prime suspect on Hanoi n=6 (63-move solutions)."""
        return self.finish_reason == "length"

    @property
    def is_empty(self) -> bool:
        return not self.raw_text

    def to_dict(self) -> dict[str, Any]:
        """Record form for attempt/trajectory logs. Includes the derived flags so
        log consumers never re-derive ``truncated`` and get it subtly wrong."""
        return {
            "raw_text": self.raw_text,
            "reasoning_text": self.reasoning_text,
            "final_text": self.final_text,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "truncated": self.truncated,
            "is_empty": self.is_empty,
        }


@runtime_checkable
class LLMClient(Protocol):
    """The call surface CORE's driver expects, so either client is drop-in.

    Parameter names and order are copied from `nvidia_client.py:70-81` and
    `153-163`. ``_build_answer_call_kwargs`` (`run_core.py:335-362`) only ever
    emits ``prompt`` / ``messages`` / ``max_new_tokens`` / ``temperature`` /
    ``top_p`` / ``system_prompt`` / ``seed`` -- all by keyword -- so those exact
    names are the contract. It never passes ``stream`` or ``reasoning_effort``:
    getting those right is the *client's* responsibility, not the caller's.

    Note the asymmetry, preserved from CORE: ``generate`` has no ``messages``
    parameter. Defaults are written ``...`` because implementations resolve
    ``None`` against their own configuration; see :class:`hexa_core.llm.VLLMClient`.

    Return contract, which CORE's retry wrappers depend on
    (`llm_utils.py:299-300`, `330-331` raise on a ``None`` result):

    ``None``
        No answer obtained: transport/infra failure, retries exhausted.
    ``Generation(raw_text="")``
        The server answered and the model emitted nothing. ``finish_reason`` and
        ``usage`` say why. This must NOT be collapsed to ``None`` or CORE's
        wrapper will spin on a request that is succeeding.
    """

    model: str

    def generate_structured(
        self,
        prompt: str | None = ...,
        max_new_tokens: int | None = ...,
        temperature: float | None = ...,
        stream: bool | None = ...,
        seed: int | None = ...,
        top_p: float | None = ...,
        system_prompt: str | None = ...,
        reasoning_effort: Any = ...,
        messages: list[dict[str, Any]] | None = ...,
    ) -> Generation | None: ...

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = ...,
        temperature: float | None = ...,
        stream: bool | None = ...,
        seed: int | None = ...,
        top_p: float | None = ...,
        system_prompt: str | None = ...,
        reasoning_effort: Any = ...,
    ) -> str | None: ...

    def manifest(self) -> dict[str, Any]: ...

    def probe(self) -> dict[str, Any]: ...


#: Keys every client's :meth:`manifest` must carry into the run manifest. A run
#: whose provenance is missing one of these cannot be compared to another run.
REQUIRED_MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "provider",
    "model",
    "base_url",
    "reasoning_effort",
    "temperature",
    "top_p",
    "max_tokens",
    "token_counter",
)


def build_manifest(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    reasoning_effort: Any,
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble a provenance dict with the required keys always present.

    Both clients go through here so the key set cannot drift between providers.
    ``token_counter`` is filled from :func:`hexa_core.tokens.counter_name`, which
    is what makes an added-token figure interpretable after the fact.
    """
    if isinstance(reasoning_effort, _DefaultReasoningEffort):
        # Never let the sentinel leak into a JSON manifest.
        reasoning_effort = repr(reasoning_effort)
    manifest: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "token_counter": counter_name(),
    }
    manifest.update(extra)
    return manifest


_BAD_REQUEST_NAMES: Final = frozenset({"BadRequestError", "UnprocessableEntityError"})
_TRANSIENT_NAMES: Final = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "APIConnectionTimeoutError",
        "RateLimitError",
        "InternalServerError",
        "APIStatusError",
        "APIError",
    }
)
_TRANSIENT_STATUS: Final = frozenset({408, 409, 429})


def status_code(exc: BaseException) -> int | None:
    """HTTP status off an openai-style exception, without importing openai."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def is_bad_request(exc: BaseException) -> bool:
    """A deterministic rejection: the request itself is wrong, retrying will not help."""
    code = status_code(exc)
    if code in (400, 422):
        return True
    return type(exc).__name__ in _BAD_REQUEST_NAMES


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: connection loss, timeout, rate limit, or a 5xx.

    Deliberately excludes deterministic 4xx (400/401/404/422). Retrying a
    rejected parameter six times with backoff burns the budget and buries the
    cause under a wall of identical tracebacks.
    """
    code = status_code(exc)
    if code is not None:
        return code in _TRANSIENT_STATUS or code >= 500
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))
