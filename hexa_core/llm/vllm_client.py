"""OpenAI-compatible client for a local vLLM server, at parity with CORE.

Parity obligations, all of which this module is required to satisfy:

* ``raw_text == <reasoning field> + content``, concatenated with **no**
  separator, via :meth:`hexa_core.llm.base.Generation.from_parts`.
* ``max_tokens=32768`` is a single budget spanning reasoning *and* final answer,
  exactly as in `nvidia_client.py:90`.
* ``temperature`` / ``top_p`` / ``seed`` pass through unchanged, and ``top_p`` /
  ``seed`` are **omitted** from the body when unset -- as CORE does
  (`nvidia_client.py:98-101`), because "omitted" and "None" are not the same
  request.
* ``reasoning_effort="high"`` actually reaches the model. Sent top-level first,
  with an automatic, latched fallback to
  ``extra_body.chat_template_kwargs.reasoning_effort`` for older vLLM builds.
* ``finish_reason`` is surfaced, so truncation at 32768 is visible instead of
  looking like a wrong answer.

Deliberate deviations from ``NvidiaClient``, none of which can change a scored
string:

===========================================  =============================================
Deviation                                    Why it is safe
===========================================  =============================================
``max_new_tokens``/``temperature``/           CORE's driver always passes these explicitly
``top_p``/``seed``/``stream`` default to      (`run_core.py:335-362`), so under CORE
``None`` = "use client config"                behaviour is identical; a standalone caller
                                              gets 32768 instead of a silent 1024 cut-off.
``stream_options={"include_usage": True}``    Metadata only; auto-dropped if rejected.
No model-name gate on ``reasoning_effort``    vLLM model strings are snapshot paths or
                                              ``--served-model-name`` aliases, so CORE's
                                              ``"gpt-oss-120b" in model`` substring test is
                                              unreliable here. ``reasoning_effort=None``
                                              disables it explicitly.
SDK ``max_retries=0``, our own backoff        We own the retry accounting; per-attempt
                                              request bodies are unchanged.
``generate()`` propagates ``None``            CORE would raise ``AttributeError`` on
                                              ``None.raw_text``; ``generate_answer``'s
                                              ``if out is None`` branch wants ``None``.
===========================================  =============================================

``openai`` is imported lazily inside :meth:`VLLMClient._make_client`, so this
module imports (and its tests run) in an environment without it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Optional

from hexa_core.llm.base import (
    DEFAULT_REASONING_EFFORT,
    PAPER_MAX_TOKENS,
    PAPER_REASONING_EFFORT,
    PAPER_TEMPERATURE,
    PAPER_TOP_P,
    Generation,
    MissingDependency,
    ProbeFailed,
    ReasoningEffortUnsupported,
    _pick_reasoning,
    _safe_text,
    build_manifest,
    is_bad_request,
    is_transient,
    status_code,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"

#: How ``reasoning_effort`` is delivered. ``top_level`` is the modern OpenAI-style
#: field; ``chat_template_kwargs`` is what older vLLM builds accept.
TRANSPORT_TOP_LEVEL = "top_level"
TRANSPORT_CHAT_TEMPLATE = "chat_template_kwargs"
TRANSPORT_UNSUPPORTED = "unsupported"

_EFFORT_COMPLAINT = re.compile(r"reasoning[_ ]?effort", re.IGNORECASE)
_CHAT_TEMPLATE_COMPLAINT = re.compile(r"chat[_ ]?template[_ ]?kwargs", re.IGNORECASE)
_UNKNOWN_FIELD_COMPLAINT = re.compile(
    r"extra_forbidden|unexpected keyword|unknown field|unrecognized|"
    r"extra inputs are not permitted|not permitted|got an unexpected",
    re.IGNORECASE,
)
_USAGE_COMPLAINT = re.compile(r"stream_options|include_usage", re.IGNORECASE)

_GARBAGE_EFFORT = "__hexa_invalid_effort__"

#: Ratio of high-effort to low-effort reasoning length below which we conclude the
#: server is ignoring the field. A judgement call: the measurements themselves go
#: into the manifest so it can be recalibrated from real data instead of re-guessed.
_DIFFERENTIAL_MIN_RATIO = 1.5


def _error_text(exc: BaseException) -> str:
    """Everything an openai-style error might hide its complaint in."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(repr(body))
    message = getattr(exc, "message", None)
    if message is not None:
        parts.append(str(message))
    return "\n".join(parts)


class VLLMClient:
    """Chat-completions client for a vLLM server, drop-in for CORE's ``NvidiaClient``.

    The constructor holds run configuration: the values used when a call does not
    override them, the provenance reported by :meth:`manifest`, and the defaults
    :meth:`probe` verifies. Defaults are the paper config (0.6 / 0.95 / 32768 /
    high).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        temperature: float = PAPER_TEMPERATURE,
        top_p: float | None = PAPER_TOP_P,
        max_tokens: int = PAPER_MAX_TOKENS,
        reasoning_effort: str | None = PAPER_REASONING_EFFORT,
        stream: bool = True,
        seed: int | None = None,
        timeout: float = 1800.0,
        max_retries: int = 6,
        retry_delay: float = 5.0,
        retry_backoff: float = 2.0,
        max_retry_delay: float = 60.0,
        include_usage: bool = True,
        raise_on_exhausted: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.getenv("HEXA_VLLM_URL") or os.getenv("VLLM_BASE_URL") or DEFAULT_BASE_URL
        # "EMPTY" matches react_agent/react_agent.py:535 -- local vLLM ignores it
        # but the SDK requires something.
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.max_retry_delay = max_retry_delay
        self.include_usage = include_usage
        self.raise_on_exhausted = raise_on_exhausted

        self._effort_transport = TRANSPORT_TOP_LEVEL
        self._transport_lock = threading.Lock()
        self._probe_result: dict[str, Any] | None = None
        self.stats: dict[str, int] = {
            "calls": 0,
            "retries": 0,
            "empty_generations": 0,
            "no_reasoning_content": 0,
            "truncated": 0,
            "failures": 0,
        }
        #: Which of ``base.REASONING_FIELDS`` the server actually populated.
        #: Surfaced in :meth:`manifest` because "the chain-of-thought was included
        #: in the scored text" is a claim a reviewer should be able to check, and
        #: the field name differs by server (NIM: ``reasoning_content``,
        #: vLLM >=0.26: ``reasoning``). ``None`` means no reasoning ever arrived.
        self.reasoning_field: str | None = None

    # ---------------------------------------------------------------- transport

    def _make_client(self) -> Any:
        """A fresh ``OpenAI`` per request.

        Mirrors `nvidia_client.py:26-32` and its reason: the Phase 2 driver
        batches with a ``ThreadPoolExecutor``, and a shared SDK client can
        cross-wire streaming responses between unrelated problems.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise MissingDependency(
                "openai package required for OpenAI-compatible models: pip install openai"
            ) from None
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            # We do our own retrying and accounting; the SDK's would double it.
            max_retries=0,
        )

    def _build_messages(
        self,
        *,
        prompt: Optional[str],
        system_prompt: Optional[str],
        messages: Optional[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Transcribed from `nvidia_client.py:53-68`, error message included."""
        if messages is not None:
            return list(messages)
        if prompt is None:
            raise ValueError("Either prompt or messages must be provided")
        built_messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            built_messages.append({"role": "system", "content": system_prompt})
        built_messages.append({"role": "user", "content": prompt})
        return built_messages

    def _apply_effort(self, params: dict[str, Any], effort: str) -> None:
        """Attach ``effort`` to ``params`` using the current transport."""
        transport = self._effort_transport
        if transport == TRANSPORT_TOP_LEVEL:
            params["reasoning_effort"] = effort
        elif transport == TRANSPORT_CHAT_TEMPLATE:
            extra_body = dict(params.get("extra_body") or {})
            # Merge rather than overwrite: a caller may already be passing
            # chat_template_kwargs for something else.
            template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            template_kwargs["reasoning_effort"] = effort
            extra_body["chat_template_kwargs"] = template_kwargs
            params["extra_body"] = extra_body
        else:
            raise ReasoningEffortUnsupported(
                f"reasoning_effort={effort!r} cannot be delivered to {self.base_url}: "
                "the server rejected both the top-level field and "
                "extra_body.chat_template_kwargs."
            )

    def _handle_bad_request(self, exc: BaseException, *, effort_active: bool) -> bool:
        """Adapt to a rejected parameter. Returns True if the request should be retried.

        Latching matters: the discovery cost is paid once (normally by
        :meth:`probe`), and ``manifest()["reasoning_effort_transport"]`` then
        records what actually worked -- which turns "we ran at high effort" from
        an assumption into an auditable claim.
        """
        text = _error_text(exc)

        if self.include_usage and _USAGE_COMPLAINT.search(text):
            self.include_usage = False
            logger.warning(
                "vLLM server rejected stream_options.include_usage; continuing without "
                "token counts (metadata only, sampling unaffected)."
            )
            return True

        if not effort_active:
            return False
        if _USAGE_COMPLAINT.search(text) and not _EFFORT_COMPLAINT.search(text):
            # Already-dropped stream_options complaining again: not effort's fault,
            # and mis-blaming it would put a wrong transport in the manifest.
            return False

        with self._transport_lock:
            transport = self._effort_transport
            if transport == TRANSPORT_TOP_LEVEL:
                blames_effort = bool(_EFFORT_COMPLAINT.search(text)) or bool(
                    _UNKNOWN_FIELD_COMPLAINT.search(text)
                )
                if not blames_effort:
                    return False
                self._effort_transport = TRANSPORT_CHAT_TEMPLATE
                logger.warning(
                    "vLLM server rejected top-level reasoning_effort; falling back to "
                    "extra_body.chat_template_kwargs for the rest of this run. "
                    "Server said: %s",
                    str(exc).strip()[:300],
                )
                return True

            if transport == TRANSPORT_CHAT_TEMPLATE:
                blames_effort = bool(_EFFORT_COMPLAINT.search(text)) or bool(
                    _CHAT_TEMPLATE_COMPLAINT.search(text)
                )
                if not blames_effort:
                    return False
                self._effort_transport = TRANSPORT_UNSUPPORTED
                raise ReasoningEffortUnsupported(
                    "Neither top-level reasoning_effort nor "
                    "extra_body.chat_template_kwargs was accepted by "
                    f"{self.base_url}. The run would silently proceed at the "
                    "server's default effort, which is the single most likely cause "
                    "of a reproduction miss. Check the vLLM version and that the "
                    "model was served with --reasoning-parser openai_gptoss.\n"
                    f"  server said: {str(exc).strip()[:500]}"
                ) from exc

        return False

    def _create(
        self,
        base_params: dict[str, Any],
        effort: str | None,
        *,
        allow_effort_fallback: bool = True,
    ) -> Any:
        """Issue one chat-completions request, adapting to rejected parameters."""
        last_exc: BaseException | None = None
        # At most: initial try + transport flip + include_usage drop.
        for _ in range(3):
            params = dict(base_params)
            if params.get("stream") and self.include_usage:
                # CORE reads usage off the Stream object, which is always None
                # (`nvidia_client.py:150`). We want completion-token counts for the
                # truncation diagnosis, so we opt in and read it off the chunks.
                params["stream_options"] = {"include_usage": True}
            if effort is not None:
                self._apply_effort(params, effort)
            try:
                return self._make_client().chat.completions.create(**params)
            except Exception as exc:  # noqa: BLE001 - re-raised unless adaptable
                last_exc = exc
                if not is_bad_request(exc):
                    raise
                if allow_effort_fallback and self._handle_bad_request(
                    exc, effort_active=effort is not None
                ):
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    # -------------------------------------------------------------- collection

    def _collect(self, response: Any, stream: bool) -> Generation:
        """Turn a response (or stream) into a :class:`Generation`.

        The streaming branch is a transcription of `nvidia_client.py:112-133` with
        one addition: ``chunk.usage`` is read *before* the ``if not chunk.choices``
        skip, because the usage chunk carries an empty ``choices`` list and CORE's
        ordering would drop it.
        """
        usage: Any = None
        if stream:
            reasoning_parts: list[str] = []
            final_parts: list[str] = []
            finish_reason: str | None = None
            for chunk in response:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                chunk_finish_reason = getattr(choice, "finish_reason", None)
                if chunk_finish_reason is not None:
                    finish_reason = str(chunk_finish_reason)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning, field = _pick_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(_safe_text(reasoning))
                    self.reasoning_field = field
                content = getattr(delta, "content", None)
                if content:
                    final_parts.append(_safe_text(content))
            reasoning_text = "".join(reasoning_parts)
            final_text = "".join(final_parts)
        else:
            choice = response.choices[0]
            message = choice.message
            reasoning_value, field = _pick_reasoning(message)
            if reasoning_value:
                self.reasoning_field = field
            reasoning_text = _safe_text(reasoning_value)
            final_text = _safe_text(getattr(message, "content", None))
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None:
                finish_reason = str(finish_reason)
            usage = getattr(response, "usage", None)

        return Generation.from_parts(reasoning_text, final_text, finish_reason, usage)

    def _sleep(self, seconds: float) -> None:
        """Isolated so tests can make backoff free."""
        time.sleep(seconds)

    def _backoff_delay(self, attempt: int) -> float:
        return min(self.max_retry_delay, self.retry_delay * self.retry_backoff ** (attempt - 1))

    # ------------------------------------------------------------------- calls

    def generate_structured(
        self,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        reasoning_effort: Any = DEFAULT_REASONING_EFFORT,
        messages: list[dict[str, Any]] | None = None,
    ) -> Generation | None:
        """One completion, with retries. Signature per CORE's ``generate_structured``.

        ``None`` arguments mean "use the configured value"; ``top_p`` and ``seed``
        resolving to ``None`` are omitted from the request body entirely.

        Return values are deliberately distinguishable:

        * ``None`` -- no answer obtained. Transport failure, retries exhausted.
        * ``Generation(raw_text="")`` -- the server answered; the model emitted
          nothing. ``finish_reason``/``usage`` say why. Never collapsed to
          ``None``, because CORE's wrapper treats ``None`` as an error and would
          retry a request that is in fact succeeding
          (`llm_utils.py:299-300`).

        Deterministic failures (bad request, effort undeliverable, missing
        ``openai``) raise rather than burning retries.
        """
        resolved_max_tokens = self.max_tokens if max_new_tokens is None else max_new_tokens
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_top_p = self.top_p if top_p is None else top_p
        resolved_seed = self.seed if seed is None else seed
        resolved_stream = self.stream if stream is None else stream
        effort = (
            self.reasoning_effort
            if reasoning_effort is DEFAULT_REASONING_EFFORT
            else reasoning_effort
        )

        base_params: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
            ),
            "temperature": resolved_temperature,
            # One budget spanning reasoning + final answer (nvidia_client.py:90).
            "max_tokens": resolved_max_tokens,
            "stream": resolved_stream,
        }
        if resolved_top_p is not None:
            base_params["top_p"] = resolved_top_p
        if resolved_seed is not None:
            base_params["seed"] = resolved_seed

        self.stats["calls"] += 1
        last_exc: BaseException | None = None
        for attempt in range(1, max(1, self.max_retries) + 1):
            try:
                response = self._create(base_params, effort)
                generation = self._collect(response, resolved_stream)
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                if not is_transient(exc):
                    self.stats["failures"] += 1
                    raise
                if attempt >= max(1, self.max_retries):
                    break
                delay = self._backoff_delay(attempt)
                self.stats["retries"] += 1
                logger.warning(
                    "vLLM request failed (attempt %d/%d): %s -- retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                self._sleep(delay)
                continue

            if generation.is_empty:
                self.stats["empty_generations"] += 1
            if not generation.reasoning_text:
                self.stats["no_reasoning_content"] += 1
            if generation.truncated:
                self.stats["truncated"] += 1
            return generation

        self.stats["failures"] += 1
        message = (
            f"vLLM request to {self.base_url} failed after {self.max_retries} attempts: "
            f"{last_exc}"
        )
        if self.raise_on_exhausted:
            raise RuntimeError(message) from last_exc
        # CORE-compatible: its wrapper converts None into an exception and retries
        # at a higher level with its own extended backoff.
        logger.error("%s -- returning None (treated upstream as an API failure)", message)
        return None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        reasoning_effort: Any = DEFAULT_REASONING_EFFORT,
    ) -> str | None:
        """The scored string. Mirrors `nvidia_client.py:153-173`.

        Unlike CORE, a failed call yields ``None`` rather than an
        ``AttributeError`` on ``None.raw_text`` -- ``generate_answer``'s
        ``if out is None: return None`` branch (`run_core.py:487`) is what wants.
        """
        generation = self.generate_structured(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream,
            seed=seed,
            top_p=top_p,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
        )
        return None if generation is None else generation.raw_text

    # ------------------------------------------------------------------- probe

    _PROBE_PROMPT = (
        "Work out 17 * 24 step by step, then state the number on its own line."
    )

    def probe(
        self,
        *,
        strict: bool = False,
        differential: bool = True,
    ) -> dict[str, Any]:
        """Verify at startup that this server can produce comparable numbers.

        Two things are checked, because each has silently invalidated whole runs:

        1. **The reasoning parser is on** (hard failure). If vLLM was started
           without ``--reasoning-parser openai_gptoss`` there is no
           any reasoning field, so ``raw_text`` drops the chain-of-thought --
           and since Hanoi's extractor takes the *last* parseable move list, every
           score in the run is then computed on the wrong string.
        2. **``reasoning_effort`` is not just accepted but honoured.** Non-rejection
           proves nothing: a server that ignores the field also does not complain
           about it. Two independent signals:

           * *garbage-value probe* -- a server that validates the field rejects
             ``reasoning_effort="__hexa_invalid_effort__"``; one that accepts it is
             not looking at the field.
           * *differential probe* -- the same prompt at ``low`` vs ``high``, comparing
             reasoning length; honoured effort shows up as markedly more thinking.

        Both signals negative is conclusive, and raises. Exactly one negative
        warns loudly and records ``effort_verified="partial"`` with both
        measurements, since a server may validate-and-clamp unknown values while
        still honouring ``"high"`` -- and a false failure that blocks a 3000-call
        run is its own kind of damage. ``strict=True`` fails on either signal.

        Costs roughly 1.2k tokens, once per run. Returns the evidence dict, which
        is also folded into :meth:`manifest`.
        """
        evidence: dict[str, Any] = {
            "base_url": self.base_url,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "strict": strict,
        }

        first = self._probe_request(max_new_tokens=96)
        evidence["reasoning_chars"] = len(first.reasoning_text)
        evidence["final_chars"] = len(first.final_text)
        evidence["finish_reason"] = first.finish_reason
        evidence["reasoning_parser"] = bool(first.reasoning_text)
        evidence["effort_transport"] = self._effort_transport
        # Which field name carried it. NIM says reasoning_content, vLLM >=0.26 says
        # reasoning; reading only one silently drops the CoT. See base.REASONING_FIELDS.
        evidence["reasoning_field"] = self.reasoning_field

        if not first.reasoning_text:
            self._probe_result = evidence
            raise ProbeFailed(
                "No reasoning channel came back from "
                f"{self.base_url} (model={self.model!r}).\n"
                "  Almost certainly the server was started without "
                "--reasoning-parser openai_gptoss.\n"
                "  Consequence: raw_text = reasoning + final silently loses the "
                "chain-of-thought, and because Tower-of-Hanoi's extractor takes the "
                "LAST parseable move list in the text, every score in the run would "
                "be computed on the wrong string.\n"
                f"  finish_reason={first.finish_reason!r} final_chars={len(first.final_text)}"
            )

        if self.reasoning_effort is None:
            evidence["effort_verified"] = "skipped(disabled)"
            self._probe_result = evidence
            logger.info("vLLM probe passed; reasoning_effort disabled by configuration.")
            return evidence

        # Signal 1: does the server validate the field at all?
        garbage_rejected = self._probe_garbage_effort()
        evidence["garbage_value_rejected"] = garbage_rejected

        # Signal 2: does effort change the amount of thinking?
        ratio: float | None = None
        if differential:
            low = self._probe_effort_size("low")
            high = self._probe_effort_size("high")
            ratio = high / max(1.0, float(low))
            evidence["differential"] = {
                "low": low,
                "high": high,
                "ratio": round(ratio, 3),
                "min_ratio": _DIFFERENTIAL_MIN_RATIO,
            }

        ignored_signals = [name for name, ignored in (
            ("garbage_value_accepted", not garbage_rejected),
            ("no_differential", ratio is not None and ratio < _DIFFERENTIAL_MIN_RATIO),
        ) if ignored]
        evidence["ignored_signals"] = ignored_signals

        conclusive = len(ignored_signals) >= 2 or (strict and ignored_signals)
        if conclusive:
            evidence["effort_verified"] = "no"
            self._probe_result = evidence
            raise ProbeFailed(
                f"{self.base_url} appears to be ignoring reasoning_effort "
                f"(signals: {', '.join(ignored_signals)}).\n"
                "  Running anyway would produce a whole run at the server's default "
                "effort while the manifest claimed 'high' -- the single most likely "
                "cause of a reproduction miss.\n"
                f"  evidence: {evidence}"
            )
        if ignored_signals:
            evidence["effort_verified"] = "partial"
            logger.warning(
                "reasoning_effort could not be fully confirmed at %s (%s). Proceeding, "
                "but treat the run's effort setting as unverified: %s",
                self.base_url,
                ", ".join(ignored_signals),
                evidence,
            )
        else:
            evidence["effort_verified"] = "yes"
            logger.info(
                "vLLM probe passed: reasoning parser active, reasoning_effort=%r honoured "
                "via %s.",
                self.reasoning_effort,
                self._effort_transport,
            )

        self._probe_result = evidence
        return evidence

    def _probe_request(
        self,
        *,
        max_new_tokens: int,
        reasoning_effort: Any = DEFAULT_REASONING_EFFORT,
    ) -> Generation:
        """One probe call, translating startup failures into actionable messages."""
        try:
            generation = self.generate_structured(
                prompt=self._PROBE_PROMPT,
                max_new_tokens=max_new_tokens,
                reasoning_effort=reasoning_effort,
            )
        except (ProbeFailed, ReasoningEffortUnsupported, MissingDependency):
            # Already precise about what is wrong and what to do about it.
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as ProbeFailed
            code = status_code(exc)
            hint = ""
            if code == 404 or re.search(r"model.*not found|does not exist", str(exc), re.I):
                hint = (
                    f"\n  The server does not know model {self.model!r}. vLLM serves the "
                    "name given by --served-model-name, defaulting to the full "
                    "--model path; pass whichever the server actually advertises "
                    "(GET /v1/models)."
                )
            raise ProbeFailed(
                f"Probe request to {self.base_url} failed: {type(exc).__name__}: {exc}{hint}"
            ) from exc
        if generation is None:
            raise ProbeFailed(
                f"Probe request to {self.base_url} returned no answer after "
                f"{self.max_retries} attempts. Is the vLLM server up and serving "
                f"model {self.model!r}?"
            )
        return generation

    def _probe_garbage_effort(self) -> bool:
        """True if the server rejected a nonsense effort value, i.e. validates it.

        Fallback is disabled here on purpose: this request is *meant* to be
        rejected, and a rejection must not be mistaken for "the transport is
        unsupported" and latch the client into the wrong mode.
        """
        base_params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say OK."}],
            "temperature": self.temperature,
            "max_tokens": 16,
            "stream": False,
        }
        try:
            response = self._create(
                base_params, _GARBAGE_EFFORT, allow_effort_fallback=False
            )
            self._collect(response, stream=False)
        except Exception as exc:  # noqa: BLE001 - the interesting outcome
            if is_bad_request(exc):
                return True
            raise ProbeFailed(
                "Garbage-value effort probe failed for a reason unrelated to "
                f"validation: {type(exc).__name__}: {exc}"
            ) from exc
        return False

    def _probe_effort_size(self, effort: str) -> int:
        """How much thinking the server does at ``effort``.

        Prefers a reported ``reasoning_tokens`` count and falls back to character
        length, so the comparison degrades rather than breaking on servers that do
        not itemise usage.
        """
        generation = self._probe_request(max_new_tokens=512, reasoning_effort=effort)
        usage = generation.usage or {}
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            tokens = details.get("reasoning_tokens")
            if isinstance(tokens, int) and tokens > 0:
                return tokens
        tokens = usage.get("reasoning_tokens")
        if isinstance(tokens, int) and tokens > 0:
            return tokens
        return len(generation.reasoning_text)

    # ---------------------------------------------------------------- manifest

    def manifest(self) -> dict[str, Any]:
        """Provenance for the run manifest, including what effort transport worked."""
        return build_manifest(
            provider="vllm",
            model=self.model,
            base_url=self.base_url,
            reasoning_effort=self.reasoning_effort,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            reasoning_effort_transport=self._effort_transport,
            # None here means no reasoning ever arrived, i.e. raw_text was the final
            # channel only and the scored text is NOT what CORE scored.
            reasoning_field=self.reasoning_field,
            stream=self.stream,
            seed=self.seed,
            include_usage=self.include_usage,
            timeout=self.timeout,
            max_retries=self.max_retries,
            probe=self._probe_result,
            stats=dict(self.stats),
        )
