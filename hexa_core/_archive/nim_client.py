"""Reference provider path: CORE's own ``NvidiaClient``, wrapped not copied.

The published CORE numbers were produced against NVIDIA NIM with this exact
client. Keeping it callable through the same interface as
:class:`hexa_core.llm.vllm_client.VLLMClient` is what makes the A/B possible:
same tasks, same prompts, same verifier, one variable changed.

So **no CORE code is reproduced here**. ``raw_text`` is taken verbatim from
CORE's generation object; :meth:`hexa_core.llm.base.Generation.from_core`
re-checks the concatenation invariant on the way past, which is a tripwire on
CORE drift rather than a transformation of the reference output.

The sharpest cross-repo hazard lives in this file. CORE's "no preference"
sentinel is a private ``object()`` compared by identity
(`nvidia_client.py:8, 103`). Ours is a *different* object, so handing it to CORE
would fall through to ``elif reasoning_effort is not None`` and serialise a bare
``object()`` into the request body. :meth:`NimClient.generate_structured`
therefore never forwards our sentinel -- see the comment there.

Imports stay lazy: ``model.nvidia_client`` does ``from openai import OpenAI`` at
module scope, and ``openai`` may be absent.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from hexa_core.core_bridge import core_git_revision, install_core_on_path
from hexa_core.llm.base import (
    DEFAULT_REASONING_EFFORT,
    PAPER_MAX_TOKENS,
    PAPER_REASONING_EFFORT,
    PAPER_TEMPERATURE,
    PAPER_TOP_P,
    Generation,
    MissingDependency,
    ProbeFailed,
    build_manifest,
)

CORE_CLIENT_PATH = "model.nvidia_client.NvidiaClient"

#: ``run_core.py:3651`` default for ``--nvidia-model``.
DEFAULT_NIM_MODEL = "openai/gpt-oss-120b"

__all__ = ["NimClient", "NvidiaClient", "NvidiaGeneration", "CORE_CLIENT_PATH", "DEFAULT_NIM_MODEL"]


@lru_cache(maxsize=1)
def _nvidia_module() -> Any:
    """Import CORE's ``model.nvidia_client``, with a diagnosable failure."""
    install_core_on_path()
    try:
        import model.nvidia_client as nvidia_client
    except ImportError as exc:
        if "openai" in str(exc):
            raise MissingDependency(
                "CORE's model.nvidia_client imports openai at module scope: "
                "pip install openai to use the NVIDIA NIM reference path "
                "(the vLLM client does not need it at import time)."
            ) from exc
        raise
    return nvidia_client


def _nvidia_client_cls() -> Any:
    return _nvidia_module().NvidiaClient


def __getattr__(name: str) -> Any:
    """Re-export CORE's classes lazily (:pep:`562`).

    ``import hexa_core.llm.nim_client`` stays clean with CORE or ``openai``
    missing, while ``from hexa_core.llm.nim_client import NvidiaClient`` works
    wherever they are present.
    """
    if name in ("NvidiaClient", "NvidiaGeneration"):
        return getattr(_nvidia_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


class NimClient:
    """CORE's ``NvidiaClient`` behind this package's client interface.

    Retries are deliberately *not* implemented: on this path CORE's own wrappers
    (`llm_utils.py:287-340`) own the retry policy, and adding a second layer would
    change the reference path's behaviour, which is the one thing it exists not to
    do.
    """

    def __init__(
        self,
        model: str = DEFAULT_NIM_MODEL,
        api_key: str | None = None,
        *,
        temperature: float = PAPER_TEMPERATURE,
        top_p: float | None = PAPER_TOP_P,
        max_tokens: int = PAPER_MAX_TOKENS,
        reasoning_effort: str | None = PAPER_REASONING_EFFORT,
        stream: bool = True,
        seed: int | None = None,
    ) -> None:
        key = api_key or os.getenv("NVIDIA_API_KEY")
        if not key:
            # Same wording as run_core.py:3650, so the fix is the familiar one.
            raise ValueError(
                "NVIDIA_API_KEY environment variable is required for NVIDIA models"
            )
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.seed = seed
        self._client = _nvidia_client_cls()(model=model, api_key=key)
        self._probe_result: dict[str, Any] | None = None

    @property
    def core_client(self) -> Any:
        """The live CORE client, for code that wants the reference object itself."""
        return self._client

    @property
    def base_url(self) -> str | None:
        # Read, never hardcoded: CORE owns this value.
        return getattr(self._client, "base_url", None)

    def _effort_kwargs(self, reasoning_effort: Any) -> dict[str, Any]:
        """Translate our sentinel into something CORE can safely receive.

        Never forward ``DEFAULT_REASONING_EFFORT``: CORE identity-compares against
        its *own* private sentinel, so ours would be serialised into the JSON body
        as a bare object. When the caller expressed no preference and our
        configured effort is CORE's own default, omit the kwarg entirely and let
        CORE decide -- that keeps this path byte-identical to a stock CORE run even
        if CORE changes its default.
        """
        if reasoning_effort is DEFAULT_REASONING_EFFORT:
            if self.reasoning_effort == PAPER_REASONING_EFFORT:
                return {}
            return {"reasoning_effort": self.reasoning_effort}
        return {"reasoning_effort": reasoning_effort}

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
        """Forward to CORE and adapt the result. ``None`` arguments use our config.

        In practice this never returns ``None``: CORE's client raises on transport
        failure rather than returning. The annotation matches the protocol so
        callers keep the same ``is None`` handling on both providers.
        """
        generation = self._client.generate_structured(
            prompt=prompt,
            max_new_tokens=self.max_tokens if max_new_tokens is None else max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
            stream=self.stream if stream is None else stream,
            seed=self.seed if seed is None else seed,
            top_p=self.top_p if top_p is None else top_p,
            system_prompt=system_prompt,
            messages=messages,
            **self._effort_kwargs(reasoning_effort),
        )
        if generation is None:
            return None
        return Generation.from_core(generation, verify=True)

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

    def probe(self) -> dict[str, Any]:
        """Check the reasoning parser only.

        There is no effort-acceptance A/B here, unlike the vLLM probe: NIM's
        handling of ``reasoning_effort`` is *definitionally* what CORE's published
        numbers were produced with, so probing it would be measuring the reference
        against itself. This method says what it verified and no more.
        """
        generation = self.generate_structured(prompt="Work out 17 * 24.", max_new_tokens=96)
        if generation is None or not generation.reasoning_text:
            raise ProbeFailed(
                f"No reasoning_content from {self.base_url} for model {self.model!r}; "
                "raw_text = reasoning + final would silently lose the chain-of-thought."
            )
        evidence = {
            "base_url": self.base_url,
            "model": self.model,
            "reasoning_parser": True,
            "reasoning_chars": len(generation.reasoning_text),
            "final_chars": len(generation.final_text),
            "finish_reason": generation.finish_reason,
            "effort_transport": "top_level",
            "effort_verified": "assumed(reference provider)",
        }
        self._probe_result = evidence
        return evidence

    def manifest(self) -> dict[str, Any]:
        return build_manifest(
            provider="nvidia-nim",
            model=self.model,
            base_url=self.base_url,
            reasoning_effort=self.reasoning_effort,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
            seed=self.seed,
            core_client=CORE_CLIENT_PATH,
            core_git_revision=core_git_revision(),
            probe=self._probe_result,
        )

    def shutdown(self) -> None:
        """Forwarded if CORE's client ever grows one (`run_core.py:3652-3653`)."""
        shutdown = getattr(self._client, "shutdown", None)
        if callable(shutdown):
            shutdown()
