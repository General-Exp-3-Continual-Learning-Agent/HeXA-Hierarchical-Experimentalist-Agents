"""LLM clients at parity with CORE's inference path.

The contract, stated once because it is the fact a reader of this package most
needs: **the scored string is ``reasoning + final``, concatenated with no
separator, and the 32768-token budget spans both.** CORE's
``generate_answer`` scores ``NvidiaGeneration.raw_text``
(`core-reasoning/scripts/run_core.py:468`), and that field is built by
`nvidia_client.py:143-144`. Chain-of-thought is verifier input, not decoration.

Two providers, one interface:

* :class:`~hexa_core.llm.vllm_client.VLLMClient` -- local vLLM serving
  ``gpt-oss-120b``. What our runs use. Its :meth:`~VLLMClient.probe` is the
  startup gate that refuses to let a run proceed with the reasoning parser off or
  ``reasoning_effort`` being ignored.
The NIM client (a wrapper over CORE's ``NvidiaClient``) has moved to
``hexa_core/_archive/``: it existed to A/B our provider against the paper's
published numbers, and the project no longer makes that comparison.

Importing this package is free of ``openai`` and of CORE: both are imported
lazily, inside the call paths that need them.
"""

from __future__ import annotations

from typing import Any

from hexa_core.llm.base import (
    DEFAULT_REASONING_EFFORT,
    PAPER_MAX_TOKENS,
    PAPER_REASONING_EFFORT,
    PAPER_TEMPERATURE,
    PAPER_TOP_P,
    REQUIRED_MANIFEST_KEYS,
    Generation,
    LLMClient,
    LLMClientError,
    MissingDependency,
    ProbeFailed,
    ReasoningEffortUnsupported,
    build_manifest,
)
from hexa_core.llm.vllm_client import VLLMClient

#: Accepted ``provider`` values for :func:`make_client`, aliases included.
PROVIDERS: dict[str, str] = {
    "vllm": "vllm",
    "local": "vllm",
}

__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "Generation",
    "LLMClient",
    "LLMClientError",
    "MissingDependency",
    "PAPER_MAX_TOKENS",
    "PAPER_REASONING_EFFORT",
    "PAPER_TEMPERATURE",
    "PAPER_TOP_P",
    "PROVIDERS",
    "ProbeFailed",
    "REQUIRED_MANIFEST_KEYS",
    "ReasoningEffortUnsupported",
    "VLLMClient",
    "build_manifest",
    "make_client",
]


def make_client(provider: str, **kwargs: Any) -> LLMClient:
    """Construct a client by provider name.

    Only vLLM remains -- the NIM path was the other half of a provider A/B that the
    project no longer runs. Kept as a function rather than inlined so adding a second
    provider later does not mean rewriting call sites.
    """
    key = PROVIDERS.get(provider.strip().lower())
    if key is None:
        raise ValueError(
            f"unknown provider {provider!r}; expected one of "
            f"{', '.join(sorted(PROVIDERS))}"
        )
    return VLLMClient(**kwargs)
