"""Token counting for the added-task-external-context metric (CORE Figure 3).

The headline number this supports is *added tokens per eval item*: the size of
the guidelines block HExA injects, measured against the byte-identical CORE base
prompt. CORE reports 0.92k, GEPA 1.29k, Episodic RAG 33.6k, MemRL 32.7k.

``tiktoken`` is used when available. The fallback is a word/char heuristic — it
is fine for enforcing an injection budget but should not be quoted in a paper, so
:func:`counter_name` is logged alongside every measurement.

**Which path is live matters and is not obvious.** ``requirements.txt`` pins
``tiktoken>=0.7.0``, but it is installed only in the ``vllm-gptoss`` serving env —
not in the system interpreter that runs the tests, nor in ``phyreagent``. So on
the machine this was written on the heuristic **is** the live path and every
added-token figure here is an estimate. Gate any published number on
:func:`is_exact` rather than on the presence of the pin.
"""

from __future__ import annotations

from functools import lru_cache

# gpt-oss models use an o200k-family tokenizer; o200k_base is the closest
# encoding tiktoken ships and is within a couple of percent for English prose.
_ENCODING_NAME = "o200k_base"

_HEURISTIC_NAME = "heuristic:words+chars(approx,not-citable)"

#: One-sentence caveat for log/manifest writers to embed verbatim, so the wording
#: is identical everywhere instead of being reinvented per call site.
TOKENIZER_NOTE = (
    "Added-token counts are only comparable to CORE's reported figures when a real "
    "tokenizer is active; see the `token_counter` field for which method produced them."
)


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding(_ENCODING_NAME)
    except Exception:
        return None


def counter_name() -> str:
    """Identifier for whichever counting method is active. Log this."""
    return f"tiktoken:{_ENCODING_NAME}" if _encoder() is not None else _HEURISTIC_NAME


def is_exact() -> bool:
    """True iff counts come from a real tokenizer.

    The predicate to gate a published added-token figure on. Note that ``tiktoken``
    being *importable* is not enough: ``get_encoding("o200k_base")`` fetches the BPE
    file over the network on first use, so an env with tiktoken but no egress also
    lands on the heuristic. This reports the outcome, not the import.
    """
    return _encoder() is not None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Heuristic: max of a word-based and a character-based estimate. Structured
    # text (JSON schemas, move lists) is char-dense; prose is word-dense.
    words = len(text.split())
    return max(int(words * 1.35), len(text) // 4)


def added_tokens_raw(base_prompt: str, augmented_prompt: str) -> int:
    """Unclamped delta, so a negative one is visible instead of only mentioned.

    :func:`added_tokens` reports ``0`` both for "nothing was injected" and for "the
    augmentation somehow shrank the prompt". Those are very different bugs; log this
    alongside if the distinction could matter.
    """
    return count_tokens(augmented_prompt) - count_tokens(base_prompt)


def added_tokens(base_prompt: str, augmented_prompt: str) -> int:
    """Tokens the injected context added, relative to CORE's base prompt.

    Clamped at zero: an augmentation should never shrink the prompt, and a
    negative number would silently flatter the Pareto curve. The clamp is a silent
    cap -- use :func:`added_tokens_raw` when you need to see it fire.
    """
    return max(0, added_tokens_raw(base_prompt, augmented_prompt))
