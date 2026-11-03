"""ReasoningBank baseline on InterPhyre.

A port of ReasoningBank (Ouyang et al., ICLR 2026, arXiv:2509.25140) to the
InterPhyre episode interface, implemented as a sibling of the Episodic RAG /
MemRL baselines so it shares their `run_stream` loop, store, embedder, and
env-reward signal — the only axis that changes is the memory mechanism.

ReasoningBank, unlike Episodic RAG (raw rollouts) and MemRL (value-ranked
summaries), distills each trajectory into at most 3 structured, generalizable
*reasoning items* (title / description / content) via an LLM, learning from both
successes (validated strategies) and failures (preventative lessons). Memory is
append-only: no capacity cap, no merging, no utility/forgetting — which is the
deliberate contrast with HeXA's bounded, hierarchical, evolve/merge bank.

Mapping to the paper:
  memory item            one MemoryRecord; its distilled items live in .content
                         and (structured) in .meta["items"]           (store.py)
  extraction             success -> SUCCESSFUL_SI, failure -> FAILED_SI, T=1.0,
                         <=3 items each                              (extract.py)
  correctness signal     the paper uses an LLM-as-judge (label-free); here it is
                         the InterPhyre environment reward, so the memory
                         mechanism is the only difference from HeXA  (see rb.py)
  retrieval              top-k experiences by embedding cosine, k=1 default
                         (the paper's default; more experiences hurt, its Fig 13)
  consolidation          append-only, no pruning            (rb.add_episode)
"""

from .extract import (
    FAILED_SI,
    SUCCESSFUL_SI,
    make_cli_extractor,
    make_rb_extractor,
    mock_extractor,
    parse_memory_items,
)
from .rb import ReasoningBank, ReasoningBankConfig, Retrieved
from .blocks import format_reasoning_bank_block

__all__ = [
    "ReasoningBank",
    "ReasoningBankConfig",
    "Retrieved",
    "format_reasoning_bank_block",
    "make_rb_extractor",
    "make_cli_extractor",
    "mock_extractor",
    "parse_memory_items",
    "SUCCESSFUL_SI",
    "FAILED_SI",
]
