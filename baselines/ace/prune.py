"""ACE's pruning trigger (`--token-cap`), which the reference code omits.

Paper, Appendix (Table 21): "Pruning Trigger (Maximum Context Length): this
threshold determines when ACE merges and prunes older or low-utility entries to
prevent unbounded growth ... performance is stable from 10K to 100K tokens".
The released code (github.com/ace-agent/ace @ 82709de) only prints
`playbook_token_budget` into the Curator prompt; nothing ever prunes.

The paper does not say how entries are ranked, so this is our choice, kept to
the two criteria it names and to its "lightweight, non-LLM" merge logic:

  * merge  -- the embedding dedup (grow-and-refine) runs first, as always;
  * prune  -- while the playbook is over the cap, drop the bullet with the
              lowest utility = helpful - harmful (the Reflector's own counters),
              breaking ties by age (lowest numeric ID = oldest) first.

A bullet nobody ever cited has utility 0, so old untagged bullets go before
newer ones, and a net-harmful bullet goes before either.
"""
from __future__ import annotations

import re

from baselines.ace import playbook as pb
from baselines.memory.blocks import count_tokens


def _age(bid: str) -> int:
    m = re.search(r"-(\d+)$", bid)
    return int(m.group(1)) if m else 0


def prune_to_cap(playbook_text: str, cap: int) -> tuple[str, list[dict]]:
    """Drop lowest-utility, then oldest, bullets until count_tokens <= cap.

    Returns (playbook, removed) where removed lists the dropped bullets."""
    if count_tokens(playbook_text) is None:
        raise SystemExit("--token-cap needs tiktoken (set TIKTOKEN_CACHE_DIR)")
    if count_tokens(playbook_text) <= cap:
        return playbook_text, []
    lines = playbook_text.split("\n")
    bullets = [(i, pb.parse_playbook_line(l)) for i, l in enumerate(lines)]
    order = sorted(
        ((i, p) for i, p in bullets if p),
        key=lambda ip: (ip[1]["helpful"] - ip[1]["harmful"], _age(ip[1]["id"])),
    )
    dropped: set[int] = set()
    removed: list[dict] = []
    for i, p in order:
        dropped.add(i)
        removed.append({"id": p["id"], "helpful": p["helpful"], "harmful": p["harmful"]})
        text = "\n".join(l for j, l in enumerate(lines) if j not in dropped)
        if count_tokens(text) <= cap:
            return text, removed
    return "\n".join(l for j, l in enumerate(lines) if j not in dropped), removed
