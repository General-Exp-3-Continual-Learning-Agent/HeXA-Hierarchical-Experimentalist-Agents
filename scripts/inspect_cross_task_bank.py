#!/usr/bin/env python
"""Read a synthesised target bank before any eval compute is spent.

The plan calls manual inspection the cheapest place to catch a broken transfer,
and Design C makes it more than a nicety: with zero target trajectories there is
no transcript to check a skill against, so nothing downstream will catch a
plausible-sounding skill built on a premise the target does not have. This script
does the mechanical part of that reading and leaves the judgement to a person.

Three checks, in increasing order of how much they need a human:

1. **Citations** -- do the cited source ids exist? Mechanical, and a hard gate.
2. **Source-entity leakage** -- did a mine, a flag, a letter or a G/Y/X mark
   survive into a Maze or Mastermind skill? Mechanical, and a hard gate.
3. **Inverted premises** -- does a skill assume something true of the source and
   false of the target? Heuristic: it flags for reading, it does not decide.

Exit code is non-zero if a hard gate fails.

    python scripts/inspect_cross_task_bank.py --dir results/design_c/maze_from_minesweeper/run0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Nouns that belong to a source task and to nothing in the target. Split by how
#: a hit should be read, because the two behave differently:
#:
#: HARD -- the word has no innocent sense in a Maze or Mastermind skill, so a hit
#: means the re-grounding step did not happen. Blocks the bank.
#: SOFT -- the word is also ordinary English ("the map revealed a wall", "a grid
#: of cells"), so a hit is worth a human read but is not by itself evidence of
#: leakage. Reported, never blocks.
#:
#: Matching is word-boundary anchored with an inflection suffix, so "flagging"
#: and "mines" hit but "examine" and "determine" do not -- an unanchored
#: substring test flagged both of those on the first Phase 3 pass.
SOURCE_ENTITIES_HARD = {
    "minesweeper": ["mine", "flag", "cascade", "adjacent count"],
    "wordle": ["vowel", "consonant", "green", "yellow", "g/y/x", "letter"],
}
SOURCE_ENTITIES_SOFT = {
    "minesweeper": ["board", "grid", "reveal", "probe"],
    "wordle": ["word", "spell"],
}


def _entity_hits(text: str, vocab: list[str]) -> list[str]:
    """Word-boundary match with an inflection suffix ("flag" -> "flagging")."""
    hits = []
    for w in vocab:
        pattern = r"\b" + re.escape(w).replace(r"\ ", r"\s+") + r"\w*\b"
        if re.search(pattern, text, re.I):
            hits.append(w)
    return hits

#: Premises the source rewards and the target does not have. Each entry is
#: (label, why it inverts, regex). These are for reading, not for rejecting: a
#: skill can mention risk and still be sound.
INVERTED_PREMISES = {
    ("minesweeper", "maze"): [
        (
            "irreversible loss",
            "Minesweeper ends the episode on one wrong probe, so 'act only where "
            "safety is proven' is correct there. Maze has NO losing move -- a wall "
            "costs a turn and nothing else -- so the same skill makes an agent "
            "stall instead of explore, which is the failure the Design B maze arm "
            "already showed (a distilled bank cost 15.2 points by replacing a "
            "working implicit policy with a cautious verbal one).",
            re.compile(r"\b(safe|safety|risk|risky|danger|dangerous|avoid guess|"
                       r"only.{0,20}certain|prove|proven|guarantee)", re.I),
        ),
        (
            "globally visible state",
            "Minesweeper redraws the whole board under every observation and "
            "reveals are permanent. Maze shows only the four adjacent cells and "
            "the agent must accumulate its own map, so a skill that says to read "
            "or scan the state assumes an observation the target never gives.",
            re.compile(r"\b(board|grid|whole|entire|全|overview|scan|all cells|"
                       r"visible state|map is shown)", re.I),
        ),
        (
            "numeric constraint system",
            "Minesweeper's numbers make a solvable constraint system. Maze returns "
            "wall/path/goal and no counts at all, so a skill about counting, "
            "arithmetic on neighbours, or constraint propagation has nothing to "
            "operate on.",
            re.compile(r"\b(count|number of|numeric|arithmetic|sum|constraint "
                       r"propagat|deduc\w+ from the numbers)", re.I),
        ),
    ],
    ("wordle", "mastermind"): [
        (
            "position-identified feedback",
            "Wordle marks each POSITION and each mark names the letter it belongs "
            "to, so 'the feedback tells you which position was right' is true "
            "there. Mastermind returns COUNTS only -- black and white pegs -- and "
            "never says which position or which digit. A skill that reads "
            "positional identity out of the feedback inverts the target's central "
            "difficulty.",
            re.compile(r"\b(which position|that position|per[- ]position|"
                       r"position[- ]by[- ]position|identif\w+ the position|"
                       r"mark(ed)? position)", re.I),
        ),
        (
            "repeat rejection",
            "Wordle rejects a repeated guess outright and wastes the turn. "
            "Mastermind accepts one, so a skill built on the environment "
            "preventing repeats is describing a guardrail the target does not "
            "have.",
            re.compile(r"\b(repeat\w*|already guessed|previously guessed)", re.I),
        ),
    ],
}

_TEXT_FIELDS = ("title", "principle", "when_to_apply", "example",
                "description", "why_it_happens", "how_to_avoid")


def _entries(bank: dict) -> list[tuple[str, dict]]:
    out = []
    for skills in bank.get("level_skills", {}).values():
        out += [("skill", s) for s in skills]
    out += [("skill", s) for s in bank.get("general_skills", [])]
    for mistakes in bank.get("level_mistakes", {}).values():
        out += [("mistake", m) for m in mistakes]
    out += [("mistake", m) for m in bank.get("common_mistakes", [])]
    return out


def _text(entry: dict) -> str:
    return " ".join(str(entry.get(f, "")) for f in _TEXT_FIELDS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--source", action="append", default=None,
                    help="repeatable, matching the synthesis. Entity and inverted-premise\n                         rules are applied for every source named.")
    args = ap.parse_args()

    provenance = {}
    prov_path = args.dir / "provenance.json"
    if prov_path.exists():
        provenance = json.loads(prov_path.read_text())
    target = args.target or provenance.get("target")
    sources = args.source or provenance.get("sources") or (
        [provenance["source"]] if provenance.get("source") else [])
    if isinstance(sources, str):
        sources = [sources]
    if not target:
        raise SystemExit("pass --target (or run this on a directory with provenance.json)")

    bank_path = args.dir / f"skill_bank_xl_{target}.json"
    audit_path = args.dir / f"skill_bank_xl_{target}_audit.json"
    if not bank_path.exists():
        raise SystemExit(f"{bank_path} not found")
    bank = json.loads(bank_path.read_text())
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    entries = _entries(bank)
    print(f"\n=== {target} <- {'+'.join(sources) or '?'}  ({len(entries)} entries) ===")
    print(f"source banks: {audit.get('source_bank_skill_counts', {})}")
    print(f"prompt variant: {audit.get('prompt_variant', '?')}")
    if audit.get("synthesis_reasoning"):
        print(f"\nteacher's own account:\n  {audit['synthesis_reasoning']}")

    failures = []

    # 1. citations
    check = audit.get("citation_check")
    print("\n-- 1. citations --")
    if check is None:
        print("  no citation_check in the audit (older synthesis run?)")
    else:
        print(f"  {check['n_citations']} cited, {check['n_invalid']} do not resolve, "
              f"{len(check['uncited_entries'])} entries cite nothing")
        for bad in check["invalid_citations"]:
            print(f"    FABRICATED {bad['entry']}: {bad['cite']} -- {bad['why']}")
        if check["n_invalid"] or check["uncited_entries"]:
            failures.append("citations do not resolve")

    # 2. entity leakage
    print("\n-- 2. source-entity leakage --")
    hard_vocab = [w for src in sources for w in SOURCE_ENTITIES_HARD.get(src, [])]
    soft_vocab = [w for src in sources for w in SOURCE_ENTITIES_SOFT.get(src, [])]
    leaked, soft = [], []
    for kind, entry in entries:
        text = _text(entry)
        title = entry.get("title") or entry.get("description")
        hard_hits = _entity_hits(text, hard_vocab)
        if hard_hits:
            leaked.append((title, hard_hits))
        soft_hits = _entity_hits(text, soft_vocab)
        if soft_hits:
            soft.append((title, soft_hits))
    for title, hits in leaked:
        print(f"    LEAKED {title!r}: {hits}")
    for title, hits in soft:
        print(f"    soft {title!r}: {hits}  (ordinary English -- read, do not auto-reject)")
    if leaked:
        failures.append("source entities survived into the target bank")
    elif not soft:
        print("  none")

    # 3. inverted premises
    print("\n-- 3. inverted premises (read these, do not auto-reject) --")
    # With several sources every source's premises can leak, so every registered
    # pair is checked -- a Minesweeper-flavoured caution skill is just as wrong in
    # a Maze bank when Wordle was also in the room.
    rules = [r for src in sources for r in INVERTED_PREMISES.get((src, target), [])]
    if not rules:
        print(f"  no rules registered for {'+'.join(sources)} -> {target}")
    flagged = 0
    for label, why, pattern in rules:
        hits = [(entry.get("title") or entry.get("description"), pattern.findall(_text(entry)))
                for _kind, entry in entries if pattern.search(_text(entry))]
        if not hits:
            continue
        flagged += len(hits)
        print(f"\n  [{label}] {why}")
        for title, matched in hits:
            words = sorted({m if isinstance(m, str) else m[0] for m in matched})
            print(f"    - {title!r}  (on: {', '.join(words[:6])})")
    if rules and not flagged:
        print("  none matched")

    # 4. the bank itself, for reading
    print("\n-- 4. the bank --")
    for kind, entry in entries:
        title = entry.get("title") or entry.get("description")
        conf = entry.get("confidence")
        print(f"\n  [{kind}] {title}" + (f"  (conf {conf})" if conf is not None else ""))
        for field in ("principle", "when_to_apply", "how_to_avoid"):
            if entry.get(field):
                print(f"      {field}: {entry[field]}")
    for record in audit.get("skills", []):
        if record.get("transfer_rationale"):
            print(f"\n  rationale [{record['skill_id']}]: {record['transfer_rationale']}")

    print("\n=== verdict ===")
    if failures:
        for f in failures:
            print(f"  BLOCKED: {f}")
        print("  Do not evaluate this bank until these are resolved.")
        return 1
    print("  mechanical checks pass. The inverted-premise flags above still need a "
          "human read before Phase 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
