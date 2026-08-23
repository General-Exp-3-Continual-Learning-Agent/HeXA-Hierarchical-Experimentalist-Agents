"""Bank audit: is the bank learning strategy, or memorising routes?

The specific degenerate outcome on this env is a "skill" that is really a
remembered path. It would show up as transfer on the 19.1% of eval instances
sharing a layout with the training pool and as nothing on the rest (RECON §10) --
so it can masquerade as a positive result. Two of the three checks in
DESIGN_PHASE3_MAZE §9 are automated here:

1. :func:`route_flags` -- textual detection of move sequences and coordinate
   literals in skill text.
2. :func:`leakage_delta` -- the empirical version, and the one that matters when a
   paraphrase slips past check 1.

The third check (does the bank find the env's genuine map-independent
regularities?) is deliberately **not** automated. Scoring skills against a rubric
we wrote is how a bank audit becomes circular. :data:`STRUCTURAL_REGULARITIES` is
a written list to read the bank against by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

_DIRECTIONS = ("up", "down", "left", "right")

#: Three or more direction words in a row, separated only by punctuation or
#: connectives -- "go left, left, then up". Two in a row is common in legitimate
#: guidance ("prefer up or left when both are open"), so the threshold is three.
_MOVE_RUN_RE = re.compile(
    r"\b(?:" + "|".join(_DIRECTIONS) + r")\b"
    r"(?:[\s,;/]+(?:then|and|followed by)?[\s,;/]*"
    r"\b(?:" + "|".join(_DIRECTIONS) + r")\b){2,}",
    re.IGNORECASE,
)

#: A coordinate literal: "(5, 2)", "(3,3)". Two or more in one skill is the flag;
#: a single one can appear in legitimately generic phrasing about notation.
_COORD_RE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")

#: The genuine, map-independent structure of this generator. A bank that finds
#: these is working; a bank full of routes is not. Read by hand -- see the module
#: docstring for why this is not a metric.
STRUCTURAL_REGULARITIES: tuple[str, ...] = (
    "The goal is always the reachable cell farthest from the start, so expanding "
    "away from the start toward unexplored frontier is systematically right "
    "(envs/maze_env_adapter.py:_find_farthest_position).",
    "The shortest path is always 7-8 moves, so past roughly 8 moves the agent has "
    "detoured and should re-plan rather than push on (the config's "
    "shortest_path_min/max_length filter).",
    "Grids are 6x6, 7x6, 7x7 or 6x7 with solid wall borders, so interior "
    "coordinates are bounded and an observed (x, y) locates the agent globally.",
    "The turn budget is 9 against a 7-8 move path, so at most 1-2 moves can be "
    "wasted: wall-bumping is nearly fatal.",
)


@dataclass
class SkillFlag:
    """One suspect skill."""

    skill_id: str
    title: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"skill_id": self.skill_id, "title": self.title, "reasons": self.reasons}


def _skill_text(skill: Any) -> str:
    parts = [
        getattr(skill, field_name, "") or ""
        for field_name in ("title", "principle", "when_to_apply", "example")
    ]
    return " ".join(parts)


def flag_skill(skill: Any) -> Optional[SkillFlag]:
    """Flag *skill* if its text looks like a remembered route."""
    text = _skill_text(skill)
    reasons: list[str] = []

    match = _MOVE_RUN_RE.search(text)
    if match:
        reasons.append(f"move sequence: {match.group(0)!r}")

    coords = _COORD_RE.findall(text)
    if len(coords) >= 2:
        reasons.append(f"coordinate literals: {coords[:4]}")

    if not reasons:
        return None
    return SkillFlag(
        skill_id=getattr(skill, "skill_id", "?"),
        title=getattr(skill, "title", ""),
        reasons=reasons,
    )


def route_flags(bank: Any, level_name: str = "maze") -> list[SkillFlag]:
    """Every suspect skill in the bank, level-specific and general alike."""
    skills = list(getattr(bank, "general_skills", []) or [])
    skills += list((getattr(bank, "level_skills", {}) or {}).get(level_name, []))
    return [flag for flag in (flag_skill(s) for s in skills) if flag is not None]


def leakage_delta(report: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Read the leaked-vs-clean episode-1 gap out of a :mod:`hexa_orbit.report` dict.

    A large positive gap means the headline is carried by instances whose layout
    the bank has seen -- recall, not strategy. Returns ``None`` when one side of
    the split is empty, since a gap is not defined then.
    """
    leaked, clean = report.get("leaked", {}), report.get("clean", {})
    if not leaked.get("n") or not clean.get("n"):
        return None
    leaked_ep1 = leaked["episode_success_rate"]["episode_1"]
    clean_ep1 = clean["episode_success_rate"]["episode_1"]
    return {
        "leaked_episode_1": leaked_ep1,
        "clean_episode_1": clean_ep1,
        "gap": leaked_ep1 - clean_ep1,
        "n_leaked": leaked["n"],
        "n_clean": clean["n"],
    }


def audit_bank(
    bank: Any,
    *,
    level_name: str = "maze",
    report: Optional[Mapping[str, Any]] = None,
    round_index: Optional[int] = None,
    known_ids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Full audit record for one round. Cheap, pure, no model calls."""
    flags = route_flags(bank, level_name)
    skills = list((getattr(bank, "level_skills", {}) or {}).get(level_name, []))
    mistakes = list((getattr(bank, "level_mistakes", {}) or {}).get(level_name, []))
    return {
        "round": round_index,
        "n_skills": len(skills),
        "n_mistakes": len(mistakes),
        "n_flagged": len(flags),
        "flagged": [f.to_dict() for f in flags],
        "flagged_fraction": len(flags) / len(skills) if skills else 0.0,
        "leakage": leakage_delta(report) if report else None,
        "source_seeds": validate_source_seeds(skills, known_ids)
        if known_ids is not None
        else None,
        "manual_check": {
            "question": "Does the bank express any of these map-independent regularities?",
            "regularities": list(STRUCTURAL_REGULARITIES),
            "note": (
                "Answer by reading the bank. Deliberately not scored -- grading skills "
                "against a rubric we wrote would make the audit circular."
            ),
        },
    }


def format_audit(audit: Mapping[str, Any]) -> str:
    """Terminal digest, flags first."""
    round_index = audit.get("round")
    where = f" (round {round_index})" if round_index is not None else ""
    lines = [
        f"bank audit{where}: "
        f"{audit['n_skills']} skills, {audit['n_mistakes']} mistakes, "
        f"{audit['n_flagged']} flagged ({audit['flagged_fraction']:.0%})"
    ]
    for flag in audit["flagged"]:
        lines.append(f"  [{flag['skill_id']}] {flag['title']}")
        for reason in flag["reasons"]:
            lines.append(f"      {reason}")
    leakage = audit.get("leakage")
    if leakage:
        lines.append(
            f"  episode-1 leaked {leakage['leaked_episode_1']:.1%} "
            f"vs clean {leakage['clean_episode_1']:.1%} "
            f"-> gap {leakage['gap']:+.1%}"
        )
    return "\n".join(lines)


# ── source_seeds provenance ──────────────────────────────────────────────
#
# A skill's confidence is the mean reward of the instances its ``source_seeds``
# cite (``distill._compute_initial_confidence``). The lookup is a plain dict
# match, and on a total miss it does **not** error -- it silently substitutes the
# mean reward over *every* trajectory in the round. So a teacher that abbreviates
# or invents an id gets the round average presented as evidence-based confidence,
# and since confidence is also the retriever's ranking key, that directly changes
# what gets injected.
#
# The risk is concrete here: instance ids are 16-hex strings
# (``132dc55db9c9f536``), which is exactly the shape a model shortens. The prompt
# says copy them verbatim, but a prompt is not a guarantee.


@dataclass
class SeedCitation:
    """One ``source_seeds`` entry and what became of it."""

    cited: str
    status: str  # "exact" | "repaired" | "unknown"
    resolved: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"cited": self.cited, "status": self.status, "resolved": self.resolved}


def resolve_citation(cited: str, known_ids: Iterable[str]) -> Optional[str]:
    """Resolve an abbreviated or extended instance id, or ``None``.

    Handles the two ways a model plausibly gets an id wrong while still meaning a
    real instance: truncating it (``132dc55d`` for ``132dc55db9c9f536``) or
    quoting the full underlying hash, of which the id is a prefix. A match is
    accepted only when it is **unique** -- an ambiguous prefix is worse than no
    match, because guessing wrong attributes a skill to the wrong evidence.
    """
    cited = (cited or "").strip()
    if not cited:
        return None
    known = list(known_ids)
    if cited in known:
        return cited
    matches = {k for k in known if k.startswith(cited) or cited.startswith(k)}
    return matches.pop() if len(matches) == 1 else None


def classify_citations(
    source_seeds: Iterable[Any], known_ids: Iterable[str]
) -> list[SeedCitation]:
    """Label every citation exact / repaired / unknown."""
    known = list(known_ids)
    out: list[SeedCitation] = []
    for cited in source_seeds or ():
        text = str(cited)
        if text in known:
            out.append(SeedCitation(text, "exact", text))
            continue
        resolved = resolve_citation(text, known)
        out.append(
            SeedCitation(text, "repaired" if resolved else "unknown", resolved)
        )
    return out


def validate_source_seeds(
    skills: Sequence[Any], known_ids: Iterable[str]
) -> dict[str, Any]:
    """Provenance report for a round's emitted skills.

    ``fell_back`` is the number that matters: those skills had *no* citation
    resolve, so their confidence is the round's mean reward wearing the costume of
    evidence. A nonzero count means the confidences from that round are not
    interpretable, whatever the rest of the report says.
    """
    known = list(known_ids)
    per_skill: list[dict[str, Any]] = []
    totals = {"exact": 0, "repaired": 0, "unknown": 0}

    for skill in skills:
        citations = classify_citations(getattr(skill, "source_seeds", []) or [], known)
        for citation in citations:
            totals[citation.status] += 1
        resolved_any = any(c.status != "unknown" for c in citations)
        per_skill.append(
            {
                "skill_id": getattr(skill, "skill_id", "?"),
                "title": getattr(skill, "title", ""),
                "citations": [c.to_dict() for c in citations],
                "fell_back": bool(citations) and not resolved_any,
                "uncited": not citations,
            }
        )

    n_citations = sum(totals.values())
    return {
        "n_skills": len(per_skill),
        "n_citations": n_citations,
        **{f"n_{k}": v for k, v in totals.items()},
        "miss_rate": (totals["unknown"] + totals["repaired"]) / n_citations
        if n_citations
        else 0.0,
        "fell_back": [s for s in per_skill if s["fell_back"]],
        "uncited": [s for s in per_skill if s["uncited"]],
        "per_skill": per_skill,
    }


def format_source_seed_report(report: Mapping[str, Any]) -> str:
    """Terminal digest. Silent on a clean round, loud otherwise."""
    if report["n_citations"] and not (report["n_repaired"] or report["n_unknown"]):
        return f"source_seeds: {report['n_citations']} citations, all exact"

    lines = [
        f"source_seeds: {report['n_citations']} citations -- "
        f"{report['n_exact']} exact, {report['n_repaired']} repaired, "
        f"{report['n_unknown']} unknown ({report['miss_rate']:.0%} miss rate)"
    ]
    for skill in report["fell_back"]:
        lines.append(
            f"  [{skill['skill_id']}] {skill['title']}: NO citation resolved -- its "
            f"confidence is the round mean, not evidence"
        )
    for skill in report["uncited"]:
        lines.append(f"  [{skill['skill_id']}] {skill['title']}: cited no instances")
    return "\n".join(lines)
