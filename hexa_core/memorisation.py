"""Detect training answers memorised into skill text (`PHASE3_DESIGN.md` §5b).

The design names this as the finding most likely to embarrass the project, and asks
for "an assertion in the pilot that no admitted skill's text contains a training
problem's answer string". This is that assertion.

It is not hypothetical. The first Stage 3a smoke run that completed a full round
produced this skill, distilled from problem ``...-train-n5-s201``::

    title:   Self-simulation verification
    example: Iterate through the list [[1,2,1],[5,0,2],...] and update the state

while s201's reference answer begins::

    moves = [[1, 2, 1], [5, 0, 2], [1, 1, 0], ...

Two verbatim moves. The prompt-side mitigation -- asking the teacher for "an
illustration of the method, not of one problem's answer" -- did not prevent it, which
is why a detector is needed rather than a better instruction.

**Why it matters even though it cannot leak to eval.** The eval split is a disjoint
problem set (6 disks vs the training split's 5), so a memorised training answer cannot
inflate the headline number. What it corrupts is *training-time* utility: a skill that
carries an answer will appear to work for reasons unrelated to the method it claims to
teach. And a reviewer who reads one bank will see it.

The check is deliberately syntactic and task-shaped rather than clever. It looks for
runs of numeric tuples shared between a skill's text and a problem's reference answer,
which is what Tower of Hanoi answers are. `min_run` is the number of consecutive
tuples that must match: 1 is far too noisy (``[1,0,2]`` appears by chance), 2 is
already strong evidence, since a specific pair of consecutive moves is unlikely to be
coincidental.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

#: A bracketed tuple of 2+ integers: ``[1, 2, 1]``, ``[5,0,2]``.
_TUPLE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]")

#: How many consecutive tuples must coincide before it is called memorisation.
DEFAULT_MIN_RUN = 2

#: Quoted strings inside a reference answer, e.g. Matchstick's
#: ``["XVI = II + XIV", "XVII = I + XVI"]``.
_QUOTED = re.compile(r'"([^"]{4,120})"')

#: An answer string short enough to appear by chance is not evidence. Matchstick
#: answers run ~13 chars ("XVI = II + XIV"); 8 excludes fragments without excluding
#: real answers.
_MIN_ANSWER_CHARS = 8

#: How many cells **of one grid row** must appear in a skill before it is memorisation.
#:
#: ZebraLogic forced this. Its reference answer is a grid whose cells are attribute
#: values -- ``"swimming"``, ``"green"``, ``"grilled cheese"`` -- and **9 of 9** of the
#: :func:`_answer_strings` candidates from the first training problem appear verbatim in
#: that problem's own prompt, because the puzzle enumerates every attribute's possible
#: values up front. A skill saying "check the FavoriteSport column" would have been
#: reported as a leak. What is secret is the *assignment*: the pairing of a person with
#: their sport, colour and food cannot be recited from the prompt. 3 is the string
#: analogue of :data:`DEFAULT_MIN_RUN`, one higher because 2 attribute values co-occur
#: plausibly in honest prose ("if the swimmer likes pizza").
DEFAULT_MIN_STRING_RUN = 3

#: A grid cell shorter than this does not count towards a row match. Excludes the house
#: number (``"1"``), which would otherwise match every skill that contains the digit,
#: and two-letter values that collide with ordinary words.
_MIN_CELL_CHARS = 3


@dataclass(frozen=True)
class Leak:
    """One skill carrying one problem's answer fragment."""

    skill_id: str
    problem_id: str
    matched: str
    n_tuples: int
    is_parent: bool
    #: ``"moves"`` (numeric run), ``"answer"`` (one verbatim string) or ``"assignment"``
    #: (a run of consecutive answer cells). Defaulted so the numeric and string-answer
    #: call sites, and existing tests, read unchanged.
    kind: str = "moves"

    def describe(self) -> str:
        origin = "its own source problem" if self.is_parent else "a DIFFERENT problem"
        if self.kind == "assignment":
            what = f"{self.n_tuples} consecutive answer cells"
        elif self.n_tuples == 1:
            what = "a verbatim answer"
        else:
            what = f"{self.n_tuples} consecutive moves"
        return (
            f"skill {self.skill_id!r} contains {what}"
            f" from {origin} ({self.problem_id}): {self.matched}"
        )


def _tuples(text: str) -> list[tuple[int, ...]]:
    """Every numeric tuple in *text*, in order, whitespace-normalised."""
    out: list[tuple[int, ...]] = []
    for match in _TUPLE.finditer(text or ""):
        try:
            out.append(tuple(int(part) for part in match.group(1).split(",")))
        except ValueError:  # pragma: no cover - the regex already constrains this
            continue
    return out


def _answer_strings(reference: str) -> list[str]:
    """Verbatim answer strings in a reference answer.

    Tower of Hanoi answers are numeric move lists, handled by :func:`_tuples`.
    Matchstick and ZebraLogic answers are *strings* -- ``"XVI = II + XIV"`` -- listed
    inside the reference text, so they need exact-substring matching instead. Both
    detectors run; a task simply yields nothing for the one that does not apply.
    """
    found = [
        candidate.strip()
        for candidate in _QUOTED.findall(reference or "")
        if len(candidate.strip()) >= _MIN_ANSWER_CHARS
    ]
    # De-duplicate while keeping order, so a repeated answer is not double-reported.
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in found:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _normalise(text: str) -> str:
    """Collapse whitespace, so a reformatted answer still matches."""
    return " ".join((text or "").split())


def _answer_rows(reference: str) -> list[tuple[str, ...]]:
    """The rows of a grid-shaped reference answer, header excluded.

    ZebraLogic's reference is ``{"header": [...], "rows": [[...], ...]}``. The row is
    the unit of secrecy: the individual cells (``"Eric"``, ``"green"``, ``"pizza"``) are
    prompt vocabulary and mostly shorter than :data:`_MIN_ANSWER_CHARS`, so
    :func:`_answer_strings` cannot see them and should not -- one cell is not a leak.
    Three cells *of the same row* is the assignment.

    The header is dropped deliberately: a skill naming three columns ("check
    FavoriteSport, then HairColor, then Color") is describing a method, not reciting an
    answer, and would otherwise be reported.

    Returns ``[]`` for any reference that is not a JSON grid, which is how the numeric
    and single-string tasks opt out.
    """
    try:
        parsed = json.loads(reference or "")
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        return []
    out: list[tuple[str, ...]] = []
    for row in rows:
        if isinstance(row, list) and len(row) >= 2:
            out.append(tuple(str(cell) for cell in row))
    return out


def _mentions_cell(text: str, cell: str) -> bool:
    """Does *text* use *cell* as a whole word?

    Plain ``in`` is wrong for grid cells: they are short and ordinary, so ``"red"``
    matches inside "predicted" and ``"pizza"`` inside a longer dish. Word boundaries and
    a length floor are what keep the row check from firing on honest prose.
    """
    cell = cell.strip()
    if len(cell) < _MIN_CELL_CHARS:
        return False
    return re.search(rf"\b{re.escape(cell)}\b", text, re.IGNORECASE) is not None


def _skill_text(skill: Any) -> str:
    """Every free-text field a teacher could hide an answer in."""
    return " ".join(
        str(getattr(skill, field, "") or "")
        for field in ("title", "principle", "when_to_apply", "example")
    )


def _runs_in_common(needle: Sequence[tuple[int, ...]],
                    haystack: Sequence[tuple[int, ...]],
                    min_run: int) -> tuple[int, list[tuple[int, ...]]]:
    """Longest run of consecutive *needle* tuples appearing consecutively in *haystack*.

    Returns ``(length, the_run)``; ``(0, [])`` when nothing reaches ``min_run``.
    """
    best_len = 0
    best: list[tuple[int, ...]] = []
    for start in range(len(needle)):
        for end in range(start + min_run, len(needle) + 1):
            window = list(needle[start:end])
            # Is this window a contiguous slice of the reference answer?
            found = any(
                list(haystack[i : i + len(window)]) == window
                for i in range(len(haystack) - len(window) + 1)
            )
            if found and len(window) > best_len:
                best_len, best = len(window), window
    return best_len, best


def references_from_task(task: Any) -> Callable[[Any], str]:
    """The reference-answer resolver for a `CoreTask`.

    ``Problem.all_valid_outputs`` is the problem *spec* (``{'num_disks': 5,
    'initial_state': [...], ...}``), **not** the answer -- only
    `format_reference_answer_for_prompt` turns it into one. Getting this wrong is how
    the first version of this module reported "clean" on a bank that provably
    contained a memorised answer, so the resolver is now explicit and there is no
    silent fallback to paper over it.
    """
    return lambda problem: str(
        task.core_task.format_reference_answer_for_prompt(problem.all_valid_outputs)
    )


def find_memorised_answers(
    skills: Iterable[Any],
    problems: Sequence[Any],
    reference_for: Callable[[Any], str],
    *,
    min_run: int = DEFAULT_MIN_RUN,
    min_string_run: int = DEFAULT_MIN_STRING_RUN,
) -> list[Leak]:
    """Skills whose text replays part of a training problem's reference answer.

    ``problems`` are the *training* problems -- the only ones a skill could have been
    distilled from. Passing eval problems here would be meaningless: a skill cannot
    have seen them.

    ``reference_for`` maps a problem to its reference-answer text; use
    :func:`references_from_task`. It is required rather than inferred, because an
    inference that quietly fails turns this check into a rubber stamp.

    An answer string that also appears **in the problem's own prompt** is discarded
    before matching. Quoting the prompt back is not memorising the answer, and on
    ZebraLogic every single-cell candidate is of exactly that kind. What survives the
    filter for a grid answer is the assignment, matched in runs of ``min_string_run``.
    """
    answers: list[tuple[str, list[tuple[int, ...]], list[str], list[tuple[str, ...]]]] = []
    filtered_as_prompt_text = 0
    for problem in problems:
        reference = reference_for(problem)
        prompt = _normalise(getattr(problem, "prompt", "") or "")
        found = _answer_strings(reference)
        # Order is preserved by the filter, so the runs below remain runs of *adjacent*
        # answer cells rather than of whatever happened to survive.
        secret = [s for s in found if _normalise(s) not in prompt]
        filtered_as_prompt_text += len(found) - len(secret)
        answers.append(
            (problem.id, _tuples(reference), secret, _answer_rows(reference))
        )

    if not any(
        tuples or strings or runs for _, tuples, strings, runs in answers
    ):
        raise ValueError(
            "no reference answer yielded any numeric tuples, answer strings or answer "
            "string runs, so this check cannot detect anything. Fix `reference_for` "
            "rather than reading the all-clear. "
            f"({filtered_as_prompt_text} candidate(s) were discarded for appearing in "
            "the problem prompt; if that is all of them, the reference resolver is "
            "probably returning the prompt.) "
            "NOTE this catches only a totally empty reference set -- a resolver that "
            "returns the wrong text containing *some* numbers (e.g. "
            "`Problem.all_valid_outputs`, the problem spec, whose `initial_state` is "
            "a numeric list) passes this check and silently misses real leaks. Use "
            "`references_from_task`."
        )

    leaks: list[Leak] = []
    for skill in skills:
        text = _skill_text(skill)
        skill_tuples = _tuples(text)
        normalised_skill = _normalise(text)
        parents = {str(s) for s in (getattr(skill, "source_seeds", None) or [])}
        for problem_id, answer_tuples, answer_strings, answer_rows in answers:
            skill_id = getattr(skill, "skill_id", "<unknown>")

            # 1. Numeric move runs (Tower of Hanoi).
            if len(answer_tuples) >= min_run and len(skill_tuples) >= min_run:
                length, run = _runs_in_common(skill_tuples, answer_tuples, min_run)
                if length >= min_run:
                    leaks.append(
                        Leak(
                            skill_id=skill_id,
                            problem_id=problem_id,
                            matched=" ".join(str(list(t)) for t in run),
                            n_tuples=length,
                            is_parent=problem_id in parents,
                        )
                    )

            # 2. Verbatim answer strings (Matchstick, ZebraLogic). One exact match is
            #    already conclusive -- unlike a single move, a full answer string does
            #    not occur by chance.
            for answer in answer_strings:
                if _normalise(answer) in normalised_skill:
                    leaks.append(
                        Leak(
                            skill_id=skill_id,
                            problem_id=problem_id,
                            matched=answer,
                            n_tuples=1,
                            is_parent=problem_id in parents,
                            kind="answer",
                        )
                    )

            # 3. Grid rows (ZebraLogic). Individually the cells are prompt vocabulary;
            #    `min_string_run` of them from the SAME row is the assignment. Set
            #    membership, not adjacency: prose reorders an assignment it recites.
            for row in answer_rows:
                hits = [
                    cell for cell in row
                    if _mentions_cell(normalised_skill, _normalise(cell))
                ]
                if len(hits) >= min_string_run:
                    leaks.append(
                        Leak(
                            skill_id=skill_id,
                            problem_id=problem_id,
                            matched=" | ".join(hits),
                            n_tuples=len(hits),
                            is_parent=problem_id in parents,
                            kind="assignment",
                        )
                    )
                    break  # one leaked row is the finding; do not enumerate the grid
    return leaks


def report(leaks: Sequence[Leak]) -> str:
    """One-block summary for the pilot diagnostics."""
    if not leaks:
        return "   no memorised answers detected (clean)"
    lines = [
        f"   {len(leaks)} MEMORISED ANSWER(S) -- a reviewer reading the bank will see this:"
    ]
    lines.extend(f"     {leak.describe()}" for leak in leaks)
    lines.append(
        "   This does not leak to eval (disjoint split), but it corrupts training-time"
    )
    lines.append(
        "   utility and is exactly what PHASE3_DESIGN.md 5b warns about."
    )
    return "\n".join(lines)
