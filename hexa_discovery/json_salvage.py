"""Salvage a truncated JSON array from the teacher, instead of discarding it.

`distill._parse_json_array` tries two regexes -- a fenced ```json [...] ``` block
and a bare `[...]` -- both of which need the CLOSING bracket. When the teacher's
reply is cut short mid-array (token limit, transport truncation) neither matches,
the function prints `[Warning] Could not parse JSON array` and returns [], and the
caller carries on with zero skills.

That is silent and expensive. Measured on `within_circle_s5rep1`: round 0's
contrastive pass returned a well-formed list of skills, was truncated before its
closing bracket, and every one was dropped. The bank started at 2 skills instead
of 8 (the 2 were failure-derived partials from a later pass) and never recovered
-- 6 skills by round 3 against a comparable run's 9. Nothing in the output said
the run was crippled; it looked like an ordinary bank.

This adds a third stage: scan for balanced top-level `{...}` objects and keep
every one that parses. A truncated array loses only its final, incomplete object
rather than all of them. If even that finds nothing, raise -- an empty
contrastive pass is a broken run, not a run with no skills, and it should stop
rather than produce a plausible-looking bank.

Installed by `patch()` into the two modules that call it.
"""

from __future__ import annotations

import json
import re
from typing import Any


class TeacherParseFailure(RuntimeError):
    """The teacher's reply yielded no parsable objects at all."""


def _objects(text: str) -> list[dict[str, Any]]:
    """Every balanced top-level {...} in `text` that parses as a JSON object."""
    out: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(obj, dict):
                            out.append(obj)
                    start = -1
    return out


def parse_json_array(text: str) -> list[dict[str, Any]]:
    """Drop-in for `distill._parse_json_array`, with truncation salvage."""
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    salvaged = _objects(text)
    if salvaged:
        print(f"  [salvage] teacher reply was malformed/truncated; recovered "
              f"{len(salvaged)} complete object(s) from it", flush=True)
        return salvaged

    raise TeacherParseFailure(
        "teacher reply yielded no parsable JSON objects; refusing to continue "
        "with an empty distillation pass. First 300 chars:\n" + text[:300]
    )


def patch() -> None:
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _ev

    _distill._parse_json_array = parse_json_array
    if hasattr(_ev, "_parse_json_array"):
        _ev._parse_json_array = parse_json_array
