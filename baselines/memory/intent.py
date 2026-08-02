"""Intent text for an InterPhyre task instance.

A task instance is `(level, seed)`.  Both baselines key their memory on an
*intent*: the text that gets embedded and compared against past episodes.
MemRL calls this the intent `z` (its ALFWorld runner uses the goal string);
Episodic RAG calls it the problem text.

Two modes, because the obvious construction does not work:

`full`
    Level description (`skillrl/core/config.py:LEVEL_DESCRIPTIONS`) plus the
    toolkit's `get_level_state()` dump.  Reads naturally, but within one level
    it is ~95% identical across seeds — same prose, same object names, same
    success condition, same world bounds, with only coordinates differing.
    Measured on 105 seeds of `catapult` and `pass_the_parcel`, pairwise cosine
    came out at mean 0.9994, std 0.0003.  At that spread MemRL's Phase A is
    all-or-nothing and its similarity z-normalization is pure noise, so
    value-aware retrieval cannot discriminate anything.

`geometry` (default)
    Only the instance-varying content: per-object coordinates and extents, plus
    derived relations between the objects the goal is about.  No prose, no
    boilerplate, no success condition — those are constant within a level and
    therefore carry no retrieval signal.  This is the honest analogue of
    MemRL's "intent": in ALFWorld the goal text differs per task, and in
    InterPhyre what differs per instance is the geometry.

Either way this costs no LLM call and no simulator rollout — the level is loaded
for its cached object table and never stepped.

Run `baselines/calibrate.py` after changing anything here; the similarity spread
it prints is the check that the intent actually discriminates instances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

MODES = ("geometry", "full")
DEFAULT_MODE = "geometry"


@dataclass(frozen=True)
class Intent:
    level: str
    seed: int
    text: str
    mode: str = DEFAULT_MODE

    @property
    def instance_id(self) -> str:
        return f"{self.level}:{self.seed}"


def _level_description(level: str) -> str:
    try:
        from skillrl.core.config import LEVEL_DESCRIPTIONS

        return (LEVEL_DESCRIPTIONS.get(level) or "").strip()
    except Exception:
        return ""


def _num(value: Any) -> str:
    """Compact fixed-point rendering; keeps token counts stable across seeds."""
    try:
        return f"{float(value):+.2f}"
    except (TypeError, ValueError):
        return "na"


def _object_line(name: str, obj: dict) -> str:
    parts = [name, _num(obj.get("x")), _num(obj.get("y"))]
    if obj.get("type") == "Ball":
        parts.append(f"r{_num(obj.get('radius'))}")
    elif obj.get("type") == "Bar":
        parts.extend(
            [
                f"l{_num(obj.get('left'))}",
                f"r{_num(obj.get('right'))}",
                f"t{_num(obj.get('top'))}",
                f"b{_num(obj.get('bottom'))}",
                f"a{_num(obj.get('angle'))}",
            ]
        )
    else:
        parts.append(f"a{_num(obj.get('angle'))}")
    parts.append("dyn" if obj.get("dynamic") else "static")
    return " ".join(parts)


def _relations(info: dict[str, dict]) -> list[str]:
    """Pairwise offsets between the coloured balls the goal conditions on.

    These are the quantities a solver actually reasons about (how far, which
    way), and they vary strongly with the seed, so they carry most of the
    retrieval signal.
    """
    import math

    keys = [n for n in ("green_ball", "blue_ball", "gray_ball") if n in info]
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            oa, ob = info[a], info[b]
            try:
                dx = float(ob["x"]) - float(oa["x"])
                dy = float(ob["y"]) - float(oa["y"])
            except (TypeError, ValueError, KeyError):
                continue
            out.append(
                f"{a}->{b} dx{_num(dx)} dy{_num(dy)} d{_num(math.hypot(dx, dy))}"
            )
    return out


def build_intent(level: str, seed: int, mode: str = DEFAULT_MODE) -> Intent:
    """Construct the intent text for one (level, seed)."""
    if mode not in MODES:
        raise ValueError(f"Unknown intent mode {mode!r}; expected one of {MODES}")

    from react_agent.tools import InterphyreToolkit

    toolkit = InterphyreToolkit(level_name=level, seed=seed)
    try:
        if mode == "full":
            parts = [f"Level: {level}", f"Seed: {seed}"]
            desc = _level_description(level)
            if desc:
                parts.append(f"Task: {desc}")
            parts.append(toolkit.get_level_state())
            text = "\n".join(parts)
        else:
            info = toolkit._level_info
            action_objects = set(toolkit._action_objects)
            lines = [
                _object_line(name, obj)
                for name, obj in sorted(info.items())
                if name not in action_objects
            ]
            lines.extend(_relations({k: v for k, v in info.items()
                                     if k not in action_objects}))
            text = "\n".join(lines)
    finally:
        close = getattr(toolkit, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    return Intent(level=level, seed=seed, text=text, mode=mode)


class IntentCache:
    """Disk-backed cache of intent texts, so repeated runs skip level loading."""

    def __init__(self, path: Optional[Path] = None, mode: str = DEFAULT_MODE):
        if mode not in MODES:
            raise ValueError(f"Unknown intent mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.path = Path(path) if path else None
        self._mem: dict[str, str] = {}
        if self.path and self.path.exists():
            try:
                self._mem = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._mem = {}

    def get(self, level: str, seed: int) -> Intent:
        key = f"{level}:{seed}"
        if key not in self._mem:
            self._mem[key] = build_intent(level, seed, self.mode).text
            self._flush()
        return Intent(level=level, seed=seed, text=self._mem[key], mode=self.mode)

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._mem, indent=2))


def cache_path(cache_dir: Path, level: str, mode: str) -> Path:
    """Per-mode cache file, so switching modes never reuses stale intents."""
    return Path(cache_dir) / f"intents_{level}_{mode}.json"
