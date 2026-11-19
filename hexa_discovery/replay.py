"""Cross-world contrast buffer for the evolver.

`evolve_skill_bank` is handed one world's episodes per round, and on this
benchmark a world is unanimous — so the batch is single-class and the evolution
prompt asks what separates solving from failing while showing one side of it.
Measured on the 2026-09-12 `logo_global_term` run, **six of eight rounds had no
contrast at all**:

    yukawa 3/0   fractional 0/3   coulomb_easy 0/3   extra_dimensions 1/2
    circle 0/3   three_species 0/3   dark_matter 0/3  oscillator 1/2

This is structural. `--seeds-per-world` cannot fix it: a seed reshuffles
observation noise and the no-bank agent is near-deterministic (reward spreads
0.05-0.11), so N seeds on one world give N of the same class. The per-class
quotas in `teacher_prompts_discovery.make_evolver_block` stop the majority class
crowding the other out; they cannot manufacture a class that is absent.

Contrast therefore has to come from pooling across WORLDS, exactly as the
round-1 bootstrap already has to. This module keeps every episode run so far and
backfills each round's batch with the missing class drawn from earlier worlds.

The buffer holds TRAIN worlds only. A held-out world's episodes reaching the
evolver would put the answer it is supposed to be blind to into the bank, so
`add` refuses them loudly rather than filtering silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

#: Seed namespace for replayed episodes. They are copied into the round's evolve
#: directory alongside this round's episodes, which reuse seed numbers across
#: worlds (every world starts at --start-seed); the teacher cites `source_seeds`,
#: so a collision would make a citation ambiguous and confidence unattributable.
REPLAY_SEED_BASE = 9000


class HeldOutLeakage(ValueError):
    """A held-out world's episode was offered to the contrast buffer."""


class ReplayBuffer:
    """Every episode run so far, across train worlds + the offline bootstrap."""

    def __init__(self, held_out_worlds: Iterable[str] = ()) -> None:
        self.held_out = set(held_out_worlds)
        self._trajs: list[dict] = []

    # ── population ─────────────────────────────────────────────────────

    def add(self, trajs: Iterable[dict]) -> int:
        added = 0
        for t in trajs:
            world = t.get("world")
            if world in self.held_out:
                raise HeldOutLeakage(
                    f"episode from HELD-OUT world {world!r} offered to the contrast "
                    "buffer; it would carry that world's answer into the bank"
                )
            self._trajs.append(t)
            added += 1
        return added

    def add_dir(self, traj_dir: Path) -> int:
        """Load `trajectory_seed*.json` from one directory into the buffer."""
        traj_dir = Path(traj_dir)
        if not traj_dir.exists():
            return 0
        loaded = []
        for f in sorted(traj_dir.glob("trajectory_seed*.json")):
            try:
                loaded.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return self.add(loaded)

    def __len__(self) -> int:
        return len(self._trajs)

    @property
    def worlds(self) -> list[str]:
        return sorted({t.get("world") for t in self._trajs if t.get("world")})

    # ── selection ──────────────────────────────────────────────────────

    def select_contrast(
        self,
        new_trajs: list[dict],
        *,
        max_trajs: int = 6,
        success_slots: int = 3,
        reward_fn: Optional[Callable[[dict], float]] = None,
    ) -> list[dict]:
        """This round's episodes, plus backfill of the MISSING class.

        Every new episode is kept — it is the evidence the round is about. The
        remaining slots go to the class this round lacks, drawn from earlier
        worlds: highest-reward successes (what solving looks like) and
        lowest-reward failures (the most instructive way to be wrong), preferring
        worlds not already in the batch so the teacher sees breadth rather than
        five variations of one world.

        Replayed copies are marked `_replay=True` and given seeds in the
        REPLAY_SEED_BASE namespace. The originals are never mutated.
        """
        if reward_fn is None:
            import skillrl.distillation.teacher_prompts as _tp
            reward_fn = _tp.compute_trajectory_reward

        chosen = list(new_trajs)
        n_succ = sum(1 for t in chosen if t.get("success"))
        n_fail = len(chosen) - n_succ

        want_succ = max(0, success_slots - n_succ)
        want_fail = max(0, (max_trajs - success_slots) - n_fail)

        seen_ids = {id(t) for t in new_trajs}
        seen_worlds = {t.get("world") for t in new_trajs}

        def pick(want: int, success: bool) -> list[dict]:
            pool = [t for t in self._trajs
                    if bool(t.get("success")) is success and id(t) not in seen_ids]
            # Best successes; worst failures.
            pool.sort(key=reward_fn, reverse=success)
            out, used_worlds = [], set(seen_worlds)
            # First pass: one per unseen world, for breadth.
            for t in pool:
                if len(out) >= want:
                    break
                if t.get("world") not in used_worlds:
                    out.append(t)
                    used_worlds.add(t.get("world"))
            # Second pass: fill any remainder in reward order.
            for t in pool:
                if len(out) >= want:
                    break
                if t not in out:
                    out.append(t)
            return out

        replayed = pick(want_succ, True) + pick(want_fail, False)

        stamped = []
        for i, t in enumerate(replayed):
            copy = dict(t)
            copy["_replay"] = True
            copy["orig_seed"] = t.get("seed")
            copy["seed"] = REPLAY_SEED_BASE + i
            stamped.append(copy)

        for t in chosen:
            t.setdefault("_replay", False)
        return chosen + stamped

    # ── materialisation ────────────────────────────────────────────────

    def write_evolve_dir(
        self, trajs: list[dict], out_dir: Path, level: str = "discovery"
    ) -> Path:
        """Write a batch to a directory `load_trajectories` can read.

        A separate directory from the round's own output, deliberately: the round
        directory is the canonical record of that world's episodes and drives
        resume, so replay copies must not land in it.
        """
        target = Path(out_dir) / level
        target.mkdir(parents=True, exist_ok=True)
        for t in trajs:
            tag = "replay" if t.get("_replay") else "new"
            path = target / f"trajectory_seed{t['seed']}_{tag}_skillrl.json"
            path.write_text(json.dumps(t, indent=2, default=str))
        return target


def describe(trajs: list[dict]) -> str:
    """One-line summary of a contrast batch, for the run log."""
    new = [t for t in trajs if not t.get("_replay")]
    rep = [t for t in trajs if t.get("_replay")]
    def counts(ts):
        s = sum(1 for t in ts if t.get("success"))
        return f"{s}S/{len(ts) - s}F"
    worlds = sorted({t.get("world") for t in rep})
    return (f"{len(trajs)} trajectories: new {counts(new)}"
            + (f" + replay {counts(rep)} from {', '.join(worlds)}" if rep else
               "  <-- no replay needed"))
