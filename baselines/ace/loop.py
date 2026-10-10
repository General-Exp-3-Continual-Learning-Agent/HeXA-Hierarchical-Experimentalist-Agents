"""ACE (Agentic Context Engineering) as an online seed stream on InterPhyre.

Maps the reference loop (`ace/ace.py:_train_single_sample`, online mode) onto
the protocol the Episodic RAG / MemRL baselines already use
(`baselines/runner/stream.py`), so the curves overlay directly:

    per seed, in order:
      1. Generator  -- ReactAgent solves the seed with the whole playbook
                       appended to the system prompt (same seam as every other
                       method, `baselines/runner/memory_runner.py`).  This first
                       attempt is the one that is scored.
      2. Reflector  -- diagnoses the trajectory from the simulator verdict and
                       tags each bullet the agent cited as helpful/harmful/neutral;
                       counters are updated in place.
      3. Curator    -- emits ADD operations, merged by ACE's own deterministic
                       code (`baselines/ace/playbook.py`).

Correspondence to the reference implementation, and the deliberate departures:

  * Same model for all three roles (paper §4.2).
  * Label-free setting: no ground truth exists on InterPhyre; the Reflector sees
    the simulator verdict and observations (paper Table 1, "GT labels ✗").
  * Reflect-and-retry (`max_num_rounds`) is OFF by default (`resolve_rounds=0`):
    in the reference loop a failed sample is re-solved up to N times with the
    reflection in context before curation.  On InterPhyre each re-solve is a full
    25-step episode with its own simulator budget, i.e. extra environment
    interaction the other methods do not get.  `resolve_rounds=N` restores it;
    retries are never scored, and their simulator calls are logged separately.
  * The post-curate re-generation (`ace.py:609-628`) is skipped: it only feeds a
    training-accuracy log and costs one more episode per seed.
  * `curate_every>1` concatenates the reflections since the last curation; the
    reference loop would silently drop all but the last.  Default 1 (= paper).
  * Grow-and-refine dedup is not run (reference default
    `use_bulletpoint_analyzer=False`), and the reference Curator only supports
    ADD, so the playbook grows monotonically.  Its size is logged every seed.
  * Offline warmup (paper Table 3, "offline warmup"): when initial trajectories
    are given, the Reflector and Curator run over them once before the stream,
    the same Off2On head start HeXA and the memory baselines get.

Resumability: a seed's trajectory JSON is written only after its curation, and
`ace_state.json` (playbook + counters) right after.  On restart a seed whose JSON
exists is skipped if the state already includes its curation, or re-curated from
the cached trajectory (never re-solved) if the crash fell between the two
writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

from baselines.ace import playbook as pb
from baselines.ace.prompts import (
    CURATOR_PROMPT,
    EMPTY_PLAYBOOK,
    FEEDBACK_FAILURE,
    FEEDBACK_SUCCESS,
    GENERATOR_BLOCK,
    REFLECTOR_PROMPT,
)
from baselines.memory.blocks import count_tokens
from baselines.memory.trajectory_fmt import (
    count_simulator_calls,
    format_trajectory,
    format_verifier_feedback,
)

METHOD = "ace"
#: `ace/core/generator.py:_extract_bullet_ids_regex`, verbatim.
BULLET_ID_RE = re.compile(r"\[([a-z]{3,}-\d{5})\]")


@dataclass
class ACEConfig:
    level: str
    seeds: Sequence[int]
    output_dir: Path
    seeds_per_round: int = 3
    max_iterations: int = 25
    temperature: float = 0.3          # actor sampling temperature (all methods)
    max_new_tokens: int = 800         # actor per-turn cap (all methods)
    is_oss: bool = False
    verbose: bool = False
    freeze_after: Optional[int] = None
    resolve_rounds: int = 0           # reference `max_num_rounds`; 0 = no re-solve
    curate_every: int = 1             # reference `curator_frequency`
    token_budget: int = 80000         # reference `playbook_token_budget` (prompt text only)
    role_max_tokens: int = 4096       # reference `max_tokens` for Reflector/Curator
    max_obs_chars: int = 600          # per-observation cap in the Reflector's trajectory
    max_traj_steps: Optional[int] = None  # None = whole trajectory to the Reflector


@dataclass
class ACEState:
    playbook: str = EMPTY_PLAYBOOK
    next_global_id: int = 1
    curated_through: int = -1         # last stream index whose curation is in `playbook`
    offline_seeds: list = field(default_factory=list)
    pending_reflections: list = field(default_factory=list)
    counters: dict = field(default_factory=lambda: {
        "reflector_calls": 0,
        "reflector_empty": 0,
        "reflector_no_json": 0,
        "tags_on_uncited_bullets": 0,   # applied anyway, as the reference does
        "curator_calls": 0,
        "curator_empty": 0,
        "curator_invalid": 0,
        "bullets_added": 0,
        "non_add_ops_ignored": 0,
        "training_episodes": 0,
        "training_simulator_calls": 0,
    })

    @classmethod
    def load(cls, path: Path) -> "ACEState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        state = cls()
        for key, value in data.items():
            if key == "counters":
                state.counters.update(value)
            elif hasattr(state, key):
                setattr(state, key, value)
        return state

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, path)


# ── pure helpers (unit-tested, no LLM) ─────────────────────────────────────

def render_block(playbook: str) -> str:
    return GENERATOR_BLOCK.format(playbook=playbook)


def cited_bullet_ids(result: dict) -> list[str]:
    """Bullet IDs the agent cited anywhere in its Thoughts, first-seen order."""
    seen: list[str] = []
    for step in result.get("trajectory") or []:
        thought = step.get("thought", "") if isinstance(step, dict) else (
            step[0] if isinstance(step, (list, tuple)) and step else ""
        )
        for bid in BULLET_ID_RE.findall(str(thought or "")):
            if bid not in seen:
                seen.append(bid)
    return seen


def extract_bullet_tags(response: str) -> list[dict]:
    """`ace/core/reflector.py:_extract_bullet_tags` (non-JSON-mode path),
    preceded by a whole-object parse so fenced or prefixed JSON also works."""
    parsed = pb.extract_json_from_text(response)
    if isinstance(parsed, dict) and isinstance(parsed.get("bullet_tags"), list):
        return [t for t in parsed["bullet_tags"] if isinstance(t, dict)]
    try:
        start_idx = response.find('"bullet_tags"')
        if start_idx != -1:
            bracket_idx = response.find("[", start_idx)
            if bracket_idx != -1:
                depth, end_idx = 0, bracket_idx
                for i in range(bracket_idx, len(response)):
                    if response[i] == "[":
                        depth += 1
                    elif response[i] == "]":
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                tags = json.loads(response[bracket_idx:end_idx])
                return [t for t in tags if isinstance(t, dict)]
    except Exception:
        pass
    return []


def validate_operations(response: str) -> list[dict]:
    """`ace/core/curator.py:_extract_and_validate_operations`; raises ValueError."""
    info = pb.extract_json_from_text(response, "operations")
    if not info or not isinstance(info, dict):
        raise ValueError("Failed to extract valid JSON from curator response")
    if "reasoning" not in info:
        raise ValueError("JSON missing required 'reasoning' field")
    if "operations" not in info:
        raise ValueError("JSON missing required 'operations' field")
    if not isinstance(info["reasoning"], str):
        raise ValueError("'reasoning' field must be a string")
    if not isinstance(info["operations"], list):
        raise ValueError("'operations' field must be a list")
    for i, op in enumerate(info["operations"]):
        if not isinstance(op, dict):
            raise ValueError(f"Operation {i} must be a dictionary")
        if "type" not in op:
            raise ValueError(f"Operation {i} missing required 'type' field")
        if op["type"] == "ADD":
            missing = {"type", "section", "content"} - set(op)
            if missing:
                raise ValueError(f"ADD operation {i} missing fields: {sorted(missing)}")
    return info["operations"]


def placement_text(result: dict) -> str:
    action = result.get("action")
    if isinstance(action, (list, tuple)) and len(action) >= 3:
        return f"red ball at x={action[0]}, y={action[1]}, radius={action[2]}"
    return "none (no valid placement submitted)"


def playbook_digest(playbook: str) -> str:
    return hashlib.sha1(playbook.encode()).hexdigest()[:12]


# ── the three roles ────────────────────────────────────────────────────────

class ACE:
    def __init__(
        self,
        config: ACEConfig,
        *,
        model_fn: Callable,
        text_llm: Callable[[str, int], str],
        task_text: Callable[[], str],
        solve: Callable[..., dict],
    ):
        self.cfg = config
        self.model_fn = model_fn
        self.llm = text_llm
        self.task_text = task_text
        self.solve = solve      # solve(seed, block) -> result dict
        self.out = Path(config.output_dir)
        self.level_dir = self.out / config.level
        self.level_dir.mkdir(parents=True, exist_ok=True)
        (self.out / "playbooks").mkdir(exist_ok=True)
        self.state_path = self.out / "ace_state.json"
        self.calls_path = self.out / "ace_calls.jsonl"
        self.episodes_path = self.out / "episodes.jsonl"
        self.state = ACEState.load(self.state_path)

    # logging --------------------------------------------------------------
    def _log_call(self, role: str, tag: str, prompt: str, response: str, **extra) -> None:
        with self.calls_path.open("a") as fh:
            fh.write(json.dumps({
                "role": role, "tag": tag, "time": time.time(),
                "prompt": prompt, "response": response, **extra,
            }, default=str) + "\n")

    # Reflector ------------------------------------------------------------
    def reflect(self, result: dict, tag: str, retry_of: Optional[str] = None) -> tuple[str, list]:
        ids = cited_bullet_ids(result)
        bullets_used = pb.extract_playbook_bullets(self.state.playbook, ids)
        feedback = (FEEDBACK_SUCCESS if result.get("success") else FEEDBACK_FAILURE)
        feedback += "\n" + format_verifier_feedback(result)
        prompt = REFLECTOR_PROMPT.format(
            question=self.task_text(),
            trajectory=format_trajectory(
                result,
                max_steps=self.cfg.max_traj_steps,
                max_obs_chars=self.cfg.max_obs_chars,
            ),
            prediction=placement_text(result),
            feedback=feedback,
            bullets_used=bullets_used,
        )
        response = self.llm(prompt, self.cfg.role_max_tokens)
        c = self.state.counters
        c["reflector_calls"] += 1
        tags: list = []
        if not response:
            c["reflector_empty"] += 1
            response = "(empty)"
        else:
            tags = extract_bullet_tags(response)
            if pb.extract_json_from_text(response) is None:
                c["reflector_no_json"] += 1
        if tags:
            c["tags_on_uncited_bullets"] += sum(
                1 for t in tags if (t.get("id") or t.get("bullet")) not in ids
            )
            self.state.playbook = pb.update_bullet_counts(self.state.playbook, tags)
        self._log_call("reflector", tag, prompt, response,
                       cited_ids=ids, bullet_tags=tags, retry_of=retry_of)
        return response, tags

    # Curator --------------------------------------------------------------
    def curate(self, reflection: str, step: int, total: int, tag: str) -> list[dict]:
        prompt = CURATOR_PROMPT.format(
            current_step=step,
            total_samples=total,
            token_budget=self.cfg.token_budget,
            playbook_stats=json.dumps(pb.get_playbook_stats(self.state.playbook), indent=2),
            recent_reflection=reflection,
            current_playbook=self.state.playbook,
            question_context=self.task_text(),
        )
        response = self.llm(prompt, self.cfg.role_max_tokens)
        c = self.state.counters
        c["curator_calls"] += 1
        applied: list[dict] = []
        error = None
        if not response:
            c["curator_empty"] += 1
            error = "empty_response"
        else:
            try:
                ops = validate_operations(response)
                applied = [op for op in ops if op.get("type") == "ADD"]
                c["non_add_ops_ignored"] += len(ops) - len(applied)
                if applied:
                    self.state.playbook, self.state.next_global_id = pb.apply_curator_operations(
                        self.state.playbook, applied, self.state.next_global_id
                    )
                    c["bullets_added"] += len(applied)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                # Reference behaviour: skip the update and keep training.
                c["curator_invalid"] += 1
                error = f"{type(exc).__name__}: {exc}"
        self._log_call("curator", tag, prompt, response, operations=applied, error=error)
        return applied

    def _maybe_curate(self, reflection: str, step: int, total: int, tag: str, force: bool = False):
        self.state.pending_reflections.append(reflection)
        if not force and len(self.state.pending_reflections) < max(1, self.cfg.curate_every):
            return []
        joined = "\n\n".join(self.state.pending_reflections)
        self.state.pending_reflections = []
        return self.curate(joined, step, total, tag)

    def _snapshot(self, name: str) -> None:
        (self.out / "playbooks" / f"{name}.txt").write_text(self.state.playbook)
        (self.out / "playbook.txt").write_text(self.state.playbook)

    # offline warmup --------------------------------------------------------
    def warmup(self, trajectories: Sequence[dict], total: int) -> None:
        todo = [t for t in sorted(trajectories, key=lambda t: int(t["seed"]))
                if int(t["seed"]) not in self.state.offline_seeds]
        for i, traj in enumerate(todo, len(self.state.offline_seeds) + 1):
            seed = int(traj["seed"])
            print(f"\n  [ACE warmup] offline seed {seed} "
                  f"({'SUCCESS' if traj.get('success') else 'FAILURE'})")
            reflection, _ = self.reflect(traj, tag=f"warmup_seed{seed}")
            ops = self._maybe_curate(reflection, i, total, f"warmup_seed{seed}",
                                     force=(i == len(trajectories)))
            print(f"    +{len(ops)} bullets  |  playbook "
                  f"{count_tokens(self.state.playbook)} tokens")
            self.state.offline_seeds.append(seed)
            self._snapshot(f"warmup_seed{seed}")
            self.state.save(self.state_path)

    # one stream step -------------------------------------------------------
    def _learn(self, result: dict, step_index: int, step_no: int, total: int) -> dict:
        """Reflect (+ optional reflect/re-solve rounds) and curate. Returns ACE info."""
        seed = result["seed"]
        info: dict[str, Any] = {"retries": []}
        reflection, tags = self.reflect(result, tag=f"seed{seed}")
        info["bullet_tags"] = tags
        current = result
        rounds = 0
        while not current.get("success") and rounds < self.cfg.resolve_rounds:
            rounds += 1
            block = render_block(self.state.playbook) + (
                "\n\n## Reflection on your previous attempt at this exact puzzle\n"
                + reflection
            )
            retry = self.solve(seed, block)
            sims = count_simulator_calls(retry)
            self.state.counters["training_episodes"] += 1
            self.state.counters["training_simulator_calls"] += sims
            info["retries"].append({
                "round": rounds, "success": bool(retry.get("success")),
                "iterations": retry.get("iterations"), "simulator_calls": sims,
                "action": retry.get("action"),
            })
            current = retry
            if current.get("success"):
                break
            reflection, _ = self.reflect(current, tag=f"seed{seed}_retry{rounds}",
                                         retry_of=f"seed{seed}")
        ops = self._maybe_curate(reflection, step_no, total, f"seed{seed}",
                                 force=(step_index == len(self.cfg.seeds) - 1))
        info["operations"] = ops
        return info

    def run(self, offline_trajectories: Sequence[dict] = (), manifest: Optional[dict] = None) -> dict:
        n_off = len(offline_trajectories)
        total = n_off + len(self.cfg.seeds)
        if offline_trajectories:
            self.warmup(offline_trajectories, total)

        records: list[dict] = []
        for step_index, seed in enumerate(self.cfg.seeds):
            step_no = n_off + step_index + 1
            traj_path = self.level_dir / f"trajectory_seed{seed}_{METHOD}.json"
            frozen = self.cfg.freeze_after is not None and step_index >= self.cfg.freeze_after

            if traj_path.exists():
                cached = json.loads(traj_path.read_text())
                if frozen or step_index <= self.state.curated_through:
                    records.append(cached)
                    print(f"\n  Seed {seed}: SKIPPING (exists — "
                          f"{'SUCCESS' if cached.get('success') else 'FAILURE'})")
                    continue
                print(f"\n  Seed {seed}: trajectory exists but its curation is "
                      f"not in the playbook — re-curating from the cached rollout")
                cached["ace"] = self._learn(cached, step_index, step_no, total)
                self.state.curated_through = step_index
                self._snapshot(f"step{step_index:03d}_seed{seed}")
                traj_path.write_text(json.dumps(cached, indent=2, default=str))
                self.state.save(self.state_path)
                records.append(cached)
                continue

            print(f"\n{'#' * 60}\n  [ace] {self.cfg.level}  seed {seed}  "
                  f"({step_index + 1}/{len(self.cfg.seeds)})"
                  f"{'  [playbook frozen]' if frozen else ''}\n{'#' * 60}")

            block = render_block(self.state.playbook)
            playbook_before = self.state.playbook
            started = time.time()
            result = self.solve(seed, block)
            result["method"] = METHOD
            result["stream_index"] = step_index
            result["memory_frozen"] = frozen
            result["retrieved_memory_ids"] = cited_bullet_ids(result)
            result["playbook_digest"] = playbook_digest(playbook_before)
            result["playbook_tokens_before"] = count_tokens(playbook_before)
            result["playbook_bullets_before"] = pb.get_playbook_stats(playbook_before)["total_bullets"]

            if not frozen:
                result["ace"] = self._learn(result, step_index, step_no, total)
                self.state.curated_through = step_index
                self._snapshot(f"step{step_index:03d}_seed{seed}")
            result["playbook_bullets_after"] = pb.get_playbook_stats(self.state.playbook)["total_bullets"]
            result["playbook_tokens_after"] = count_tokens(self.state.playbook)

            traj_path.write_text(json.dumps(result, indent=2, default=str))
            self.state.save(self.state_path)
            with self.episodes_path.open("a") as fh:
                fh.write(json.dumps({
                    "method": METHOD, "level": self.cfg.level, "seed": seed,
                    "stream_index": step_index, "success": bool(result.get("success")),
                    "iterations": result.get("iterations"),
                    "simulator_calls": result.get("simulator_calls"),
                    "added_context_tokens": result.get("added_context_tokens"),
                    "cited_bullet_ids": result["retrieved_memory_ids"],
                    "bullets_added": len((result.get("ace") or {}).get("operations") or []),
                    "playbook_bullets_after": result["playbook_bullets_after"],
                    "playbook_tokens_after": result["playbook_tokens_after"],
                    "memory_frozen": frozen,
                    "elapsed_time": result.get("elapsed_time"),
                }, default=str) + "\n")
            records.append(result)
            print(f"  -> {'SUCCESS' if result.get('success') else 'FAILURE'} in "
                  f"{result.get('iterations', '?')} iters, "
                  f"{result.get('simulator_calls', '?')} sims, "
                  f"{time.time() - started:.0f}s  |  playbook "
                  f"{result['playbook_bullets_after']} bullets / "
                  f"{result['playbook_tokens_after']} tokens")

        summary = self.summarize(records, manifest)
        (self.out / f"summary_{METHOD}.json").write_text(json.dumps(summary, indent=2, default=str))
        from baselines.runner.stream import print_summary
        print_summary(summary)
        return summary

    def summarize(self, records: Sequence[dict], manifest: Optional[dict]) -> dict:
        from baselines.runner.stream import summarize as stream_summarize

        stats = pb.get_playbook_stats(self.state.playbook)
        shim_store = SimpleNamespace(stats=lambda: {
            "playbook_bullets": stats["total_bullets"],
            "playbook_tokens": count_tokens(self.state.playbook),
            **{k: stats[k] for k in ("high_performing", "problematic", "unused")},
        })
        shim_cfg = SimpleNamespace(method=METHOD, level=self.cfg.level,
                                   seeds=self.cfg.seeds,
                                   seeds_per_round=self.cfg.seeds_per_round)
        summary = stream_summarize(records, shim_cfg, shim_store, manifest=manifest)
        summary["ace"] = {
            "counters": dict(self.state.counters),
            "offline_warmup_seeds": list(self.state.offline_seeds),
            "final_playbook_stats": stats,
            "final_playbook_tokens": count_tokens(self.state.playbook),
            "playbook_growth": [
                {"seed": r.get("seed"),
                 "bullets_after": r.get("playbook_bullets_after"),
                 "tokens_after": r.get("playbook_tokens_after")}
                for r in records
            ],
        }
        return summary
