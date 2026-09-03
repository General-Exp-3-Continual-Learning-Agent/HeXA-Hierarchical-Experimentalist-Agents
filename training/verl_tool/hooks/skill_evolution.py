"""Step-end skill evolution hook for the continuous-process evolving variant.

Called by the verl trainer fit() loop after every `skill_evolve_freq` steps.
It is *synchronous*: the trainer blocks until the bank file has been rewritten,
so the next __getitem__ in rl_dataset sees a coherent updated bank.

Inputs (read from disk, written by the reward manager during training):
  - TRAJ_LOG_FILE: JSONL with one entry per rollout (when TRAJ_LOG_ALL=1).
  - skill_bank_path: the JSON the dataset injects at runtime.

Outputs:
  - skill_bank_path is atomically rewritten with the evolved bank.

Failure mode: if anything goes wrong (teacher down, parse error, ...), the
function logs and returns without modifying the bank file. Training continues
with the previous bank.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Module-level state: tracks files in FULL_TRAJ_DIR already consumed so each
# fire only feeds *new* rollouts to the teacher. Files are written by the
# reward manager and never deleted, so a set of paths is sufficient.
_processed_files: set = set()


def _ensure_skillrl_path():
    """Make `skillrl.*` importable. The repo has a symlink skillrl -> SKILL-RL
    at the project root, so inserting cwd is sufficient when training is
    launched from the repo root (as our SLURM scripts do)."""
    if "skillrl" in sys.modules:
        return
    here = Path.cwd()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def _parse_react_steps(text: str) -> list[dict]:
    """Parse ReAct text into thought/action/observation dicts.

    Mirrors the logic in SKILL-RL/tools/convert_training_trajs.py:parse_react_steps
    so the converted trajectories look identical to the existing batch tool.
    """
    import re
    steps = []
    chunks = re.split(r"(?=Thought:)", text.strip())
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        thought_m = re.search(r"Thought:(.*?)(?=Action:|$)", chunk, re.DOTALL)
        action_m  = re.search(r"Action:(.*?)(?=Action Input:|Observation:|Thought:|$)", chunk, re.DOTALL)
        obs_m     = re.search(r"Observation:(.*?)(?=Thought:|$)", chunk, re.DOTALL)
        thought = thought_m.group(1).strip() if thought_m else ""
        action  = action_m.group(1).strip()  if action_m  else ""
        obs     = obs_m.group(1).strip()     if obs_m     else ""
        if thought or action:
            steps.append({"thought": thought, "action": action, "observation": obs})
    return steps


def _full_to_skillrl_record(full: dict) -> dict:
    """Convert a FULL_TRAJ_DIR JSON (full ReAct text) into the SKILL-RL
    trajectory record format the teacher consumes."""
    return {
        "success":      float(full.get("score", 0.0)) >= 1.0,
        "seed":         full.get("seed"),
        "iterations":   full.get("n_turns", 0),
        "level":        full.get("level", "unknown"),
        "trajectory":   _parse_react_steps(full.get("raw_response", "")),
        "raw_response": full.get("raw_response", ""),
    }


def _select_new_train_files(full_traj_dir: Path, level_name: str) -> list[Path]:
    """Find new train rollout files in FULL_TRAJ_DIR since last fire.
    Returns paths under full_traj_dir/step_*/train/. Updates the module-level
    processed-set in place so the next call only sees new files."""
    if not full_traj_dir.exists():
        return []
    new_files = []
    for p in sorted(full_traj_dir.glob("step_*/train/*.json")):
        if p in _processed_files:
            continue
        _processed_files.add(p)
        new_files.append(p)
    return new_files


def _stage_one_per_seed(new_files: list[Path], staging_dir: Path, level_name: str) -> int:
    """Group new train rollouts by seed, pick the highest-scoring one per seed,
    convert to SKILL-RL format, and write `trajectory_seed{N}_step{S}.json`
    files into staging_dir for the teacher to glob.

    Returns the number of files written (= number of distinct seeds).
    """
    by_seed: dict = {}  # seed -> (score, full_dict)
    for p in new_files:
        try:
            full = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if full.get("level") != level_name:
            continue
        seed = full.get("seed")
        if seed is None:
            continue
        score = float(full.get("score", 0.0))
        prev = by_seed.get(seed)
        # Higher score wins; on tie, keep the first one seen (deterministic by glob sort).
        if prev is None or score > prev[0]:
            by_seed[seed] = (score, full)

    if not by_seed:
        return 0

    staging_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for seed, (_score, full) in sorted(by_seed.items()):
        rec = _full_to_skillrl_record(full)
        step = full.get("step", 0)
        # Filename MUST start with "trajectory_seed" — load_trajectories in
        # SKILL-RL/distillation/distill.py globs for "trajectory_seed*.json".
        fname = f"trajectory_seed{seed}_step{step}.json"
        (staging_dir / fname).write_text(json.dumps(rec, indent=2))
        written += 1
    return written


def run_skill_evolution(
    *,
    skill_bank_path: str,
    level_name: str,
    teacher_endpoint: str,
    full_traj_dir: Optional[str] = None,
    work_dir: Optional[str] = None,
    max_skills: int = 10,
    max_mistakes: int = 5,
    global_step: int = 0,
    **_legacy,  # absorb traj_log_path and other deprecated kwargs without breaking callers
) -> None:
    """Run one round of skill evolution. Called by the trainer hook.

    Parameters
    ----------
    skill_bank_path : the JSON file rl_dataset reads at __getitem__.
    level_name      : level being trained (single-level runs only for now).
    teacher_endpoint: e.g. "http://gpu013:8000".
    full_traj_dir   : directory of per-rollout JSON files written by the
                      reward manager. Defaults to $FULL_TRAJ_DIR.
    work_dir        : scratch dir for per-step trajectory JSONs.
    max_skills      : SkillBank cap.
    max_mistakes    : SkillBank cap.
    global_step     : current trainer step (for logging / output naming).
    """
    full_traj_dir = Path(full_traj_dir or os.environ.get("FULL_TRAJ_DIR", ""))
    if not str(full_traj_dir) or not full_traj_dir.exists():
        logger.warning("[skill_evolution] no FULL_TRAJ_DIR — skipping evolution at step %d", global_step)
        return

    work_root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="skillevolve_"))
    step_dir = work_root / f"step_{global_step}" / level_name

    new_files = _select_new_train_files(full_traj_dir, level_name)
    n = _stage_one_per_seed(new_files, step_dir, level_name)
    if n == 0:
        logger.info("[skill_evolution] no new train trajectories at step %d — keeping bank", global_step)
        return

    _ensure_skillrl_path()
    try:
        from skillrl.core.skill_bank import SkillBank  # type: ignore
        from skillrl.distillation.evolving_distill import evolve_skill_bank  # type: ignore
    except Exception as e:
        logger.warning("[skill_evolution] skillrl import failed (%s: %s) — skipping", type(e).__name__, e)
        return

    bank_path = Path(skill_bank_path)
    if not bank_path.exists():
        logger.warning("[skill_evolution] skill bank %s missing — cannot evolve", bank_path)
        return

    try:
        prev_bank = SkillBank.load(bank_path)
    except Exception as e:
        logger.warning("[skill_evolution] failed to load prev bank: %s: %s", type(e).__name__, e)
        return

    # Write the evolved bank to a temp path, then atomic rename so readers
    # never see a half-written file.
    tmp_path = bank_path.with_suffix(bank_path.suffix + ".tmp")
    try:
        evolve_skill_bank(
            level_name=level_name,
            prev_bank=prev_bank,
            new_trajs_dir=step_dir,
            output_path=tmp_path,
            max_skills=max_skills,
            max_mistakes=max_mistakes,
            teacher_model=f"vllm:{teacher_endpoint}",
        )
    except Exception as e:
        logger.warning("[skill_evolution] evolve_skill_bank raised: %s: %s — keeping previous bank",
                       type(e).__name__, e)
        if tmp_path.exists():
            try: tmp_path.unlink()
            except OSError: pass
        return

    if not tmp_path.exists():
        logger.warning("[skill_evolution] evolve_skill_bank did not write %s — keeping previous bank", tmp_path)
        return

    # Atomic publish.
    os.replace(tmp_path, bank_path)
    # Snapshot the post-evolve bank into this step's folder so the bank that
    # produced the next K steps' rollouts is preserved alongside the inputs
    # that produced it.
    try:
        shutil.copy2(bank_path, step_dir / "bank_out.json")
    except Exception as e:
        logger.warning("[skill_evolution] failed to snapshot bank_out.json at step %d: %s: %s",
                       global_step, type(e).__name__, e)
    logger.info("[skill_evolution] step %d: rewrote %s using %d new trajectories", global_step, bank_path, n)
