"""
Reward manager for Interphyre physics puzzle RL training.

Reward signal:
  +1.0  — agent called `finish` and the simulation succeeded (SUCCESS in [FINAL_RESULT])
   0.0  — agent called `finish` but simulation failed, or never called finish

Trajectory logging:
  Set TRAJ_LOG_FILE env var to a .jsonl path. Per training step, the best,
  worst, and one random trajectory are appended to the file for inspection.
"""

import json
import os
import random
import re
from pathlib import Path

import torch
import logging
from collections import defaultdict

from verl.workers.reward_manager import register

logger = logging.getLogger(__name__)

# Matches the [FINAL_RESULT] block injected by the tool server on `finish`
_FINAL_RESULT_RE = re.compile(r"\[FINAL_RESULT\](.*?)(?=Thought:|Action:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", re.DOTALL)


def _compute_score(response_str: str) -> float:
    match = _FINAL_RESULT_RE.search(response_str)
    if match is None:
        return 0.0
    return 1.0 if "SUCCESS" in match.group(1) else 0.0


def _summarise_tool_calls(tool_interact_info: list) -> list:
    """Extract compact tool call summary from agent loop interaction log."""
    calls = []
    for info in tool_interact_info:
        action = info.get("action", "")
        m = _ACTION_RE.search(action)
        tool_name = m.group(1) if m else "unknown"
        calls.append({
            "tool": tool_name,
            "valid": info.get("valid_action", True),
            "done": info.get("done", False),
            "obs_snippet": str(info.get("obs", ""))[:200],
        })
    return calls


@register("interphyre")
class InterphyreRewardManager:
    """Reward manager for Interphyre physics puzzle training."""

    name = "interphyre"

    def __init__(self, tokenizer, num_examine: int = 1, compute_score=None, **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # Always use our own scoring — ignore injected compute_score (wrong signature)
        self._examined = 0
        self._call_count = 0
        self._traj_log_path = os.environ.get("TRAJ_LOG_FILE")
        # Evolving variant sets TRAJ_LOG_ALL=1 so the skill-evolution hook sees
        # all n=4 rollouts per step (success + failure on same seed). Default
        # off → keep current best/worst/random subsampling for non-evolving runs.
        self._log_all_rollouts = os.environ.get("TRAJ_LOG_ALL", "0") == "1"
        # Evolving variant: write each rollout's FULL ReAct response to
        # FULL_TRAJ_DIR/step_{S}/{phase}/seed{N}_w{w}.json. Canonical input for
        # the skill-evolution hook (full text, no 2000-char cap, one file per
        # rollout). Unset → no-op (non-evolving variants unaffected).
        self._full_traj_dir = os.environ.get("FULL_TRAJ_DIR")
        # rollout_idx is always 0 in async ReAct (reward manager invoked
        # per-rollout with len(data)==1) — use a per-(step,phase) counter for
        # filename uniqueness.
        self._write_counter: dict = defaultdict(int)

    def _read_global_step(self, data) -> int:
        meta_step = (data.meta_info or {}).get("global_steps")
        if meta_step is not None:
            try:
                return int(meta_step)
            except (TypeError, ValueError):
                pass
        return self._call_count

    def _write_traj_log(self, entries: list):
        if not self._traj_log_path or not entries:
            return
        os.makedirs(os.path.dirname(self._traj_log_path), exist_ok=True)
        with open(self._traj_log_path, "a", buffering=1) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def __call__(self, data, return_dict: bool = False):
        self._call_count += 1
        current_step = self._read_global_step(data)
        # Tag train vs val so the skill-evolution hook can route them to
        # separate folders and only feed train rollouts to the teacher.
        # verl sets data.meta_info["validate"] = True in _validate().
        is_val = False
        try:
            is_val = bool((data.meta_info or {}).get("validate", False))
        except Exception:
            pass
        phase = "val" if is_val else "train"

        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {k: data.non_tensor_batch[k] for k in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        traj_entries = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attn_mask = data_item.batch["attention_mask"]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(attn_mask[prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            score = _compute_score(response_str)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = score

            reward_extra_info["score"].append(score)
            reward_extra_info["success"].append(1.0 if score > 0 else 0.0)
            reward_extra_info["response_length"].append(valid_response_length)

            # Extract level name for logging
            level_name = "unknown"
            try:
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                if isinstance(extra_info, dict):
                    level_name = extra_info.get("level_name", "unknown")
            except Exception:
                pass

            # Tool call summary from agent loop
            tool_interact_info = []
            try:
                raw = data_item.non_tensor_batch.get("tool_interact_info", [])
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                # np.array([list_of_dicts], dtype=object).tolist() → [list_of_dicts]
                if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
                    raw = raw[0]
                if isinstance(raw, list):
                    tool_interact_info = raw
            except Exception:
                pass

            # Debug print for first few
            if self._examined < self.num_examine:
                self._examined += 1
                logger.info(
                    f"[InterphyreReward] step={current_step} level={level_name} "
                    f"score={score:.2f} resp_len={valid_response_length}\n"
                    f"--- response (last 500 chars) ---\n{response_str[-500:]}\n"
                )

            seed_val = None
            try:
                if isinstance(extra_info, dict):
                    seed_val = extra_info.get("seed")
            except Exception:
                pass

            # Per-rollout full-text dump for the skill-evolution hook.
            # No-op unless FULL_TRAJ_DIR is set (evolving variant only).
            if self._full_traj_dir:
                key = (current_step, phase)
                w = self._write_counter[key]
                self._write_counter[key] = w + 1
                # Verl spawns multiple reward manager instances across agent
                # loop worker processes, each with its own _write_counter
                # starting at 0. Without pid in the filename, w=0 from each
                # process collides on disk and only the last write survives.
                pid = os.getpid()
                out_dir = Path(self._full_traj_dir) / f"step_{current_step}" / phase
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"seed{seed_val}_pid{pid}_w{w}.json").write_text(json.dumps({
                    "step": current_step,
                    "phase": phase,
                    "level": level_name,
                    "seed": seed_val,
                    "rollout_idx": i,
                    "write_idx": w,
                    "score": round(float(score), 4),
                    "response_length": valid_response_length,
                    "n_turns": len(tool_interact_info),
                    "raw_response": response_str,
                }))

            # Compact inspection log (still truncated to 2000 chars).
            if self._traj_log_path:
                traj_entries.append({
                    "step": current_step,
                    "phase": phase,
                    "level": level_name,
                    "seed": seed_val,
                    "rollout_idx": i,
                    "score": round(float(score), 4),
                    "response_length": valid_response_length,
                    "tool_calls": _summarise_tool_calls(tool_interact_info),
                    "n_turns": len(tool_interact_info),
                    "response_snippet": response_str[-2000:],
                })

        if traj_entries:
            if self._log_all_rollouts:
                self._write_traj_log(traj_entries)
            else:
                # Default: best, worst, and one random trajectory per step.
                scores = [e["score"] for e in traj_entries]
                best_idx = int(max(range(len(scores)), key=lambda k: scores[k]))
                worst_idx = int(min(range(len(scores)), key=lambda k: scores[k]))
                rand_idx = random.randrange(len(traj_entries))
                to_log = {best_idx, worst_idx, rand_idx}
                self._write_traj_log([traj_entries[k] for k in sorted(to_log)])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        return reward_tensor
