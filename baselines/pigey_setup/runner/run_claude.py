"""Run the unmodified Pigey harness through HeXA's Claude Code CLI pattern."""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zlib

SETUP_ROOT = Path(__file__).resolve().parents[1]  # baselines/pigey_setup
if str(SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(SETUP_ROOT))

from runner.claude_code import ClaudeCodeBackend, ClaudeCodeError  # noqa: E402
from runner.perception import LOCAL_ER_MODEL, LocalER, SamClient, claude_yesno, route_local_er, route_sam  # noqa: E402
from hexa import teacher as hexa_teacher  # noqa: E402  (puts the repository root on sys.path)


def load_harness(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("pigey_agent_sim", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def is_claude_model(model: str) -> bool:
    return model.startswith("claude-") or model in ("sonnet", "opus", "haiku", "fable")


def suite_of(remaining: list[str]) -> str:
    """The --suite forwarded to agent_sim.py, with Pigey's own default."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite", default="libero_spatial")
    return parser.parse_known_args(remaining)[0].suite


def trial_seed(remaining: list[str]) -> str:
    """HeXA's seed for this trial, ``t<task>_e<episode>``, from the forwarded Pigey args."""
    from hexa.trajectory import seed_of

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    known = parser.parse_known_args(remaining)[0]
    return seed_of(known.task, known.episode)


def skill_injection(bank_path: str, level_name: str, budget_tokens: int, seed: str | None = None) -> dict:
    """Render the HeXA skills block for this suite and describe what it holds.

    ``parent_overlap`` lists injected entries distilled from this very trial (its
    seed appears in their ``source_seeds``). HeXA's frozen eval must never see one;
    in training it is expected when an init state is replayed. Reported, not
    filtered, so a leak is visible in the trial record rather than silently fixed.
    """
    from skillrl.core.skill_bank import SkillBank
    from hexa.teacher import render_block

    block = render_block(SkillBank.load(bank_path), level_name, budget_tokens=budget_tokens)
    if not block.block_text:
        print(f"[hexa] skill bank {bank_path} offers nothing for {level_name}; the prompt is unchanged", file=sys.stderr)
    overlap = [item.skill_id if hasattr(item, "skill_id") else item.mistake_id
               for item in [*block.skills, *block.mistakes]
               if seed is not None and seed in {str(s) for s in item.source_seeds}]
    if overlap:
        print(f"[hexa] WARNING: trial {seed} is a source of injected entries {overlap}; "
              f"this is contamination if {seed} is an eval episode", file=sys.stderr)
    return {"skill_bank": bank_path, "level": level_name, "block_tokens": block.tokens,
            "skill_ids": [skill.skill_id for skill in block.skills],
            "mistake_ids": [mistake.mistake_id for mistake in block.mistakes],
            "dropped_skills": block.dropped_skills, "dropped_mistakes": block.dropped_mistakes,
            "seed": seed, "parent_overlap": overlap, "block_text": block.block_text}


def smoke_test(backend: ClaudeCodeBackend, verifier: ClaudeCodeBackend | None = None):
    # A real native image input, including the tool-result nesting used by
    # Pigey after every action. No simulator, checkpoints, or perception key.
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 32, 32, 8, 2, 0, 0, 0)) +
           chunk(b'IDAT', zlib.compress((b'\0' + bytes([255, 0, 0]) * 32) * 32)) + chunk(b'IEND', b''))
    tools = [{"name": "Perceive", "description": "Observe the scene.",
              "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}}]
    messages = [
        {"role": "user", "content": "Inspect the image returned by Perceive."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "previous", "name": "Perceive", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "previous", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
            {"type": "text", "text": "Set reasoning to the single lowercase word for the dominant image color. Select Perceive with empty input."},
        ]}]},
    ]
    result = backend("", backend.model, messages, "This is a vision and robot-tool transport check.", tools)
    assert result["content"][0]["text"].strip().lower() == "red", result
    calls = [block for block in result["content"] if block["type"] == "tool_use"]
    assert len(calls) == 1 and calls[0]["name"] == "Perceive" and calls[0]["input"] == {}, result
    report = {"status": "passed", "backend": "claude-code", "model": backend.model,
              "vision": "red", "tool": "Perceive", "cli_calls": backend.calls}
    if verifier is not None:
        verdict = claude_yesno(verifier)(None, png, "Is this image a solid red square?")
        assert verdict == "YES", verdict
        report.update(verify_model=verifier.model, verify_verdict=verdict, verify_cli_calls=verifier.calls)
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Other options are forwarded to Pigey's agent_sim.py.")
    parser.add_argument("--model", "--claude-model", default=os.environ.get("PIGEY_CLAUDE_MODEL", "claude-haiku-4-5"))
    parser.add_argument("--claude-timeout", type=int, default=int(os.environ.get("CLAUDE_CLI_TIMEOUT", "300")))
    parser.add_argument("--claude-max-turns", type=int, default=int(os.environ.get("PIGEY_CLAUDE_MAX_TURNS", "3")),
                        help="CLI agent-loop turns per Pigey step (env PIGEY_CLAUDE_MAX_TURNS; object_task's HeXA arm used 10)")
    parser.add_argument("--mode", choices=["harness"], default="harness")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--perception", choices=["local", "gemini"], default=os.environ.get("PIGEY_PERCEPTION", "local"),
                        help="local: open-weight VLM via serve_perception.sh; gemini: upstream Gemini Robotics-ER")
    parser.add_argument("--er-url", default=os.environ.get("PIGEY_ER_URL"),
                        help="ER server URL; serve_perception.sh writes it to perception.env")
    parser.add_argument("--er-model", default=os.environ.get("PIGEY_ER_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--verifier", choices=["claude", "gemini"], default=os.environ.get("PIGEY_VERIFIER", "claude"),
                        help="claude: VerifyCandidate through Claude Code; gemini: upstream Gemini Flash-Lite")
    parser.add_argument("--verify-model", default=os.environ.get("PIGEY_VERIFY_MODEL", "claude-haiku-4-5"))
    parser.add_argument("--sam", choices=["service", "inprocess"], default=os.environ.get("PIGEY_SAM", "service"),
                        help="service: Pigey's SAM on CUDA via sam_server.py; inprocess: upstream (CPU-only here, falls back)")
    parser.add_argument("--sam-url", default=os.environ.get("PIGEY_SAM_URL"),
                        help="SAM server URL; serve_perception.sh writes it to perception.env")
    parser.add_argument("--phase0", choices=["skip", "run"], default=os.environ.get("PIGEY_PHASE0", "skip"),
                        help="skip: agent acts first, as in the paper's Algorithm 1 (sets CORTEX_NO_PHASE0=1); "
                             "run: upstream code's silent pi0.5 attempt before the agent")
    parser.add_argument("--skill-bank", default=os.environ.get("PIGEY_SKILL_BANK"),
                        help="HeXA skill bank JSON; its skills block is appended to Pigey's system prompt")
    parser.add_argument("--skill-level", default=None, help="Bank level to retrieve (default: the --suite)")
    parser.add_argument("--skill-budget-tokens", type=int,
                        default=int(os.environ.get("PIGEY_SKILL_BUDGET", str(hexa_teacher.DEFAULT_INJECTION_BUDGET))),
                        help="Token budget for the injected skills block")
    parser.add_argument("--cli-smoke-test", action="store_true", help="One live camera-image/tool-selection check; run under Slurm")
    parser.add_argument("--check-cli", action="store_true", help="Check the installed CLI's authentication without model inference")
    args, remaining = parser.parse_known_args(argv)
    if not is_claude_model(args.model) or (args.verifier == "claude" and not is_claude_model(args.verify_model)):
        parser.error("Use a Claude model ID or a Claude Code alias such as sonnet or opus.")
    root = Path(os.environ.get("PIGEY_ROOT", "/scratch3/workspace/udhanuka_umass_edu-pigey"))
    executable = os.environ.get("CLAUDE_BINARY", "claude")
    backend = ClaudeCodeBackend(root / "tmp/claude-code", model=args.model, timeout=args.claude_timeout,
                                max_turns=args.claude_max_turns, executable=executable)
    verifier = None
    if args.verifier == "claude":
        verifier = ClaudeCodeBackend(root / "tmp/claude-code", model=args.verify_model, timeout=args.claude_timeout,
                                     max_turns=args.claude_max_turns, executable=executable)
    if args.check_cli:
        return subprocess.run([backend.executable, "auth", "status"], timeout=30).returncode
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run Pigey and its live CLI smoke check inside a Slurm allocation.")
    if args.cli_smoke_test:
        try:
            smoke_test(backend, verifier)
        except (ClaudeCodeError, AssertionError) as exc:
            errors = backend.errors + (verifier.errors if verifier else [])
            print(json.dumps({"status": "failed", "message": str(exc), "backend_errors": errors}), file=sys.stderr)
            return 1
        return 0
    if args.perception == "local" and not args.er_url:
        parser.error("--perception local needs --er-url or PIGEY_ER_URL; source perception.env from serve_perception.sh.")
    if args.sam == "service" and not args.sam_url:
        parser.error("--sam service needs --sam-url or PIGEY_SAM_URL; source perception.env from serve_perception.sh.")
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if "gemini" in (args.perception, args.verifier) and not has_gemini_key:
        parser.error("GEMINI_API_KEY is required for --perception gemini or --verifier gemini.")
    pigey = Path(os.environ.get("PIGEY_REPO", str(SETUP_ROOT.parent / "Pigey")))
    module = load_harness(pigey / "sim/agent_sim.py")
    module.call_claude = backend
    if verifier is not None:
        module.LiberoRunner._vlm_yesno = claude_yesno(verifier)
    injection = None
    if args.skill_bank:
        # run_harness resolves get_system_prompt from the module at call time, so
        # the skills block is appended without touching the upstream prompt.
        injection = skill_injection(args.skill_bank, args.skill_level or suite_of(remaining), args.skill_budget_tokens,
                                    seed=trial_seed(remaining))
        upstream_prompt = module.get_system_prompt
        block_text = injection["block_text"]
        module.get_system_prompt = lambda task_description, max_steps_per_rollout: (
            upstream_prompt(task_description, max_steps_per_rollout) + ("\n\n" + block_text if block_text else ""))
        print(f"[hexa] injecting {len(injection['skill_ids'])} skills and {len(injection['mistake_ids'])} mistakes "
              f"({injection['block_tokens']} tokens) from {args.skill_bank}")
    sam = None
    if args.sam == "service":
        sam = SamClient(args.sam_url)
        route_sam(sam)
    local_er = None
    # Process-local overrides, restored after the run. The sentinels only pass
    # upstream key-presence checks; the CLI child removes the Anthropic one, and
    # local ER requests never send the Gemini one.
    # HARNESS_PERCEPTION=gemini_er is Pigey's documented default, but only an
    # explicit value makes post-action images get a fresh detection instead of
    # boxes from an earlier frame.
    overrides = {"ANTHROPIC_API_KEY": "pigey-claude-code-unused", "HARNESS_PERCEPTION": "gemini_er",
                 "CORTEX_NO_PHASE0": "1" if args.phase0 == "skip" else "0"}
    if args.perception == "local":
        import litellm
        local_er = LocalER(args.er_url, args.er_model)
        route_local_er(litellm, local_er)
        overrides["GEMINI_ER_MODEL"] = LOCAL_ER_MODEL
        if not has_gemini_key:
            overrides["GEMINI_API_KEY"] = "pigey-local-er-unused"
    original_log = module.TrialLog

    def all_errors():
        return (backend.errors + (verifier.errors if verifier else [])
                + (local_er.errors if local_er else []) + (sam.errors if sam else []))

    class CLITrialLog(original_log):
        def finish(self, success, env_steps, llm_steps, duration_s):
            errors = all_errors()
            self.data.update(backend="claude-code", claude_code_calls=backend.calls,
                             perception=args.perception, verifier=args.verifier, sam=args.sam, phase0=args.phase0,
                             backend_errors=errors, status="backend_error" if errors else "completed",
                             rate_limit_waits=[wait for source in (backend, verifier) if source is not None
                                               for wait in getattr(source, "rate_limit_waits", [])])
            if verifier is not None:
                self.data.update(verify_model=verifier.model, verify_calls=verifier.calls)
            if local_er is not None:
                self.data.update(er_model=local_er.model, er_calls=local_er.calls)
            if sam is not None:
                self.data.update(sam_calls=sam.calls)
            if injection is not None:
                self.data.update(skill_injection=injection)
            return super().finish(None if errors else success, env_steps, llm_steps, duration_s)

    module.TrialLog = CLITrialLog
    previous_env = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    previous_argv = sys.argv
    sys.argv = [str(pigey / "sim/agent_sim.py"), *remaining, "--mode", "harness", "--model", args.model,
                "--out-dir", args.out_dir or str(root / "results/claude-code")]
    try:
        code = module.main()
        return 1 if all_errors() else code
    finally:
        sys.argv = previous_argv
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    sys.exit(main())
