#!/usr/bin/env python
"""Phase 3: HExA Design B on ORBIT Maze.

Runs the round loop over training instances, then a frozen-bank eval on held-out
instances. Companion to ``run_orbit_maze_baseline.py``, which produces the no-bank
arm this is diffed against -- same instances, same sampling seeds, so the two are
paired (DESIGN_PHASE3_MAZE §14).

    # server up first:  sbatch scripts/serve_gpt_oss_vllm.sh
    python scripts/run_orbit_maze_hexa.py --rounds 10 --batch-size 20 \
        --eval-size 256 --out results/orbit_maze_hexa

    # plumbing check, no server, oracle plays and the teacher is canned:
    python scripts/run_orbit_maze_hexa.py --rounds 2 --batch-size 2 \
        --eval-size 2 --stub --out /tmp/hexa_smoke

Serial (D5). Resumable: re-run the same ``--out`` and completed rounds are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import loop, oracle, pools, report  # noqa: E402
from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from hexa_orbit.adapter import MazeTask  # noqa: E402

logger = logging.getLogger("hexa_orbit")

#: Sampling seed for an instance, identical across arms. This is what makes the
#: no-bank and frozen-bank arms *paired*, which McNemar assumes -- at temperature
#: 0.6 the same arm twice gives different runs, so without a shared per-instance
#: seed the two arms are independent samples and the test is wrong (§14, and
#: `hexa_core/WEAKNESSES.md:300` for the precedent of getting this wrong).
def paired_seed(instance_id: str, run_index: int = 0) -> int:
    return (int(instance_id[:8], 16) + run_index) % (2**31)


class OracleStub:
    """Plays the known-map optimum. Proves the plumbing without a server."""

    def __init__(self, instance):
        self.instance = instance
        self.queue: list[str] = []

    def generate_structured(self, *, messages, **_):
        from hexa_core.llm.base import Generation

        if not self.queue:
            self.queue = oracle.instance_optimal_actions(self.instance)
        return Generation.from_parts("", oracle.boxed(self.queue.pop(0)), finish_reason="stop")


def install_stub_teacher() -> None:
    """Canned teacher responses, so --stub needs no server for evolution either."""
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving

    skills = json.dumps([{
        "title": "Expand away from the start",
        "principle": "Prefer directions that increase distance from the start cell.",
        "when_to_apply": "when two or more openings are available",
        "example": "a corridor already walked is spent",
        "source_seeds": [], "confidence": 0.8, "is_new": True,
    }])
    evolved = json.dumps({"skills": json.loads(skills), "mistakes": [],
                          "removed_skill_titles": [], "reasoning": "stub"})
    mistakes = json.dumps({"mistakes": [], "partial_skills": []})

    def call_teacher(prompt: str, model: str = "stub") -> str:
        if "CURRENT SKILL BANK" in prompt:
            return evolved
        if "partial_skills" in prompt:
            return mistakes
        return skills

    _distill.call_teacher = call_teacher
    _evolving.call_teacher = call_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maze-config", default=pools.DEFAULT_MAZE_CONFIG,
        choices=[pools.DEFAULT_MAZE_CONFIG, pools.PAPER_QWEN_CONFIG],
        help="Which ORBIT Maze val config to draw pools from. The default is the "
             "historical pool (total_step_cap 45, ORBIT's openai.sh). "
             "multi_task_multi_episode_config.yaml is total_step_cap 27 -- the pool "
             "behind the paper's Qwen3-8B table. The two pools are DISJOINT; never "
             "pair or pool results across them.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=512,
                        help="training pool drawn; --rounds x --batch-size are consumed")
    parser.add_argument("--eval-size", type=int, default=pools.EVAL_POOL_SIZE,
                        help="§14: the full 256 is what the headline claim needs")
    parser.add_argument("--out", type=Path, default=Path("results/orbit_maze_hexa"))
    parser.add_argument("--teacher-model", default=ActorConfig.model)
    parser.add_argument("--model", default=ActorConfig.model)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=ActorConfig.temperature)
    parser.add_argument("--top-p", type=float, default=ActorConfig.top_p)
    parser.add_argument("--max-response-tokens", type=int,
                        default=ActorConfig.max_response_tokens)
    parser.add_argument(
        "--reasoning-effort", default=ActorConfig.reasoning_effort,
        help="pass '' for models without a reasoning channel (Qwen2.5, Llama, ...). "
             "Sending this to a model that does not understand it makes vLLM reject "
             "the request or ignore it silently, depending on version.",
    )
    parser.add_argument("--injection-budget-tokens", type=int,
                        default=loop.INJECTION_BUDGET_TOKENS)
    parser.add_argument("--admission", action="store_true",
                        help="enable M5 gating (off by default: the criterion needs "
                             "the Phase 2 base rate to calibrate -- see admission.py)")
    parser.add_argument("--admission-trials", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-eval", action="store_true",
                        help="train only; run the frozen eval later from the same --out")
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="instances in flight at once. Maze generation is serialised by a lock "
             "(the only place ORBIT touches the global RNG), so results are "
             "bit-identical to --concurrency 1; see DECISIONS.md D5. With Qwen3 "
             "thinking enabled an instance takes ~3.5 min, so 8-16 is the difference "
             "between a 15-hour and a 1-2 hour 256-instance arm.",
    )
    parser.add_argument(
        "--transcript-mode", choices=["final", "raw"], default="final",
        help="what the assistant turn carries. 'final' = answer channel only "
             "(~15 tokens/turn); 'raw' = reasoning + answer, which is what ORBIT's "
             "own agent stores (~800 tokens/turn with thinking on). See "
             "ActorConfig.transcript_mode -- this materially changes the numbers.",
    )
    parser.add_argument(
        "--trajectory-token-budget", type=int, default=None,
        help="total completion tokens per instance. ORBIT uses 31744 as a "
             "TRAJECTORY budget (data.max_response_length); ours is per-call and "
             "cannot bind. Set 31744 to impose theirs.",
    )
    parser.add_argument(
        "--truncation-scoring", choices=["exclude", "orbit"], default="exclude",
        help="how to score episodes an instance never reached. 'exclude' omits them; "
             "'orbit' counts them as failures, reproducing "
             "MultiEpisodeEnv.get_metrics's early-termination branch.",
    )
    parser.add_argument("--stub", action="store_true",
                        help="oracle actor + canned teacher; no server needed")
    parser.add_argument(
        "--evolver-v2", action="store_true",
        help="iteration-2 evolver: outcome-class trajectory selection (so the "
             "evolver actually sees cross-episode learners), graded failure rewards "
             "(so 46%% of instances stop scoring identically), and a prompt that "
             "quotes the actor's system prompt so skills stop restating it. Off by "
             "default -- it changes what the teacher sees, so a run must not switch "
             "mid-flight.",
    )
    parser.add_argument(
        "--max-trajs", type=int, default=None,
        help="transcripts the evolver block is given per round. Defaults to "
             "hexa_orbit.teacher.DEFAULT_MAX_TRAJS (6). Raising it is half of the "
             "Mastermind v3 recipe: with a smaller --batch-size the evolver "
             "otherwise sees a thinner slice of each round.",
    )
    parser.add_argument(
        "--per-episode-injection", action="store_true",
        help="inject only episode-1-usable skills during episode 1, and the whole "
             "bank from episode 2 on. Roughly 60%% of v3's block was phrased "
             "'based on prior exploration data' and inert in episode 1.")
    parser.add_argument(
        "--inject-top-k", type=int, default=None,
        help="inject only the K highest-confidence skills (and no mistakes) "
             "instead of the whole bank. Separates 'does the bank contain useful "
             "skills' from 'does injecting all of them help'.")
    parser.add_argument("--placebo", choices=["scramble", "foreign"], default=None,
                        help="arm C (§13): evaluate a length-matched placebo bank "
                             "instead of the real one. 'scramble' deranges principles "
                             "across skills; 'foreign' swaps in off-task donors. Writes "
                             "to eval_placebo/ so it cannot overwrite arm B.")
    parser.add_argument("--placebo-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    config = ActorConfig(
        model=args.model, base_url=args.base_url, temperature=args.temperature,
        top_p=args.top_p, max_response_tokens=args.max_response_tokens,
        # '' from the CLI means 'this model has no reasoning channel'. It must
        # become None, not stay an empty string: VLLMClient omits the argument
        # only when it is None, and would otherwise send reasoning_effort="".
        reasoning_effort=args.reasoning_effort or None,
        transcript_mode=args.transcript_mode,
        trajectory_token_budget=args.trajectory_token_budget,
        truncation_scoring=args.truncation_scoring,
    )
    if args.stub:
        install_stub_teacher()

    task = MazeTask(config_name=args.maze_config)
    train_specs = task.train_specs(args.train_size)
    eval_specs = task.eval_specs(args.eval_size)
    pools.assert_disjoint(train_specs, eval_specs)

    needed = args.rounds * args.batch_size
    if needed > len(train_specs):
        logger.warning(
            "--rounds x --batch-size = %d exceeds the %d-instance training pool; "
            "the run will stop early", needed, len(train_specs))

    def actor_factory(guidelines, instance=None):
        """``client=None`` builds a real ``VLLMClient`` lazily.

        Under ``--stub`` the oracle needs the live instance to read the map, which
        is exactly why the loop hands it over rather than leaving a stub to reach
        for one.
        """
        actor = MazeActor(config, guidelines=guidelines)
        if args.stub and instance is not None:
            actor._client = OracleStub(instance)
        return actor

    # `None` means "let the loop install the teacher as it always has", so this
    # stays a no-op unless --max-trajs is passed.
    teacher_install = None
    if args.max_trajs is not None:
        from hexa_orbit import teacher as _teacher

        def teacher_install() -> None:  # noqa: F811
            _teacher.install(reward_bin_selection=args.evolver_v2,
                             max_trajs=args.max_trajs)

    started = time.monotonic()
    result = loop.run_hexa_loop(
        task, train_specs, actor_factory, args.out,
        rounds=args.rounds, batch_size=args.batch_size,
        teacher_model=args.teacher_model,
        admission_enabled=args.admission,
        admission_trials=args.admission_trials,
        injection_budget_tokens=args.injection_budget_tokens,
        concurrency=args.concurrency,
        evolver_v2=args.evolver_v2,
        teacher_install=teacher_install,
        per_episode_injection=args.per_episode_injection,
        resume=not args.no_resume,
        manifest={"actor": config.to_dict(), "teacher_model": args.teacher_model,
                  "stub": args.stub,
                  # The loop knobs were absent from run.json, so recovering what a
                  # run actually did meant counting round dirs and grepping logs.
                  "loop": {"rounds": args.rounds, "batch_size": args.batch_size,
                           "max_trajs": args.max_trajs, "evolver_v2": args.evolver_v2,
                           "per_episode_injection": args.per_episode_injection,
                           "inject_top_k": args.inject_top_k},
                  **task.manifest()},
    )

    print()
    for record in result.rounds:
        print(f"round {record.round_index}: skills {record.n_skills_before}"
              f"->{record.n_skills_after}  bins {record.reward_bins}"
              f"  injected {record.injection['tokens']}tok"
              f"  flagged {record.audit['n_flagged']}")

    if args.skip_eval:
        print(f"\ntrained only; wrote {args.out}")
        return 0

    next_round, bank, _ = loop.resume_state(args.out)
    if bank is None:
        print("\nno bank produced; nothing to evaluate")
        return 1

    eval_dir, arm_label = args.out / "eval_frozen", "frozen bank"
    if args.placebo:
        from hexa_orbit import placebo as placebo_mod

        if args.placebo == "scramble":
            real_bank = bank
            try:
                bank, manifest = placebo_mod.scramble_bank(
                    real_bank, seed=args.placebo_seed
                )
            except ValueError as exc:
                # A 1-skill bank has no derangement. Refusing is correct -- the
                # "placebo" would be the real bank -- but say so rather than
                # traceback.
                raise SystemExit(f"cannot build a placebo: {exc}") from None
            # Compare against the in-memory original, not a re-read from disk.
            placebo_mod.assert_scrambled(real_bank, bank)
        else:
            raise SystemExit(
                "--placebo foreign needs a donor bank; pass one via "
                "hexa_orbit.placebo.foreign_bank in a script for now"
            )
        eval_dir = args.out / "eval_placebo"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "placebo_manifest.json").write_text(json.dumps(manifest, indent=2))
        arm_label = f"PLACEBO ({args.placebo}, seed {args.placebo_seed})"
        print(f"\nplacebo bank: {manifest['tokens_real']} -> "
              f"{manifest['tokens_placebo']} tokens, "
              f"length_matched={manifest['length_matched']}")

    summary = loop.frozen_eval(
        task, eval_specs, actor_factory, bank, eval_dir,
        train_specs=train_specs, injection_budget_tokens=args.injection_budget_tokens,
        concurrency=args.concurrency,
        per_episode_injection=args.per_episode_injection,
        inject_top_k=args.inject_top_k,
    )
    print()
    print(report.format_summary(
        summary["report"],
        title=f"Maze HExA Design B, {arm_label} "
              f"({'ORACLE STUB' if args.stub else config.model})",
    ))
    print(f"\nwall clock {time.monotonic() - started:.0f}s")
    print(f"wrote {args.out}")
    print("\nnext: diff against the no-bank arm on `clean` episode-1, paired by "
          "instance (§14). Run run_orbit_maze_baseline.py on the same --eval-size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
