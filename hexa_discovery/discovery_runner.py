"""Run one DiscoverPhysics episode (optionally skill-augmented) and emit a
HeXA-schema trajectory JSON — the DiscoverPhysics analog of HeXA's
`run_skill_augmented` / `run_batch`.

Skill injection is non-invasive: we build a normal DiscoveryAgent, then append
the retrieved skills block to `agent._system` before `agent.run()`. No edits to
scienceagent.

IMPORTANT: scienceagent loads its prompt files by relative path, so agents
must run with HeXA/discovery-agents as the working directory.
`run_batch_discovery` switches into it for the episodes (output paths are made
absolute first); launch the entry points from the HeXA root.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from hexa_discovery import _paths

_paths.bootstrap()

from hexa_discovery.config import (
    DISCOVERY_LEVEL,
    FIT_WORLDS,
    PASS_ERR_THRESHOLD,
    PASS_SCORE_THRESHOLD,
    WORLD_ENGINE,
    WORLD_VARS,
)
from hexa_discovery.reward import blended_reward
from hexa_discovery.traj_adapter import build_trajectory

# HeXA skill-bank plumbing (agent-agnostic).
from skillrl.core.skill_bank import SkillBank
from skillrl.core.retriever import retrieve_skills


_SKILLS_PREAMBLE = (
    "The following experimentation and reasoning skills were learned from "
    "discovery episodes on OTHER physics worlds. Treat them as reusable methods, "
    "not answers: this world's law may differ, so use the skills to design "
    "experiments and always verify against the data you observe.\n\n"
)


@contextmanager
def _cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def retrieve_skills_and_mistakes(
    bank: Optional[SkillBank],
    *,
    max_skills: int = 12,
    max_mistakes: int = 6,
):
    """Return the (skills, mistakes) lists that would be injected for this bank.

    `max_skills` defaults to 12 to match the bank capacity (`--max-skills`), not
    the old 8. With a cap of 8 and a bank of 10, the two lowest-confidence skills
    never reached the agent -- and they were consistently the SUBMISSION-stage
    ones ("hard-code well-measured constants", "only declare genuinely uncertain
    quantities as fittable"), because submission hygiene scores lower than
    experiment design on confidence and is therefore always first over the side.
    The same bias was measured at the pi_ev synthesis step. Worse, the bank grows
    each evolution round while injection stayed fixed, so the injected FRACTION
    shrank as training progressed.

    Split out from build_skills_block so callers can record which specific
    skills/mistakes were shown to the agent (ids + titles), not just whether
    injection happened.
    """
    if bank is None:
        return [], []
    skills = retrieve_skills(
        bank, DISCOVERY_LEVEL, max_general=0, max_specific=max_skills
    )
    mistakes = bank.get_mistakes_for_level(DISCOVERY_LEVEL)[:max_mistakes]
    return skills, mistakes


def render_skills_block(bank: SkillBank, skills: list, mistakes: list) -> str:
    """Render already-retrieved skills+mistakes into the injected prompt block.

    Split from retrieval so callers can record exactly which skills/mistakes
    were shown (ids + titles) using the SAME lists that got rendered — avoids
    two independent retrieval calls drifting apart.
    """
    if not skills and not mistakes:
        return ""
    block = bank.format_skills_as_prompt(
        DISCOVERY_LEVEL, skills=skills, mistakes=mistakes
    )
    if not block:
        return ""
    # Relabel the auto-generated "Discovery-Specific Skills" header honestly.
    block = block.replace(
        "## Discovery-Specific Skills",
        "## Cross-World Experimentation Skills (learned on other worlds — verify here)",
    )
    return _SKILLS_PREAMBLE + block


def build_skills_block(
    bank: Optional[SkillBank],
    *,
    max_skills: int = 12,
    max_mistakes: int = 6,
) -> str:
    """Retrieve + render the cross-world skills+mistakes block in one call."""
    skills, mistakes = retrieve_skills_and_mistakes(
        bank, max_skills=max_skills, max_mistakes=max_mistakes
    )
    if bank is None:
        return ""
    return render_skills_block(bank, skills, mistakes)


def _make_evaluator(world: str, executor):
    """Select the world-appropriate evaluator (mirrors run_discovery.py)."""
    from scienceagent.evaluator import (
        Evaluator,
        CircleEvaluator,
        SpeciesEvaluator,
        ThreeSpeciesEvaluator,
        DarkMatterEvaluator,
        EtherEvaluator,
        HubbleEvaluator,
    )
    mapping = {
        "circle": CircleEvaluator,
        "species": SpeciesEvaluator,
        "three_species": ThreeSpeciesEvaluator,
        "dark_matter": DarkMatterEvaluator,
        "ether": EtherEvaluator,
        "hubble": HubbleEvaluator,
    }
    return mapping.get(world, Evaluator)(executor)


def evaluate_episode(
    world: str,
    world_dict: dict,
    agent,
    law_source: Optional[str],
    judge_model: str,
) -> dict:
    """Evaluate a finished episode → metrics dict with graded-reward inputs."""
    from scienceagent.evaluator import (
        ExplanationJudge,
        _extract_training_trajectories,
    )

    if law_source is None:
        return {
            "mean_pos_error": float("inf"),
            "norm_mse": float("inf"),
            "expl_score": 0.0,
            "expl_reasoning": "",
            "passed_benchmark": False,
            "no_law": True,
        }

    executor = world_dict["executor"]
    evaluator = _make_evaluator(world, executor)

    eval_kwargs = {"verbose": False}
    if world in FIT_WORLDS:
        eval_kwargs["training_trajectories"] = _extract_training_trajectories(
            agent.conversation_log
        )
    results = evaluator.evaluate(law_source, **eval_kwargs)

    mean_pos_error = results.get("mean_pos_error", float("inf"))
    var = WORLD_VARS.get(world)
    norm_mse = mean_pos_error / var if var else mean_pos_error

    # Score the law a second time WITHOUT the evaluator's parameter refit, i.e.
    # on the constants the agent itself submitted.
    #
    # On FIT_WORLDS the evaluator re-optimises the law's declared parameters
    # against the agent's own TRAINING trajectories, then scores on HELD-OUT
    # test cases -- so a well-measured constant can be overfitted away.
    # Measured over 16 sampled circle/fractional episodes: the refit was worse
    # in 10, better in 4, median 150x worse, worst cases 3e5-7e5x. Laws scoring
    # 1e-9..1e-6 on their own constants were pushed past the 0.1 pass line.
    #
    # `fit_parameters()` is OPTIONAL, so this gap is actionable: a constant the
    # agent has already pinned down should not be declared fittable. The teacher
    # could never learn that, because only the post-refit number was stored.
    norm_mse_own = None
    if world in FIT_WORLDS and eval_kwargs.get("training_trajectories"):
        try:
            own = evaluator.evaluate(law_source, verbose=False)
            e_own = own.get("mean_pos_error", float("inf"))
            norm_mse_own = e_own / var if var else e_own
        except Exception as exc:          # never let the diagnostic fail a run
            print(f"  [own-constants eval failed: {str(exc)[:70]}]")

    judge = ExplanationJudge(judge_model=judge_model)
    expl_result = judge.score(
        agent_explanation=agent.discovered_explanation,
        optimal_explanation=world_dict.get("optimal_explanation", ""),
        rubric=world_dict.get("explanation_rubric", ""),
        verbose=False,
    )
    expl_score = expl_result.get("score")
    expl_score = 0.0 if expl_score is None else float(expl_score)
    # The judge's critique names exactly which rubric dimension was missed
    # ("they never identify the fractional Laplacian operator..."). Half the
    # reward is this score, and until now the teacher saw only the scalar --
    # no signal at all about WHY an explanation lost points, so it could not
    # distil a lesson about it. Kept here; redacted in the adapter.
    expl_reasoning = expl_result.get("reasoning") or ""

    passed = (norm_mse < PASS_ERR_THRESHOLD) and (expl_score >= PASS_SCORE_THRESHOLD)
    return {
        "mean_pos_error": mean_pos_error,
        "norm_mse": norm_mse,
        "norm_mse_own": norm_mse_own,
        "expl_score": expl_score,
        "passed_benchmark": bool(passed),
        "expl_reasoning": expl_reasoning,
        "no_law": False,
    }


def run_episode(
    world: str,
    seed: int,
    bank: Optional[SkillBank],
    *,
    model: str = "claude-opus-4-6",
    judge_model: str = "claude-opus-4-6",
    noise_std: float = 0.075,
    max_rounds: Optional[int] = None,
    min_rounds: Optional[int] = None,
    max_tokens: int = 8192,
    use_critic: bool = False,
    critic_model: str = "claude-haiku-4-5-20251001",
    max_skills: int = 12,
    max_mistakes: int = 6,
    prompt_block: Optional[str] = None,
    redact: bool = True,
    verbose: bool = False,
) -> dict:
    """Run a single episode; return a HeXA-schema trajectory dict.

    `prompt_block` is an already-rendered block appended to the system prompt in
    place of a skill bank -- the seam the memory baselines in `baselines/discovery`
    use, so a retrieved-rollout block costs context in exactly the place HeXA's
    skills block does and the two arms are comparable at equal injection point.
    It is mutually exclusive with `bank`; passing both is a caller error.
    """
    if bank is not None and prompt_block:
        raise ValueError("pass a bank or a prompt_block, not both")
    from scienceagent.worlds import get_world
    from scienceagent.agent import DiscoveryAgent

    engine = WORLD_ENGINE.get(world, "nbody")
    world_dict = get_world(
        world, engine=engine, noise_std=noise_std, noise_seed=seed
    )

    critic = None
    if use_critic:
        from scienceagent.critic import CriticAgent
        critic = CriticAgent(model=critic_model)

    agent_kwargs = dict(
        model=model,
        executor=world_dict["executor"],
        mission=world_dict["mission"],
        max_tokens=max_tokens,
        verbose=verbose,
        system_prompt_path=world_dict["system_prompt"],
        instructions_path=world_dict["instructions"],
        law_stub=world_dict["law_stub"],
        experiment_format=world_dict["experiment_format"],
        critic=critic,
    )
    if max_rounds is not None:
        agent_kwargs["max_rounds"] = max_rounds
    # ScienceAgent's MIN_ROUNDS default is 2, and the actor's system prompt
    # states it verbatim ("You must run at least 2 round(s)"). Measured on
    # circle: caution-flavoured banks drive the agent to 2.3 rounds -- the hard
    # floor -- against a 4.3-round no-bank baseline, i.e. they submit at the
    # first legal opportunity. Raising the floor is a deviation from the
    # benchmark's actor configuration and makes an arm non-comparable with any
    # arm run at the default, so it is opt-in and recorded per run.
    if min_rounds is not None:
        agent_kwargs["min_rounds"] = min_rounds

    agent = DiscoveryAgent(**agent_kwargs)

    # ── Skill injection (non-invasive): append to the built system prompt ──
    injected_skills, injected_mistakes = retrieve_skills_and_mistakes(
        bank, max_skills=max_skills, max_mistakes=max_mistakes
    )
    skills_block = (
        render_skills_block(bank, injected_skills, injected_mistakes)
        if bank is not None else (prompt_block or "")
    )
    if skills_block:
        agent._system = agent._system.rstrip() + "\n\n" + skills_block

    start = time.perf_counter()
    law_source = agent.run()
    elapsed = time.perf_counter() - start

    metrics = evaluate_episode(world, world_dict, agent, law_source, judge_model)
    reward = blended_reward(metrics.get("expl_score"), metrics.get("norm_mse"))

    traj = build_trajectory(
        seed=seed,
        world=world,
        conversation_log=agent.conversation_log,
        explanation=agent.discovered_explanation,
        law_source=law_source,
        metrics=metrics,
        max_rounds=agent.max_rounds,
        reward=reward,
        redact=redact,
    )
    traj["elapsed_time"] = elapsed
    traj["model"] = model
    traj["noise_std"] = noise_std
    traj["skills_injected"] = bool(skills_block)
    # Exactly which bank entries were shown to the agent this episode — lets
    # you confirm/inspect injection directly instead of diffing prompt text.
    traj["skill_ids_injected"] = [s.skill_id for s in injected_skills]
    traj["skill_titles_injected"] = [s.title for s in injected_skills]
    traj["mistake_ids_injected"] = [m.mistake_id for m in injected_mistakes]
    return traj


def run_batch_discovery(
    world: str,
    seeds: list[int],
    bank: Optional[SkillBank],
    eval_dir: Path,
    *,
    level_dirname: str = DISCOVERY_LEVEL,
    use_cli: bool = True,
    cli_model: Optional[str] = None,
    **episode_kwargs,
) -> list[dict]:
    """Run episodes over seeds, writing trajectory_seed{N}_skillrl.json files.

    Files land in `eval_dir/<level_dirname>/` so HeXA's load_trajectories (which
    globs trajectory_seed*.json) can read them for distill/evolve. Existing files
    are reused (resume-friendly).

    use_cli=True (default) routes the actor / judge / critic through the Claude
    Code CLI (no API key), matching HeXA. cli_model, if given, overrides the
    model sent to the CLI for all three.
    """
    if use_cli:
        from hexa_discovery.cli_client import patch_llm_client
        patch_llm_client(model_override=cli_model)
        if cli_model:
            # cli_model forces every `claude -p --model ...` call regardless of
            # what string run_episode passes in — so the trajectory's own
            # "model"/judge bookkeeping must match, or the JSON records a model
            # that was never actually invoked (e.g. default claude-opus-4-6
            # while every real call went to sonnet).
            episode_kwargs["model"] = cli_model
            episode_kwargs.setdefault("judge_model", cli_model)

    # Absolute before the chdir into DISCOVERY_ROOT below, so a relative
    # eval_dir still resolves against the caller's working directory.
    eval_dir = Path(eval_dir).resolve()
    out_dir = eval_dir / level_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    with _cwd(_paths.DISCOVERY_ROOT):
        for seed in seeds:
            traj_path = out_dir / f"trajectory_seed{seed}_skillrl.json"
            if traj_path.exists():
                try:
                    results.append(json.loads(traj_path.read_text()))
                    print(f"  seed {seed}: reuse existing "
                          f"({'PASS' if results[-1].get('success') else 'FAIL'})")
                    continue
                except (json.JSONDecodeError, OSError):
                    pass  # corrupted — re-run
            print(f"\n{'#'*60}\n  {world}  seed={seed}\n{'#'*60}")
            traj = run_episode(world, seed, bank, **episode_kwargs)
            traj_path.write_text(json.dumps(traj, indent=2, default=str))
            results.append(traj)
            print(f"  -> {'PASS' if traj.get('success') else 'FAIL'}  "
                  f"norm_mse={traj.get('norm_mse')}  expl={traj.get('expl_score')}  "
                  f"reward={traj.get('reward'):+.2f}")

    n_pass = sum(1 for r in results if r.get("success"))
    print(f"\n  {world}: {n_pass}/{len(results)} passed")
    return results
