"""Simple viewer for visualizing Interphyre levels and solutions."""

import argparse
import json

from interphyre import InterphyreEnv, SimulationConfig
from interphyre.render.pygame import PygameRenderer
from interphyre.render.video import VideoRecorder, generate_video_filename


def _make_renderer(
    level_name: str,
    seed: int,
    label: str,
    record_video: bool,
    video_format: str,
    output_dir: str,
):
    """Return a renderer — VideoRecorder if recording, PygameRenderer otherwise."""
    if record_video:
        video_path = generate_video_filename(
            level_name, seed, output_dir, video_format, label=label
        )
        print(f"Recording to: {video_path}")
        return VideoRecorder(
            width=600,
            height=600,
            ppm=60,
            video_format=video_format,
            fps=60,
            output_path=video_path,
        )
    return PygameRenderer(width=600, height=600, ppm=60)


def view_action(
    level_name: str,
    seed: int,
    action: list[float] | tuple[float, float, float],
    pause_time: float = 2.0,
    record_video: bool = False,
    video_format: str = "mp4",
    output_dir: str = "outputs",
) -> bool:
    """Replay a specific placement on a level.

    Returns True if the level was solved.
    """
    if not (isinstance(action, (list, tuple)) and len(action) == 3):
        raise ValueError(f"Action must be [x, y, r] or (x, y, r), got {action}")
    action = tuple(action)

    print(f"Viewing {level_name} (seed={seed}): {action}")

    renderer = _make_renderer(
        level_name, seed, "action", record_video, video_format, output_dir
    )
    config = SimulationConfig(fps=60, time_step=1 / 60)
    env = InterphyreEnv(level_name, seed=seed, config=config)
    env.renderer = renderer

    try:
        env.reset()
        obs, reward, terminated, truncated, info = env.step([action])
        success = info.get("success", False)
        print(f"Result: {'SUCCESS' if success else 'FAIL'} (reward={reward})")
        if not record_video and hasattr(renderer, "wait"):
            renderer.wait(int(pause_time * 1000))
        return success
    finally:
        renderer.close()
        env.close()


def view_bundle_solution(
    level_name: str,
    seed: int,
    pause_time: float = 2.0,
    record_video: bool = False,
    video_format: str = "mp4",
    output_dir: str = "outputs",
) -> bool:
    """Replay the certified solution stored in the bundle for (level_name, seed).

    Raises ValueError if no valid solution exists for this pair (impossible seed
    or oracle-only level with no stored placement coordinates).

    Returns True if the replay succeeded (should always be True for valid entries).
    """
    from interphyre.validation import _get_registry

    registry = _get_registry()
    entry = registry.get_valid_entry(level_name, seed)

    if entry is None or entry.get("status") != "valid":
        raise ValueError(
            f"No valid bundle solution for {level_name} seed={seed}. "
            "Only seeds 0–10000 are bundled; impossible seeds have no solution."
        )
    solution = entry["solution"]
    if solution is None:
        raise ValueError(
            f"Bundle entry for {level_name} seed={seed} has no stored solution "
            "(oracle-only level — placement coordinates were not recorded)."
        )

    print(f"Playing bundled solution for {level_name} seed={seed}: {solution}")

    renderer = _make_renderer(
        level_name, seed, "solution", record_video, video_format, output_dir
    )
    config = SimulationConfig(fps=60, time_step=1 / 60)
    env = InterphyreEnv(level_name, seed=seed, config=config)
    env.renderer = renderer

    try:
        env.reset()
        _, reward, _, _, info = env.step(solution)
        success = info.get("success", False)
        print(f"Result: {'SUCCESS' if success else 'FAIL'} (reward={reward})")
        if not record_video and hasattr(renderer, "wait"):
            renderer.wait(int(pause_time * 1000))
        return success
    finally:
        renderer.close()
        env.close()


def view_solutions_from_file(
    solutions_file: str,
    level_name: str | None = None,
    pause_time: float = 2.0,
    record_video: bool = False,
    video_format: str = "mp4",
    output_dir: str = "outputs",
) -> None:
    """Replay solutions from a JSON file.

    The file must contain a list of entries, each with keys:
        {"level": str, "seed": int, "action": [x, y, r]}

    If level_name is given, only entries for that level are replayed.
    If level_name is None, all entries in the file are replayed.
    """
    with open(solutions_file) as fh:
        solutions = json.load(fh)

    entries = [
        sol for sol in solutions if level_name is None or sol.get("level") == level_name
    ]

    scope = f"{level_name}" if level_name else "all levels"
    print(f"Replaying {len(entries)} solution(s) from {solutions_file} ({scope})")

    successful = 0
    for i, sol in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}]")
        success = view_action(
            level_name=sol["level"],
            seed=sol["seed"],
            action=sol["action"],
            pause_time=pause_time,
            record_video=record_video,
            video_format=video_format,
            output_dir=output_dir,
        )
        if success:
            successful += 1

    print(f"\nResults: {successful}/{len(entries)} successful")


def run_random_demo(
    level_name: str,
    seed: int | None = None,
    max_trials: int = 20,
    pause_time: float = 1.0,
    record_video: bool = False,
    video_format: str = "mp4",
    output_dir: str = "outputs",
):
    """Run a random-placement search on level_name, up to max_trials attempts."""
    import numpy as np

    rng = np.random.default_rng(seed)
    level_seed = seed if seed is not None else int(rng.integers(0, 100000))

    print(f"Random demo: {level_name} (max {max_trials} trials)")

    config = SimulationConfig(fps=60, time_step=1 / 60)
    if record_video:
        renderer = _make_renderer(
            level_name, level_seed, "demo", record_video, video_format, output_dir
        )
        env = InterphyreEnv(level_name, seed=level_seed, config=config)
        env.renderer = renderer
    else:
        env = InterphyreEnv(
            level_name, seed=level_seed, config=config, render_mode="human"
        )

    try:
        for trial in range(1, max_trials + 1):
            env.reset()
            env.render()

            # Sample until a geometrically valid placement is found.
            for _ in range(100):
                candidate = env.action_space.sample()
                action = (float(candidate[0]), float(candidate[1]), float(candidate[2]))
                try:
                    obs, reward, terminated, truncated, info = env.step([action])
                except ValueError:
                    continue
                if not info.get("invalid_action", False):
                    break
            else:
                print("Could not find a valid placement after 100 attempts")
                break

            success = info.get("success", False)
            print(f"\nTrial {trial}: {action}")
            print(f"  {'SUCCESS' if success else 'FAIL'} (reward={reward})")

            if success:
                print(f"\nSolved in {trial} trials!")
                if not record_video and hasattr(env.renderer, "wait"):
                    env.renderer.wait(int(pause_time * 2000))
                break

            if not record_video and hasattr(env.renderer, "wait"):
                env.renderer.wait(int(pause_time * 1000))
    finally:
        env.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interphyre viewer — replay actions, bundle solutions, or file outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # replay the certified bundle solution for a seed (--bundle is the default)
  python -m interphyre.viewer catapult --seed 42
  python -m interphyre.viewer catapult --seed 42 --bundle

  # replay a specific placement
  python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.5

  # replay all solutions written to a file by an agent
  python -m interphyre.viewer --file results.json

  # replay file entries for one level only
  python -m interphyre.viewer catapult --file results.json

  # random-placement demo
  python -m interphyre.viewer catapult --demo --trials 20
""",
    )
    parser.add_argument(
        "level",
        nargs="?",
        default=None,
        help=(
            "Level name. Required for --action, --bundle, and --demo. "
            "Optional filter for --file (omit to replay all levels in the file)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--action",
        nargs=3,
        type=float,
        metavar=("X", "Y", "R"),
        help="Replay a specific placement: x y radius",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Replay the certified bundle solution for LEVEL --seed",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Replay solutions from a JSON file written by an agent",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a random-placement search on LEVEL",
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Max trials for --demo (default: 20)"
    )
    parser.add_argument(
        "--pause", type=float, default=2.0, help="Pause after each replay (seconds)"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record video instead of displaying"
    )
    parser.add_argument(
        "--format",
        default="mp4",
        choices=["mp4", "gif"],
        help="Video format (default: mp4)",
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Output directory for recorded videos"
    )

    args = parser.parse_args()

    # Validate: --action, --bundle, --demo all require a level name.
    if (args.action or args.bundle or args.demo) and args.level is None:
        parser.error("LEVEL is required for --action, --bundle, and --demo")

    # Default: level + seed with no explicit mode → replay the bundle solution.
    if args.level and not any([args.action, args.bundle, args.file, args.demo]):
        args.bundle = True

    if args.file:
        view_solutions_from_file(
            args.file,
            level_name=args.level,
            pause_time=args.pause,
            record_video=args.record,
            video_format=args.format,
            output_dir=args.output_dir,
        )
    elif args.bundle:
        view_bundle_solution(
            args.level,
            args.seed,
            args.pause,
            args.record,
            args.format,
            args.output_dir,
        )
    elif args.action:
        view_action(
            args.level,
            args.seed,
            args.action,
            args.pause,
            args.record,
            args.format,
            args.output_dir,
        )
    elif args.demo:
        run_random_demo(
            args.level,
            args.seed,
            args.trials,
            args.pause,
            args.record,
            args.format,
            args.output_dir,
        )
    else:
        parser.error("Specify one of: --action, --bundle, --file, --demo")


if __name__ == "__main__":
    main()
