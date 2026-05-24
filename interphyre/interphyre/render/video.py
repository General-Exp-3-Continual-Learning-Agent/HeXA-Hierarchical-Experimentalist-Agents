from __future__ import annotations

import logging
import os
import argparse
import json
import cv2
import numpy as np
from interphyre.render.opencv import OpenCVRenderer
from interphyre.render.base import Renderer

logger = logging.getLogger(__name__)


class VideoRecorder(Renderer):
    """Renderer that captures frames during simulation and saves them as video.

    This renderer wraps OpenCVRenderer and captures frames during render() calls,
    then saves them as MP4 or GIF files. Designed for headless operation on servers.

    Attributes:
        width (int): Width of the video in pixels
        height (int): Height of the video in pixels
        ppm (float): Pixels per Box2D unit (scaling factor)
        video_format (str): Output format ('mp4' or 'gif')
        fps (int): Target frames per second for the video
        output_path (str): Path where the video will be saved
        frames (list): List of captured frames
        opencv_renderer (OpenCVRenderer): Internal OpenCV renderer
    """

    def __init__(
        self,
        width: int = 600,
        height: int = 600,
        ppm: float = 60,
        video_format: str = "mp4",
        fps: int = 30,
        output_path: str | None = None,
    ):
        """Initialize the video recorder.

        Args:
            width: Width of the video in pixels (default: 600)
            height: Height of the video in pixels (default: 600)
            ppm: Pixels per Box2D unit (scaling factor) (default: 60)
            video_format: Output format, 'mp4' or 'gif' (default: 'mp4')
            fps: Target frames per second for the video (default: 30)
            output_path: Path where the video will be saved. If None, must be set via set_output_path()
        """
        self.width = width
        self.height = height
        self.ppm = ppm
        self.video_format = video_format.lower()
        self.fps = fps
        self.output_path = output_path
        self.frames: list[np.ndarray] = []
        self._closed = False
        self.opencv_renderer = OpenCVRenderer(width=width, height=height, ppm=ppm)

        if self.video_format not in ["mp4", "gif"]:
            raise ValueError(
                f"Unsupported video format: {video_format}. Use 'mp4' or 'gif'"
            )

    def set_output_path(self, path: str) -> None:
        """Set the output path for the video file.

        Args:
            path: Path where the video will be saved
        """
        self.output_path = path

    def render(self, engine) -> None:
        """Render the current state and capture the frame.

        Args:
            engine: The Box2DEngine containing the physics world to render
        """
        frame = self.opencv_renderer.render(engine)
        # Frame is RGB (height × width × 3); copy to avoid aliasing with the renderer buffer.
        self.frames.append(frame.copy())

    def close(self) -> None:
        """Close the recorder and save the video file.

        This method is idempotent - it's safe to call multiple times.
        After the first call, subsequent calls will do nothing.
        """
        if self._closed:
            return

        if not self.frames:
            if self.output_path:
                logger.warning("No frames captured, skipping video save")
            self._closed = True
            return

        if not self.output_path:
            raise ValueError(
                f"VideoRecorder has {len(self.frames)} captured frames but output_path "
                "is None. Call set_output_path() before close(), or pass output_path to "
                "the constructor."
            )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        if self.video_format == "mp4":
            self._save_mp4()
        elif self.video_format == "gif":
            self._save_gif()

        self.output_path = None
        self.opencv_renderer.close()
        self.frames.clear()
        self._closed = True

    def _save_mp4(self) -> None:
        """Save frames as MP4 video using OpenCV VideoWriter."""
        # output_path is guaranteed to be non-None by close() method
        assert self.output_path is not None, (
            "output_path must be set before calling _save_mp4"
        )
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            self.output_path, fourcc, float(self.fps), (self.width, self.height)
        )

        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.output_path}")

        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"Saved MP4 video to: {self.output_path} ({len(self.frames)} frames)")

    def _save_gif(self) -> None:
        """Save frames as GIF using imageio (with Pillow fallback)."""
        # output_path is guaranteed to be non-None by close() method
        assert self.output_path is not None, (
            "output_path must be set before calling _save_gif"
        )

        # Convert frames to uint8 if needed and ensure they're in the right format
        frames_uint8 = []
        for frame in self.frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames_uint8.append(frame)

        try:
            import imageio

            # Save as GIF
            # Note: GIF format stores frame delays in centiseconds (1/100th second),
            # so exact frame rates like 60 FPS may be slightly approximated.
            # This is acceptable as long as playback speed is reasonable.
            imageio.mimsave(  # type: ignore[arg-type]
                self.output_path,
                frames_uint8,
                fps=self.fps,
                loop=0,  # Loop forever
            )
        except ImportError:
            from PIL import Image

            frames = [Image.fromarray(frame) for frame in frames_uint8]
            duration_ms = int(1000 / max(self.fps, 1))
            frames[0].save(
                self.output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )

        print(f"Saved GIF to: {self.output_path} ({len(self.frames)} frames)")

    def wait(self, duration: int) -> None:
        """Wait for specified duration (compatibility method).

        Args:
            duration: Duration in milliseconds
        """
        import time

        time.sleep(duration / 1000.0)

    def get_frame_count(self) -> int:
        """Get the number of frames captured so far.

        Returns:
            Number of frames captured
        """
        return len(self.frames)


def generate_video_filename(
    level_name: str,
    seed: int | None = None,
    output_dir: str = "outputs",
    video_format: str = "mp4",
    label: str | None = None,
) -> str:
    """Generate a filename for a video file.

    Args:
        level_name: Name of the level
        seed: Optional seed value
        output_dir: Base output directory (default: outputs)
        video_format: Video format ('mp4' or 'gif')
        label: Optional label (e.g., 'success' or 'failure')

    Returns:
        Full path to the video file
    """
    format_dir = os.path.join(output_dir, video_format)
    os.makedirs(format_dir, exist_ok=True)

    parts = [level_name]
    if seed is not None:
        parts.append(str(seed))
    if label:
        parts.append(label)

    filename = "_".join(parts) + f".{video_format}"
    return os.path.join(format_dir, filename)


def get_all_levels(data_dir: str = "data") -> list[str]:
    """Get all level names from the data directory that have both successes and failures.

    Args:
        data_dir: Directory containing level data (default: "data")

    Returns:
        List of level names that have both successes.json and failures.json files.
    """
    if not os.path.exists(data_dir):
        return []

    levels = []
    for item in os.listdir(data_dir):
        level_dir = os.path.join(data_dir, item)
        if os.path.isdir(level_dir):
            successes_path = os.path.join(level_dir, "successes.json")
            failures_path = os.path.join(level_dir, "failures.json")
            if os.path.exists(successes_path) and os.path.exists(failures_path):
                levels.append(item)

    return sorted(levels)


def get_first_seed_from_file(solutions_file: str, level_name: str) -> int | None:
    """Get the first seed from a solutions file.

    Args:
        solutions_file: Path to the solutions JSON file
        level_name: Name of the level

    Returns:
        First seed value as an integer, or None if not found.
    """
    if not os.path.exists(solutions_file):
        return None

    with open(solutions_file, "r") as f:
        solutions_data = json.load(f)

    if level_name not in solutions_data:
        return None

    level_data = solutions_data[level_name]
    if "solutions" not in level_data:
        return None

    solutions = level_data["solutions"]
    if not solutions:
        return None

    # Get the first seed (sorted for consistency)
    first_seed_str = sorted(solutions.keys(), key=int)[0]
    return int(first_seed_str)


def export_videos_for_level(
    level_name: str,
    data_dir: str,
    output_dir: str,
    video_fps: int = 30,
    formats: list[str] | None = None,
):
    """Export videos for both success and failure solutions for a level.

    Args:
        level_name: Name of the level to export
        data_dir: Directory containing level data
        output_dir: Base output directory for videos
        video_fps: Video frame rate (default: 30)
        formats: List of video formats to export (default: ["mp4", "gif"])
    """
    if formats is None:
        formats = ["mp4", "gif"]

    from interphyre.viewer._viewer import visualize_solution_from_file

    level_dir = os.path.join(data_dir, level_name)
    successes_file = os.path.join(level_dir, "successes.json")
    failures_file = os.path.join(level_dir, "failures.json")

    if not os.path.exists(successes_file) or not os.path.exists(failures_file):
        print(f"Skipping {level_name}: missing solutions files")
        return

    # Get first seed from each file
    success_seed = get_first_seed_from_file(successes_file, level_name)
    failure_seed = get_first_seed_from_file(failures_file, level_name)

    if success_seed is None or failure_seed is None:
        print(f"Skipping {level_name}: no seeds found")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing level: {level_name}")
    print(f"{'=' * 60}")

    # Export success videos
    print(f"\nExporting SUCCESS videos (seed {success_seed})...")
    for fmt in formats:
        print(f"  - {fmt.upper()}...")
        try:
            visualize_solution_from_file(
                solutions_file=successes_file,
                level_name=level_name,
                seed=success_seed,
                pause_time=0.0,
                record_video=True,
                video_format=fmt,
                video_fps=video_fps,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"    Error exporting {fmt}: {e}")

    # Export failure videos
    print(f"\nExporting FAILURE videos (seed {failure_seed})...")
    for fmt in formats:
        print(f"  - {fmt.upper()}...")
        try:
            visualize_solution_from_file(
                solutions_file=failures_file,
                level_name=level_name,
                seed=failure_seed,
                pause_time=0.0,
                record_video=True,
                video_format=fmt,
                video_fps=video_fps,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"    Error exporting {fmt}: {e}")


def main():
    """CLI entry point for exporting videos for all levels."""
    parser = argparse.ArgumentParser(
        description="Export MP4 and GIF videos for success and failure solutions for all levels"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing level data (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory for videos (default: outputs/). Videos will be saved in outputs/mp4/ or outputs/gif/",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["mp4", "gif"],
        default=["mp4", "gif"],
        help="Video formats to export (default: mp4 gif)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=str,
        help="Specific levels to export (default: all levels)",
    )

    args = parser.parse_args()

    # Get levels to process
    if args.levels:
        levels = args.levels
    else:
        levels = get_all_levels(args.data_dir)

    if not levels:
        print(f"No levels found in {args.data_dir}")
        return

    print(f"Found {len(levels)} levels to process")
    print(f"Formats: {', '.join(args.formats)}")
    print(f"Output directory: {args.output_dir}")

    # Process each level
    successful = 0
    failed = 0

    for i, level_name in enumerate(levels, 1):
        print(f"\n[{i}/{len(levels)}] Processing {level_name}...")
        try:
            export_videos_for_level(
                level_name,
                args.data_dir,
                args.output_dir,
                args.video_fps,
                args.formats,
            )
            successful += 1
        except Exception as e:
            print(f"Error processing {level_name}: {e}")
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPORT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total levels: {len(levels)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
