from interphyre.render.base import (
    COLORS as COLORS,
    DISCRETE_COLORS as DISCRETE_COLORS,
    RGB_TO_DISCRETE as RGB_TO_DISCRETE,
    Renderer as Renderer,
)
from interphyre.render.opencv import OpenCVRenderer as OpenCVRenderer
from interphyre.render.pygame import PygameRenderer as PygameRenderer
from interphyre.render.video import VideoRecorder as VideoRecorder


def save_obs_as_image(obs, filename, image_size=None):
    """Convert observation to RGB image and save to file.

    Works with both discrete and RGB observations:
    - Discrete observations (single-channel, values 0-7) are converted to RGB
    - RGB observations are saved directly

    Args:
        obs: Observation array (discrete or RGB)
        filename: Output filename (e.g., 'observation.png')
        image_size: Optional tuple (width, height). If None, uses obs.shape

    Example:
        env = InterphyreEnv(level=level, observation_type='image', discrete_colors=True)
        obs, reward, terminated, truncated, info = env.step(action)
        save_obs_as_image(obs, 'my_observation.png')
    """
    import cv2

    if image_size is None:
        height, width = obs.shape[:2]
    else:
        width, height = image_size

    if len(obs.shape) == 2:  # Discrete observation
        renderer = OpenCVRenderer(width=width, height=height, ppm=60)
        rgb_image = renderer.discrete_to_rgb(obs)
        cv2.imwrite(filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        renderer.close()
    else:  # RGB observation
        cv2.imwrite(filename, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
