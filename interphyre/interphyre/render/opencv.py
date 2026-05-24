import cv2
import numpy as np
from interphyre.render.base import Renderer, DISCRETE_COLORS, RGB_TO_DISCRETE
from Box2D import b2PolygonShape, b2CircleShape


class OpenCVRenderer(Renderer):
    """OpenCV-based renderer for generating images of physics simulations without real-time display.

    Attributes:
        width (int): Width of the rendered image in pixels
        height (int): Height of the rendered image in pixels
        ppm (float): Pixels per Box2D unit (scaling factor)
        image (np.ndarray): Current rendered image buffer
    """

    def __init__(self, width: int = 600, height: int = 600, ppm: float = 60):
        """Initialize the OpenCV renderer.

        Args:
            width: Width of the image in pixels (default: 600)
            height: Height of the image in pixels (default: 600)
            ppm: Pixels per Box2D unit (scaling factor) (default: 60)
        """
        self.width = width
        self.height = height
        self.ppm = ppm
        self.image = np.zeros((height, width, 3), dtype=np.uint8)

    def render(self, engine) -> np.ndarray:
        """
        Render the current state of the simulation to an image.

        Each fixture is rendered after applying the body transform to its local coordinates.

        Args:
            engine: The Box2DEngine containing the physics world to render

        Returns:
            np.ndarray: RGB image as numpy array (height, width, 3)
        """
        self.image.fill(255)

        # Sort bodies by y-position (bottom to top) so objects above are drawn last
        sorted_bodies = sorted(
            engine.bodies.items(), key=lambda item: item[1].position.y
        )

        for name, body in sorted_bodies:
            color = self._get_object_color(body, engine)
            if color is None:
                continue
            bgr_color = (color[2], color[1], color[0])

            for fixture in body.fixtures:
                if fixture.sensor:
                    continue

                shape = fixture.shape
                if isinstance(shape, b2CircleShape):
                    position = body.transform * shape.pos
                    radius = max(1, round(shape.radius * self.ppm))
                    screen_pos = self.world_to_screen((position[0], position[1]))
                    cv2.circle(self.image, screen_pos, radius, bgr_color, -1)

                elif isinstance(shape, b2PolygonShape):
                    vertices = [body.transform * v for v in shape.vertices]
                    pts = np.array(
                        [self.world_to_screen((v[0], v[1])) for v in vertices],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(self.image, [pts], bgr_color)
                else:
                    raise ValueError(f"Unsupported shape type: {type(shape)}")

        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def render_discrete(self, engine) -> np.ndarray:
        """Render simulation state to discrete color image.

        Returns single-channel image where pixel values represent:
        - 0: Background
        - 1-7: Object colors (green, red, blue, black, gray, purple)

        Args:
            engine: Physics engine containing simulation state

        Returns:
            Single-channel discrete color image (height, width)
        """
        discrete_image = np.zeros((self.height, self.width), dtype=np.uint8)

        sorted_bodies = sorted(
            engine.bodies.items(), key=lambda item: item[1].position.y
        )

        for name, body in sorted_bodies:
            color = self._get_object_color(body, engine)
            if color is None:
                continue
            discrete_idx = RGB_TO_DISCRETE.get(color, 0)

            for fixture in body.fixtures:
                if fixture.sensor:
                    continue

                shape = fixture.shape
                if isinstance(shape, b2CircleShape):
                    position = body.transform * shape.pos
                    radius = max(1, round(shape.radius * self.ppm))
                    screen_pos = self.world_to_screen((position[0], position[1]))
                    cv2.circle(discrete_image, screen_pos, radius, discrete_idx, -1)  # type: ignore

                elif isinstance(shape, b2PolygonShape):
                    vertices = [body.transform * v for v in shape.vertices]
                    pts = np.array(
                        [self.world_to_screen((v[0], v[1])) for v in vertices],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(discrete_image, [pts], discrete_idx)  # type: ignore
                else:
                    raise ValueError(f"Unsupported shape type: {type(shape)}")

        return discrete_image

    def discrete_to_rgb(self, discrete_image: np.ndarray) -> np.ndarray:
        """Convert discrete color image to RGB.

        Args:
            discrete_image: Single-channel discrete color image

        Returns:
            RGB image (height, width, 3)
        """
        height, width = discrete_image.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        for idx, rgb_color in DISCRETE_COLORS.items():
            mask = discrete_image == idx
            rgb_image[mask] = rgb_color

        return rgb_image

    def close(self) -> None:
        """Close and clean up renderer resources."""
        pass

    def wait(self, duration: int) -> None:
        """Wait for specified duration (compatibility method)."""
        import time

        time.sleep(duration / 1000.0)
