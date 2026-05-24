import pygame
from interphyre.render.base import Renderer, COLORS
from Box2D import b2PolygonShape, b2CircleShape


class PygameRenderer(Renderer):
    """Pygame-based renderer for visualizing physics simulations in real-time.

    Attributes:
        width (int): Width of the rendering window in pixels
        height (int): Height of the rendering window in pixels
        ppm (float): Pixels per Box2D unit (scaling factor)
        screen: Pygame screen surface for drawing
        clock: Pygame clock for frame rate control
        fps (int): Target frames per second for rendering
    """

    def __init__(self, width: int = 600, height: int = 600, ppm: float = 60):
        """Initialize the Pygame renderer.

        Args:
            width: Width of the window in pixels (default: 600)
            height: Height of the window in pixels (default: 600)
            ppm: Pixels per Box2D unit (scaling factor) (default: 60)
        """
        pygame.init()
        self.width = width
        self.height = height
        self.ppm = ppm
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Interphyre Simulation")
        self.clock = pygame.time.Clock()
        self.fps = 60
        self._closed = False

    def render(self, engine) -> None:
        """
        Render the current state of the simulation.

        Each fixture is rendered after applying the body transform to its local coordinates.
        Returns immediately if the window has been closed.
        """
        if self._closed:
            return

        self.screen.fill(COLORS["white"])

        # Sort bodies by y-position (bottom to top) so objects above are drawn last.
        sorted_bodies = sorted(
            engine.bodies.items(), key=lambda item: item[1].position.y
        )

        for name, body in sorted_bodies:
            color = self._get_object_color(body, engine)
            if color is None:
                continue
            for fixture in body.fixtures:
                # Do not render sensor fixtures, they are only used for detection and measurement purposes
                if fixture.sensor:
                    continue

                shape = fixture.shape
                if isinstance(shape, b2CircleShape):
                    position = body.transform * shape.pos
                    radius = max(1, round(shape.radius * self.ppm))
                    screen_pos = self.world_to_screen((position[0], position[1]))
                    pygame.draw.circle(
                        self.screen,
                        color,
                        screen_pos,
                        radius,
                    )
                elif isinstance(shape, b2PolygonShape):
                    vertices = [body.transform * v for v in shape.vertices]
                    pts = [self.world_to_screen((v[0], v[1])) for v in vertices]
                    pygame.draw.polygon(self.screen, color, pts)
                else:
                    raise ValueError(f"Unsupported shape type: {type(shape)}")

        pygame.display.flip()
        pygame.event.pump()
        self.clock.tick(self.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

    def close(self) -> None:
        self._closed = True
        pygame.quit()

    def wait(self, duration: int) -> None:
        """Wait for specified duration while processing pygame events to keep window responsive."""
        if self._closed:
            return
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            pygame.time.wait(10)
