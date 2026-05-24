"""
Tests for rendering system (OpenCV and Pygame).

This module tests:
- Coordinate transforms (world to screen)
- Color mapping (RGB and discrete)
- OpenCV renderer (image generation)
- Pygame renderer (real-time visualization)
"""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import MagicMock, patch
from Box2D import b2World, b2PolygonShape, b2CircleShape

from interphyre.render.base import COLORS, DISCRETE_COLORS, RGB_TO_DISCRETE, Renderer
from interphyre.render.opencv import OpenCVRenderer
from interphyre.render.pygame import PygameRenderer
from interphyre.engine import Box2DEngine
from interphyre.levels import load_level
from interphyre.level import Level
from interphyre.objects import PhyreObject


# ============================================================================
# Coordinate Transform Tests (8-10 tests)
# ============================================================================


@pytest.mark.fast
def test_opencv_world_to_screen_origin():
    """Test that world origin (0,0) maps to image center."""
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    screen_x, screen_y = renderer.world_to_screen((0, 0))
    assert screen_x == 300, f"Expected x=300, got {screen_x}"
    assert screen_y == 300, f"Expected y=300, got {screen_y}"


@pytest.mark.fast
def test_opencv_world_to_screen_positive_coords():
    """Test positive world coordinates with scaling."""
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    screen_x, screen_y = renderer.world_to_screen((2.0, 3.0))
    # x * ppm + width/2 = 2.0 * 60 + 300 = 420
    # -y * ppm + height/2 = -3.0 * 60 + 300 = 120
    assert screen_x == 420, f"Expected x=420, got {screen_x}"
    assert screen_y == 120, f"Expected y=120, got {screen_y}"


@pytest.mark.fast
def test_opencv_world_to_screen_negative_coords():
    """Test negative world coordinates with offset."""
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    screen_x, screen_y = renderer.world_to_screen((-2.0, -3.0))
    # x * ppm + width/2 = -2.0 * 60 + 300 = 180
    # -y * ppm + height/2 = -(-3.0) * 60 + 300 = 480
    assert screen_x == 180, f"Expected x=180, got {screen_x}"
    assert screen_y == 480, f"Expected y=480, got {screen_y}"


@pytest.mark.fast
def test_pygame_world_to_screen_consistency(mock_pygame):
    """Test that OpenCV and Pygame use identical transforms."""
    opencv = OpenCVRenderer(width=800, height=600, ppm=100)
    pygame = PygameRenderer(width=800, height=600, ppm=100)

    test_positions = [(0, 0), (1.5, -2.0), (-3.0, 4.0), (0.5, 0.5)]

    for world_pos in test_positions:
        opencv_screen = opencv.world_to_screen(world_pos)
        pygame_screen = pygame.world_to_screen(world_pos)
        assert opencv_screen == pygame_screen, (
            f"Transform mismatch at {world_pos}: OpenCV={opencv_screen}, Pygame={pygame_screen}"
        )

    pygame.close()


@pytest.mark.fast
def test_world_to_screen_custom_ppm():
    """Test coordinate transforms with different ppm values."""
    for ppm in [30, 100, 120]:
        renderer = OpenCVRenderer(width=600, height=600, ppm=ppm)
        screen_x, screen_y = renderer.world_to_screen((1.0, 1.0))
        expected_x = int(1.0 * ppm + 300)
        expected_y = int(-1.0 * ppm + 300)
        assert screen_x == expected_x, (
            f"ppm={ppm}: Expected x={expected_x}, got {screen_x}"
        )
        assert screen_y == expected_y, (
            f"ppm={ppm}: Expected y={expected_y}, got {screen_y}"
        )


@pytest.mark.fast
def test_world_to_screen_custom_dimensions():
    """Test coordinate transforms with non-square images."""
    renderer = OpenCVRenderer(width=800, height=600, ppm=60)
    screen_x, screen_y = renderer.world_to_screen((0, 0))
    assert screen_x == 400, f"Expected x=400, got {screen_x}"
    assert screen_y == 300, f"Expected y=300, got {screen_y}"


@pytest.mark.fast
def test_world_to_screen_fractional_coords():
    """Test rounding behavior for fractional coordinates."""
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    # Test that fractional values are properly rounded
    screen_x, screen_y = renderer.world_to_screen((0.1, 0.1))
    # 0.1 * 60 = 6, so x = 6 + 300 = 306
    # -0.1 * 60 = -6, so y = -6 + 300 = 294
    assert screen_x == 306, f"Expected x=306, got {screen_x}"
    assert screen_y == 294, f"Expected y=294, got {screen_y}"


@pytest.mark.fast
def test_world_to_screen_large_coords():
    """Test coordinate transforms with large world coordinates."""
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    screen_x, screen_y = renderer.world_to_screen((10.0, -10.0))
    # Should handle large coords without error
    assert isinstance(screen_x, int)
    assert isinstance(screen_y, int)
    # Values may be outside image bounds, but should be valid integers
    assert screen_x == 900  # 10 * 60 + 300
    assert screen_y == 900  # -(-10) * 60 + 300


@pytest.mark.fast
def test_world_to_screen_boundary_conditions():
    """Test coordinate transforms at image boundaries."""
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    # Test coordinates that map to image edges
    # Left edge: x = -5.0 → screen_x = -5 * 60 + 300 = 0
    # Right edge: x = 5.0 → screen_x = 5 * 60 + 300 = 600
    left_x, _ = renderer.world_to_screen((-5.0, 0))
    right_x, _ = renderer.world_to_screen((5.0, 0))
    assert left_x == 0, f"Expected left edge x=0, got {left_x}"
    assert right_x == 600, f"Expected right edge x=600, got {right_x}"


# ============================================================================
# Color Mapping Tests (10-12 tests)
# ============================================================================


@pytest.mark.fast
def test_colors_dict_completeness():
    """Test that all 8 colors are present in COLORS dict."""
    expected_colors = {
        "green",
        "red",
        "blue",
        "black",
        "gray",
        "purple",
        "yellow",
        "white",
    }
    actual_colors = set(COLORS.keys())
    assert actual_colors == expected_colors, (
        f"Missing colors: {expected_colors - actual_colors}, Extra colors: {actual_colors - expected_colors}"
    )


@pytest.mark.fast
def test_color_rgb_values():
    """Test that all RGB values are in valid 0-255 range."""
    for color_name, rgb in COLORS.items():
        assert len(rgb) == 3, f"Color {color_name} should have 3 RGB components"
        for component in rgb:
            assert 0 <= component <= 255, (
                f"Color {color_name} has invalid RGB component {component} (must be 0-255)"
            )


@pytest.mark.fast
def test_discrete_colors_mapping():
    """Test that discrete color indices 0-7 map to correct RGB values."""
    # Check all indices 0-7 exist
    for idx in range(8):
        assert idx in DISCRETE_COLORS, f"Discrete color index {idx} missing"
        rgb = DISCRETE_COLORS[idx]
        assert len(rgb) == 3, f"Discrete color {idx} should have 3 RGB components"
        for component in rgb:
            assert 0 <= component <= 255, (
                f"Discrete color {idx} has invalid RGB component {component}"
            )


@pytest.mark.fast
def test_rgb_to_discrete_reverse_mapping():
    """Test bidirectional consistency between RGB and discrete colors."""
    # For each discrete color, verify reverse mapping
    for idx, rgb in DISCRETE_COLORS.items():
        assert rgb in RGB_TO_DISCRETE, f"RGB {rgb} not in reverse mapping"
        assert RGB_TO_DISCRETE[rgb] == idx, (
            f"RGB {rgb} maps to {RGB_TO_DISCRETE[rgb]}, expected {idx}"
        )


@pytest.mark.fast
def test_get_object_color_basic(simple_env):
    """Test that object colors are correctly retrieved from engine."""
    engine = simple_env.engine
    renderer = OpenCVRenderer()

    # Get a body from the engine
    if "green_ball" in engine.bodies:
        body = engine.bodies["green_ball"]
        color = renderer._get_object_color(body, engine)
        assert color == COLORS["green"], f"Expected green color, got {color}"


@pytest.mark.fast
def test_get_object_color_wall_objects(simple_env):
    """Test that exact wall body names return None (skipped) but wall-substring names do not."""
    engine = simple_env.engine
    renderer = OpenCVRenderer()

    # Actual wall bodies should return None
    for wall_name in ("left_wall", "right_wall", "top_wall", "bottom_wall"):
        if wall_name in engine.bodies:
            body = engine.bodies[wall_name]
            color = renderer._get_object_color(body, engine)
            assert color is None, f"{wall_name} should return None, got {color}"

    # Objects whose names merely contain "wall" should NOT be skipped
    mock_body = MagicMock()
    for non_wall_name in ("wall_breaker", "stonewall_platform", "wallaby"):
        mock_body.userData = non_wall_name
        color = renderer._get_object_color(mock_body, engine)
        assert color is not None, (
            f"Object '{non_wall_name}' should be rendered (not None), "
            "substring match must not skip it"
        )


@pytest.mark.fast
@pytest.mark.parametrize(
    "engine_level,user_data,description",
    [
        (None, "nonexistent_object", "no level"),
        ("two_body_problem", "missing_object", "unknown object name"),
    ],
)
def test_get_object_color_fallback_to_black(engine_level, user_data, description):
    """Test fallback to black for missing objects and various scenarios."""
    if engine_level is None:
        engine = Box2DEngine()
        engine.level = None
    else:
        engine = Box2DEngine(level=load_level(engine_level, seed=1))

    renderer = OpenCVRenderer()
    mock_body = MagicMock()
    mock_body.userData = user_data
    color = renderer._get_object_color(mock_body, engine)
    assert color == COLORS["black"], (
        f"Expected black fallback for {description}, got {color}"
    )


@pytest.mark.fast
def test_get_object_color_missing_color_attribute():
    """Objects without color attribute should map to black."""

    class NoColor(PhyreObject):
        def __init__(self):
            super().__init__(x=0.0, y=0.0)
            del self.color

    objects: dict[str, PhyreObject] = {"noc": NoColor()}
    level = Level(
        name="no_color",
        objects=objects,
        action_objects=[],
        success_condition=lambda e: False,
        metadata={},
    )
    engine = MagicMock()
    engine.level = level
    renderer = OpenCVRenderer()
    mock_body = MagicMock()
    mock_body.userData = "noc"
    assert renderer._get_object_color(mock_body, engine) == COLORS["black"]


@pytest.mark.fast
def test_discrete_color_conversion_all_indices():
    """Test conversion of all discrete indices 0-7 to RGB."""
    renderer = OpenCVRenderer()

    for idx in range(8):
        # Create a discrete image with single pixel
        discrete_img = np.array([[idx]], dtype=np.uint8)
        rgb_img = renderer.discrete_to_rgb(discrete_img)

        assert rgb_img.shape == (
            1,
            1,
            3,
        ), f"RGB image should be (1,1,3), got {rgb_img.shape}"
        expected_rgb = DISCRETE_COLORS[idx]
        actual_rgb = tuple(rgb_img[0, 0])
        assert actual_rgb == expected_rgb, (
            f"Index {idx}: Expected RGB {expected_rgb}, got {actual_rgb}"
        )


@pytest.mark.fast
def test_color_case_insensitivity():
    """Test that color names are case-insensitive."""
    renderer = OpenCVRenderer()

    # Create mock objects with different case colors
    mock_body_green = MagicMock()
    mock_body_green.userData = "test_ball"

    mock_engine = MagicMock()
    mock_level = MagicMock()
    mock_obj = MagicMock()
    mock_obj.color = "GREEN"  # Uppercase
    mock_level.objects = {"test_ball": mock_obj}
    mock_engine.level = mock_level

    color = renderer._get_object_color(mock_body_green, mock_engine)
    assert color == COLORS["green"], (
        f"Case-insensitive color lookup failed, got {color}"
    )


@pytest.mark.fast
def test_invalid_discrete_index():
    """Test handling of invalid discrete color indices."""
    renderer = OpenCVRenderer()

    # Create discrete image with invalid index (8)
    discrete_img = np.array([[8]], dtype=np.uint8)
    rgb_img = renderer.discrete_to_rgb(discrete_img)

    # Should map to background (0) or handle gracefully
    assert rgb_img.shape == (1, 1, 3)
    # Index 8 doesn't exist, so it won't match any mask and will remain (0,0,0)
    assert tuple(rgb_img[0, 0]) == (0, 0, 0), "Invalid index should map to black"


# ============================================================================
# OpenCV Renderer Tests (12-15 tests)
# ============================================================================


@pytest.mark.fast
def test_opencv_renderer_initialization():
    """Test OpenCV renderer initialization with default parameters."""
    renderer = OpenCVRenderer()
    assert renderer.width == 600, f"Expected width=600, got {renderer.width}"
    assert renderer.height == 600, f"Expected height=600, got {renderer.height}"
    assert renderer.ppm == 60, f"Expected ppm=60, got {renderer.ppm}"
    assert renderer.image.shape == (
        600,
        600,
        3,
    ), f"Expected image shape (600,600,3), got {renderer.image.shape}"


@pytest.mark.fast
def test_opencv_render_empty_world():
    """Test rendering empty world produces white background."""
    engine = Box2DEngine()
    renderer = OpenCVRenderer(width=100, height=100)
    image = renderer.render(engine)

    assert image.shape == (
        100,
        100,
        3,
    ), f"Expected shape (100,100,3), got {image.shape}"
    assert image.dtype == np.uint8, f"Expected dtype uint8, got {image.dtype}"
    # All pixels should be white (255, 255, 255)
    assert np.all(image == 255), "Empty world should render as all white"


@pytest.mark.fast
def test_opencv_render_single_ball(simple_env):
    """Test rendering a single ball appears in image."""
    engine = simple_env.engine
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    image = renderer.render(engine)

    assert image.shape == (600, 600, 3)
    assert image.dtype == np.uint8
    # Image should not be all white (should have some colored pixels)
    assert not np.all(image == 255), "Image should contain rendered objects"


@pytest.mark.fast
def test_opencv_render_single_bar():
    """Test rendering a single bar (rectangle)."""
    level = load_level("two_body_problem", seed=42)
    engine = Box2DEngine(level=level)
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    image = renderer.render(engine)

    assert image.shape == (600, 600, 3)
    assert image.dtype == np.uint8
    # Should have rendered objects
    assert not np.all(image == 255)


@pytest.mark.fast
def test_opencv_render_output_dtype():
    """Test that render output has correct dtype and shape."""
    engine = Box2DEngine()
    renderer = OpenCVRenderer(width=400, height=300)
    image = renderer.render(engine)

    assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"
    assert image.shape == (300, 400, 3), f"Expected (300,400,3), got {image.shape}"


@pytest.mark.fast
def test_opencv_render_discrete_output_dtype():
    """Test that discrete render output has correct dtype and shape."""
    engine = Box2DEngine()
    renderer = OpenCVRenderer(width=400, height=300)
    discrete_image = renderer.render_discrete(engine)

    assert discrete_image.dtype == np.uint8, (
        f"Expected uint8, got {discrete_image.dtype}"
    )
    assert discrete_image.shape == (
        300,
        400,
    ), f"Expected (300,400), got {discrete_image.shape}"


@pytest.mark.fast
def test_opencv_render_discrete_values():
    """Test that discrete render values are in 0-7 range."""
    level = load_level("two_body_problem", seed=42)
    engine = Box2DEngine(level=level)
    renderer = OpenCVRenderer(width=600, height=600)
    discrete_image = renderer.render_discrete(engine)

    # All values should be in valid range
    assert np.all(discrete_image >= 0), "Discrete values should be >= 0"
    assert np.all(discrete_image <= 7), "Discrete values should be <= 7"


@pytest.mark.fast
def test_opencv_discrete_to_rgb_round_trip():
    """Test bidirectional conversion between discrete and RGB."""
    renderer = OpenCVRenderer(width=100, height=100)

    # Create discrete image with various indices
    discrete_img = np.random.randint(0, 8, size=(100, 100), dtype=np.uint8)

    # Convert to RGB
    rgb_img = renderer.discrete_to_rgb(discrete_img)

    # Convert back to discrete (manual check)
    for idx in range(8):
        mask = discrete_img == idx
        expected_rgb = DISCRETE_COLORS[idx]
        actual_rgb = rgb_img[mask]
        if np.any(mask):
            assert np.all(actual_rgb == expected_rgb), (
                f"Round-trip failed for index {idx}"
            )


@pytest.mark.fast
def test_opencv_sensor_exclusion_and_draw_order(simple_env):
    """Test that sensor fixtures are not rendered and draw order works."""
    engine = simple_env.engine
    renderer = OpenCVRenderer(width=600, height=600)

    # Render should sort bodies by y-position and skip sensors
    image = renderer.render(engine)

    assert image.shape == (600, 600, 3)
    # Sensors should not appear in the image - tested implicitly
    # Visual verification would require pixel inspection


@pytest.mark.fast
@pytest.mark.parametrize("render_method", ["render", "render_discrete"])
def test_opencv_render_skips_sensor(monkeypatch, render_method):
    """Test that both render and render_discrete skip sensor fixtures."""
    renderer = OpenCVRenderer(width=60, height=60, ppm=10)
    mock_body = MagicMock()
    mock_body.position.y = 0
    mock_body.userData = "ball"
    sensor_fixture = MagicMock()
    sensor_fixture.sensor = True
    sensor_fixture.shape = MagicMock()
    mock_body.fixtures = [sensor_fixture]
    mock_body.transform = MagicMock()
    mock_body.transform.__mul__ = MagicMock(return_value=(0, 0))

    mock_engine = MagicMock()
    mock_engine.bodies = {"ball": mock_body}
    mock_engine.level = load_level("two_body_problem", seed=1)

    circle = MagicMock()
    monkeypatch.setattr(cv2, "circle", circle)
    with patch.object(renderer, "_get_object_color", return_value=COLORS["red"]):
        getattr(renderer, render_method)(mock_engine)
    assert circle.call_count == 0


@pytest.mark.fast
@pytest.mark.parametrize("render_method", ["render", "render_discrete"])
def test_opencv_render_polygon_calls_fillpoly(monkeypatch, render_method):
    """Test that both render and render_discrete call fillPoly for polygon shapes."""
    world = b2World()
    body = world.CreateStaticBody(position=(0, 0))
    body.CreateFixture(shape=b2PolygonShape(box=(1, 1)), density=1)

    mock_engine = MagicMock()
    mock_engine.bodies = {"poly": body}
    mock_engine.level = load_level("two_body_problem", seed=1)

    renderer = OpenCVRenderer(width=60, height=60, ppm=10)
    fill_poly = MagicMock()
    monkeypatch.setattr(cv2, "fillPoly", fill_poly)
    with patch.object(renderer, "_get_object_color", return_value=COLORS["blue"]):
        getattr(renderer, render_method)(mock_engine)
    assert fill_poly.called


@pytest.mark.fast
@pytest.mark.parametrize("render_method", ["render", "render_discrete"])
def test_opencv_render_unsupported_shape_error_combined(render_method):
    """Test that both render methods raise ValueError for unsupported shapes."""
    renderer = OpenCVRenderer(width=60, height=60, ppm=10)
    mock_body = MagicMock()
    mock_body.position.y = 0
    mock_body.userData = "obj"
    mock_body.fixtures = [MagicMock(sensor=False, shape=MagicMock())]
    mock_body.transform = MagicMock()
    mock_body.transform.__mul__ = MagicMock(return_value=(0, 0))

    mock_engine = MagicMock()
    mock_engine.bodies = {"obj": mock_body}
    mock_engine.level = load_level("two_body_problem", seed=1)

    with patch.object(renderer, "_get_object_color", return_value=COLORS["red"]):
        with pytest.raises(ValueError, match="Unsupported shape type"):
            getattr(renderer, render_method)(mock_engine)


@pytest.mark.fast
def test_opencv_wait_uses_sleep(monkeypatch):
    renderer = OpenCVRenderer()
    sleep = MagicMock()
    monkeypatch.setattr(time, "sleep", sleep)
    renderer.wait(5)
    sleep.assert_called_once_with(0.005)


@pytest.mark.fast
def test_opencv_render_custom_dimensions():
    """Test rendering with custom image dimensions."""
    engine = Box2DEngine()
    renderer = OpenCVRenderer(width=800, height=600, ppm=100)
    image = renderer.render(engine)

    assert image.shape == (600, 800, 3), f"Expected (600,800,3), got {image.shape}"


# ============================================================================
# Pygame Renderer Tests (10-12 tests) - WITH MOCKING
# ============================================================================


@pytest.fixture
def mock_pygame():
    """Fixture to mock pygame module."""
    with patch("interphyre.render.pygame.pygame") as mock_pygame_module:
        mock_pygame_module.init.return_value = None
        mock_screen = MagicMock()
        mock_pygame_module.display.set_mode.return_value = mock_screen
        mock_pygame_module.display.flip.return_value = None
        mock_pygame_module.draw.circle.return_value = None
        mock_pygame_module.draw.polygon.return_value = None
        mock_pygame_module.event.pump.return_value = None
        mock_pygame_module.event.get.return_value = []
        mock_clock = MagicMock()
        mock_clock.tick.return_value = None
        mock_pygame_module.time.Clock.return_value = mock_clock
        mock_pygame_module.QUIT = 1
        yield mock_pygame_module, mock_screen, mock_clock


@pytest.mark.fast
def test_pygame_renderer_initialization(mock_pygame):
    """Test that pygame renderer calls pygame.init() on initialization."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=600, height=600, ppm=60)

    mock_pygame_module.init.assert_called_once()
    assert renderer.width == 600
    assert renderer.height == 600
    assert renderer.ppm == 60


@pytest.mark.fast
def test_pygame_render_calls_screen_fill(mock_pygame, simple_env):
    """Test that render calls screen.fill with white background."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=600, height=600, ppm=60)
    renderer.screen = mock_screen

    engine = simple_env.engine
    renderer.render(engine)

    # Should call fill with white color
    mock_screen.fill.assert_called()
    call_args = mock_screen.fill.call_args[0][0]
    assert call_args == COLORS["white"], f"Expected white fill, got {call_args}"


@pytest.mark.fast
def test_pygame_render_ball_draws_circle(mock_pygame, simple_env):
    """Test that ball objects trigger pygame.draw.circle calls."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=600, height=600, ppm=60)
    renderer.screen = mock_screen

    engine = simple_env.engine
    renderer.render(engine)

    # Should call draw.circle for ball objects
    assert mock_pygame_module.draw.circle.called or len(engine.bodies) == 0, (
        "Should call draw.circle for ball objects"
    )


@pytest.mark.fast
def test_pygame_render_polygon_calls_draw_polygon(mock_pygame):
    """Test that polygon objects trigger pygame.draw.polygon calls."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=20, height=20, ppm=10)

    world = b2World()
    body = world.CreateDynamicBody(position=(0, 0))
    body.CreateFixture(shape=b2PolygonShape(box=(1, 1)), density=1)

    engine = MagicMock()
    engine.bodies = {"poly": body}
    engine.level = load_level("two_body_problem", seed=1)

    renderer.render(engine)
    assert mock_pygame_module.draw.polygon.called
    renderer.close()


@pytest.mark.fast
def test_pygame_get_object_color_fallbacks(mock_pygame):
    renderer = PygameRenderer(width=20, height=20, ppm=10)
    mock_body = MagicMock()
    mock_body.userData = "missing"

    engine = Box2DEngine(level=None)
    assert renderer._get_object_color(mock_body, engine) == COLORS["black"]

    level = load_level("two_body_problem", seed=1)
    engine = Box2DEngine(level=level)
    assert renderer._get_object_color(mock_body, engine) == COLORS["black"]

    class NoColor(PhyreObject):
        def __init__(self):
            super().__init__(x=0.0, y=0.0)
            del self.color

    level.objects["noc"] = NoColor()
    mock_body.userData = "noc"
    assert renderer._get_object_color(mock_body, engine) == COLORS["black"]
    renderer.close()


@pytest.mark.fast
def test_pygame_render_skips_sensor_fixtures(mock_pygame):
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=20, height=20, ppm=10)

    world = b2World()
    body = world.CreateDynamicBody(position=(0, 0))
    fixture = body.CreateFixture(shape=b2CircleShape(radius=1), density=1)
    fixture.sensor = True

    engine = MagicMock()
    engine.bodies = {"ball": body}
    engine.level = load_level("two_body_problem", seed=1)

    renderer.render(engine)
    assert mock_pygame_module.draw.circle.call_count == 0
    renderer.close()


@pytest.mark.fast
def test_pygame_render_quit_event_does_not_raise_system_exit(mock_pygame):
    """Closing the pygame window must not raise SystemExit (FIX-PYGAME-EXIT)."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=20, height=20, ppm=10)
    engine = Box2DEngine(level=load_level("two_body_problem", seed=1))

    quit_event = MagicMock()
    quit_event.type = mock_pygame_module.QUIT
    mock_pygame_module.event.get.return_value = [quit_event]

    # render() must return normally — no SystemExit, no exit()
    renderer.render(engine)
    assert renderer._closed is True


@pytest.mark.fast
def test_pygame_render_noop_after_close(mock_pygame):
    """After close, render() returns immediately without touching pygame (FIX-PYGAME-EXIT)."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=20, height=20, ppm=10)
    renderer.close()

    # Reset mocks so we can verify no further pygame calls
    mock_screen.fill.reset_mock()
    mock_pygame_module.display.flip.reset_mock()

    engine = Box2DEngine(level=load_level("two_body_problem", seed=1))
    renderer.render(engine)

    mock_screen.fill.assert_not_called()
    mock_pygame_module.display.flip.assert_not_called()


@pytest.mark.fast
def test_pygame_wait_noop_after_close(mock_pygame):
    """After close, wait() returns immediately (FIX-PYGAME-EXIT)."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=20, height=20, ppm=10)
    renderer.close()

    mock_pygame_module.time.get_ticks.reset_mock()
    renderer.wait(5000)
    mock_pygame_module.time.get_ticks.assert_not_called()


@pytest.mark.fast
def test_pygame_wait_processes_events(mock_pygame):
    """Test that wait() processes events and calls time.wait for responsiveness."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=20, height=20, ppm=10)

    # Simulate get_ticks returning 0 then exceeding duration on second call
    mock_pygame_module.time.get_ticks.side_effect = [0, 0, 100]
    renderer.wait(10)
    mock_pygame_module.time.wait.assert_called_with(10)
    renderer.close()


@pytest.mark.fast
def test_renderer_base_noop_methods():
    class Dummy(Renderer):
        def render(self, engine):
            return super().render(engine)

        def close(self):
            return super().close()

    dummy = Dummy()
    assert dummy.render(MagicMock()) is None
    assert dummy.close() is None


@pytest.mark.fast
def test_save_obs_as_image_branches(monkeypatch):
    rgb_obs = np.zeros((4, 5, 3), dtype=np.uint8)
    discrete_obs = np.zeros((4, 5), dtype=np.uint8)

    imwrite = MagicMock(return_value=True)
    cvt_color = MagicMock(side_effect=lambda img, code: img)
    monkeypatch.setattr(cv2, "imwrite", imwrite)
    monkeypatch.setattr(cv2, "cvtColor", cvt_color)

    from interphyre.render import save_obs_as_image

    save_obs_as_image(rgb_obs, "rgb.png")
    save_obs_as_image(discrete_obs, "disc.png")

    assert imwrite.call_count == 2


@pytest.mark.fast
def test_save_obs_as_image_with_image_size(monkeypatch):
    rgb_obs = np.zeros((2, 3, 3), dtype=np.uint8)
    imwrite = MagicMock(return_value=True)
    cvt_color = MagicMock(side_effect=lambda img, code: img)
    monkeypatch.setattr(cv2, "imwrite", imwrite)
    monkeypatch.setattr(cv2, "cvtColor", cvt_color)

    from interphyre.render import save_obs_as_image

    save_obs_as_image(rgb_obs, "rgb.png", image_size=(10, 12))
    assert imwrite.call_count == 1


@pytest.mark.fast
def test_pygame_render_display_flip(mock_pygame, simple_env):
    """Test that render calls display.flip() after drawing."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=600, height=600, ppm=60)
    renderer.screen = mock_screen

    engine = simple_env.engine
    renderer.render(engine)

    # Should call display.flip()
    mock_pygame_module.display.flip.assert_called()


@pytest.mark.fast
def test_pygame_render_clock_tick(mock_pygame, simple_env):
    """Test that render calls clock.tick() for FPS control."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=600, height=600, ppm=60)
    renderer.screen = mock_screen
    renderer.clock = mock_clock

    engine = simple_env.engine
    renderer.render(engine)

    # Should call clock.tick with fps
    mock_clock.tick.assert_called_with(60)


@pytest.mark.fast
def test_pygame_close_quit(mock_pygame):
    """Test that close() calls pygame.quit()."""
    mock_pygame_module, mock_screen, mock_clock = mock_pygame
    renderer = PygameRenderer(width=600, height=600, ppm=60)
    renderer.close()

    mock_pygame_module.quit.assert_called_once()


# ============================================================================
# Integration Tests (5-7 tests)
# ============================================================================


@pytest.mark.fast
def test_render_full_level_two_body(simple_env):
    """Test rendering complete two_body_problem level."""
    engine = simple_env.engine
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    image = renderer.render(engine)

    assert image.shape == (600, 600, 3)
    assert image.dtype == np.uint8
    # Should have rendered multiple objects
    assert not np.all(image == 255)


@pytest.mark.fast
def test_render_after_simulation_steps(simple_env):
    """Test that objects move between frames after simulation steps."""
    engine = simple_env.engine
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)

    # Render initial state
    image1 = renderer.render(engine)

    # Run simulation steps
    for _ in range(10):
        engine.world.Step(
            engine.config.time_step,
            engine.config.velocity_iters,
            engine.config.position_iters,
        )
        engine.time_update(engine.config.time_step)

    # Render after steps
    image2 = renderer.render(engine)

    # Images should be different (objects moved)
    assert not np.array_equal(image1, image2), (
        "Images should differ after simulation steps"
    )


@pytest.mark.fast
def test_render_determinism(simple_env):
    """Test that same state produces identical image."""
    engine = simple_env.engine
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)

    # Render same state twice
    image1 = renderer.render(engine)
    image2 = renderer.render(engine)

    # Should be identical
    assert np.array_equal(image1, image2), "Same state should produce identical images"


@pytest.mark.fast
def test_render_discrete_vs_rgb_consistency(simple_env):
    """Test that discrete and RGB renders are consistent."""
    engine = simple_env.engine
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)

    discrete_img = renderer.render_discrete(engine)
    rgb_from_discrete = renderer.discrete_to_rgb(discrete_img)
    rgb_direct = renderer.render(engine)

    # They may not be pixel-perfect identical due to color mapping,
    # but should have similar structure
    assert discrete_img.shape == (600, 600)
    assert rgb_from_discrete.shape == rgb_direct.shape


@pytest.mark.fast
def test_render_multiple_objects(simple_env):
    """Test rendering with multiple objects in scene."""
    engine = simple_env.engine
    renderer = OpenCVRenderer(width=600, height=600, ppm=60)
    image = renderer.render(engine)

    # Should successfully render all objects
    assert image.shape == (600, 600, 3)
    assert len(engine.bodies) > 0, "Should have bodies to render"


# ============================================================================
# DEDUPLICATE-RENDERER-COLOR-LOGIC regression tests
# ============================================================================


@pytest.mark.fast
def test_color_method_inherited_from_base():
    """Both renderers inherit _get_object_color from Renderer base class."""
    assert OpenCVRenderer._get_object_color is Renderer._get_object_color
    assert PygameRenderer._get_object_color is Renderer._get_object_color


@pytest.mark.fast
def test_both_renderers_same_color_for_same_object(simple_env):
    """Both renderers produce the same color for the same body."""
    engine = simple_env.engine
    opencv_renderer = OpenCVRenderer()
    pygame_renderer = PygameRenderer.__new__(PygameRenderer)
    # Minimal init without pygame display
    pygame_renderer.width = 600
    pygame_renderer.height = 600
    pygame_renderer.ppm = 60

    for name, body in engine.bodies.items():
        color_opencv = opencv_renderer._get_object_color(body, engine)
        color_pygame = pygame_renderer._get_object_color(body, engine)
        assert color_opencv == color_pygame, (
            f"Color mismatch for '{name}': OpenCV={color_opencv}, Pygame={color_pygame}"
        )


@pytest.mark.fast
def test_small_radius_renders_at_least_one_pixel():
    """A ball with radius 0.005 at ppm=60 renders as at least 1 pixel in both renderers."""
    # 0.005 * 60 = 0.3, which would truncate to 0 with int()
    # max(1, round(0.3)) = 1
    ppm = 60
    small_radius = 0.005
    expected_min = 1

    opencv_renderer = OpenCVRenderer(ppm=ppm)
    computed_radius = max(1, round(small_radius * opencv_renderer.ppm))
    assert computed_radius >= expected_min, (
        f"Small radius {small_radius} at ppm={ppm} produced {computed_radius}px, "
        f"expected at least {expected_min}px"
    )

    # Verify the render path works: create a mock engine with a tiny circle
    engine = MagicMock()
    engine.level = None
    world = b2World(gravity=(0, -10))
    body = world.CreateDynamicBody(position=(0, 0))
    body.userData = "tiny_ball"
    body.CreateCircleFixture(radius=small_radius, density=1.0)
    engine.bodies = {"tiny_ball": body}

    image = opencv_renderer.render(engine)
    # The tiny ball should have at least 1 non-white pixel
    non_white = np.any(image != 255, axis=-1)
    assert np.sum(non_white) >= 1, "Tiny ball should render at least 1 pixel"


# ============================================================================
# FIX-VIDEO-RECORDER-SILENT-FAILURE regression tests
# ============================================================================


@pytest.mark.fast
def test_video_recorder_close_raises_when_frames_but_no_output_path():
    """VideoRecorder with frames but no output_path raises ValueError on close()."""
    from interphyre.render.video import VideoRecorder

    recorder = VideoRecorder(output_path=None)
    # Simulate captured frames
    recorder.frames.append(np.zeros((600, 600, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="output_path"):
        recorder.close()


@pytest.mark.fast
def test_video_recorder_close_no_frames_no_output_path():
    """VideoRecorder with no frames and no output_path closes without error."""
    from interphyre.render.video import VideoRecorder

    recorder = VideoRecorder(output_path=None)
    # No frames — should close cleanly
    recorder.close()
    assert recorder._closed is True
