# Rendering

Rendering utilities live in `interphyre/render`.

## Renderer

Abstract base class that defines:

- `render(engine)`
- `close()`

## PygameRenderer

Real-time renderer for interactive visualization.

```python
from interphyre.render import PygameRenderer
renderer = PygameRenderer(width=600, height=600, ppm=60)
```

## OpenCVRenderer

Offscreen renderer for RGB or discrete images.

```python
from interphyre.render import OpenCVRenderer
renderer = OpenCVRenderer(width=600, height=600, ppm=60)
```

Key methods:

- `render(engine) -> np.ndarray`
- `render_discrete(engine) -> np.ndarray`
- `discrete_to_rgb(discrete_image) -> np.ndarray`

## Utility helpers

`save_obs_as_image(obs, filename, image_size=None)` writes RGB or discrete observations to disk.

## World bounds

`interphyre.render` defines `MIN_X`, `MAX_X`, `MIN_Y`, `MAX_Y`, `WORLD_WIDTH`, and `WORLD_HEIGHT` for common bounds.
