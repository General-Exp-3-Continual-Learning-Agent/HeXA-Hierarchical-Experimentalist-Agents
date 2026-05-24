# Objects

Physics objects for building levels. All objects inherit from `PhyreObject` and can be used with `InterphyreEnv.add_object()` or in custom level definitions.

## Quick Start

```python
from interphyre.objects import Ball, Bar, Basket

# Create objects
ball = Ball(x=0, y=3, radius=0.5, color="green", dynamic=True)
platform = Bar(x=0, y=0, length=4.0, thickness=0.2, angle=0, dynamic=False)
basket = Basket(x=2, y=-2, total_width=1.5, total_height=1.0)
```

## PhyreObject (Base Class)

All objects inherit from `PhyreObject`, which provides common physical properties.

### Common Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `x` | float | required | X position |
| `y` | float | required | Y position |
| `angle` | float | 0.0 | Rotation angle in degrees |
| `color` | str | "gray" | Visual color |
| `dynamic` | bool | True | If True, affected by physics; if False, static |
| `friction` | float | 0.3 | Surface friction coefficient |
| `restitution` | float | 0.2 | Bounciness (0 = no bounce, 1 = full bounce) |
| `density` | float | 1.0 | Mass density |

## Ball

Circular physics object.

```python
from interphyre.objects import Ball

# Basic ball
ball = Ball(x=0, y=5, radius=1.0, color="red")

# Static platform ball
platform = Ball(x=0, y=-3, radius=2.0, dynamic=False, color="gray")

# Ball with custom physics
bouncy = Ball(x=0, y=5, radius=0.5, restitution=0.9, friction=0.1)
```

### Ball Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | float | required | X position |
| `y` | float | required | Y position |
| `radius` | float | 0.5 | Ball radius |
| `color` | str | "gray" | Visual color |
| `dynamic` | bool | True | Physics-enabled or static |

## Bar

Rectangular bar or platform with flexible initialization.

### From Center Point

```python
from interphyre.objects import Bar

# Horizontal platform
platform = Bar(x=0, y=-3, length=6.0, thickness=0.3, angle=0)

# Tilted ramp
ramp = Bar(x=0, y=0, length=4.0, thickness=0.2, angle=-30)

# Vertical wall
wall = Bar(x=-4, y=0, length=6.0, thickness=0.2, angle=90)
```

### From Endpoints

```python
# Define by start and end points
bar = Bar(x1=0, y1=0, x2=4, y2=2, thickness=0.2)
```

### From Bounding Box

```python
# Horizontal bar from left/right edges
bar = Bar(left=-2, right=2, y=0, thickness=0.2)

# Vertical bar from top/bottom edges
bar = Bar(top=2, bottom=-2, x=0, thickness=0.2)
```

### Bar Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x`, `y` | float | - | Center position (for center-based init) |
| `length` | float | 2.0 | Bar length |
| `thickness` | float | 0.2 | Bar thickness |
| `angle` | float | 0.0 | Rotation in degrees |
| `x1`, `y1`, `x2`, `y2` | float | - | Endpoint coordinates |
| `left`, `right` | float | - | Horizontal bounds |
| `top`, `bottom` | float | - | Vertical bounds |
| `dynamic` | bool | False | Physics-enabled or static |

### Bar Properties

After creation, bars expose computed properties:

```python
bar.x1, bar.y1  # First endpoint
bar.x2, bar.y2  # Second endpoint
bar.left, bar.right  # Bounding box
bar.top, bar.bottom  # Bounding box
```

## Basket

U-shaped container for catching balls.

```python
from interphyre.objects import Basket

basket = Basket(
    x=2, y=-2,
    total_width=1.5,
    total_height=1.0,
    wall_thickness=0.1,
    color="gray",
    dynamic=False
)
```

### Basket Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x`, `y` | float | required | Center position |
| `total_width` | float | 1.0 | Outer width |
| `total_height` | float | 1.0 | Outer height |
| `wall_thickness` | float | 0.1 | Thickness of walls |
| `opening` | str | "top" | Which side is open |
| `dynamic` | bool | False | Usually static |

## Adding Objects to Simulation

### In Custom Levels

```python
from interphyre.level import Level
from interphyre.objects import Ball, Bar

level = Level(
    name="my_level",
    objects={
        "ball": Ball(x=0, y=3, radius=0.5, color="green", dynamic=True),
        "platform": Bar(x=0, y=0, length=4.0, thickness=0.2, dynamic=False),
    },
    action_objects=["ball"],
    success_condition=lambda engine: engine.bodies["ball"].position.y < -2,
)
```

### During Simulation

```python
from interphyre import InterphyreEnv
from interphyre.objects import Ball

env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)
env.reset()

# Add new object
env.add_object(
    "helper",
    Ball(x=0, y=3, radius=0.5, color="blue", dynamic=True),
    impulse=(5.0, 0.0)  # Optional initial impulse
)
```

## Color Reference

Standard colors available:

- `"red"` - Action objects (user-placeable)
- `"green"` - Target objects
- `"blue"` - Goal/target objects
- `"gray"` - Platforms and obstacles
- `"black"` - Static barriers
- `"white"` - Background elements
- `"yellow"` - Highlight objects
- `"purple"` - Special objects

## Low-Level Creation Functions

For direct Box2D body creation (advanced use):

```python
from interphyre.objects import create_ball, create_bar, create_basket, create_walls

# Create physics body from object
body = create_ball(world, ball, "ball_name", use_ccd=False)
body = create_bar(world, bar, "bar_name", use_ccd=False)
body = create_basket(world, basket, "basket_name", use_ccd=False)

# Create boundary walls
left, right, top, bottom = create_walls(world, wall_thickness=0.5, room_width=10, room_height=10)
```

## See Also

- [Level Model](level.md) - Creating custom levels
- [Environment](environment.md) - Adding objects during simulation
- [Examples: Custom Levels](../examples/custom_levels.md) - Level building examples
