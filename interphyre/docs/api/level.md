# Level Model

## Level

`Level` is the core data structure for a physics puzzle. It bundles objects, action objects, success conditions, and metadata.

Constructor:

```python
from interphyre.level import Level
level = Level(
    name="my_level",
    objects={"green_ball": green_ball},
    action_objects=["red_ball"],
    success_condition=success_condition,
    metadata={"description": "Goal statement"},
)
```

Attributes:

- `name`: unique level identifier
- `objects`: mapping of object names to `PhyreObject` instances
- `action_objects`: list of object names the agent can place
- `success_condition`: callable `success_condition(engine) -> bool`
- `metadata`: optional dictionary (typically includes `description`)

Methods:

- `move_object(obj_name, x, y)`
- `set_angle(obj_name, angle)`
- `change_color(obj_name, color)`
- `remove_object(obj_name)`
- `set_dynamic(obj_name, dynamic)`
- `set_restitution(obj_name, restitution)`
- `set_friction(obj_name, friction)`
- `clone(new_name=None)`
