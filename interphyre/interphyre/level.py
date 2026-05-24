from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from interphyre.objects import PhyreObject

if TYPE_CHECKING:
    from interphyre.engine import Box2DEngine


@dataclass
class Level:
    """Represents a physics puzzle level with objects and success conditions.

    A level defines the initial state of a physics puzzle, including all objects
    in the scene, which objects can be controlled by the agent, and how success
    is determined. This is the core data structure for defining puzzle levels.

    Attributes:
        name (str): Unique identifier for the level
        objects (dict[str, PhyreObject]): Dictionary mapping object names to physics objects
        action_objects (list[str]): List of object names that can be controlled by the agent
        success_condition (Callable[[Box2DEngine], bool]): Function that determines if the level is solved
        metadata (dict | None): Additional level information (default: empty dict)
    """

    name: str
    objects: dict[str, PhyreObject]
    action_objects: list[str]
    success_condition: Callable[[Box2DEngine], bool]
    metadata: dict | None = field(default_factory=dict)

    def __post_init__(self):
        """Validate the level configuration after initialization.

        Raises:
            ValueError: If success_condition is not callable
        """
        if not callable(self.success_condition):
            raise ValueError(
                f"Level '{self.name}' must define a success_condition function."
            )

    def move_object(self, obj_name: str, x: float, y: float):
        """Move an object to a new position.

        Args:
            obj_name: Name of the object to move
            x: New x-coordinate
            y: New y-coordinate

        Raises:
            ValueError: If object doesn't exist in the level
        """
        if obj_name in self.objects:
            self.objects[obj_name].x, self.objects[obj_name].y = x, y
        else:
            raise ValueError(f"No object named '{obj_name}' in level.")

    def set_angle(self, obj_name: str, angle: float):
        """Set the rotation angle of an object.

        Args:
            obj_name: Name of the object to rotate
            angle: New angle in degrees

        Raises:
            ValueError: If object doesn't exist in the level
        """
        if obj_name in self.objects:
            self.objects[obj_name].angle = angle
        else:
            raise ValueError(f"No object named '{obj_name}' in level.")

    def change_color(self, obj_name: str, color: str):
        """Change the color of an object.

        Args:
            obj_name: Name of the object to recolor
            color: New color name

        Raises:
            ValueError: If object doesn't exist in the level
        """
        if obj_name in self.objects:
            self.objects[obj_name].color = color
        else:
            raise ValueError(f"No object named '{obj_name}' in level.")

    def remove_object(self, obj_name: str):
        """Remove an object from the level.

        Args:
            obj_name: Name of the object to remove

        Raises:
            ValueError: If object doesn't exist in the level
        """
        if obj_name in self.objects:
            del self.objects[obj_name]
            if obj_name in self.action_objects:
                self.action_objects.remove(obj_name)
        else:
            raise ValueError(f"Cannot remove: No object named '{obj_name}' in level.")

    def set_dynamic(self, obj_name: str, dynamic: bool):
        """Set whether an object is affected by physics forces.

        Args:
            obj_name: Name of the object to modify
            dynamic: True for physics-affected, False for static

        Raises:
            ValueError: If object doesn't exist in the level
        """
        if obj_name in self.objects:
            self.objects[obj_name].dynamic = dynamic
        else:
            raise ValueError(f"No object named '{obj_name}' in level.")

    def set_restitution(self, obj_name: str, restitution: float):
        """Set the bounciness (restitution) of an object.

        Args:
            obj_name: Name of the object to modify
            restitution: Bounciness factor (0.0 = no bounce, 1.0 = perfect bounce)

        Raises:
            ValueError: If object doesn't exist in the level
        """
        if obj_name in self.objects:
            self.objects[obj_name].restitution = restitution
        else:
            raise ValueError(f"No object named '{obj_name}' in level.")

    def set_friction(self, obj_name: str, friction: float):
        """Set the friction coefficient of an object.

        Args:
            obj_name: Name of the object to modify
            friction: Friction coefficient (0.0 = no friction, higher = more friction)

        Raises:
            ValueError: If object doesn't exist in the level
        """
        if obj_name in self.objects:
            self.objects[obj_name].friction = friction
        else:
            raise ValueError(f"No object named '{obj_name}' in level.")

    def clone(self, new_name: str | None = None):
        """Create a deep copy of the level.

        Args:
            new_name: Name for the cloned level (default: original_name + "_clone")

        Returns:
            Level: A new level with deep-copied objects and metadata
        """
        import copy

        return Level(
            name=new_name or self.name + "_clone",
            objects=copy.deepcopy(self.objects),
            action_objects=self.action_objects[:],
            success_condition=self.success_condition,
            metadata=copy.deepcopy(self.metadata),
        )
