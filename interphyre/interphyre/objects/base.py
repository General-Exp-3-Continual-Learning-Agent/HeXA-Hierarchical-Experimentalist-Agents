class PhyreObject:
    """Base class for all physics objects in the world.

    This class defines the common properties that all physics objects share,
    including position, orientation, and physical properties like friction
    and restitution.

    Attributes:
        x (float): X-coordinate of the object's center position
        y (float): Y-coordinate of the object's center position
        angle (float): Rotation angle in degrees (default: 0.0)
        color (str): Visual color of the object (default: "black")
        dynamic (bool): Whether the object is affected by physics forces (default: True)
        restitution (float): Bounciness factor, 0.0 = no bounce, 1.0 = perfect bounce (default: 0.2)
        friction (float): Surface friction coefficient (default: 0.5)
        linear_damping (float): Linear velocity damping factor (default: 0.0)
        angular_damping (float): Angular velocity damping factor (default: 0.01)
        density (float): Density of the object (default: 0.25)
    """

    def __init__(
        self,
        x: float,
        y: float,
        angle: float = 0.0,
        color: str = "black",
        dynamic: bool = True,
        restitution: float = 0.2,
        friction: float = 0.5,
        linear_damping: float = 0.0,
        angular_damping: float = 0.01,
        density: float = 0.25,
    ):
        """Initialize a PhyreObject with position and physical properties.

        Args:
            x, y: Position coordinates
            angle: Rotation angle in degrees (default: 0.0)
            color: Visual color (default: "black")
            dynamic: Whether affected by physics forces (default: True)
            restitution: Bounciness factor, 0.0 = no bounce, 1.0 = perfect bounce (default: 0.2)
            friction: Surface friction coefficient (default: 0.5)
            linear_damping: Linear velocity damping factor (default: 0.0)
            angular_damping: Angular velocity damping factor (default: 0.01)
            density: Density of the object (default: 0.25)
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.color = color
        self.dynamic = dynamic
        self.restitution = restitution
        self.friction = friction
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.density = density

    def _repr_dimensions(self) -> str:
        """Subclass-specific dimensions for __repr__. Override in subclasses."""
        return ""

    def __repr__(self) -> str:
        dims = self._repr_dimensions()
        suffix = f", {dims}" if dims else ""
        return f"{type(self).__name__}(color={self.color!r}, pos=({self.x:.2f}, {self.y:.2f}){suffix})"
