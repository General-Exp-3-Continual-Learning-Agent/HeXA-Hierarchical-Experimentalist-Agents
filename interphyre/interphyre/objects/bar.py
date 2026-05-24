import math

from Box2D import b2_pi, b2World

from interphyre.config import PRECISION

from .base import PhyreObject


class Bar(PhyreObject):
    """A rectangular physics object representing a bar, beam, or platform.

    Bars are rectangular objects that can be positioned and oriented in 2D space.
    They support multiple initialization patterns for convenient level design.

    Attributes:
        length (float): Length of the bar along its main axis
        thickness (float): Thickness of the bar perpendicular to its main axis
        x1, y1 (float): First endpoint coordinates (read-only)
        x2, y2 (float): Second endpoint coordinates (read-only)
        left, right (float): Bounding box left/right edges (read-only)
        top, bottom (float): Bounding box top/bottom edges (read-only)

    Initialization Patterns:
        # From center point and dimensions
        bar = Bar(x=0, y=0, length=4.0, angle=45.0, thickness=0.2)

        # From endpoints
        bar = Bar(x1=0, y1=0, x2=4, y2=0, thickness=0.2)

        # From bounding box (horizontal)
        bar = Bar(left=-2, right=2, y=0, thickness=0.2)

        # From bounding box (vertical)
        bar = Bar(top=2, bottom=-2, x=0, thickness=0.2)

    Examples:
        # Create a horizontal platform
        platform = Bar(x=0, y=-3, length=6.0, angle=0.0, thickness=0.3)

        # Create a diagonal ramp
        ramp = Bar.from_endpoints(0, 0, 4, 2, thickness=0.2)

        # Create a vertical wall
        wall = Bar.touching_wall("left", 0, thickness=0.1)
    """

    def __init__(
        self,
        x=None,
        y=None,
        length=2.0,
        angle=0.0,
        thickness=0.2,
        x1=None,
        y1=None,
        x2=None,
        y2=None,
        left=None,
        right=None,
        top=None,
        bottom=None,
        **kwargs,
    ):
        """Initialize a Bar with flexible positioning options.

        The Bar can be initialized using several different patterns:
        - Center point: (x, y, length, angle, thickness)
        - Endpoints: (x1, y1, x2, y2, thickness)
        - Bounding box: (left, right, y, thickness) or (top, bottom, x, thickness)

        Args:
            x, y (float, optional): Center coordinates
            length (float): Bar length (default: 2.0)
            angle (float): Rotation angle in degrees (default: 0.0)
            thickness (float): Bar thickness (default: 0.2)
            x1, y1, x2, y2 (float, optional): Endpoint coordinates
            left, right, top, bottom (float, optional): Bounding box coordinates
            **kwargs: Additional PhyreObject properties (color, dynamic, etc.)

        Raises:
            ValueError: If insufficient positioning information is provided
        """
        # Handle different initialization patterns
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            # Initialize from endpoints
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        elif left is not None and right is not None and y is not None:
            # Initialize from left/right
            x = (left + right) / 2
            length = right - left
            angle = 0.0
        elif top is not None and bottom is not None and x is not None:
            # Initialize from top/bottom
            y = (top + bottom) / 2
            length = top - bottom
            angle = 90.0
        elif x is None or y is None:
            raise ValueError(
                "Must provide either (x,y) or endpoints or left/right or top/bottom"
            )

        # Initialize private attributes first
        self._x = x
        self._y = y
        self._angle = angle
        self._length = length
        self._thickness = thickness

        # Call parent constructor with the same values
        super().__init__(x=x, y=y, angle=angle, **kwargs)

        # Update endpoints after everything is initialized
        self._update_endpoints()

    def _repr_dimensions(self) -> str:
        return f"length={self.length:.2f}, thickness={self.thickness:.2f}"

    def _update_endpoints(self):
        """Update endpoint coordinates based on center, length, and angle"""
        angle_rad = math.radians(self._angle)
        dx = (self._length / 2) * math.cos(angle_rad)
        dy = (self._length / 2) * math.sin(angle_rad)
        self._x1 = self._x - dx
        self._y1 = self._y - dy
        self._x2 = self._x + dx
        self._y2 = self._y + dy

    def _update_center_from_endpoints(self):
        """Update center, length, and angle from endpoints"""
        self._x = (self._x1 + self._x2) / 2
        self._y = (self._y1 + self._y2) / 2
        self._length = math.hypot(self._x2 - self._x1, self._y2 - self._y1)
        self._angle = math.degrees(math.atan2(self._y2 - self._y1, self._x2 - self._x1))

    # Endpoint properties
    @property
    def x1(self):
        return self._x1

    @property
    def y1(self):
        return self._y1

    @property
    def x2(self):
        return self._x2

    @property
    def y2(self):
        return self._y2

    @x1.setter
    def x1(self, value):
        self._x1 = value
        self._update_center_from_endpoints()

    @y1.setter
    def y1(self, value):
        self._y1 = value
        self._update_center_from_endpoints()

    @x2.setter
    def x2(self, value):
        self._x2 = value
        self._update_center_from_endpoints()

    @y2.setter
    def y2(self, value):
        self._y2 = value
        self._update_center_from_endpoints()

    # Bounding box properties (read-only for convenience)
    @property
    def left(self):
        return min(self._x1, self._x2)

    @property
    def right(self):
        return max(self._x1, self._x2)

    @property
    def top(self):
        return max(self._y1, self._y2)

    @property
    def bottom(self):
        return min(self._y1, self._y2)

    # Override properties to update endpoints when changed
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._update_endpoints()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self._update_endpoints()

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self._update_endpoints()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self._update_endpoints()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value

    @classmethod
    def from_endpoints(cls, x1, y1, x2, y2, thickness=0.2, **kwargs):
        """Create bar connecting two points.

        Args:
            x1, y1 (float): First endpoint coordinates
            x2, y2 (float): Second endpoint coordinates
            thickness (float): Bar thickness (default: 0.2)
            **kwargs: Additional Bar properties (color, dynamic, etc.)

        Returns:
            Bar: Bar object positioned and angled to connect the points
        """
        return cls(x1=x1, y1=y1, x2=x2, y2=y2, thickness=thickness, **kwargs)

    @classmethod
    def from_point_and_angle(cls, x, y, angle, length, thickness=0.2, **kwargs):
        """Create bar from center point, angle, and length.

        Args:
            x, y (float): Center point coordinates
            angle (float): Bar angle in degrees
            length (float): Bar length
            thickness (float): Bar thickness (default: 0.2)
            **kwargs: Additional Bar properties

        Returns:
            Bar: Bar object positioned at center with specified angle and length
        """
        return cls(x=x, y=y, angle=angle, length=length, thickness=thickness, **kwargs)

    @classmethod
    def from_corner(cls, corner_x, corner_y, angle, length, thickness=0.2, **kwargs):
        """
        Create bar starting from corner point.

        Args:
            corner_x, corner_y: Starting corner coordinates
            angle: Bar angle in degrees
            length: Bar length
            thickness: Bar thickness
            **kwargs: Additional Bar properties

        Returns:
            Bar object positioned starting from corner at specified angle
        """
        # Calculate center position from corner
        angle_rad = math.radians(angle)
        center_x = corner_x + (length / 2) * math.cos(angle_rad)
        center_y = corner_y + (length / 2) * math.sin(angle_rad)

        return cls(
            x=center_x,
            y=center_y,
            angle=angle,
            length=length,
            thickness=thickness,
            **kwargs,
        )

    @classmethod
    def ramp_to_wall(cls, start_x, start_y, angle, wall_side, thickness=0.2, **kwargs):
        """
        Create ramp from point to wall at angle.

        Args:
            start_x, start_y: Starting point coordinates
            angle: Ramp angle in degrees
            wall_side: Which wall to reach ('left', 'right', 'top', 'bottom')
            thickness: Bar thickness
            **kwargs: Additional Bar properties

        Returns:
            Bar object from start point to wall
        """
        # Calculate distance to wall based on angle
        angle_rad = math.radians(angle)

        if wall_side == "left":
            # Distance to left wall (x = -5)
            distance = (-5 - start_x) / math.cos(angle_rad)
        elif wall_side == "right":
            # Distance to right wall (x = 5)
            distance = (5 - start_x) / math.cos(angle_rad)
        elif wall_side == "top":
            # Distance to top wall (y = 5)
            distance = (5 - start_y) / math.sin(angle_rad)
        elif wall_side == "bottom":
            # Distance to bottom wall (y = -5)
            distance = (-5 - start_y) / math.sin(angle_rad)
        else:
            raise ValueError(
                f"Invalid wall_side: {wall_side}. Must be 'left', 'right', 'top', or 'bottom'"
            )

        if distance <= 0:
            raise ValueError(
                f"Angle {angle}° does not point toward the {wall_side} wall from "
                f"({start_x}, {start_y})"
            )

        # Calculate center position
        center_x = start_x + (distance / 2) * math.cos(angle_rad)
        center_y = start_y + (distance / 2) * math.sin(angle_rad)

        return cls(
            x=center_x,
            y=center_y,
            angle=angle,
            length=distance,
            thickness=thickness,
            **kwargs,
        )

    @classmethod
    def touching_wall(cls, wall_side, angle, offset=0, thickness=0.2, **kwargs):
        """
        Create bar that touches a specific wall at given angle.

        Args:
            wall_side: Which wall to touch ('left', 'right', 'top', 'bottom')
            angle: Bar angle in degrees
            offset: Distance from wall (positive = away from wall)
            thickness: Bar thickness
            **kwargs: Additional Bar properties

        Returns:
            Bar object that touches the specified wall
        """
        angle_rad = math.radians(angle)

        if wall_side == "left":
            # Bar touching left wall
            wall_x = -5 + offset
            wall_y = 0  # Center vertically
            # Calculate length to reach opposite wall
            distance = (5 - wall_x) / math.cos(angle_rad)
        elif wall_side == "right":
            # Bar touching right wall
            wall_x = 5 - offset
            wall_y = 0  # Center vertically
            # Calculate length to reach opposite wall
            distance = (wall_x - (-5)) / math.cos(angle_rad)
        elif wall_side == "top":
            # Bar touching top wall
            wall_x = 0  # Center horizontally
            wall_y = 5 - offset
            # Calculate length to reach opposite wall
            distance = (wall_y - (-5)) / math.sin(angle_rad)
        elif wall_side == "bottom":
            # Bar touching bottom wall
            wall_x = 0  # Center horizontally
            wall_y = -5 + offset
            # Calculate length to reach opposite wall
            distance = (5 - wall_y) / math.sin(angle_rad)
        else:
            raise ValueError(
                f"Invalid wall_side: {wall_side}. Must be 'left', 'right', 'top', or 'bottom'"
            )

        # Calculate center position
        center_x = wall_x + (distance / 2) * math.cos(angle_rad)
        center_y = wall_y + (distance / 2) * math.sin(angle_rad)

        return cls(
            x=center_x,
            y=center_y,
            angle=angle,
            length=distance,
            thickness=thickness,
            **kwargs,
        )

    @classmethod
    def support_leg(cls, top_x, top_y, bottom_x, bottom_y, thickness=0.2, **kwargs):
        """
        Create support leg from top to bottom points.

        Args:
            top_x, top_y: Top connection point
            bottom_x, bottom_y: Bottom connection point
            thickness: Bar thickness
            **kwargs: Additional Bar properties

        Returns:
            Bar object representing the support leg
        """
        return cls(
            x1=top_x, y1=top_y, x2=bottom_x, y2=bottom_y, thickness=thickness, **kwargs
        )

    @classmethod
    def offset_along_angle(
        cls, base_x, base_y, angle, distance, thickness=0.2, **kwargs
    ):
        """
        Create bar offset along an angle from a base point.

        Args:
            base_x, base_y: Base point coordinates
            angle: Direction angle in degrees
            distance: Distance to offset
            thickness: Bar thickness
            **kwargs: Additional Bar properties

        Returns:
            Bar object offset along the specified angle
        """
        angle_rad = math.radians(angle)
        offset_x = distance * math.cos(angle_rad)
        offset_y = distance * math.sin(angle_rad)

        center_x = base_x + offset_x
        center_y = base_y + offset_y

        return cls(
            x=center_x,
            y=center_y,
            angle=angle,
            length=distance,
            thickness=thickness,
            **kwargs,
        )


def create_bar(world: b2World, bar: "Bar", name: str, use_ccd: bool = False):
    """Create a Box2D physics body from a Bar object.

    Converts a Bar data object into a Box2D physics body that can be
    simulated in the physics world.

    Args:
        world (b2World): The Box2D physics world to create the body in
        bar (Bar): The Bar object containing position and physical properties
        name (str): Unique identifier for the physics body
        use_ccd (bool): Whether to enable continuous collision detection (bullet mode) (default: False)

    Returns:
        b2Body: The created Box2D physics body

    Note:
        All floating point values are rounded to the configured PRECISION to ensure determinism.
    """
    x = round(float(bar.x), PRECISION)
    y = round(float(bar.y), PRECISION)
    angle_deg = round(float(bar.angle), PRECISION)
    angle = round(angle_deg * b2_pi / 180, PRECISION)
    length = round(float(bar.length), PRECISION)
    thickness = round(float(bar.thickness), PRECISION)
    density = round(float(bar.density), PRECISION)
    friction = round(float(bar.friction), PRECISION)
    restitution = round(float(bar.restitution), PRECISION)
    linear_damping = round(float(bar.linear_damping), PRECISION)
    angular_damping = round(float(bar.angular_damping), PRECISION)

    body = (
        world.CreateDynamicBody(
            position=(x, y),
            angle=angle,
            bullet=use_ccd,
        )
        if bar.dynamic
        else world.CreateStaticBody(position=(x, y), angle=angle)
    )
    body.CreatePolygonFixture(
        box=(round(length / 2, PRECISION), round(thickness / 2, PRECISION)),
        density=density,
        friction=friction,
        restitution=restitution,
    )

    body.linearDamping = linear_damping
    body.angularDamping = angular_damping
    body.userData = name
    return body


def circle_intersects_bar(cx: float, cy: float, radius: float, bar: "Bar") -> bool:
    """Return True if a circle overlaps a (possibly rotated) bar.

    Transforms the circle center into the bar's local frame (rotating by
    -bar.angle), then finds the nearest point on the bar's axis-aligned
    bounding box and compares the squared distance against radius².

    This is the authoritative implementation shared by InterphyreEnv
    (placement validation) and the oracle registry (solvability checking).

    Args:
        cx: Circle center x-coordinate.
        cy: Circle center y-coordinate.
        radius: Circle radius.
        bar: Bar instance providing x, y, angle, length, and thickness attributes.

    Returns:
        True if the circle intersects the bar; False otherwise.
    """
    angle_rad = math.radians(-bar.angle)
    dx, dy = cx - bar.x, cy - bar.y
    local_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    local_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    closest_x = max(-bar.length / 2, min(local_x, bar.length / 2))
    closest_y = max(-bar.thickness / 2, min(local_y, bar.thickness / 2))
    return (local_x - closest_x) ** 2 + (local_y - closest_y) ** 2 <= radius**2
