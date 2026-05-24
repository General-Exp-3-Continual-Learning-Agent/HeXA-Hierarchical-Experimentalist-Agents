from Box2D import b2_pi, b2PolygonShape, b2World

from interphyre.config import PRECISION

from .base import PhyreObject


class Basket(PhyreObject):
    """U-shaped container with configurable dimensions.

    The basket geometry is defined by bottom_width, top_width, and height.
    Use explicit dimensions for precise control, or the scale parameter to
    create baskets with standard proportions at different sizes.

    Positioning uses an anchor system where (x, y) can refer to different
    reference points (e.g., bottom_center, center, top_center).

    Attributes:
        bottom_width (float): Interior width at the bottom of the basket
        top_width (float): Interior width at the top of the basket
        height (float): Interior height of the basket
        scale (float): Proportional scaling factor (alternative to explicit dimensions)
        wall_thickness (float): Thickness of the basket walls (default: 0.175)
        floor_thickness (float): Thickness of the basket floor (default: wall_thickness)
        anchor (str): Reference point for positioning (default: "bottom_center")
        double_walls (bool): Whether to create anti-tunneling double walls (default: False)
        enable_sensor (bool): Whether to create a sensor for success detection (default: True)
        sensor_margin (float): Margin around the sensor area (default: 0.05)
        sensor_height_ratio (float): Height of sensor as ratio of basket height (default: 0.3)

    Examples:
        # Explicit dimensions
        basket = Basket(x=0, y=-3, bottom_width=2.0, top_width=2.4, height=2.5)

        # Using scale
        basket = Basket(x=0, y=-3, scale=1.5)

        # Using convenience method
        basket = Basket.from_width_and_flare(x=0, y=-3, bottom_width=2.0, flare_ratio=1.2)
    """

    def __init__(
        self,
        x: float,
        y: float,
        bottom_width: float | None = None,
        top_width: float | None = None,
        height: float | None = None,
        scale: float | None = None,
        wall_thickness: float | None = None,
        floor_thickness: float | None = None,
        anchor: str = "bottom_center",
        double_walls: bool = False,
        enable_sensor: bool = True,
        sensor_margin: float = 0.05,
        sensor_height_ratio: float = 0.3,
        **kwargs,
    ):
        """Initialize a Basket with flexible dimension options.

        Args:
            x, y: Position coordinates
            bottom_width, top_width, height: Explicit dimensions
            scale: Proportional scaling factor (alternative to explicit dimensions)
            wall_thickness: Thickness of side walls (default: 0.175)
            floor_thickness: Thickness of floor (defaults to wall_thickness)
            anchor: Reference point for positioning (default: "bottom_center")
            **kwargs: Additional PhyreObject properties (color, dynamic, etc.)
        """
        # Initialize parent class first
        super().__init__(x=x, y=y, **kwargs)

        # Set basic properties
        self.anchor = anchor
        self.double_walls = double_walls
        self.enable_sensor = enable_sensor
        self.sensor_margin = sensor_margin
        self.sensor_height_ratio = sensor_height_ratio

        # Store scale for reference
        self.scale = scale

        # Initialize dimensions from scale or apply defaults
        if scale is not None:
            self.wall_thickness = (
                wall_thickness
                if wall_thickness is not None
                else min(0.175, scale * 0.22)
            )
            if bottom_width is None:
                self.bottom_width = 1.083 * scale
            else:
                self.bottom_width = bottom_width
            if top_width is None:
                self.top_width = 1.083 * scale * 1.25
            else:
                self.top_width = top_width
            if height is None:
                self.height = 2 * scale
            else:
                self.height = height
        else:
            # Apply defaults for any None values
            self.wall_thickness = (
                wall_thickness if wall_thickness is not None else 0.175
            )
            self.bottom_width = bottom_width if bottom_width is not None else 2.0
            self.top_width = top_width if top_width is not None else 2.2
            self.height = height if height is not None else 3.2

        # Handle floor_thickness
        if floor_thickness is None:
            self.floor_thickness = self.wall_thickness
        else:
            self.floor_thickness = floor_thickness

    def _repr_dimensions(self) -> str:
        return f"width={self.bottom_width:.2f}, height={self.height:.2f}"

    @classmethod
    def from_width_and_flare(
        cls, x, y, bottom_width, flare_ratio=1.2, height=None, **kwargs
    ):
        """Create basket from bottom width and flare ratio.

        Args:
            x: X-coordinate
            y: Y-coordinate
            bottom_width: Interior width at bottom
            flare_ratio: Ratio of top_width to bottom_width (default 1.2)
            height: Interior height (defaults to bottom_width * 1.5)
            **kwargs: Additional Basket parameters
        """
        top_width = bottom_width * flare_ratio
        if height is None:
            height = bottom_width * 1.5
        return cls(
            x=x,
            y=y,
            bottom_width=bottom_width,
            top_width=top_width,
            height=height,
            **kwargs,
        )

    @staticmethod
    def calculate_dimensions(scale: float, wall_thickness: float | None = None):
        """Calculate basket dimensions from scale parameter.

        This helper method computes the total dimensions of a basket given its scale,
        useful for positioning calculations before basket creation.

        Args:
            scale: Proportional scaling factor for the basket
            wall_thickness: Optional custom wall thickness (defaults to min(0.175, scale * 0.22))

        Returns:
            dict with keys:
                - 'bottom_width': Interior width at bottom
                - 'top_width': Interior width at top
                - 'height': Interior height
                - 'wall_thickness': Wall thickness
                - 'floor_thickness': Floor thickness (same as wall_thickness)
                - 'total_width': Total width including walls
                - 'total_height': Total height including floor
        """
        if wall_thickness is None:
            wall_thickness = min(0.175, scale * 0.22)

        bottom_width = 1.083 * scale
        top_width = 1.083 * scale * 1.25
        height = 2 * scale
        floor_thickness = wall_thickness
        total_width = bottom_width + 2 * wall_thickness
        total_height = height + floor_thickness

        return {
            "bottom_width": bottom_width,
            "top_width": top_width,
            "height": height,
            "wall_thickness": wall_thickness,
            "floor_thickness": floor_thickness,
            "total_width": total_width,
            "total_height": total_height,
        }

    @property
    def total_width(self):
        return self.bottom_width + 2 * self.wall_thickness

    @property
    def total_height(self):
        return self.height + self.floor_thickness

    def get_anchor_offset(self):
        """Get offset from bottom-center to the specified anchor point."""
        if self.anchor == "bottom_center":
            return (0, 0)
        elif self.anchor == "center":
            return (0, -self.height / 2)
        elif self.anchor == "top_center":
            return (0, -self.height)
        elif self.anchor == "bottom_left":
            return (self.bottom_width / 2, 0)
        elif self.anchor == "bottom_right":
            return (-self.bottom_width / 2, 0)
        elif self.anchor == "top_left":
            return (self.top_width / 2, -self.height)
        elif self.anchor == "top_right":
            return (-self.top_width / 2, -self.height)
        else:
            raise ValueError(f"Unknown anchor: {self.anchor}")


def circle_intersects_basket(
    cx: float, cy: float, radius: float, basket: "Basket"
) -> bool:
    """Return True if a circle overlaps any wall segment of a basket.

    Four axis-aligned wall slabs are checked: floor, left wall, right wall, and a
    virtual top cap.  The test uses nearest-point clamping to find the closest
    point on each slab to the circle center, then compares the squared distance
    against radius².

    The top cap has no corresponding physics fixture (baskets are open at the top),
    but it blocks action-ball placements inside the basket interior — which would
    be an exploit in levels where the goal is to land a ball in the basket.

    Anchor awareness: basket.x/y is the anchor point, not the geometric center.
    get_anchor_offset() returns the vector from bottom-center to the anchor in
    local space, which must be subtracted from the anchor position to recover the
    actual floor-bottom world coordinate.

    This is the authoritative implementation shared by InterphyreEnv (placement
    validation) and the oracle registry (solvability checking).  It intentionally
    uses only the Python standard library so that basket.py remains numpy-free.

    Args:
        cx: Circle center x-coordinate.
        cy: Circle center y-coordinate.
        radius: Circle radius.
        basket: Basket instance providing x, y, total_width, total_height,
            wall_thickness, and get_anchor_offset() attributes.

    Returns:
        True if the circle intersects at least one wall slab; False otherwise.
    """
    half_width = basket.total_width / 2
    wall_thickness = basket.wall_thickness

    # Apply the anchor offset to get the actual floor-bottom world coordinate.
    # basket.x/y is the anchor point; the local geometry is built with anchor_offset
    # applied so that the floor bottom sits at (basket.x + offset_x, basket.y + offset_y).
    anchor_offset_x, anchor_offset_y = basket.get_anchor_offset()

    basket_left = basket.x + anchor_offset_x - half_width
    basket_right = basket.x + anchor_offset_x + half_width
    basket_bottom = basket.y + anchor_offset_y
    basket_top = basket.y + anchor_offset_y + basket.total_height

    # Four axis-aligned wall slabs: floor, left, right, virtual top cap.
    walls = [
        (basket_left, basket_bottom, basket_right, basket_bottom + wall_thickness),
        (basket_left, basket_bottom, basket_left + wall_thickness, basket_top),
        (basket_right - wall_thickness, basket_bottom, basket_right, basket_top),
        (basket_left, basket_top - wall_thickness, basket_right, basket_top),
    ]

    radius_sq = radius**2
    for left, bottom, right, top in walls:
        nearest_x = max(left, min(cx, right))
        nearest_y = max(bottom, min(cy, top))
        if (cx - nearest_x) ** 2 + (cy - nearest_y) ** 2 <= radius_sq:
            return True
    return False


def create_basket(world: b2World, basket: "Basket", name: str, use_ccd: bool = False):
    """Create a Box2D physics body from a Basket object.

    The basket is built in local coordinates with bottom-center at origin,
    then positioned according to the basket's anchor point.

    Args:
        world (b2World): The Box2D physics world to create the basket in
        basket (Basket): The Basket object with geometry parameters
        name (str): Name to assign to the body's userData
        use_ccd (bool): Whether to enable continuous collision detection (bullet mode) (default: False)

    Returns:
        b2Body: Box2D body with basket fixtures

    Note:
        All floating point values are rounded to the configured PRECISION to ensure determinism.
    """
    x = round(float(basket.x), PRECISION)
    y = round(float(basket.y), PRECISION)
    angle_deg = round(float(basket.angle), PRECISION)
    angle_rad = round(angle_deg * b2_pi / 180, PRECISION)
    bw = round(float(basket.bottom_width), PRECISION)
    tw = round(float(basket.top_width), PRECISION)
    h = round(float(basket.height), PRECISION)
    wt = round(float(basket.wall_thickness), PRECISION)
    ft = round(float(basket.floor_thickness), PRECISION)
    anchor_offset_x, anchor_offset_y = basket.get_anchor_offset()
    anchor_offset_x = round(float(anchor_offset_x), PRECISION)
    anchor_offset_y = round(float(anchor_offset_y), PRECISION)
    density = round(float(basket.density), PRECISION)
    friction = round(float(basket.friction), PRECISION)
    restitution = round(float(basket.restitution), PRECISION)

    if basket.dynamic:
        body = world.CreateDynamicBody(position=(x, y), angle=angle_rad, bullet=use_ccd)
    else:
        body = world.CreateStaticBody(position=(x, y), angle=angle_rad, bullet=use_ccd)

    # Floor
    floor_shape = b2PolygonShape()
    floor_half_width = round((bw + 2 * wt) / 2, PRECISION)
    floor_half_height = round(ft / 2, PRECISION)
    floor_center_x = anchor_offset_x
    floor_center_y = round(anchor_offset_y + ft / 2, PRECISION)
    floor_shape.SetAsBox(
        floor_half_width,
        floor_half_height,
        (floor_center_x, floor_center_y),
        0,
    )
    body.CreateFixture(
        shape=floor_shape,
        density=density,
        friction=friction,
        restitution=restitution,
    )

    # Left wall (trapezoid)
    left_wall_vertices = [
        (
            round(-bw / 2 - wt + anchor_offset_x, PRECISION),
            round(ft + anchor_offset_y, PRECISION),
        ),
        (
            round(-tw / 2 - wt + anchor_offset_x, PRECISION),
            round(ft + h + anchor_offset_y, PRECISION),
        ),
        (
            round(-tw / 2 + anchor_offset_x, PRECISION),
            round(ft + h + anchor_offset_y, PRECISION),
        ),
        (
            round(-bw / 2 + anchor_offset_x, PRECISION),
            round(ft + anchor_offset_y, PRECISION),
        ),
    ]
    left_wall_shape = b2PolygonShape(vertices=left_wall_vertices)
    body.CreateFixture(
        shape=left_wall_shape,
        density=density,
        friction=friction,
        restitution=restitution,
    )

    # Right wall (trapezoid)
    right_wall_vertices = [
        (
            round(bw / 2 + wt + anchor_offset_x, PRECISION),
            round(ft + anchor_offset_y, PRECISION),
        ),
        (
            round(bw / 2 + anchor_offset_x, PRECISION),
            round(ft + anchor_offset_y, PRECISION),
        ),
        (
            round(tw / 2 + anchor_offset_x, PRECISION),
            round(ft + h + anchor_offset_y, PRECISION),
        ),
        (
            round(tw / 2 + wt + anchor_offset_x, PRECISION),
            round(ft + h + anchor_offset_y, PRECISION),
        ),
    ]
    right_wall_shape = b2PolygonShape(vertices=right_wall_vertices)
    body.CreateFixture(
        shape=right_wall_shape,
        density=density,
        friction=friction,
        restitution=restitution,
    )

    # Optional double walls for anti-tunneling
    if basket.double_walls:
        inner_gap = round(0.03, PRECISION)
        left_inner_vertices = [
            (
                round(-bw / 2 + inner_gap + anchor_offset_x, PRECISION),
                round(ft + anchor_offset_y, PRECISION),
            ),
            (
                round(-tw / 2 + inner_gap + anchor_offset_x, PRECISION),
                round(ft + h + anchor_offset_y, PRECISION),
            ),
            (
                round(-tw / 2 + inner_gap + wt / 2 + anchor_offset_x, PRECISION),
                round(ft + h + anchor_offset_y, PRECISION),
            ),
            (
                round(-bw / 2 + inner_gap + wt / 2 + anchor_offset_x, PRECISION),
                round(ft + anchor_offset_y, PRECISION),
            ),
        ]
        left_inner_shape = b2PolygonShape(vertices=left_inner_vertices)
        body.CreateFixture(
            shape=left_inner_shape,
            density=round(0.1, PRECISION),
            friction=friction,
            restitution=restitution,
        )

        right_inner_vertices = [
            (
                round(bw / 2 - inner_gap + anchor_offset_x, PRECISION),
                round(ft + anchor_offset_y, PRECISION),
            ),
            (
                round(bw / 2 - inner_gap - wt / 2 + anchor_offset_x, PRECISION),
                round(ft + anchor_offset_y, PRECISION),
            ),
            (
                round(tw / 2 - inner_gap - wt / 2 + anchor_offset_x, PRECISION),
                round(ft + h + anchor_offset_y, PRECISION),
            ),
            (
                round(tw / 2 - inner_gap + anchor_offset_x, PRECISION),
                round(ft + h + anchor_offset_y, PRECISION),
            ),
        ]
        right_inner_shape = b2PolygonShape(vertices=right_inner_vertices)
        body.CreateFixture(
            shape=right_inner_shape,
            density=round(0.1, PRECISION),
            friction=friction,
            restitution=restitution,
        )

    # Optional sensor fixture for success detection
    if basket.enable_sensor:
        sensor_height_ratio = round(float(basket.sensor_height_ratio), PRECISION)
        sensor_margin = round(float(basket.sensor_margin), PRECISION)
        sensor_height = round(h * sensor_height_ratio, PRECISION)
        sensor_width = round(bw - 2 * sensor_margin, PRECISION)
        sensor_center_y = round(ft + sensor_height / 2, PRECISION)

        sensor_shape = b2PolygonShape()
        sensor_shape.SetAsBox(
            round(sensor_width / 2, PRECISION),
            round(sensor_height / 2, PRECISION),
            (anchor_offset_x, round(anchor_offset_y + sensor_center_y, PRECISION)),
            0,
        )
        sensor_fixture = body.CreateFixture(
            shape=sensor_shape,
            density=0,
            isSensor=True,
        )
        sensor_fixture.userData = f"{name}_sensor"

    body.userData = name
    return body
