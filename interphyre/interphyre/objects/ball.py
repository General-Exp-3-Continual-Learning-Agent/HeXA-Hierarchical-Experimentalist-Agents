from Box2D import b2Vec2, b2World

from interphyre.config import PRECISION

from .base import PhyreObject


class Ball(PhyreObject):
    """A circular physics object.

    Represents a ball or sphere in the physics simulation. Balls can be
    dynamic (affected by gravity and collisions) or static (fixed in place).

    Attributes:
        radius (float): Radius of the ball in simulation units (default: 0.5)

    Examples:
        # Create a dynamic red ball
        ball = Ball(x=0, y=5, radius=1.0, color="red")

        # Create a static platform ball
        platform = Ball(x=0, y=-3, radius=2.0, dynamic=False, color="gray")
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 0.5,
        **kwargs,
    ):
        """Initialize a Ball with position and radius.

        Args:
            x, y: Position coordinates
            radius: Radius of the ball in simulation units (default: 0.5)
            **kwargs: Additional PhyreObject properties (color, dynamic, etc.)
        """
        super().__init__(x=x, y=y, **kwargs)
        self.radius = radius

    def _repr_dimensions(self) -> str:
        return f"radius={self.radius:.2f}"


def create_ball(world: b2World, ball: Ball, name: str, use_ccd: bool = False):
    """Create a Box2D physics body from a Ball object.

    Converts a Ball data object into a Box2D physics body that can be
    simulated in the physics world.

    Args:
        world (b2World): The Box2D physics world to create the body in
        ball (Ball): The Ball object containing position and physical properties
        name (str): Unique identifier for the physics body
        use_ccd (bool): Whether to enable continuous collision detection (bullet mode) (default: False)

    Returns:
        b2Body: The created Box2D physics body

    Note:
        All floating point values are rounded to the configured PRECISION
        to ensure determinism. Box2D converts to float32 internally.
        Continuous collision detection (CCD) can be enabled via use_ccd parameter, but may
        reduce determinism.
    """
    x = round(float(ball.x), PRECISION)
    y = round(float(ball.y), PRECISION)
    radius = round(float(ball.radius), PRECISION)
    density = round(float(ball.density), PRECISION)
    friction = round(float(ball.friction), PRECISION)
    restitution = round(float(ball.restitution), PRECISION)
    linear_damping = round(float(ball.linear_damping), PRECISION)
    angular_damping = round(float(ball.angular_damping), PRECISION)

    body = (
        world.CreateDynamicBody(
            position=b2Vec2(x, y),
            angle=0,
            fixedRotation=False,
            bullet=use_ccd,
        )
        if ball.dynamic
        else world.CreateStaticBody(
            position=b2Vec2(x, y),
            angle=0,
            fixedRotation=False,
            bullet=use_ccd,
        )
    )
    body.CreateCircleFixture(
        radius=radius,
        density=density,
        friction=friction,
        restitution=restitution,
    )

    body.linearDamping = linear_damping
    body.angularDamping = angular_damping
    body.userData = name
    return body
