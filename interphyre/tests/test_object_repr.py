"""Tests for PhyreObject __repr__ across all object types."""

from interphyre.objects.ball import Ball
from interphyre.objects.bar import Bar
from interphyre.objects.basket import Basket


def test_ball_repr_contains_color_and_position():
    ball = Ball(x=1.5, y=3.0, radius=0.75, color="red")
    r = repr(ball)
    assert "Ball" in r
    assert "red" in r
    assert "1.50" in r
    assert "3.00" in r
    assert "radius=0.75" in r


def test_bar_repr_contains_color_and_dimensions():
    bar = Bar(x=0.0, y=-2.0, length=4.0, thickness=0.3, color="blue")
    r = repr(bar)
    assert "Bar" in r
    assert "blue" in r
    assert "0.00" in r
    assert "-2.00" in r
    assert "length=4.00" in r
    assert "thickness=0.30" in r


def test_basket_repr_contains_color_and_dimensions():
    basket = Basket(x=0.0, y=-3.0, bottom_width=2.0, height=3.2, color="gray")
    r = repr(basket)
    assert "Basket" in r
    assert "gray" in r
    assert "width=2.00" in r
    assert "height=3.20" in r


def test_base_repr_default_color():
    """Default color is 'black' and should appear in repr."""
    ball = Ball(x=0.0, y=0.0)
    r = repr(ball)
    assert "black" in r
