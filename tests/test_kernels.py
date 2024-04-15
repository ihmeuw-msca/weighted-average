"""Tests for kernel functions.

In general, kernel functions should satisfy the following properties:
1. k(x, y) is real-valued, finite, and nonnegative
2. k(x, y) <= k(x', y') if d(x, y) > d(x', y')
   k(x, y) >= k(x', y') if d(x, y) < d(x', y')

"""

from hypothesis import given, settings
from hypothesis.strategies import floats, integers
import numpy as np

from weave.kernels import exponential, depth, tricubic, inverse

# Hypothesis types
my_dist = floats(
    min_value=0.0,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
my_level = integers(min_value=1, max_value=10)
my_radius1 = floats(
    min_value=1e2,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
my_radius2 = floats(
    min_value=0.5,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    exclude_min=True,
    exclude_max=True,
)
my_exponent = floats(
    min_value=1e-1,
    max_value=1e1,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
my_version = ["codem", "stgpr"]


# Property 1: Output is a real-valued, finite, nonnegative float
def property_1(weight):
    """Output satisfies property 1."""
    assert np.isreal(weight)
    assert np.isfinite(weight)
    assert weight >= 0.0
    assert isinstance(weight, np.float32)


@settings(deadline=None)
@given(my_dist, my_radius1)
def test_exponential_type(distance, radius):
    """Exponential output satisfies property 1."""
    weight = exponential(distance, radius)
    property_1(weight)


@settings(deadline=None)
@given(my_dist, my_level, my_radius2)
def test_depth_type(distance, levels, radius):
    """Depth output satisfies property 1."""
    for version in my_version:
        weight = depth(distance, levels, radius, version)
        property_1(weight)


@settings(deadline=None)
@given(my_dist, my_radius1, my_exponent)
def test_tricubic_type(distance, radius, exponent):
    """Tricubic output satisfies property 1."""
    weight = tricubic(distance, radius, exponent)
    property_1(weight)


@settings(deadline=None)
@given(my_dist, my_radius1)
def test_inverse_type(distance, radius):
    """Inverse output satisfies property 1."""
    inverse_term = inverse(distance, radius)
    property_1(inverse_term)


# Property 2: Output decreases as distance increases
def property_2(distance_a, distance_b, weight_a, weight_b):
    """Output satisfies property 2."""
    if distance_a > distance_b:
        assert weight_a <= weight_b
    if distance_a < distance_b:
        assert weight_a >= weight_b


@given(my_dist, my_dist, my_radius1)
def test_exponential_direction(distance_a, distance_b, radius):
    """Exponential output satisfies property 2."""
    weight_a = exponential(distance_a, radius)
    weight_b = exponential(distance_b, radius)
    property_2(distance_a, distance_b, weight_a, weight_b)


@given(my_dist, my_dist, my_level, my_radius2)
def test_depth_direction(distance_a, distance_b, levels, radius):
    """Depth output satisfies property 2."""
    for version in my_version:
        weight_a = depth(distance_a, levels, radius, version)
        weight_b = depth(distance_b, levels, radius, version)
        property_2(distance_a, distance_b, weight_a, weight_b)


@given(my_dist, my_dist, my_radius1, my_exponent)
def test_tricubic_direction(distance_a, distance_b, radius, exponent):
    """Tricubic output satisfies property 2."""
    weight_a = tricubic(distance_a, radius, exponent)
    weight_b = tricubic(distance_b, radius, exponent)
    property_2(distance_a, distance_b, weight_a, weight_b)


@given(my_dist, my_dist, my_radius1)
def test_inverse_direction(distance_a, distance_b, radius):
    """Inverse output satisfies property 2."""
    weight_a = 1 / (inverse(distance_a, radius) + 0.1)
    weight_b = 1 / (inverse(distance_b, radius) + 0.1)
    property_2(distance_a, distance_b, weight_a, weight_b)


# Test specific output values
def test_same_country():
    """Test depth kernel with same country."""
    assert np.isclose(depth(0, 3, 0.9, "codem"), 0.9)
    assert np.isclose(depth(0, 3, 0.9, "stgpr"), 1)


def test_same_region():
    """Test depth kernel with same region."""
    assert np.isclose(depth(1, 3, 0.9, "codem"), 0.09)
    assert np.isclose(depth(1, 3, 0.9, "stgpr"), 0.9)


def test_same_super_region():
    """Test depth kernel with same super region."""
    assert np.isclose(depth(2, 3, 0.9, "codem"), 0.01)
    assert np.isclose(depth(2, 3, 0.9, "stgpr"), 0.81)


def test_different_super_region():
    """Test depth kernel with different super regions."""
    assert np.isclose(depth(3, 3, 0.9, "codem"), 0)
    assert np.isclose(depth(3, 3, 0.9, "stgpr"), 0)
