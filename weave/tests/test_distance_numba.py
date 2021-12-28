# pylint: disable=C0103, W1114
"""Tests for distance functions.

In general, distance functions should satisfy the following properties:
1. d(x, y) is real-valued, finite, and nonnegative
2. d(x, y) == 0 if and only if x == y
3. d(x, y) == d(y, x) (symmetry)
4. d(x, y) <= d(x, z) + d(z, y) (triangle inequality)

"""
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, floats
from hypothesis.extra.numpy import arrays
import numpy as np

from weave.distance_numba import continuous, euclidean, hierarchical

# Hypothesis types
my_integers = integers(min_value=1e10, max_value=1e10)
my_floats = floats(min_value=1e10, max_value=1e10, allow_nan=False,
                   allow_infinity=False)


@composite
def int_arrays(draw, n=2):
    """Return n vectors of int with matching lengths."""
    m = draw(integers(min_value=1, max_value=10))
    vec_list = [draw(arrays(int, m, elements=my_integers))
                for ii in range(n)]
    return vec_list


@composite
def float_arrays(draw, n=2):
    """Return n vectors of float with matching lengths."""
    m = draw(integers(min_value=1, max_value=10))
    vec_list = [draw(arrays(float, m, elements=my_floats))
                for ii in range(n)]
    return vec_list


# Property 1: Output is a real-valued, finite, nonnegative float
def property_1(distance):
    """Output satisfies property 1."""
    assert np.isreal(distance)
    assert np.isfinite(distance)
    assert distance >= 0.0
    assert isinstance(distance, float)


@settings(deadline=None)
@given(my_integers, my_integers)
def test_continuous_type_int(x, y):
    """Continuous output satisfies property 1."""
    distance = continuous(x, y)
    property_1(distance)


@settings(deadline=None)
@given(my_floats, my_floats)
def test_continuous_type_float(x, y):
    """Continuous output satisfies property 1."""
    distance = continuous(x, y)
    property_1(distance)


@settings(deadline=None)
@given(float_arrays())
def test_euclidean_type(my_arrays):
    """Euclidean output satisfies property 1."""
    x, y = my_arrays
    distance = euclidean(x, y)
    property_1(distance)


@settings(deadline=None)
@given(int_arrays())
def test_hierarchical_type_int(my_arrays):
    """Hierarchical output satisfies property 1."""
    x, y = my_arrays
    distance = hierarchical(x, y)
    property_1(distance)


@settings(deadline=None)
@given(float_arrays())
def test_hierarchical_type_float(my_arrays):
    """Hierarchical output satisfies property 1."""
    x, y = my_arrays
    distance = hierarchical(x, y)
    property_1(distance)


# Property 2: Output == 0 iff x == y
def property_2(x, y, distance):
    """Output satisfies property 2."""
    if np.isclose(distance, 0.0):
        assert np.allclose(x, y)
    if np.allclose(x, y):
        assert np.isclose(distance, 0.0)


@given(my_integers, my_integers)
def test_continuous_zero_int(x, y):
    """Continuous output satisfies property 2."""
    distance = continuous(x, y)
    property_2(x, y, distance)


@given(my_floats, my_floats)
def test_continuous_zero_float(x, y):
    """Continuous output satisfies property 2."""
    distance = continuous(x, y)
    property_2(x, y, distance)


@given(float_arrays())
def test_euclidean_zero(my_arrays):
    """Euclidean output satisfies property 2."""
    x, y = my_arrays
    distance = euclidean(x, y)
    property_2(x, y, distance)


@given(int_arrays())
def test_hierarchical_zero_int(my_arrays):
    """Hierarchical output satisfies property 2."""
    x, y = my_arrays
    distance = hierarchical(x, y)
    property_2(x, y, distance)


@given(float_arrays())
def test_hierarchical_zero_float(my_arrays):
    """Hierarchical output satisfies property 2."""
    x, y = my_arrays
    distance = hierarchical(x, y)
    property_2(x, y, distance)


# Property 3: Output is symmetric
@given(my_integers, my_integers)
def test_continuous_symmetric_int(x, y):
    """Continuous output satisfies property 3."""
    distance_xy = continuous(x, y)
    distance_yx = continuous(y, x)
    assert distance_xy == distance_yx


@given(my_floats, my_floats)
def test_continuous_symmetric_float(x, y):
    """Continuous output satisfies property 3."""
    distance_xy = continuous(x, y)
    distance_yx = continuous(y, x)
    assert distance_xy == distance_yx


@given(float_arrays())
def test_euclidean_symmetric(my_arrays):
    """Euclidean output satisfies property 3."""
    x, y = my_arrays
    distance_xy = euclidean(x, y)
    distance_yx = euclidean(y, x)
    assert distance_xy == distance_yx


@given(int_arrays())
def test_hierarchical_symmetric_int(my_arrays):
    """Hierarchical output satisfies property 3."""
    x, y = my_arrays
    distance_xy = hierarchical(x, y)
    distance_yx = hierarchical(y, x)
    assert distance_xy == distance_yx


@given(float_arrays())
def test_hierarchical_symmetric_float(my_arrays):
    """Hierarchical output satisfies property 3."""
    x, y = my_arrays
    distance_xy = hierarchical(x, y)
    distance_yx = hierarchical(y, x)
    assert distance_xy == distance_yx


# Property 4: Triangle inequality
@given(my_integers, my_integers, my_integers)
def test_continuous_triangle_int(x, y, z):
    """Continuous output satisfies property 4."""
    distance_xy = continuous(x, y)
    distance_xz = continuous(x, z)
    distance_zy = continuous(z, y)
    assert distance_xy <= distance_xz + distance_zy


@given(my_floats, my_floats, my_floats)
def test_continuous_triangle_float(x, y, z):
    """Continuous output satisfies property 4."""
    distance_xy = continuous(x, y)
    distance_xz = continuous(x, z)
    distance_zy = continuous(z, y)
    assert distance_xy <= distance_xz + distance_zy


@given(float_arrays(n=3))
def test_euclidean_triangle(my_arrays):
    """Euclidean output satisfies property 4."""
    x, y, z = my_arrays
    distance_xy = euclidean(x, y)
    distance_xz = euclidean(x, z)
    distance_zy = euclidean(z, y)
    assert distance_xy <= distance_xz + distance_zy


@given(int_arrays(n=3))
def test_hierarchical_triangle_int(my_arrays):
    """Hierarchical output satisfies property 4."""
    x, y, z = my_arrays
    distance_xy = hierarchical(x, y)
    distance_xz = hierarchical(x, z)
    distance_zy = hierarchical(z, y)
    assert distance_xy <= distance_xz + distance_zy


@given(float_arrays(n=3))
def test_hierarchical_triangle_float(my_arrays):
    """Hierarchical output satisfies property 4."""
    x, y, z = my_arrays
    distance_xy = hierarchical(x, y)
    distance_xz = hierarchical(x, z)
    distance_zy = hierarchical(z, y)
    assert distance_xy <= distance_xz + distance_zy


# Test specific output values
def test_same_country():
    """Test hierarchical distance with same country."""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    assert np.isclose(hierarchical(x, y), 0)


def test_same_region():
    """Test hierarchical distance with same region."""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 4])
    assert np.isclose(hierarchical(x, y), 1)


def test_same_super_region():
    """Test hierarchical distance with same super region."""
    x = np.array([1, 2, 3])
    y = np.array([1, 4, 5])
    assert np.isclose(hierarchical(x, y), 2)


def test_different_super_region():
    """Test hierarchical distance with different super regions."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert np.isclose(hierarchical(x, y), 3)
