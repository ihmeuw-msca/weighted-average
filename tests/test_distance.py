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
import pytest

from weave.distance import euclidean, tree


# Hypothesis types
@composite
def my_floats(draw):
    """Return float32 rounded to 5 decimals."""
    my_float = draw(
        floats(
            min_value=-1e-5,
            max_value=1e5,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    return np.float32(np.around(my_float, decimals=5))


@composite
def my_arrays(draw, n=2):
    """Return n vectors of float32 with matching lengths."""
    m = draw(integers(min_value=1, max_value=5))
    array_list = [draw(arrays(np.float32, m, elements=my_floats())) for ii in range(n)]
    return array_list


# Property 1: Output is a real-valued, finite, nonnegative float
def property_1(distance):
    """Output satisfies property 1."""
    assert np.isreal(distance)
    assert np.isfinite(distance)
    assert distance >= 0.0
    assert isinstance(distance, np.float32)


@settings(deadline=None)
@given(my_arrays())
def test_euclidean_type(xy):
    """Euclidean output satisfies property 1."""
    x, y = xy
    distance = euclidean(x, y)
    property_1(distance)


@settings(deadline=None)
@given(my_arrays())
def test_tree_type(xy):
    """Tree output satisfies property 1."""
    x, y = xy
    x[0] = y[0]  # same roots
    distance = tree(x, y)
    property_1(distance)


# Property 2: Output == 0 iff x == y
def property_2(x, y, distance):
    """Output satisfies property 2."""
    if np.isclose(distance, 0.0):
        assert np.allclose(x, y, rtol=1e-6)
    if np.allclose(x, y, rtol=1e0 - 6):
        assert np.isclose(distance, 0.0)


@given(my_arrays())
def test_euclidean_zero(xy):
    """Euclidean output satisfies property 2."""
    x, y = xy
    distance = euclidean(x, y)
    property_2(x, y, distance)


@given(my_arrays())
def test_tree_zero(xy):
    """Tree output satisfies property 2."""
    x, y = xy
    x[0] = y[0]  # same roots
    distance = tree(x, y)
    property_2(x, y, distance)


# Property 3: Output is symmetric
@given(my_arrays())
def test_euclidean_symmetric(xy):
    """Euclidean output satisfies property 3."""
    x, y = xy
    distance_xy = euclidean(x, y)
    distance_yx = euclidean(y, x)
    assert np.isclose(distance_xy, distance_yx)


@given(my_arrays())
def test_tree_symmetric(xy):
    """Tree output satisfies property 3."""
    x, y = xy
    x[0] = y[0]  # same roots
    distance_xy = tree(x, y)
    distance_yx = tree(y, x)
    assert np.isclose(distance_xy, distance_yx)


# Property 4: Triangle inequality
def property_4(distance_xy, distance_xz, distance_zy):
    """Output satisfies property 4."""
    distance_xy = np.around(distance_xy, decimals=5)
    distance_xzy = np.around(distance_xz + distance_zy, decimals=5)
    try:
        assert distance_xy <= distance_xzy
    except AssertionError:
        assert np.isclose(distance_xy, distance_xzy)


@given(my_arrays(n=3))
def test_euclidean_triangle(xyz):
    """Euclidean output satisfies property 4."""
    x, y, z = xyz
    distance_xy = euclidean(x, y)
    distance_xz = euclidean(x, z)
    distance_zy = euclidean(z, y)
    property_4(distance_xy, distance_xz, distance_zy)


@given(my_arrays(n=3))
def test_tree_triangle(xyz):
    """Tree output satisfies property 4."""
    x, y, z = xyz
    x[0], y[0] = z[0], z[0]  # same roots
    distance_xy = tree(x, y)
    distance_xz = tree(x, z)
    distance_zy = tree(z, y)
    property_4(distance_xy, distance_xz, distance_zy)


# Test specific output values
def test_same_country():
    """Test tree output with same country."""
    x = np.array([1, 2, 4])
    y = np.array([1, 2, 4])
    assert np.isclose(tree(x, y), 0.0)


def test_same_region():
    """Test tree output with same region."""
    x = np.array([1, 2, 4])
    y = np.array([1, 2, 5])
    assert np.isclose(tree(x, y), 1.0)


def test_same_super_region():
    """Test tree output with same super region."""
    x = np.array([1, 2, 4])
    y = np.array([1, 3, 6])
    assert np.isclose(tree(x, y), 2.0)


def test_different_super_region():
    """Test tree output with different super regions."""
    x = np.array([1, 2, 4])
    y = np.array([7, 8, 9])
    assert np.isclose(tree(x, y), 3.0)
