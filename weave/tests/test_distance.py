# pylint: disable=C0103, W1114
"""Tests for distance functions.

In general, distance functions should satisfy the following properties:
1. d(x, y) is real-valued, finite, and nonnegative
2. d(x, y) == 0 if and only if x == y
3. d(x, y) == d(y, x) (symmetry)
4. d(x, y) <= d(x, z) + d(z, y) (triangle inequality)

TODO:
* Add tests for `dictionary` distance function

"""
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, floats
from hypothesis.extra.numpy import arrays
import numpy as np

from weave.distance import euclidean, hierarchical

# Hypothesis types
my_integers = integers(min_value=-1e5, max_value=1e5)


@composite
def my_floats(draw):
    """Return float rounded to 5 decimals."""
    my_float = draw(floats(min_value=-1e-5, max_value=1e5, allow_nan=False,
                           allow_infinity=False, allow_subnormal=False))
    return np.around(my_float, decimals=5)


@composite
def my_arrays(draw, n=2):
    """Return n vectors of float with matching lengths."""
    m = draw(integers(min_value=1, max_value=5))
    vec_list = [draw(arrays(float, m, elements=my_floats()))
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
@given(my_arrays())
def test_euclidean_type(xy):
    """Euclidean output satisfies property 1."""
    x, y = xy
    distance = euclidean(x, y)
    property_1(distance)


@settings(deadline=None)
@given(my_arrays())
def test_hierarchical_type(xy):
    """Hierarchical output satisfies property 1."""
    x, y = xy
    distance = hierarchical(x, y)
    property_1(distance)


# Property 2: Output == 0 iff x == y
def property_2(x, y, distance):
    """Output satisfies property 2."""
    if np.isclose(distance, 0.0, rtol=0):
        assert np.allclose(x, y)
    if np.allclose(x, y):
        assert np.isclose(distance, 0.0, rtol=0)


@given(my_arrays())
def test_euclidean_zero(xy):
    """Euclidean output satisfies property 2."""
    x, y = xy
    distance = euclidean(x, y)
    property_2(x, y, distance)


@given(my_arrays())
def test_hierarchical_zero(xy):
    """Hierarchical output satisfies property 2."""
    x, y = xy
    distance = hierarchical(x, y)
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
def test_hierarchical_symmetric(xy):
    """Hierarchical output satisfies property 3."""
    x, y = xy
    distance_xy = hierarchical(x, y)
    distance_yx = hierarchical(y, x)
    assert np.isclose(distance_xy, distance_yx)


# Property 4: Triangle inequality
def property_4(distance_xy, distance_xz, distance_zy):
    """Output satisfies property 4."""
    distance_xy = np.around(distance_xy, decimals=5)
    distance_xzy = np.around(distance_xz + distance_zy, decimals=5)
    assert distance_xy <= distance_xzy


@given(my_arrays(n=3))
def test_euclidean_triangle(xyz):
    """Euclidean output satisfies property 4."""
    x, y, z = xyz
    distance_xy = euclidean(x, y)
    distance_xz = euclidean(x, z)
    distance_zy = euclidean(z, y)
    property_4(distance_xy, distance_xz, distance_zy)


@given(my_arrays(n=3))
def test_hierarchical_triangle(xyz):
    """Hierarchical output satisfies property 4."""
    x, y, z = xyz
    distance_xy = hierarchical(x, y)
    distance_xz = hierarchical(x, z)
    distance_zy = hierarchical(z, y)
    property_4(distance_xy, distance_xz, distance_zy)


# Test specific output values
def test_same_country():
    """Test hierarchical distance with same country."""
    x = np.array([1., 2., 3.])
    y = np.array([1., 2., 3.])
    assert np.isclose(hierarchical(x, y), 0.)


def test_same_region():
    """Test hierarchical distance with same region."""
    x = np.array([1., 2., 3.])
    y = np.array([1., 2., 4.])
    assert np.isclose(hierarchical(x, y), 1.)


def test_same_super_region():
    """Test hierarchical distance with same super region."""
    x = np.array([1., 2., 3.])
    y = np.array([1., 4., 5.])
    assert np.isclose(hierarchical(x, y), 2.)


def test_different_super_region():
    """Test hierarchical distance with different super regions."""
    x = np.array([1., 2., 3.])
    y = np.array([4., 5., 6.])
    assert np.isclose(hierarchical(x, y), 3.)
