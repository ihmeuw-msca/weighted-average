# pylint: disable=C0103, W1114
"""Tests for distance functions.

In general, distance functions should satisfy the following properties:
1. d(x, y) is real-valued, finite, and nonnegative
2. d(x, y) == 0 if and only if x == y
3. d(x, y) == d(y, x) (symmetry)
4. d(x, y) <= d(x, z) + d(z, y) (triangle inequality)

Tests for `dictionary` distance function are not included because it is
based on user-supplied values.

TODO:
* Add tests for `get_typed_dict()`

"""
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, floats
from hypothesis.extra.numpy import arrays
import numpy as np
import pytest

from weave.distance import _check_dict, euclidean, tree

# Hypothesis types
my_integers = integers(min_value=-1e5, max_value=1e5)

# Lists of wrong types to test exceptions
not_numeric = ['dummy', True, None, [], (), {}]
not_dict = [1, 1.0, 'dummy', True, None, [], ()]
not_tuple = [1, 1.0, 'dummy', True, None, [], {}]


@composite
def my_floats(draw):
    """Return float rounded to 5 decimals."""
    my_float = draw(floats(min_value=-1e-4, max_value=1e4, allow_nan=False,
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
    assert isinstance(distance, np.float32)


@settings(deadline=None)
@given(my_arrays())
def test_euclidean_type(xy):
    """Euclidean output satisfies property 1."""
    x, y = xy
    distance = euclidean(x, np.atleast_2d(y))[0]
    property_1(distance)


@settings(deadline=None)
@given(my_arrays())
def test_tree_type(xy):
    """Tree output satisfies property 1."""
    x, y = xy
    distance = tree(x, np.atleast_2d(y))[0]
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
    distance = euclidean(x, np.atleast_2d(y))[0]
    property_2(x, y, distance)


@given(my_arrays())
def test_tree_zero(xy):
    """Tree output satisfies property 2."""
    x, y = xy
    distance = tree(x, np.atleast_2d(y))[0]
    property_2(x, y, distance)


# Property 3: Output is symmetric
@given(my_arrays())
def test_euclidean_symmetric(xy):
    """Euclidean output satisfies property 3."""
    x, y = xy
    distance_xy = euclidean(x, np.atleast_2d(y))[0]
    distance_yx = euclidean(y, np.atleast_2d(x))[0]
    assert np.isclose(distance_xy, distance_yx)


@given(my_arrays())
def test_tree_symmetric(xy):
    """Tree output satisfies property 3."""
    x, y = xy
    distance_xy = tree(x, np.atleast_2d(y))[0]
    distance_yx = tree(y, np.atleast_2d(x))[0]
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
    distance_xy = euclidean(x, np.atleast_2d(y))[0]
    distance_xz = euclidean(x, np.atleast_2d(z))[0]
    distance_zy = euclidean(z, np.atleast_2d(y))[0]
    property_4(distance_xy, distance_xz, distance_zy)


@given(my_arrays(n=3))
def test_tree_triangle(xyz):
    """Tree output satisfies property 4."""
    x, y, z = xyz
    distance_xy = tree(x, np.atleast_2d(y))[0]
    distance_xz = tree(x, np.atleast_2d(z))[0]
    distance_zy = tree(z, np.atleast_2d(y))[0]
    property_4(distance_xy, distance_xz, distance_zy)


# Test output shape
@given(my_arrays(n=3))
def test_euclidean_shape(xyz):
    """Euclidean output is length `y`."""
    x, y, z = xyz
    distance = euclidean(x, np.array([y, z]))
    assert distance.shape == (2,)


@given(my_arrays(n=3))
def test_tree_shape(xyz):
    """Tree output is length `y`."""
    x, y, z = xyz
    distance = tree(x, np.array([y, z]))
    assert distance.shape == (2,)


# Test specific output values
def test_same_country():
    """Test tree distance with same country."""
    x = np.array([1., 2., 3.])
    y = np.array([[1., 2., 3.]])
    assert np.isclose(tree(x, y)[0], 0.)


def test_same_region():
    """Test tree distance with same region."""
    x = np.array([1., 2., 3.])
    y = np.array([[1., 2., 4.]])
    assert np.isclose(tree(x, y)[0], 1.)


def test_same_super_region():
    """Test tree distance with same super region."""
    x = np.array([1., 2., 3.])
    y = np.array([[1., 4., 5.]])
    assert np.isclose(tree(x, y)[0], 2.)


def test_different_super_region():
    """Test tree distance with different super regions."""
    x = np.array([1., 2., 3.])
    y = np.array([[4., 5., 6.]])
    assert np.isclose(tree(x, y)[0], 3.)


# Test `_check_dict()`
@pytest.mark.parametrize('distance_dict', not_dict)
def test_dict_type(distance_dict):
    """Raise TypeError if `distance_dict` is not a dict."""
    with pytest.raises(TypeError):
        _check_dict(distance_dict)


@pytest.mark.parametrize('key', not_tuple)
@pytest.mark.parametrize('value', [1, 1.0])
def test_key_type(key, value):
    """Raise TypeError if keys are not all tuples."""
    with pytest.raises(TypeError):
        _check_dict({key: value})


@pytest.mark.parametrize('key1', not_numeric)
@pytest.mark.parametrize('key2', not_numeric)
@pytest.mark.parametrize('value', [1, 1.0])
def test_key_element_type(key1, key2, value):
    """Raise TypeError if key elements are not all int or float."""
    with pytest.raises(TypeError):
        _check_dict({(key1, key2): value})


@pytest.mark.parametrize('key1', [1, 1.0])
@pytest.mark.parametrize('key2', [1, 1.0])
@pytest.mark.parametrize('value', not_numeric)
def test_value_type(key1, key2, value):
    """Raise TypeError if values are not all int or float."""
    with pytest.raises(TypeError):
        _check_dict({(key1, key2): value})


def test_empty_dict():
    """Raise ValueError if `distance_dict` is empty."""
    with pytest.raises(ValueError):
        _check_dict({})


@pytest.mark.parametrize('key', [(), (1, ), (1., ), (1, 2, 3), (1., 2., 3.)])
@pytest.mark.parametrize('value', [1, 1.0])
def test_key_length(key, value):
    """Raise ValueError if keys are not all length 2."""
    with pytest.raises(ValueError):
        _check_dict({key: value})


@pytest.mark.parametrize('key1', [1, 1.0])
@pytest.mark.parametrize('key2', [1, 1.0])
@pytest.mark.parametrize('value', [-2, -2.0, -1, -1.0])
def test_value_nonnegative(key1, key2, value):
    """Raise ValueError if values are not all nonnegative."""
    with pytest.raises(ValueError):
        _check_dict({(key1, key2): value})
