# pylint: disable=E0611
"""Tests for kernel functions.

In general, kernel functions should satisfy the following properties:
1. k(x, y) is real-valued, finite, and nonnegative
2. k(x, y) <= k(x', y') if d(x, y) > d(x', y')
   k(x, y) >= k(x', y') if d(x, y) < d(x', y')

TODO:
* Add tests for `get_typed_pars()`
* Add test for positive integer parameter

"""
from hypothesis import given, settings
from hypothesis.strategies import integers, floats
import numpy as np
import pytest

from weave.kernels import exponential, tricubic, depth, _check_pars

# Hypothesis types
my_int = integers(min_value=1, max_value=5)
my_pos = floats(min_value=0.0, max_value=1e5, allow_nan=False,
                allow_infinity=False, allow_subnormal=False, exclude_min=True)
my_nonpos = floats(min_value=-1e5, max_value=0.0, allow_nan=False,
                   allow_infinity=False, allow_subnormal=False)
my_nonneg = floats(min_value=0.0, max_value=1e5, allow_nan=False,
                   allow_infinity=False, allow_subnormal=False)
my_frac = floats(min_value=0.0, max_value=1.0, allow_nan=False,
                 allow_infinity=False, allow_subnormal=False, exclude_min=True,
                 exclude_max=True)
my_notfrac = floats(min_value=1.0, max_value=1e5, allow_nan=False,
                    allow_infinity=False, allow_subnormal=False)

# Lists of wrong types to test exceptions
not_dict = [1, 1.0, 'dummy', True, None, [], ()]
not_int = [1.0, 'dummy', True, None, [], (), {}]
not_float = [1, 'dummy', True, None, [], (), {}]
not_numeric = ['dummy', True, None, [], (), {}]
not_bool = [1, 1.0, 'dummy', None, [], (), {}]


# Property 1: Output is a real-valued, finite, nonnegative float
def property_1(weight):
    """Output satisfies property 1."""
    assert np.isreal(weight)
    assert np.isfinite(weight)
    assert weight >= 0.0
    assert isinstance(weight, (float, np.floating))


@settings(deadline=None)
@given(my_nonneg, my_frac, my_int)
def test_depth_type(distance, radius, levels):
    """Depth output satisfies property 1."""
    weight = depth(np.float32(distance), np.float32(radius), np.int32(levels))
    property_1(weight)


@settings(deadline=None)
@given(my_nonneg, my_pos)
def test_exponential_type(distance, radius):
    """Exponential output satisfies property 1."""
    weight = exponential(distance, radius)
    property_1(weight)


@settings(deadline=None)
@given(my_nonneg, my_pos, my_pos)
def test_tricubic_type(distance, radius, exponent):
    """Tricubic output satisfies property 1."""
    weight = tricubic(distance, radius, exponent)
    property_1(weight)


# Property 2: Output decreases as distance increases
def property_2(distance_a, distance_b, weight_a, weight_b):
    """Output satisfies property 2."""
    if distance_a > distance_b:
        assert weight_a <= weight_b
    if distance_a < distance_b:
        assert weight_a >= weight_b


@given(my_nonneg, my_nonneg, my_frac, my_int)
def test_depth_direction(distance_a, distance_b, radius, levels):
    """Depth output satisfies property 2."""
    radius = np.float32(radius)
    levels = np.int32(levels)
    weight_a = depth(np.float32(distance_a), radius, levels)
    weight_b = depth(np.float32(distance_b), radius, levels)
    property_2(distance_a, distance_b, weight_a, weight_b)


@given(my_nonneg, my_nonneg, my_pos)
def test_exponential_direction(distance_a, distance_b, radius):
    """Exponential output satisfies property 2."""
    weight_a = exponential(distance_a, radius)
    weight_b = exponential(distance_b, radius)
    property_2(distance_a, distance_b, weight_a, weight_b)


@given(my_nonneg, my_nonneg, my_pos, my_pos)
def test_tricubic_direction(distance_a, distance_b, radius, exponent):
    """Tricubic output satisfies property 2."""
    weight_a = tricubic(distance_a, radius, exponent)
    weight_b = tricubic(distance_b, radius, exponent)
    property_2(distance_a, distance_b, weight_a, weight_b)


# Test specific output values
def test_same_country():
    """Test depth kernel with same country."""
    distance = np.float32(0)
    radius = np.float32(0.9)
    levels = np.int32(3)
    weight = depth(distance, radius, levels)
    assert np.isclose(weight, 0.9)


def test_same_region():
    """Test depth kernel with same region."""
    distance = np.float32(1)
    radius = np.float32(0.9)
    levels = np.int32(3)
    weight = depth(distance, radius, levels)
    assert np.isclose(weight, 0.09)


def test_same_super_region():
    """Test depth kernel with same super region."""
    distance = np.float32(2)
    radius = np.float32(0.9)
    levels = np.int32(3)
    weight = depth(distance, radius, levels)
    assert np.isclose(weight, 0.01)


def test_different_super_region():
    """Test depth kernel with different super regions."""
    distance = np.float32(3)
    radius = np.float32(0.9)
    levels = np.int32(3)
    weight = depth(distance, radius, levels)
    assert np.isclose(weight, 0.0)


# Test `_check_pars()`
@pytest.mark.parametrize('pars', not_dict)
def test_pars_type(pars):
    """Raise TypeError if `kernel_pars` is not a dict."""
    with pytest.raises(TypeError):
        _check_pars(pars, 'radius', 'pos_num')


@given(my_pos)
def test_pars_missing(par_val):
    """Raise KeyError if `pars` is missing a kernel parameter."""
    with pytest.raises(KeyError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'radius', 'pos_num')


@pytest.mark.parametrize('par_val', not_numeric)
def test_pars_num(par_val):
    """Raise TypeError if kernel parameter is not an int or float."""
    with pytest.raises(TypeError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_num')


@pytest.mark.parametrize('par_val', not_float)
def test_pars_float(par_val):
    """Raise TypeError if kernel parameter is not a float."""
    with pytest.raises(TypeError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_frac')


@pytest.mark.parametrize('par_val', not_bool)
def test_pars_bool(par_val):
    """Raise TypeError if kernel parameter is not a bool."""
    with pytest.raises(TypeError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'bool')


@given(my_nonpos)
def test_pars_pos_num(par_val):
    """Raise ValueError if kernel parameter is not positive."""
    with pytest.raises(ValueError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_num')


@given(my_nonpos)
def test_pars_pos_frac(par_val):
    """Raise ValueError if kernel parameter is not positive."""
    with pytest.raises(ValueError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_frac')


@given(my_notfrac)
def test_pars_frac(par_val):
    """Raise ValueError if kernel parameter is >= 1."""
    with pytest.raises(ValueError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_frac')
