# pylint: disable=C0103, E0611
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

from weave.kernels import depth, exponential, tricubic, _check_pars

# Hypothesis types
pos_int = integers(min_value=1, max_value=5)
npos_int = integers(min_value=-5, max_value=0)
pos_float = floats(min_value=0.0, max_value=1e5, allow_nan=False,
                   allow_infinity=False, allow_subnormal=False,
                   exclude_min=True)
npos_float = floats(min_value=-1e5, max_value=0.0, allow_nan=False,
                    allow_infinity=False, allow_subnormal=False)
nneg_float = floats(min_value=0.0, max_value=1e5, allow_nan=False,
                    allow_infinity=False, allow_subnormal=False)
pos_frac = floats(min_value=0.0, max_value=1.0, allow_nan=False,
                  allow_infinity=False, allow_subnormal=False,
                  exclude_min=True, exclude_max=True)
not_frac = floats(min_value=1.0, max_value=1e5, allow_nan=False,
                  allow_infinity=False, allow_subnormal=False)

# Lists of wrong types to test exceptions
not_dict = [1, 1.0, 'dummy', True, None, [], ()]
not_int = [1.0, 'dummy', True, None, [], (), {}]
not_float = [1, 'dummy', True, None, [], (), {}]
not_numeric = ['dummy', True, None, [], (), {}]
not_bool = [1, 1.0, 'dummy', None, [], (), {}]

# Radius values to test depth kernel
radius_vals = 0.1*np.arange(1, 10).astype(np.float32)


# Property 1: Output is a real-valued, finite, nonnegative float
def property_1(weight):
    """Output satisfies property 1."""
    assert np.isreal(weight)
    assert np.isfinite(weight)
    assert weight >= 0.0
    assert isinstance(weight, (float, np.floating))


@settings(deadline=None)
@given(nneg_float, pos_frac, pos_int)
def test_depth_type(distance, radius, levels):
    """Depth output satisfies property 1."""
    weight = depth(np.float32(distance), np.float32(radius), np.int32(levels))
    property_1(weight)


@settings(deadline=None)
@given(nneg_float, pos_float)
def test_exponential_type(distance, radius):
    """Exponential output satisfies property 1."""
    weight = exponential(distance, radius)
    property_1(weight)


@settings(deadline=None)
@given(nneg_float, pos_float, pos_float)
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


@given(nneg_float, nneg_float, pos_frac, pos_int)
def test_depth_direction(distance_a, distance_b, radius, levels):
    """Depth output satisfies property 2."""
    radius = np.float32(radius)
    levels = np.int32(levels)
    weight_a = depth(np.float32(distance_a), radius, levels)
    weight_b = depth(np.float32(distance_b), radius, levels)
    property_2(distance_a, distance_b, weight_a, weight_b)


@given(nneg_float, nneg_float, pos_float)
def test_exponential_direction(distance_a, distance_b, radius):
    """Exponential output satisfies property 2."""
    weight_a = exponential(distance_a, radius)
    weight_b = exponential(distance_b, radius)
    property_2(distance_a, distance_b, weight_a, weight_b)


@given(nneg_float, nneg_float, pos_float, pos_float)
def test_tricubic_direction(distance_a, distance_b, radius, exponent):
    """Tricubic output satisfies property 2."""
    weight_a = tricubic(distance_a, radius, exponent)
    weight_b = tricubic(distance_b, radius, exponent)
    property_2(distance_a, distance_b, weight_a, weight_b)


# Test specific output values
@pytest.mark.parametrize('radius', radius_vals)
def test_depth_levels_1(radius):
    """Depth kernel with 1 level."""
    levels = np.int32(1)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [1, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


@pytest.mark.parametrize('radius', radius_vals)
def test_depth_levels_2(radius):
    """Depth kernel with 2 levels."""
    levels = np.int32(2)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [radius, 1 - radius, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


@pytest.mark.parametrize('radius', radius_vals)
def test_depth_levels_3(radius):
    """Depth kernel with 3 levels."""
    levels = np.int32(3)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [radius, radius*(1 - radius), (1 - radius)**2, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


@pytest.mark.parametrize('radius', radius_vals)
def test_depth_levels_4(radius):
    """Depth kernel with 4 levels."""
    levels = np.int32(4)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [radius, radius*(1 - radius), radius*(1 - radius)**2,
               (1 - radius)**3, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


@pytest.mark.parametrize('radius', radius_vals)
def test_depth_levels_5(radius):
    """Depth kernel with 5 levels."""
    levels = np.int32(5)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [radius, radius*(1 - radius), radius*(1 - radius)**2,
               radius*(1 - radius)**3, (1 - radius)**4, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


# Test `_check_pars()`
@pytest.mark.parametrize('pars', not_dict)
def test_pars_type(pars):
    """Raise TypeError if `kernel_pars` is not a dict."""
    with pytest.raises(TypeError):
        _check_pars(pars, 'radius', 'pos_num')


@given(pos_float)
def test_pars_missing(par_val):
    """Raise KeyError if `pars` is missing a kernel parameter."""
    with pytest.raises(KeyError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'radius', 'pos_num')


@pytest.mark.parametrize('par_val', not_bool)
def test_pars_bool(par_val):
    """Raise TypeError if kernel parameter is not a bool."""
    with pytest.raises(TypeError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'bool')


@pytest.mark.parametrize('par_val', not_numeric)
def test_pars_num(par_val):
    """Raise TypeError if kernel parameter is not an int or float."""
    with pytest.raises(TypeError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_num')


@pytest.mark.parametrize('par_val', not_int)
def test_pars_int(par_val):
    """Raise TypeError if kernel parameter is not an int."""
    with pytest.raises(TypeError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_int')


@pytest.mark.parametrize('par_val', not_float)
def test_pars_float(par_val):
    """Raise TypeError if kernel parameter is not a float."""
    with pytest.raises(TypeError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_frac')


@given(npos_int)
def test_pars_pos_num_1(par_val):
    """Raise ValueError if kernel parameter is not positive."""
    with pytest.raises(ValueError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_num')


@given(npos_float)
def test_pars_pos_num_2(par_val):
    """Raise ValueError if kernel parameter is not positive."""
    with pytest.raises(ValueError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_num')


@given(npos_int)
def test_pars_pos_int(par_val):
    """Raise ValueError if kernel parameter is not positive."""
    with pytest.raises(ValueError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_int')


@given(npos_float)
def test_pars_frac_1(par_val):
    """Raise ValueError if kernel parameter is <= 0."""
    with pytest.raises(ValueError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_frac')


@given(not_frac)
def test_pars_frac_2(par_val):
    """Raise ValueError if kernel parameter is >= 1."""
    with pytest.raises(ValueError):
        pars = {'dummy': par_val}
        _check_pars(pars, 'dummy', 'pos_frac')
