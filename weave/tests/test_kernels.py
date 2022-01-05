# pylint: disable=E0611
"""Tests for kernel functions.

In general, kernel functions should satisfy the following properties:
1. k(x, y) is real-valued, finite, and nonnegative
2. k(x, y) <= k(x', y') if d(x, y) > d(x', y')
   k(x, y) >= k(x', y') if d(x, y) < d(x', y')

"""
from hypothesis import given, settings
from hypothesis.strategies import floats
from numba.typed import Dict
import numpy as np

from weave.kernels import exponential, tricubic, depth

# Hypothesis types
my_nonneg = floats(min_value=0.0, max_value=1e5, allow_nan=False,
                   allow_infinity=False, allow_subnormal=False)
my_pos = floats(min_value=0.0, max_value=1e5, allow_nan=False,
                allow_infinity=False, allow_subnormal=False,
                exclude_min=True)
my_frac = floats(min_value=0.0, max_value=1.0, allow_nan=False,
                 allow_infinity=False, allow_subnormal=False,
                 exclude_min=True, exclude_max=True)


# Property 1: Output is a real-valued, finite, nonnegative float
def property_1(weight):
    """Output satisfies property 1."""
    assert np.isreal(weight)
    assert np.isfinite(weight)
    assert weight >= 0.0
    assert isinstance(weight, float)


@settings(deadline=None)
@given(my_nonneg, my_pos)
def test_exponential_type(distance, radius):
    """Exponential output satisfies property 1."""
    pars = Dict()
    pars['radius'] = radius
    weight = exponential(distance, pars)
    property_1(weight)


@settings(deadline=None)
@given(my_nonneg, my_pos, my_pos)
def test_tricubic_type(distance, radius, exponent):
    """Tricubic output satisfies property 1."""
    pars = Dict()
    pars['radius'] = radius
    pars['exponent'] = exponent
    weight = tricubic(distance, pars)
    property_1(weight)


@settings(deadline=None)
@given(my_nonneg, my_frac)
def test_depth_type(distance, radius):
    """Depth output satisfies property 1."""
    pars = Dict()
    pars['radius'] = radius
    weight = depth(distance, pars)
    property_1(weight)


# Property 2: Output decreases as distance increases
def property_2(distance_a, distance_b, weight_a, weight_b):
    """Output satisfies property 2."""
    if distance_a > distance_b:
        assert weight_a <= weight_b
    if distance_a < distance_b:
        assert weight_a >= weight_b


@given(my_nonneg, my_nonneg, my_pos)
def test_exponential_direction(distance_a, distance_b, radius):
    """Exponential output satisfies property 2."""
    pars = Dict()
    pars['radius'] = radius
    weight_a = exponential(distance_a, pars)
    weight_b = exponential(distance_b, pars)
    property_2(distance_a, distance_b, weight_a, weight_b)


@given(my_nonneg, my_nonneg, my_pos, my_pos)
def test_tricubic_direction(distance_a, distance_b, radius, exponent):
    """Tricubic output satisfies property 2."""
    pars = Dict()
    pars['radius'] = radius
    pars['exponent'] = exponent
    weight_a = tricubic(distance_a, pars)
    weight_b = tricubic(distance_b, pars)
    property_2(distance_a, distance_b, weight_a, weight_b)


@given(my_nonneg, my_nonneg, my_pos)
def test_depth_direction(distance_a, distance_b, radius):
    """Depth output satisfies property 2."""
    pars = Dict()
    pars['radius'] = radius
    weight_a = depth(distance_a, pars)
    weight_b = depth(distance_b, pars)
    property_2(distance_a, distance_b, weight_a, weight_b)


# Test specific output values
def test_same_country():
    """Test depth kernel with same country."""
    distance = 0
    pars = Dict()
    pars['radius'] = 0.9
    weight = depth(distance, pars)
    assert np.isclose(weight, 0.9)


def test_same_region():
    """Test depth kernel with same region."""
    distance = 1
    pars = Dict()
    pars['radius'] = 0.9
    weight = depth(distance, pars)
    assert np.isclose(weight, 0.09)


def test_same_super_region():
    """Test depth kernel with same super region."""
    distance = 2
    pars = Dict()
    pars['radius'] = 0.9
    weight = depth(distance, pars)
    assert np.isclose(weight, 0.01)


def test_different_super_region():
    """Test depth kernel with different super regions."""
    distance = 3
    pars = Dict()
    pars['radius'] = 0.9
    weight = depth(distance, pars)
    assert np.isclose(weight, 0.0)
