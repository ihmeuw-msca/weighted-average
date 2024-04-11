"""Tests for general utility functions."""

from hypothesis import given
from hypothesis.strategies import integers, floats
import numpy as np
import pytest

from weave.utils import as_list, flatten, is_number, is_int, is_float

# Example types
value_list = [1, 1.0, "dummy", True, None, (), {}]
not_int = [1.0, np.nan, np.inf, "dummy", True, None, [], (), {}]
not_float = [1, np.nan, np.inf, "dummy", True, None, (), {}]
not_number = [np.nan, np.inf, "dummy", True, None, [], (), {}]

# Hypothesis types
my_integers = integers(min_value=-1e5, max_value=1e5)
my_floats = floats(
    min_value=-1e-5, max_value=1e5, allow_nan=False, allow_infinity=False
)


# Test `as_list()`
@pytest.mark.parametrize("values", value_list)
def test_list_single_type(values):
    """Cast `values` as list if not already."""
    assert isinstance(as_list(values), list)


@pytest.mark.parametrize("values", value_list)
def test_list_single_value(values):
    """Returns list of `values`."""
    result = as_list(values)
    assert result[0] == values


@pytest.mark.parametrize("values", [[value] for value in value_list])
def test_list_multi(values):
    """Return `values` if already a list."""
    assert as_list(values) == values


# Test `flatten()`
@pytest.mark.parametrize("values", value_list)
def test_flatten_type(values):
    """Raise TypeError if `values` is not a list."""
    with pytest.raises(TypeError):
        flatten(values)


def test_flatten_not_flat():
    """Return a flattened list."""
    values = [["age", "year"], [["sup_reg", "reg", "loc"]]]
    assert flatten(values) == ["age", "year", "sup_reg", "reg", "loc"]


def test_flatten_flat():
    """Return `values` if already flattened."""
    values = ["age", "year", "location"]
    assert flatten(values) == values


# Test `is_int()`
@given(my_integers)
def test_int_true(value):
    """Return True if `value` is an int."""
    assert is_int(value)


@pytest.mark.parametrize("value", not_int)
def test_int_false(value):
    """Return False if `value` is not an int."""
    assert not is_int(value)


# Test `is_float()`
@given(my_floats)
def test_float_true(value):
    """Return True if `value` is a float."""
    assert is_float(value)


@pytest.mark.parametrize("value", not_float)
def test_float_false(value):
    """Return False if `value` is not a float."""
    assert not is_float(value)


# Test `is_number()`
@given(my_integers)
def test_number_int_true(value):
    """Return True if `value` is an int."""
    assert is_number(value)


@given(my_floats)
def test_number_float_true(value):
    """Return True if `value` is a float."""
    assert is_number(value)


@pytest.mark.parametrize("value", not_number)
def test_number_false(value):
    """Return False if `value` is not an int or float."""
    assert not is_number(value)
