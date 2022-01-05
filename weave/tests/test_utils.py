"""Tests for general utility functions."""
import pytest

from weave.utils import as_list, flatten_list

value_list = [1, 1.0, 'dummy', True, None, (), {}]


# Test `as_list()`
@pytest.mark.parametrize('values', value_list)
def test_list_single_type(values):
    """Cast `values` as list if not already."""
    assert isinstance(as_list(values), list)


@pytest.mark.parametrize('values', value_list)
def test_list_single_value(values):
    """Returns list of `values`."""
    result = as_list(values)
    assert result[0] == values


@pytest.mark.parametrize('values', [[value] for value in value_list])
def test_list_multi(values):
    """Return `values` if already a list."""
    assert as_list(values) == values


# Test `flatten_list()`
@pytest.mark.parametrize('values', value_list)
def test_flatten_type(values):
    """Raise TypeError if `values` is not a list."""
    with pytest.raises(TypeError):
        flatten_list(values)


def test_flatten_not_flat():
    """Return a flattened list."""
    values = [['age', 'year'], [['sup_reg', 'reg', 'loc']]]
    assert flatten_list(values) == ['age', 'year', 'sup_reg', 'reg', 'loc']


def test_flatten_flat():
    """Return `values` if already flattened."""
    values = ['age', 'year', 'location']
    assert flatten_list(values) == values
