"""Tests for general utility functions."""
import pytest

from weave.utils import as_list

value_list = [1, 1.0, 'dummy', True, None, (), {}]


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
