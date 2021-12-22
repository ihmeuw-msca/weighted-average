"""Tests for general utility functions."""
import pytest

from weave.utils import as_list

from examples import test_dict


@pytest.mark.parametrize('values', test_dict['list'])
def test_list_single_type(values):
    """Cast `values` as list if not already."""
    assert isinstance(as_list(values), list)


@pytest.mark.parametrize('values', test_dict['list'])
def test_list_single_value(values):
    """Returns list of `values`."""
    result = as_list(values)
    assert result[0] == values


@pytest.mark.parametrize('values', [[value] for value in test_dict['list']])
def test_list_multi(values):
    """Return `values` if already a list."""
    assert as_list(values) == values
