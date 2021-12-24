"""Tests for distance functions."""
import numpy as np
import pytest

from weave.distance_numba import continuous, euclidean, hierarchical

from examples import data, levels

year_id = data['year_id'].values  # int
age_mid = data['age_mid'].values  # float
location = data[levels].values    # vector


# Test output type
def test_continuous_int():
    """Output is an int."""
    x = year_id[1]
    y = year_id[3]
    assert isinstance(continuous(x, y), int)


def test_continuous_float():
    """Output is a float."""
    x = age_mid[1]
    y = age_mid[3]
    assert isinstance(continuous(x, y), float)


def test_euclidean_float():
    """Output is a float."""
    x = location[1].astype(float)
    y = location[3].astype(float)
    assert isinstance(euclidean(x, y), float)


def test_hierarchical_int():
    """Output is an int."""
    x = location[1]
    y = location[3]
    assert isinstance(hierarchical(x, y), int)


# Test output is symmetric
@pytest.mark.parametrize('dim', [year_id, age_mid])
def test_continuous_symmetric(dim):
    """Output is symmetric."""
    x = dim[1]
    y = dim[3]
    assert np.isclose(continuous(x, y), continuous(y, x))


def test_euclidean_symmetric():
    """Output is symmetric."""
    x = location[1].astype(float)
    y = location[3].astype(float)
    assert np.isclose(euclidean(x, y), euclidean(y, x))


def test_hierarchical_symmetric():
    """Output is symmetric."""
    x = location[1]
    y = location[3]
    assert np.isclose(hierarchical(x, y), hierarchical(y, x))


# Test output values
def test_year_value():
    """Test continuous distance on year."""
    x = year_id[1]
    y = year_id[3]
    assert np.isclose(continuous(x, y), 20)


def test_age_value():
    """Test continuous distance on age."""
    x = age_mid[1]
    y = age_mid[3]
    assert np.isclose(continuous(x, y), 2.0)


def test_euclidean_value():
    """Test euclidean distance on location."""
    x = location[1].astype(float)
    y = location[3].astype(float)
    assert np.isclose(euclidean(x, y), 2.23606797749979)


def test_same_country():
    """Test hierarchical distance with same country."""
    x = location[0]
    y = location[1]
    assert np.isclose(hierarchical(x, y), 0)


def test_same_region():
    """Test hierarchical distance with same region."""
    x = location[0]
    y = location[2]
    assert np.isclose(hierarchical(x, y), 1)


def test_same_super_region():
    """Test hierarchical distance with same super region."""
    x = location[0]
    y = location[3]
    assert np.isclose(hierarchical(x, y), 2)


def test_different_super_region():
    """Test hierarchical distance with different super regions."""
    x = location[0]
    y = location[4]
    assert np.isclose(hierarchical(x, y), 3)
