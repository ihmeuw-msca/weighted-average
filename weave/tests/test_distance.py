"""Tests for distance functions."""
import numpy as np
import pytest

from examples import data, distance_dict, levels


# Test input length
@pytest.mark.parametrize('dist', distance_dict.values())
def test_scalar_length(dist):
    """Raise ValueError if input lengths do not match."""
    with pytest.raises(ValueError):
        dist(1, [1, 2])


@pytest.mark.parametrize('dist', distance_dict.values())
def test_vector_length(dist):
    """Raise ValueError if input lenghts do not match."""
    with pytest.raises(ValueError):
        dist([1, 2], [1, 2, 3])


# Test output type
@pytest.mark.parametrize('dim', ['age_mid', 'year_id'])
def test_continuous_scalar_type(dim):
    """Output is float."""
    x = data.iloc[1][dim]
    y = data.iloc[3][dim]
    assert isinstance(distance_dict['cont'](x, y), (float, np.floating))


def test_continuous_vector_type():
    """Output is float."""
    x = data.iloc[1][['age_mid', 'year_id']]
    y = data.iloc[3][['age_mid', 'year_id']]
    assert isinstance(distance_dict['cont'](x, y), (float, np.floating))


def test_hierarchical_scalar_type():
    """Output is int."""
    x = data.iloc[1]['name']
    y = data.iloc[3]['name']
    assert isinstance(distance_dict['hier'](x, y), (int, np.integer))


def test_hierarchical_vector_type():
    """Output is int."""
    x = data.iloc[1][levels]
    y = data.iloc[3][levels]
    assert isinstance(distance_dict['hier'](x, y), (int, np.integer))


# Test output symmetric
@pytest.mark.parametrize('dim', ['age_mid', 'year_id'])
def test_continuous_scalar_symmetric(dim):
    """Output is symmetric."""
    x = data.iloc[1][dim]
    y = data.iloc[3][dim]
    assert distance_dict['cont'](x, y) == distance_dict['cont'](y, x)


def test_continuous_vector_symmetric():
    """Output is symmetric."""
    x = data.iloc[1][['age_mid', 'year_id']]
    y = data.iloc[3][['age_mid', 'year_id']]
    assert distance_dict['cont'](x, y) == distance_dict['cont'](y, x)


def test_hierarchical_scalar_symmetric():
    """Output is symmetric."""
    x = data.iloc[1]['name']
    y = data.iloc[3]['name']
    assert distance_dict['hier'](x, y) == distance_dict['hier'](y, x)


def test_hierarchical_vector_symmetric():
    """Output is symmetric."""
    x = data.iloc[1][levels]
    y = data.iloc[3][levels]
    assert distance_dict['hier'](x, y) == distance_dict['hier'](y, x)


# Test output values
def test_age_example():
    """Test continuous distance on age."""
    x = data.iloc[1]['age_mid']
    y = data.iloc[3]['age_mid']
    assert np.isclose(distance_dict['cont'](x, y), 2.0)


def test_year_example():
    """Test continuous distance on year."""
    x = data.iloc[1]['year_id']
    y = data.iloc[3]['year_id']
    assert np.isclose(distance_dict['cont'](x, y), 20.0)


def test_age_year_example():
    """Test continuous distance on age and year."""
    x = data.iloc[1][['age_mid', 'year_id']]
    y = data.iloc[3][['age_mid', 'year_id']]
    assert np.isclose(distance_dict['cont'](x, y), 20.09975124224178)


def test_same_name():
    """Test hierarchical distance with same name."""
    x = data.iloc[0]['name']
    y = data.iloc[0]['name']
    assert np.isclose(distance_dict['hier'](x, y), 0)


def test_different_name():
    """Test hierarchical distance with different name."""
    x = data.iloc[1]['name']
    y = data.iloc[3]['name']
    assert np.isclose(distance_dict['hier'](x, y), 1)


def test_same_country():
    """Test hierarchical distance with same country."""
    x = data.iloc[0][levels]
    y = data.iloc[1][levels]
    assert np.isclose(distance_dict['hier'](x, y), 0)


def test_same_region():
    """Test hierarchical distance with same region."""
    x = data.iloc[0][levels]
    y = data.iloc[2][levels]
    assert np.isclose(distance_dict['hier'](x, y), 1)


def test_same_super_region():
    """Test hierarchical distance with same super region."""
    x = data.iloc[0][levels]
    y = data.iloc[3][levels]
    assert np.isclose(distance_dict['hier'](x, y), 2)


def test_different_super_region():
    """Test hierarchical distance with different super region."""
    x = data.iloc[0][levels]
    y = data.iloc[4][levels]
    assert np.isclose(distance_dict['hier'](x, y), 3)
