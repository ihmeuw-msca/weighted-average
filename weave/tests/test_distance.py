"""Tests for distance functions."""
import numpy as np
import pytest

from weave.distance import Continuous, Hierarchical

from examples import data, levels

cont = Continuous()
hier = Hierarchical()


# Test input length
@pytest.mark.parametrize('dist', [cont, hier])
def test_scalar_length(dist):
    """Raise ValueError if input lengths do not match."""
    with pytest.raises(ValueError):
        dist(1, [1, 2])


@pytest.mark.parametrize('dist', [cont, hier])
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
    assert isinstance(cont(x, y), (float, np.floating))


def test_continuous_vector_type():
    """Output is float."""
    x = data.iloc[1][['age_mid', 'year_id']]
    y = data.iloc[3][['age_mid', 'year_id']]
    assert isinstance(cont(x, y), (float, np.floating))


def test_hierarchical_scalar_type():
    """Output is int."""
    x = data.iloc[1]['name']
    y = data.iloc[3]['name']
    assert isinstance(hier(x, y), (int, np.integer))


def test_hierarchical_vector_type():
    """Output is int."""
    x = data.iloc[1][levels]
    y = data.iloc[3][levels]
    assert isinstance(hier(x, y), (int, np.integer))


# Test output symmetric
@pytest.mark.parametrize('dim', ['age_mid', 'year_id'])
def test_continuous_scalar_symmetric(dim):
    """Output is symmetric."""
    x = data.iloc[1][dim]
    y = data.iloc[3][dim]
    assert cont(x, y) == cont(y, x)


def test_continuous_vector_symmetric():
    """Output is symmetric."""
    x = data.iloc[1][['age_mid', 'year_id']]
    y = data.iloc[3][['age_mid', 'year_id']]
    assert cont(x, y) == cont(y, x)


def test_hierarchical_scalar_symmetric():
    """Output is symmetric."""
    x = data.iloc[1]['name']
    y = data.iloc[3]['name']
    assert hier(x, y) == hier(y, x)


def test_hierarchical_vector_symmetric():
    """Output is symmetric."""
    x = data.iloc[1][levels]
    y = data.iloc[3][levels]
    assert hier(x, y) == hier(y, x)


# Test output values
def test_age_example():
    """Test continuous distance on age."""
    x = data.iloc[1]['age_mid']
    y = data.iloc[3]['age_mid']
    assert np.isclose(cont(x, y), 2.0)


def test_year_example():
    """Test continuous distance on year."""
    x = data.iloc[1]['year_id']
    y = data.iloc[3]['year_id']
    assert np.isclose(cont(x, y), 20.0)


def test_age_year_example():
    """Test continuous distance on age and year."""
    x = data.iloc[1][['age_mid', 'year_id']]
    y = data.iloc[3][['age_mid', 'year_id']]
    assert np.isclose(cont(x, y), 20.09975124224178)


def test_same_name():
    """Test hierarchical distance with same name."""
    x = data.iloc[0]['name']
    y = data.iloc[0]['name']
    assert np.isclose(hier(x, y), 0)


def test_different_name():
    """Test hierarchical distance with different name."""
    x = data.iloc[1]['name']
    y = data.iloc[3]['name']
    assert np.isclose(hier(x, y), 1)


def test_same_country():
    """Test hierarchical distance with same country."""
    x = data.iloc[0][levels]
    y = data.iloc[1][levels]
    assert np.isclose(hier(x, y), 0)


def test_same_region():
    """Test hierarchical distance with same region."""
    x = data.iloc[0][levels]
    y = data.iloc[2][levels]
    assert np.isclose(hier(x, y), 1)


def test_same_super_region():
    """Test hierarchical distance with same super region."""
    x = data.iloc[0][levels]
    y = data.iloc[3][levels]
    assert np.isclose(hier(x, y), 2)


def test_different_super_region():
    """Test hierarchical distance with different super region."""
    x = data.iloc[0][levels]
    y = data.iloc[4][levels]
    assert np.isclose(hier(x, y), 3)
