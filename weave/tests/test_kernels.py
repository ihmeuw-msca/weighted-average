"""Tests for kernel functions."""
import numpy as np
import pytest

from weave.distance import Continuous
from weave.kernels import Exponential, Tricubic, Depth

from examples import data, distance_dict, levels, kernel_dict, test_dict


# Test constructor types
@pytest.mark.parametrize('kernel', [Exponential, Tricubic])
@pytest.mark.parametrize('dist', test_dict['other'])
def test_distance_type(kernel, dist):
    """Raise TypeError if `distance` not a distance function."""
    if dist is not None:
        with pytest.raises(TypeError):
            kernel(0.5, distance=dist)


@pytest.mark.parametrize('kernel', [Exponential, Tricubic])
@pytest.mark.parametrize('radius', test_dict['numeric'])
@pytest.mark.parametrize('dist', distance_dict.values())
def test_radius_type(kernel, radius, dist):
    """Raise TypeError if `radius` not an int or float."""
    with pytest.raises(TypeError):
        kernel(radius, distance=dist)


@pytest.mark.parametrize('radius', test_dict['float'])
def test_depth_radius_type(radius):
    """Raise TypeError if `radius` not a float."""
    with pytest.raises(TypeError):
        Depth(radius)


@pytest.mark.parametrize('lam', test_dict['numeric'])
@pytest.mark.parametrize('dist', distance_dict.values())
def test_lam_type(lam, dist):
    """Raise TypeError if `lam` not an int or float."""
    with pytest.raises(TypeError):
        Tricubic(0.5, lam, dist)


# Test constructor values
@pytest.mark.parametrize('kernel', [Exponential, Tricubic])
@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
@pytest.mark.parametrize('dist', distance_dict.values())
def test_radius_value(kernel, radius, dist):
    """Raise ValueError if `radius` not positive."""
    with pytest.raises(ValueError):
        kernel(radius, distance=dist)


@pytest.mark.parametrize('radius', [-1.0, 0.0, 1.0, 2.0])
def test_depth_radius_value(radius):
    """Raise ValueError if `radius` not in (0, 1)."""
    with pytest.raises(ValueError):
        Depth(radius)


@pytest.mark.parametrize('lam', [-1, -1.0, 0, 0.0])
@pytest.mark.parametrize('dist', distance_dict.values())
def test_lam_value(lam, dist):
    """Raise ValueError if `lam` not positive."""
    with pytest.raises(ValueError):
        Tricubic(0.5, lam, dist)


# Test input length
@pytest.mark.parametrize('kernel', kernel_dict.values())
def test_scalar_length(kernel):
    """Raise ValueError if input lengths do not match."""
    with pytest.raises(ValueError):
        kernel(1, [1, 2])


@pytest.mark.parametrize('kernel', kernel_dict.values())
def test_vector_length(kernel):
    """Raise ValueError if input lenghts do not match."""
    with pytest.raises(ValueError):
        kernel([1, 2], [1, 2, 3])


# Test input values
@pytest.mark.parametrize('dist', distance_dict.values())
def test_dist_rad(dist):
    """Raise ValueError if distance greater than `radius`."""
    with pytest.raises(ValueError):
        if isinstance(dist, Continuous):
            radius = 3.0
            columns = 'age_mid'
        else:
            radius = 2
            columns = levels
        kernel = Tricubic(radius, distance=dist)
        x = data.iloc[0][columns]
        y = data.iloc[4][columns]
        kernel(x, y)


# Test output type
@pytest.mark.parametrize('kernel', kernel_dict.values())
@pytest.mark.parametrize('columns', ['age_mid', levels])
def test_output_type(kernel, columns):
    """Output is a float."""
    x = data.iloc[0][columns]
    y = data.iloc[4][columns]
    assert isinstance(kernel(x, y), (float, np.floating))


# Test output values
def test_exponential_continuous():
    """Test exponential kernel with continuous distance."""
    x = data.iloc[0]['age_mid']
    y = data.iloc[4]['age_mid']
    kernel = kernel_dict['exp_cont']
    assert np.isclose(kernel(x, y), 0.00033546262790251185)


def test_exponential_hierarchical():
    """Test exponential kernel with hierarchical distance."""
    x = data.iloc[0][levels]
    y = data.iloc[4][levels]
    kernel = kernel_dict['exp_hier']
    assert np.isclose(kernel(x, y), 0.0024787521766663585)


def test_tricubic_continuous():
    """Test tricubic kernel with continuous distance."""
    x = data.iloc[0]['age_mid']
    y = data.iloc[4]['age_mid']
    kernel = kernel_dict['tri_cont']
    assert np.isclose(kernel(x, y), 0.5381833400915066)


def test_tricubic_hierarchical():
    """Test tricubic kernel with hierarchical distance."""
    x = data.iloc[0][levels]
    y = data.iloc[4][levels]
    kernel = kernel_dict['tri_hier']
    assert np.isclose(kernel(x, y), 0.0)


def test_same_name():
    """Test depth kernel with same name."""
    x = data.iloc[0]['name']
    y = data.iloc[0]['name']
    kernel = kernel_dict['depth']
    assert np.isclose(kernel(x, y), 0.9)


def test_different_name():
    """Test depth kernel with different name."""
    x = data.iloc[0]['name']
    y = data.iloc[4]['name']
    kernel = kernel_dict['depth']
    assert np.isclose(kernel(x, y), 0.09)


def test_same_country():
    """Test depth kernel with same country."""
    x = data.iloc[0][levels]
    y = data.iloc[1][levels]
    kernel = kernel_dict['depth']
    assert np.isclose(kernel(x, y), 0.9)


def test_same_region():
    """Test depth kernel with same region."""
    x = data.iloc[0][levels]
    y = data.iloc[2][levels]
    kernel = kernel_dict['depth']
    assert np.isclose(kernel(x, y), 0.09)


def test_same_super_region():
    """Test depth kernel with same super region."""
    x = data.iloc[0][levels]
    y = data.iloc[3][levels]
    kernel = kernel_dict['depth']
    assert np.isclose(kernel(x, y), 0.01)


def test_different_super_region():
    """Test depth kernel with different super regions."""
    x = data.iloc[0][levels]
    y = data.iloc[4][levels]
    kernel = kernel_dict['depth']
    assert np.isclose(kernel(x, y), 0.0)
