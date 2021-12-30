"""Tests for Dimension class.

Length of dimension should also correspond to distance function:
* 'continuous': dimension = 1
* 'euclidean': dimension > 1
* 'hierarchical': dimension > 1

"""
import pytest

from weave.dimension_numba import Dimension

from examples import test_dict

dimension_list = ['dummy', ['dummy1', 'dummy2']]
kernel_list = ['exponential', 'tricubic', 'depth']
distance_list = ['continuous', 'euclidean', 'hierarchical']


# Test constructor types
@pytest.mark.parametrize('dimension', test_dict['dimension'])
def test_dimension_type(dimension):
    """Raise TypeError if `dimension` is not a str or list of str."""
    with pytest.raises(TypeError):
        Dimension(dimension, 'exponential', radius=0.5)


@pytest.mark.parametrize('kernel', test_dict['str'])
def test_kernel_type(kernel):
    """Raise TypeError if `kernel` is not a str."""
    with pytest.raises(TypeError):
        Dimension('dummy', kernel, radius=0.5)


@pytest.mark.parametrize('distance', test_dict['str'])
def test_distance_type(distance):
    """Raise TypeError if `distance` is not a str."""
    if distance is not None:
        with pytest.raises(TypeError):
            Dimension('dummy', 'exponential', distance, radius=0.5)


@pytest.mark.parametrize('radius', test_dict['numeric'])
def test_exponential_radius_type(radius):
    """Raise TypeError if `radius` is an invalid type."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'exponential', radius=radius)


@pytest.mark.parametrize('radius', test_dict['numeric'])
def test_tricubic_radius_type(radius):
    """Raise TypeError if `radius` is an invalid type."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'tricubic', radius=radius, exponent=3)


@pytest.mark.parametrize('exponent', test_dict['numeric'])
def test_tricubic_exponent_type(exponent):
    """Raise TypeError if `exponent` is an invalid type."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'tricubic', radius=0.5, exponent=exponent)


@pytest.mark.parametrize('radius', test_dict['numeric'])
def test_depth_radius_type(radius):
    """Raise TypeError if `radius` is an invalid type."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'depth', radius=radius)


# Test constructor values
def test_dimension_duplicates():
    """Raise ValueError if duplicates found in `dimension`."""
    pass


def test_kernel_value():
    """Raise ValueError if `kernel` is not valid."""
    pass


def test_exponential_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    pass


def test_tricubic_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    pass


def test_tricubic_exponent_exist():
    """Raise KeyError if `exponent` is not passed."""
    pass


def test_depth_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    pass


def test_exponential_radius_value():
    """Raise ValueError if `radius` is not valid."""
    pass


def test_tricubic_radius_value():
    """Raise ValueError if `radius` is not valid."""
    pass


def test_tricubic_exponent_value():
    """Raise ValueError if `exponenent` is not valid."""
    pass


def test_depth_radius_value():
    """Raise ValueError if `radius` is not valid."""
    pass


def test_no_extra_pars():
    """Only relevant parameters saved to `pars`."""
    pass


def test_distance_value():
    """Raise ValueError if `distance` is not valid."""
    pass


def test_exponential_distance_default():
    """`distance` is set to 'continuous' if not supplied."""
    pass


def test_tricubic_distance_default():
    """`distanc` is set to 'continuous' if not supplied."""
    pass


def test_depth_distance_default():
    """`distance` is set to 'hierarchical' if not supplied."""
    pass


def test_distance_changed():
    """`distance` is changed to 'hierarchical'.

    When `kernel` == 'depth', enforce `distance` == 'hierarchical'.

    """
    pass


def test_distance_warning():
    """Warn when `distance` is changed to 'hierarchical'.

    When `kernel` == 'depth', enforce `distance` == 'hierarchical'.
    If `distance` is changed, produce a warning.

    """
    pass


# Test getter behavior
def test_dimension_len():
    """Return values based on number of dimensions.

    Return a str if only one value in `dimension`, otherwise return a
    list of str.

    """
    pass


# Test setter behavior
def test_kernel_pars_deleted():
    """Delete `pars` when `kernel` is changed."""
    pass


def test_kernel_pars_warning():
    """Warn that `pars` deleted when `kernel` is changed."""
    pass


def test_kernel_distance_changed():
    """`distance` is changed to 'hierarchical'.

    When `kernel` is changed to 'depth', enforce `distance` ==
    'hierarchical'.

    """
    pass


def test_kernel_distance_warning():
    """Warn when `distance` is changed to 'hierarchical'.

    When `kernel` is changed to 'depth', enforce `distance` ==
    'hierarchical'. If `distance` is changed, produce a warning.

    """
    pass


# Test equality
def test_dimension_equal():
    """Return True if `dimension` equal."""
    pass


def test_dimension_not_equal():
    """Return False if `dimension` not equal."""
    pass
