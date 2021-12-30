"""Tests for Dimension class.

Length of dimension should also correspond to distance function:
* 'continuous': dimension = 1
* 'euclidean': dimension > 1
* 'hierarchical': dimension > 1
* 'depth' -> 'hierarchical': dimension > 1

Where would I check this? Can I make `dimension` immutable?
Can I just set self._dimension = tuple()?
Solution: Add error in setter if already exists.
Add to documentation somewhere? Rethink where checks happen for
dimension/kernel/distance combinations?

Currently only testing a simple example for most cases, not testing
other examples of valid input. Could do this with parametrize or
hypothesis. Examples of valid input:
* 'dummy', 'exponential', None, radius=0.5
* 'dummy', 'exponential', 'continuous', radius=0.5
* ['dummy1', 'dummy2'], 'exponential', 'euclidean', radius=0.5
* ['dummy1', 'dummy2'], 'exponential', 'hierarchical', radius=0.5
* 'dummy', 'tricubic', None, radius=0.5, exponent=3
* 'dummy', 'tricubic', 'continuous', radius=0.5, exponent=3
* ['dummy1', 'dummy2'], 'tricubic', 'euclidean', radius=0.5, exponent=3
* ['dummy1', 'dummy2'], 'tricubic', 'hierarchical', radius=0.5, exponent=3
* ['dummy1', 'dummy2'], 'depth', None, radius=0.5
* ['dummy1', 'dummy2'], 'depth', 'hierarchical', radius=0.5

"""
import pytest

from weave.dimension_numba import Dimension

from examples import test_dict


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
        Dimension('dummy', kernel)


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
    with pytest.raises(ValueError):
        Dimension(['dummy', 'dummy'], 'exponential', radius=0.5)


def test_kernel_value():
    """Raise ValueError if `kernel` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'dummy')


def test_exponential_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', 'exponential')


def test_tricubic_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', 'tricubic', exponent=3)


def test_tricubic_exponent_exist():
    """Raise KeyError if `exponent` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', 'tricubic', radius=0.5)


def test_depth_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension(['dummy1', 'dummy2'], 'depth')


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_exponential_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'exponential', radius=radius)


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_tricubic_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'tricubic', radius=radius, exponent=3)


@pytest.mark.parametrize('exponent', [-1, -1.0, 0, 0.0])
def test_tricubic_exponent_value(exponent):
    """Raise ValueError if `exponenent` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'tricubic', radius=0.5, exponent=exponent)


@pytest.mark.parametrize('radius', [-1.0, 0.0, 1.0, 2.0])
def test_depth_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension(['dummy1', 'dummy2'], 'depth', radius=radius)


@pytest.mark.parametrize('kernel', ['exponential', 'tricubic'])
def test_no_extra_pars(kernel):
    """Only relevant parameters saved to `pars`."""
    dim = Dimension('dummy', kernel, radius=0.5, exponent=3, dummy=100)
    if kernel == 'tricubic':
        assert 'exponent' in dim.pars
    else:
        assert 'exponent' not in dim.pars
    assert 'radius' in dim.pars
    assert 'dummy' not in dim.pars


def test_distance_value():
    """Raise ValueError if `distance` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'exponential', 'dummy', radius=0.5)


def test_exponential_distance_default():
    """`distance` is set to 'continuous' if not supplied."""
    dim = Dimension('dummy', 'exponential', radius=0.5)
    assert dim.distance == 'continuous'


def test_tricubic_distance_default():
    """`distance` is set to 'continuous' if not supplied."""
    dim = Dimension('dummy', 'tricubic', radius=0.5, exponent=3)
    assert dim.distance == 'continuous'


def test_depth_distance_default():
    """`distance` is set to 'hierarchical' if not supplied."""
    dim = Dimension(['dummy1', 'dummy2'], 'depth', radius=0.5)
    assert dim.distance == 'hierarchical'


@pytest.mark.filterwarnings("ignore:`kernel` == 'depth'")
def test_distance_changed():
    """`distance` is changed to 'hierarchical'.

    When `kernel` == 'depth', enforce `distance` == 'hierarchical'.

    """
    dim = Dimension(['dummy1', 'dummy2'], 'depth', 'euclidean', radius=0.5)
    assert dim.distance == 'hierarchical'


def test_distance_warning():
    """Warn when `distance` is changed to 'hierarchical'.

    When `kernel` == 'depth', enforce `distance` == 'hierarchical'.
    If `distance` is changed, produce a warning.

    """
    with pytest.warns(UserWarning):
        Dimension(['dummy1', 'dummy2'], 'depth', 'euclidean', radius=0.5)


# Test getter behavior
def test_dimension_len():
    """Return values based on number of dimensions.

    Return a str if only one value in `dimension`, otherwise return a
    list of str.

    """
    dim1 = Dimension('dummy', 'exponential', radius=0.5)
    dim2 = Dimension(['dummy1', 'dummy2'], 'exponential', 'euclidean',
                     radius=0.5)
    assert isinstance(dim1.dimension, str)
    assert isinstance(dim2.dimension, list)
    assert len(dim2.dimension) == 2


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
