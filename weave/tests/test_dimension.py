"""Tests for Dimension class.

Currently only testing a simple example for most cases, not testing
other examples of valid input. Could do this with 'parametrize' or
'hypothesis'. Examples of valid input:
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

from weave.dimension import Dimension

# Lists of wrong types to test exceptions
not_float = [1, 'dummy', True, None, [], (), {}]
not_numeric = ['dummy', True, None, [], (), {}]
not_str = [1, 1.0, True, None, [], (), {}]
not_dimension = not_str + [[value] for value in not_str]


# Test constructor types
@pytest.mark.parametrize('dimension', not_dimension)
def test_dimension_type(dimension):
    """Raise TypeError if `dimension` is not a str or list of str."""
    with pytest.raises(TypeError):
        Dimension(dimension, 'exponential', radius=0.5)


@pytest.mark.parametrize('kernel', not_str)
def test_kernel_type(kernel):
    """Raise TypeError if `kernel` is not a str."""
    with pytest.raises(TypeError):
        Dimension('dummy', kernel)


@pytest.mark.parametrize('distance', not_str)
def test_distance_type(distance):
    """Raise TypeError if `distance` is not a str."""
    if distance is not None:
        with pytest.raises(TypeError):
            Dimension('dummy', 'exponential', distance, radius=0.5)


@pytest.mark.parametrize('radius', not_numeric)
def test_exponential_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'exponential', radius=radius)


@pytest.mark.parametrize('radius', not_numeric)
def test_tricubic_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'tricubic', radius=radius, exponent=3)


@pytest.mark.parametrize('exponent', not_numeric)
def test_tricubic_exponent_type(exponent):
    """Raise TypeError if `exponent` is not an int or float."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'tricubic', radius=0.5, exponent=exponent)


@pytest.mark.parametrize('radius', not_float)
def test_depth_radius_type(radius):
    """Raise TypeError if `radius` is not a float."""
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


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_exponential_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'exponential', radius=radius)


def test_tricubic_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', 'tricubic', exponent=3)


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_tricubic_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'tricubic', radius=radius, exponent=3)


def test_tricubic_exponent_exist():
    """Raise KeyError if `exponent` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', 'tricubic', radius=0.5)


@pytest.mark.parametrize('exponent', [-1, -1.0, 0, 0.0])
def test_tricubic_exponent_value(exponent):
    """Raise ValueError if `exponenent` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'tricubic', radius=0.5, exponent=exponent)


def test_depth_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension(['dummy1', 'dummy2'], 'depth')


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


def test_exponential_distance_scalar_default():
    """`distance` is set to 'continuous' if not supplied."""
    dim = Dimension('dummy', 'exponential', radius=0.5)
    assert dim.distance == 'continuous'


def test_exponential_distance_vector_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension(['dummy1', 'dummy2'], 'exponential', radius=0.5)
    assert dim.distance == 'euclidean'


def test_tricubic_distance_scalar_default():
    """`distance` is set to 'continuous' if not supplied."""
    dim = Dimension('dummy', 'tricubic', radius=0.5, exponent=3)
    assert dim.distance == 'continuous'


def test_tricubic_distance_vector_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension(['dummy1', 'dummy2'], 'tricubic', radius=0.5, exponent=3)
    assert dim.distance == 'euclidean'


def test_depth_distance_default():
    """`distance` is set to 'hierarchical' if not supplied."""
    dim = Dimension(['dummy1', 'dummy2'], 'depth', radius=0.5)
    assert dim.distance == 'hierarchical'


@pytest.mark.filterwarnings('ignore:`kernel`')
def test_distance_changed_hierarchical():
    """`distance` is changed to 'hierarchical'.

    When `kernel` == 'depth', enforce `distance` == 'hierarchical'.

    """
    dim = Dimension(['dummy1', 'dummy2'], 'depth', 'euclidean', radius=0.5)
    assert dim.distance == 'hierarchical'


def test_distance_warning_hierarchical():
    """Warn when `distance` is changed to 'hierarchical'.

    When `kernel` == 'depth', enforce `distance` == 'hierarchical'.
    If `distance` is changed, produce a warning.

    """
    with pytest.warns(UserWarning):
        Dimension(['dummy1', 'dummy2'], 'depth', 'euclidean', radius=0.5)


@pytest.mark.filterwarnings('ignore:`dimension`')
@pytest.mark.parametrize('kernel', ['exponential', 'tricubic'])
def test_distance_changed_euclidean(kernel):
    """`distance` is changed to 'euclidean'.

    When `kernel` in {'exponential', 'tricubic'} and `dimension` is a
    list of str, enforce `distance` != 'continuous'.

    """
    dim = Dimension(['dummy1', 'dummy2'], kernel, 'continuous', radius=0.5,
                    exponent=3)
    assert dim.distance == 'euclidean'


@pytest.mark.parametrize('kernel', ['exponential', 'tricubic'])
def test_distance_warning_euclidean(kernel):
    """Warn when `distance` is changed to 'euclidean'.

    When `kernel` in {'exponential', 'tricubic'} and `dimension` is a
    list of str, enforce `distance` != 'continuous'. If `distance` is
    changed, produce a warning.

    """
    with pytest.warns(UserWarning):
        Dimension(['dummy1', 'dummy2'], kernel, 'continuous', radius=0.5,
                  exponent=3)


# Test getter behavior
def test_dimension_len():
    """Return values based on number of dimensions.

    Return a str if only one value in `dimension`, otherwise return a
    list of str.

    """
    dim1 = Dimension('dummy', 'exponential', radius=0.5)
    dim2 = Dimension(['dummy1', 'dummy2'], 'exponential', radius=0.5)
    assert isinstance(dim1.dimension, str)
    assert isinstance(dim2.dimension, list)
    assert len(dim2.dimension) == 2


# Test setter behavior
@pytest.mark.filterwarnings('ignore:`kernel`')
def test_kernel_pars_deleted():
    """Delete `pars` when `kernel` is changed."""
    dim = Dimension('dummy', 'exponential', radius=0.5)
    dim.kernel = 'tricubic'
    assert hasattr(dim, 'pars') is False


def test_kernel_pars_warning():
    """Warn that `pars` deleted when `kernel` is changed."""
    with pytest.warns(UserWarning):
        dim = Dimension('dummy', 'exponential', radius=0.5)
        dim.kernel = 'tricubic'


@pytest.mark.filterwarnings('ignore:`kernel`')
def test_setter_distance_changed_hierarchical():
    """`distance` is changed to 'hierarchical'.

    When `kernel` is changed to 'depth', enforce `distance` ==
    'hierarchical'.

    """
    dim = Dimension(['dummy1', 'dummy2'], 'exponential', radius=0.5)
    dim.kernel = 'depth'
    assert dim.distance == 'hierarchical'


def test_setter_distance_warning_hierarchical():
    """Warn when `distance` is changed to 'hierarchical'.

    When `kernel` is changed to 'depth', enforce `distance` ==
    'hierarchical'. If `distance` is changed, produce a warning.

    """
    with pytest.warns(UserWarning):
        dim = Dimension(['dummy1', 'dummy2'], 'exponential', radius=0.5)
        dim.kernel = 'depth'


@pytest.mark.filterwarnings('ignore:`dimension`')
@pytest.mark.parametrize('kernel', ['exponential', 'tricubic'])
def test_setter_distance_changed_euclidean(kernel):
    """`distance` is changed to 'euclidean'.

    When `kernel` in {'exponential', 'tricubic'} and `dimension` is a
    list of str, enforce `distance` != 'continuous'.

    """
    dim = Dimension(['dummy1', 'dummy2'], kernel, radius=0.5, exponent=3)
    dim.distance = 'continuous'
    assert dim.distance == 'euclidean'


@pytest.mark.parametrize('kernel', ['exponential', 'tricubic'])
def test_setter_distance_warning_euclidean(kernel):
    """Warn when `distance` is changed to 'euclidean'.

    When `kernel` in {'exponential', 'tricubic'} and `dimension` is a
    list of str, enforce `distance` != 'continuous'. If `distance` is
    changed, produce a warning.

    """
    with pytest.warns(UserWarning):
        dim = Dimension(['dummy1', 'dummy2'], kernel, radius=0.5, exponent=3)
        dim.distance = 'continuous'
