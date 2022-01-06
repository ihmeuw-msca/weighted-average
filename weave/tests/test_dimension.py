"""Tests for Dimension class.

Currently only testing a simple example for most cases, not testing
other examples of valid input. Could do this with 'parametrize' or
'hypothesis'. Examples of valid input:
* `kernel` == 'exponential'
    - `dimension` in {'dummy', ['dummy1', 'dummy2']}
    - `pars` == {'radius': 0.5}
    - `distance` in {'euclidean', 'hierarchical', None}
* `kernel == 'tricubic'
    - `dimension` in {'dummy', ['dummy1', 'dummy2']}
    - `pars` == {'radius': 0.5, 'exponent': 3}
    - `distance` in {'euclidean', 'hierarchical', one}
* `kernel == 'depth'
    - `dimension` in {'dummy', ['dummy1', 'dummy2']}
    - `pars` == {'radius': 0.5}
    - `distance` in {'hierarchical', None}

"""
import pytest

from weave.dimension import Dimension

# Lists of wrong types to test exceptions
not_float = [1, 'dummy', True, None, [], (), {}]
not_numeric = ['dummy', True, None, [], (), {}]
not_str = [1, 1.0, True, None, [], (), {}]
not_dimension = not_str + [[value] for value in not_str]
pars = {'radius': 0.5, 'exponent': 3}


# Test constructor types
@pytest.mark.parametrize('dimension', not_dimension)
def test_dimension_type(dimension):
    """Raise TypeError if `dimension` is not a str or list of str."""
    with pytest.raises(TypeError):
        Dimension(dimension, 'exponential', pars)


@pytest.mark.parametrize('kernel', not_str)
def test_kernel_type(kernel):
    """Raise TypeError if `kernel` is not a str."""
    with pytest.raises(TypeError):
        Dimension('dummy', kernel, pars)


@pytest.mark.parametrize('distance', not_str)
def test_distance_type(distance):
    """Raise TypeError if `distance` is not a str."""
    if distance is not None:
        with pytest.raises(TypeError):
            Dimension('dummy', 'exponential', pars, distance)


@pytest.mark.parametrize('radius', not_numeric)
def test_exponential_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'exponential', {'radius': radius})


@pytest.mark.parametrize('radius', not_numeric)
def test_tricubic_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'tricubic', {'radius': radius, 'exponent': 3})


@pytest.mark.parametrize('exponent', not_numeric)
def test_tricubic_exponent_type(exponent):
    """Raise TypeError if `exponent` is not an int or float."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'tricubic', {'radius': 0.5, 'exponent': exponent})


@pytest.mark.parametrize('radius', not_float)
def test_depth_radius_type(radius):
    """Raise TypeError if `radius` is not a float."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'depth', {'radius': radius})


# Test constructor values
def test_dimension_duplicates():
    """Raise ValueError if duplicates found in `dimension`."""
    with pytest.raises(ValueError):
        Dimension(['dummy', 'dummy'], 'exponential', pars)


def test_kernel_value():
    """Raise ValueError if `kernel` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'dummy', pars)


def test_exponential_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', 'exponential', {'dummy': 100})


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_exponential_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'exponential', {'radius': radius, 'exponent': 3})


def test_tricubic_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', 'tricubic', {'exponent': 3})


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_tricubic_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'tricubic', {'radius': radius, 'exponent': 3})


def test_tricubic_exponent_exist():
    """Raise KeyError if `exponent` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', 'tricubic', {'radius': 0.5})


@pytest.mark.parametrize('exponent', [-1, -1.0, 0, 0.0])
def test_tricubic_exponent_value(exponent):
    """Raise ValueError if `exponenent` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'tricubic', {'radius': 0.5, 'exponent': exponent})


def test_depth_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension(['dummy1', 'dummy2'], 'depth', {'dummy', 100})


@pytest.mark.parametrize('radius', [-1.0, 0.0, 1.0, 2.0])
def test_depth_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': radius, 'exponent': 3}
        Dimension(['dummy1', 'dummy2'], 'depth', bad_pars)


@pytest.mark.parametrize('kernel', ['exponential', 'tricubic'])
def test_no_extra_pars(kernel):
    """Only relevant parameters saved to `pars`."""
    bad_pars = {'radius': 0.5, 'exponent': 3, 'dummy': 100}
    dim = Dimension('dummy', kernel, bad_pars)
    if kernel == 'tricubic':
        assert 'exponent' in dim.pars
    else:
        assert 'exponent' not in dim.pars
    assert 'radius' in dim.pars
    assert 'dummy' not in dim.pars


def test_distance_value():
    """Raise ValueError if `distance` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'exponential', pars, 'dummy')


def test_exponential_distance_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension('dummy', 'exponential', pars)
    assert dim.distance == 'euclidean'


def test_tricubic_distance_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension('dummy', 'tricubic', pars)
    assert dim.distance == 'euclidean'


def test_depth_distance_default():
    """`distance` is set to 'hierarchical' if not supplied."""
    dim = Dimension(['dummy1', 'dummy2'], 'depth', pars)
    assert dim.distance == 'hierarchical'


@pytest.mark.filterwarnings('ignore:`kernel`')
def test_distance_changed_hierarchical():
    """`distance` is changed to 'hierarchical'.

    When `kernel` == 'depth', enforce `distance` == 'hierarchical'.

    """
    dim = Dimension(['dummy1', 'dummy2'], 'depth', pars, 'euclidean')
    assert dim.distance == 'hierarchical'


def test_distance_warning_hierarchical():
    """Warn when `distance` is changed to 'hierarchical'.

    When `kernel` == 'depth', enforce `distance` == 'hierarchical'.
    If `distance` is changed, produce a warning.

    """
    with pytest.warns(UserWarning):
        Dimension(['dummy1', 'dummy2'], 'depth', pars, 'euclidean')


# Test getter behavior
def test_dimension_len():
    """Return values based on number of dimensions.

    Return a str if only one value in `dimension`, otherwise return a
    list of str.

    """
    dim1 = Dimension('dummy', 'exponential', pars)
    dim2 = Dimension(['dummy1', 'dummy2'], 'exponential', pars)
    assert isinstance(dim1.dimension, str)
    assert isinstance(dim2.dimension, list)
    assert len(dim2.dimension) == 2


# Test setter behavior
@pytest.mark.filterwarnings('ignore:`kernel`')
def test_kernel_pars_deleted():
    """Delete `pars` when `kernel` is changed."""
    dim = Dimension('dummy', 'exponential', pars)
    dim.kernel = 'tricubic'
    assert hasattr(dim, 'pars') is False


def test_kernel_pars_warning():
    """Warn that `pars` deleted when `kernel` is changed."""
    with pytest.warns(UserWarning):
        dim = Dimension('dummy', 'exponential', pars)
        dim.kernel = 'tricubic'


@pytest.mark.filterwarnings('ignore:`kernel`')
def test_setter_distance_changed_hierarchical():
    """`distance` is changed to 'hierarchical'.

    When `kernel` is changed to 'depth', enforce `distance` ==
    'hierarchical'.

    """
    dim = Dimension(['dummy1', 'dummy2'], 'exponential', pars)
    dim.kernel = 'depth'
    assert dim.distance == 'hierarchical'


def test_setter_distance_warning_hierarchical():
    """Warn when `distance` is changed to 'hierarchical'.

    When `kernel` is changed to 'depth', enforce `distance` ==
    'hierarchical'. If `distance` is changed, produce a warning.

    """
    with pytest.warns(UserWarning):
        dim = Dimension(['dummy1', 'dummy2'], 'exponential', pars)
        dim.kernel = 'depth'
