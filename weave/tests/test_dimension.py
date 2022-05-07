"""Tests for Dimension class.

Currently only testing a simple example for most cases, not testing
other examples of valid input. Could do this with 'parametrize' or
'hypothesis'.

TODO:
* Add test for `get_typed_dimension()`

"""
import pytest

from weave.dimension import Dimension

# Lists of wrong types to test exceptions
not_int = [1.0, 'dummy', True, None, [], (), {}]
not_float = [1, 'dummy', True, None, [], (), {}]
not_numeric = ['dummy', True, None, [], (), {}]
not_str = [1, 1.0, True, None, [], (), {}]
not_bool = [1, 1.0, 'dummy', None, [], (), {}]
not_tuple = [1, 1.0, 'dummy', True, None, [], {}]
not_coordinates = not_str + [[value] for value in not_str]
not_dict = [1, 1.0, 'dummy', True, None, [], ()]

# Example kernel parameters and distance dictionary
kernels = ['depth', 'exponential', 'identity', 'tricubic']
kernel_pars = {'radius': 0.5, 'levels': 3, 'exponent': 3, 'normalize': True}
distance_dict = {(1.0, 1.0): 1.0}


# Test constructor types
@pytest.mark.parametrize('name', not_str)
def test_name_type(name):
    """Raise TypeError if `name` not a str."""
    with pytest.raises(TypeError):
        Dimension(name)


@pytest.mark.parametrize('coordinates', not_coordinates)
def test_coordinates_type(coordinates):
    """Raise TypeError if `coordinates` is not a str or list of str."""
    if coordinates is not None and coordinates != []:
        with pytest.raises(TypeError):
            Dimension('dummy', coordinates)


@pytest.mark.parametrize('kernel', not_str)
def test_kernel_type(kernel):
    """Raise TypeError if `kernel` is not a str."""
    with pytest.raises(TypeError):
        Dimension('dummy', kernel=kernel)


@pytest.mark.parametrize('radius', not_float)
def test_depth_radius_type(radius):
    """Raise TypeError if `radius` is not a float."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': radius, 'levels': 3}
        Dimension('dummy', kernel='depth', kernel_pars=bad_pars)


@pytest.mark.parametrize('levels', not_int)
def test_depth_levels_type(levels):
    """Raise TypeError if `levels` is not an int."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': 0.5, 'levels': levels}
        Dimension('dummy', kernel='depth', kernel_pars=bad_pars)


@pytest.mark.parametrize('normalize', not_bool)
def test_depth_normalize_type(normalize):
    """Raise TypeError if `normalize` is not a bool."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': 0.5, 'levels': 3, 'normalize': normalize}
        Dimension('dummy', kernel='depth', kernel_pars=bad_pars)


@pytest.mark.parametrize('radius', not_numeric)
def test_exponential_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': radius}
        Dimension('dummy', kernel='exponential', kernel_pars=bad_pars)


@pytest.mark.parametrize('normalize', not_bool)
def test_exponential_normalize_type(normalize):
    """Raise TypeError if `normalize` is not a bool."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': 0.5, 'normalize': normalize}
        Dimension('dummy', kernel='exponential', kernel_pars=bad_pars)


@pytest.mark.parametrize('normalize', not_bool)
def test_identity_normalize_type(normalize):
    """Raise TypeError if `normalize` is not a bool."""
    with pytest.raises(TypeError):
        bad_pars = {'normalize': normalize}
        Dimension('dummy', kernel='identity', kernel_pars=bad_pars)


@pytest.mark.parametrize('radius', not_numeric)
def test_tricubic_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': radius, 'exponent': 3}
        Dimension('dummy', kernel='tricubic', kernel_pars=bad_pars)


@pytest.mark.parametrize('exponent', not_int)
def test_tricubic_exponent_type(exponent):
    """Raise TypeError if `exponent` is not an int."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': 0.5, 'exponent': exponent}
        Dimension('dummy', kernel='tricubic', kernel_pars=bad_pars)


@pytest.mark.parametrize('normalize', not_bool)
def test_tricubic_normalize_type(normalize):
    """Raise TypeError if `normalize` is not a bool."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': 0.5, 'exponent': 3, 'normalize': normalize}
        Dimension('dummy', kernel='tricubic', kernel_pars=bad_pars)


@pytest.mark.parametrize('distance', not_str)
def test_distance_type(distance):
    """Raise TypeError if `distance` is not a str."""
    if distance is not None:
        with pytest.raises(TypeError):
            Dimension('dummy', distance=distance)


@pytest.mark.parametrize('bad_dict', not_dict)
def test_distance_dict_type(bad_dict):
    """Raise TypeError if `distance_dict` not a dict."""
    if bad_dict is not None:
        with pytest.raises(TypeError):
            Dimension('dummy', distance='dictionary', distance_dict=bad_dict)


@pytest.mark.parametrize('key', not_tuple)
@pytest.mark.parametrize('value', [1, 1.0])
def test_distance_dict_key_type(key, value):
    """Raise TypeError if `distance_dict` keys are not tuples."""
    with pytest.raises(TypeError):
        bad_dict = {key: value}
        Dimension('dummy', distance='dictionary', distance_dict=bad_dict)


@pytest.mark.parametrize('key1', not_numeric)
@pytest.mark.parametrize('key2', not_numeric)
@pytest.mark.parametrize('value', [1, 1.0])
def test_distance_dict_key_element_type(key1, key2, value):
    """Raise TypeError if `distance_dict` keys contain invalid values."""
    with pytest.raises(TypeError):
        bad_dict = {(key1, key2): value}
        Dimension('dummy', distance='dictionary', distance_dict=bad_dict)


@pytest.mark.parametrize('key1', [1, 1.0])
@pytest.mark.parametrize('key2', [1, 1.0])
@pytest.mark.parametrize('value', not_numeric)
def test_distance_dict_value_type(key1, key2, value):
    """Raise TypeError if `distance_dict` values not all int or float."""
    with pytest.raises(TypeError):
        bad_dict = {(key1, key2): value}
        Dimension('dummy', distance='dictionary', distance_dict=bad_dict)


# Test constructor values
def test_coordinates_empty():
    """Raise ValueError if `coordinates` is an empty list."""
    with pytest.raises(ValueError):
        Dimension('dummy', [])


def test_coordinates_duplicates():
    """Raise ValueError if duplicates found in `coordinates`."""
    with pytest.raises(ValueError):
        Dimension('dummy', ['dummy', 'dummy'])


def test_coordinates_default():
    """`coordinates` set to [`name`] if not supplied."""
    dim = Dimension('dummy')
    assert dim.coordinates == [dim.name]


def test_kernel_value():
    """Raise ValueError if `kernel` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', kernel='dummy')


def test_kernel_default():
    """`kernel` set to 'identity' if not supplied."""
    dim = Dimension('dummy')
    assert dim.kernel == 'identity'


def test_depth_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', kernel='depth', kernel_pars={'levels': 3})


@pytest.mark.parametrize('radius', [-1.0, 0.0, 1.0, 2.0])
def test_depth_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': radius, 'exponent': 3}
        Dimension('dummy', kernel='depth', kernel_pars=bad_pars)


def test_depth_levels_exist():
    """Raise KeyError if `levels` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', kernel='depth', kernel_pars={'radius': 0.5})


@pytest.mark.parametrize('levels', [-1, 0])
def test_depth_levels_value(levels):
    """Raise ValueError if `levels` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': 0.5, 'levels': levels}
        Dimension('dummy', kernel='depth', kernel_pars=bad_pars)


def test_depth_normalize_default():
    """`normalize` set to True if not supplied."""
    pars = {'radius': 0.5, 'levels': 3}
    dim = Dimension('dummy', kernel='depth', kernel_pars=pars)
    assert dim.kernel_pars['normalize'] is True


def test_exponential_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', kernel='exponential', kernel_pars={'dummy': 100})


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_exponential_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': radius}
        Dimension('dummy', kernel='exponential', kernel_pars=bad_pars)


def test_exponential_normalize_default():
    """`normalize` set to False if not supplied."""
    pars = {'radius': 0.5}
    dim = Dimension('dummy', kernel='exponential', kernel_pars=pars)
    assert dim.kernel_pars['normalize'] is False


def test_identity_normalize_default():
    """`normalize` set to False if not supplied."""
    dim = Dimension('dummy', kernel='identity')
    assert dim.kernel_pars['normalize'] is False


def test_tricubic_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', kernel='tricubic', kernel_pars={'exponent': 3})


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_tricubic_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': radius, 'exponent': 3}
        Dimension('dummy', kernel='tricubic', kernel_pars=bad_pars)


def test_tricubic_exponent_exist():
    """Raise KeyError if `exponent` is not passed."""
    with pytest.raises(KeyError):
        Dimension('dummy', kernel='tricubic', kernel_pars={'radius': 0.5})


@pytest.mark.parametrize('exponent', [-1, 0])
def test_tricubic_exponent_value(exponent):
    """Raise ValueError if `exponenent` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': 0.5, 'exponent': exponent}
        Dimension('dummy', kernel='tricubic', kernel_pars=bad_pars)


def test_tricubic_normalize_default():
    """`normalize` set to False if not supplied."""
    pars = {'radius': 0.5, 'exponent': 3}
    dim = Dimension('dummy', kernel='tricubic', kernel_pars=pars)
    assert dim.kernel_pars['normalize'] is False


@pytest.mark.parametrize('kernel', kernels)
def test_no_extra_pars(kernel):
    """Only relevant parameters saved to `kernel_pars`."""
    extra_pars = {'radius': 0.5, 'levels': 3, 'exponent': 3, 'dummy': 100}
    dim = Dimension('dummy', kernel=kernel, kernel_pars=extra_pars)
    if kernel == 'depth':
        assert 'levels' in dim.kernel_pars
    else:
        assert 'levels' not in dim.kernel_pars
    if kernel == 'identity':
        assert 'radius' not in dim.kernel_pars
    else:
        assert 'radius' in dim.kernel_pars
    if kernel == 'tricubic':
        assert 'exponent' in dim.kernel_pars
    else:
        assert 'exponent' not in dim.kernel_pars
    assert 'normalize' in dim.kernel_pars
    assert 'dummy' not in dim.kernel_pars


def test_distance_value():
    """Raise ValueError if `distance` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', distance='dummy')


def test_depth_distance_default():
    """`distance` is set to 'tree' if not supplied."""
    dim = Dimension('dummy', kernel='depth', kernel_pars=kernel_pars)
    assert dim.distance == 'tree'


def test_exponential_distance_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension('dummy', kernel='exponential', kernel_pars=kernel_pars)
    assert dim.distance == 'euclidean'


def test_identity_distance_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension('dummy', kernel='identity')
    assert dim.distance == 'euclidean'


def test_tricubic_distance_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension('dummy', kernel='tricubic', kernel_pars=kernel_pars)
    assert dim.distance == 'euclidean'


def test_dictionary_distance_dict():
    """Raise ValueError if `distance_dict` if not passed."""
    with pytest.raises(ValueError):
        Dimension('dummy', distance='dictionary')


def test_dictionary_empty():
    """Raise ValueError if `distance_dict` is an empty dict."""
    with pytest.raises(ValueError):
        Dimension('dummy', kernel='dictionary', distance_dict={})


@pytest.mark.parametrize('key', [(), (1, ), (1., ), (1, 2, 3), (1., 2., 3.)])
@pytest.mark.parametrize('value', [1, 1.0])
def test_distance_dict_key_length(key, value):
    """Raise ValueError if `distance_dict` keys not all length 2."""
    with pytest.raises(ValueError):
        bad_dict = {key: value}
        Dimension('dummy', distance='dictionary', distance_dict=bad_dict)


@pytest.mark.parametrize('key1', [1, 1.0])
@pytest.mark.parametrize('key2', [1, 1.0])
@pytest.mark.parametrize('value', [-1, -1.0])
def test_distance_dict_value_nonnegative(key1, key2, value):
    """Raise ValueError if `distance_dict` values not all nonnegative."""
    with pytest.raises(ValueError):
        bad_dict = {(key1, key2): value}
        Dimension('dummy', distance='dictionary', distance_dict=bad_dict)


def test_distance_dict_coordinates():
    """Raise ValueError if `coordinates` not 1D."""
    with pytest.raises(ValueError):
        Dimension('dummy', ['dummy1', 'dummy2'], distance='dictionary',
                  distance_dict=distance_dict)


# Test setter behavior
def test_name_immutable():
    """Raise AttributeError if attempt to reset `name`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy')
        dim.name = 'spam'


def test_coordinates_immutable():
    """Raise AttributeError if attempt to reset `coordinates`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy')
        dim.coordinates = 'spam'


def test_kernel_immutable():
    """Raise AttributeError if attempt to reset `kernel`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy')
        dim.kernel = 'exponential'


def test_distance_immutable():
    """Raise AttributeError if attempt to reset `distance`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy')
        dim.distance = 'tree'


def test_distance_dict_immutable():
    """Raise AttributeError if attempt to reset `distance_dict`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy', distance='dictionary',
                        distance_dict=distance_dict)
        dim.distance_dict = {(2.0, 2.0): 2.0}
