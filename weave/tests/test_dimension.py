"""Tests for Dimension class.

Currently only testing a simple example for most cases, not testing
other examples of valid input. Could do this with 'parametrize' or
'hypothesis'. Examples of valid input:
* `kernel` == 'exponential'
    - `dimension` in {'dummy', ['dummy1', 'dummy2']}
    - `pars` == {'radius': 0.5}
    - `distance` in {'dictionary', 'euclidean', 'hierarchical', None}
* `kernel == 'tricubic'
    - `dimension` in {'dummy', ['dummy1', 'dummy2']}
    - `pars` == {'radius': 0.5, 'exponent': 3}
    - `distance` in {'dictionary', 'euclidean', 'hierarchical', None}
* `kernel == 'depth'
    - `dimension` in {'dummy', ['dummy1', 'dummy2']}
    - `pars` == {'radius': 0.5, 'normalize': True}
    - `distance` in {'dictionary', 'euclidean', 'hierarchical', None}

TODO:
* Add tests for TypedDimension

"""
import pytest

from weave.dimension import Dimension

# Lists of wrong types to test exceptions
not_float = [1, 'dummy', True, None, [], (), {}]
not_numeric = ['dummy', True, None, [], (), {}]
not_str = [1, 1.0, True, None, [], (), {}]
not_bool = [1, 1.0, 'dummy', None, [], (), {}]
not_tuple = [1, 1.0, 'dummy', True, None, [], {}]
not_columns = not_str + [[value] for value in not_str]
not_dict = [1, 1.0, 'dummy', True, None, [], ()]

# Example kernel parameters and distance dictionary
kernel_pars = {'radius': 0.5, 'exponent': 3, 'normalize': True}
distance_dict = {(1.0, 1.0): 1.0}


# Test constructor types
@pytest.mark.parametrize('name', not_str)
def test_name_type(name):
    """Raise TypeError if `name` not a str."""
    with pytest.raises(TypeError):
        Dimension(name, 'dummy', 'exponential', kernel_pars)


@pytest.mark.parametrize('columns', not_columns)
def test_columns_type(columns):
    """Raise TypeError if `columns` is not a str or list of str."""
    if columns != []:
        with pytest.raises(TypeError):
            Dimension('dummy', columns, 'exponential', kernel_pars)


@pytest.mark.parametrize('kernel', not_str)
def test_kernel_type(kernel):
    """Raise TypeError if `kernel` is not a str."""
    with pytest.raises(TypeError):
        Dimension('dummy', 'dummy', kernel, kernel_pars)


@pytest.mark.parametrize('radius', not_numeric)
def test_exponential_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': radius}
        Dimension('dummy', 'dummy', 'exponential', bad_pars)


@pytest.mark.parametrize('radius', not_numeric)
def test_tricubic_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': radius, 'exponent': 3}
        Dimension('dummy', 'dummy', 'tricubic', bad_pars)


@pytest.mark.parametrize('exponent', not_numeric)
def test_tricubic_exponent_type(exponent):
    """Raise TypeError if `exponent` is not an int or float."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': 0.5, 'exponent': exponent}
        Dimension('dummy', 'dummy', 'tricubic', bad_pars)


@pytest.mark.parametrize('radius', not_float)
def test_depth_radius_type(radius):
    """Raise TypeError if `radius` is not a float."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': radius}
        Dimension('dummy', 'dummy', 'depth', bad_pars)


@pytest.mark.parametrize('normalize', not_bool)
def test_depth_normalize_type(normalize):
    """Raise TypeError if `normalize` is not a bool."""
    with pytest.raises(TypeError):
        bad_pars = {'radius': 0.5, 'normalize': not_bool}
        Dimension('dummy', 'dummy', 'depth', bad_pars)


@pytest.mark.parametrize('distance', not_str)
def test_distance_type(distance):
    """Raise TypeError if `distance` is not a str."""
    if distance is not None:
        with pytest.raises(TypeError):
            Dimension('dummy', 'dummy', 'exponential', kernel_pars, distance)


@pytest.mark.parametrize('bad_dict', not_dict)
def test_distance_dict_type(bad_dict):
    """Raise TypeError if `distance_dict` not a dict."""
    if bad_dict is not None:
        with pytest.raises(TypeError):
            Dimension('dummy', 'dummy', 'exponential', kernel_pars,
                      'dictionary', bad_dict)


@pytest.mark.parametrize('key', not_tuple)
@pytest.mark.parametrize('value', [1, 1.0])
def test_distance_dict_key_type(key, value):
    """Raise TypeError if `distance_dict` keys are not tuples."""
    with pytest.raises(TypeError):
        bad_dict = {key: value}
        Dimension('dummy', 'dummy', 'exponential', kernel_pars, 'dictionary',
                  bad_dict)


@pytest.mark.parametrize('key1', not_numeric)
@pytest.mark.parametrize('key2', not_numeric)
@pytest.mark.parametrize('value', [1, 1.0])
def test_distance_dict_key_element_type(key1, key2, value):
    """Raise TypeError if `distance_dict` keys contain invalid values."""
    with pytest.raises(TypeError):
        bad_dict = {(key1, key2): value}
        Dimension('dummy', 'dummy', 'exponential', kernel_pars, 'dictionary',
                  bad_dict)


@pytest.mark.parametrize('key1', [1, 1.0])
@pytest.mark.parametrize('key2', [1, 1.0])
@pytest.mark.parametrize('value', not_numeric)
def test_distance_dict_value_type(key1, key2, value):
    """Raise TypeError if `distance_dict` values not all int or float."""
    with pytest.raises(TypeError):
        bad_dict = {(key1, key2): value}
        Dimension('dummy', 'dummy', 'exponential', kernel_pars, 'dictionary',
                  bad_dict)


# Test constructor values
def test_columns_empty():
    """Raise ValueError if `columns` is an empty list."""
    with pytest.raises(ValueError):
        Dimension('dummy', [], 'exponential', kernel_pars)


def test_columns_duplicates():
    """Raise ValueError if duplicates found in `columns`."""
    with pytest.raises(ValueError):
        Dimension('dummy', ['dummy', 'dummy'], 'exponential', kernel_pars)


def test_kernel_value():
    """Raise ValueError if `kernel` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'dummy', 'dummy', kernel_pars)


def test_exponential_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        bad_pars = {'dummy': 100}
        Dimension('dummy', 'dummy', 'exponential', bad_pars)


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_exponential_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': radius, 'exponent': 3}
        Dimension('dummy', 'dummy', 'exponential', bad_pars)


def test_tricubic_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        bad_pars = {'exponent': 3}
        Dimension('dummy', 'dummy', 'tricubic', bad_pars)


@pytest.mark.parametrize('radius', [-1, -1.0, 0, 0.0])
def test_tricubic_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': radius, 'exponent': 3}
        Dimension('dummy', 'dummy', 'tricubic', bad_pars)


def test_tricubic_exponent_exist():
    """Raise KeyError if `exponent` is not passed."""
    with pytest.raises(KeyError):
        bad_pars = {'radius': 0.5}
        Dimension('dummy', 'dummy', 'tricubic', bad_pars)


@pytest.mark.parametrize('exponent', [-1, -1.0, 0, 0.0])
def test_tricubic_exponent_value(exponent):
    """Raise ValueError if `exponenent` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': 0.5, 'exponent': exponent}
        Dimension('dummy', 'dummy', 'tricubic', bad_pars)


def test_depth_radius_exist():
    """Raise KeyError if `radius` is not passed."""
    with pytest.raises(KeyError):
        bad_pars = {'dummy': 100}
        Dimension('dummy', ['dummy1', 'dummy2'], 'depth', bad_pars)


@pytest.mark.parametrize('radius', [-1.0, 0.0, 1.0, 2.0])
def test_depth_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        bad_pars = {'radius': radius, 'exponent': 3}
        Dimension('dummy', ['dummy1', 'dummy2'], 'depth', bad_pars)


def test_depth_normalize_default():
    """`normalize` set to True if not supplied."""
    dim = Dimension('dummy', 'dummy', 'depth', {'radius': 0.5})
    assert dim.kernel_pars['normalize'] is True


@pytest.mark.parametrize('kernel', ['exponential', 'tricubic'])
def test_no_extra_pars(kernel):
    """Only relevant parameters saved to `kernel_pars`."""
    bad_pars = {'radius': 0.5, 'exponent': 3, 'dummy': 100}
    dim = Dimension('dummy', 'dummy', kernel, bad_pars)
    if kernel == 'tricubic':
        assert 'exponent' in dim.kernel_pars
    else:
        assert 'exponent' not in dim.kernel_pars
    assert 'radius' in dim.kernel_pars
    assert 'dummy' not in dim.kernel_pars


def test_distance_value():
    """Raise ValueError if `distance` is not valid."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'dummy', 'exponential', kernel_pars, 'dummy')


def test_exponential_distance_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension('dummy', 'dummy', 'exponential', kernel_pars)
    assert dim.distance == 'euclidean'


def test_tricubic_distance_default():
    """`distance` is set to 'euclidean' if not supplied."""
    dim = Dimension('dummy', 'dummy', 'tricubic', kernel_pars)
    assert dim.distance == 'euclidean'


def test_depth_distance_default():
    """`distance` is set to 'hierarchical' if not supplied."""
    dim = Dimension('dummy', ['dummy1', 'dummy2'], 'depth', kernel_pars)
    assert dim.distance == 'hierarchical'


def test_dictionary_columns_one():
    """Raise ValueError if length of `columns` > 1."""
    with pytest.raises(ValueError):
        Dimension('dummy', ['dummy1', 'dummy2'], 'dictionary', kernel_pars,
                  distance_dict)


def test_dictionary_distance_dict():
    """Raise ValueError if `distance_dict` if not passed."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'dummy', 'exponential', kernel_pars, 'dictionary')


def test_dictionary_empty():
    """Raise ValueError if `distance_dict` is an empty dict."""
    with pytest.raises(ValueError):
        Dimension('dummy', 'dummy', 'exponential', kernel_pars, 'dictionary',
                  {})


@pytest.mark.parametrize('key', [(), (1, ), (1., ), (1, 2, 3), (1., 2., 3.)])
@pytest.mark.parametrize('value', [1, 1.0])
def test_distance_dict_key_length(key, value):
    """Raise ValueError if `distance_dict` keys not all length 2."""
    with pytest.raises(ValueError):
        bad_dict = {key: value}
        Dimension('dummy', 'dummy', 'exponential', kernel_pars, 'dictionary',
                  bad_dict)


@pytest.mark.parametrize('key1', [1, 1.0])
@pytest.mark.parametrize('key2', [1, 1.0])
@pytest.mark.parametrize('value', [-1, -1.0])
def test_distance_dict_value_nonnegative(key1, key2, value):
    """Raise ValueError if `distance_dict` values not all nonnegative."""
    with pytest.raises(ValueError):
        bad_dict = {(key1, key2): value}
        Dimension('dummy', 'dummy', 'exponential', kernel_pars, 'dictionary',
                  bad_dict)


# Test setter behavior
def test_name_immutable():
    """Raise AttributeError if attempt to reset `name`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy', 'dummy', 'exponential', kernel_pars)
        dim.name = 'spam'


def test_columns_immutable():
    """Raise AttributeError if attempt to reset `columns`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy', 'dummy', 'exponential', kernel_pars)
        dim.columns = ['dummy1', 'dummy2']


def test_kernel_immutable():
    """Raise AttributeError if attempt to reset `kernel`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy', 'dummy', 'exponential', kernel_pars)
        dim.kernel = 'tricubic'


def test_distance_immutable():
    """Raise AttributeError if attempt to reset `distance`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy', 'dummy', 'exponential', kernel_pars)
        dim.distance = 'hierarchical'


def test_distance_dict_immutable():
    """Raise AttributeError if attempt to reset `distance_dict`."""
    with pytest.raises(AttributeError):
        dim = Dimension('dummy', 'dummy', 'exponential', kernel_pars,
                        'dictionary', distance_dict)
        dim.distance_dict = {(2.0, 2.0): 2.0}
