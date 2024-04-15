"""Tests for Dimension class."""

import pytest

from weave.dimension import Dimension

# Lists of wrong types to test exceptions
not_int = [1.0, "dummy", True, None, [], (), {}]
not_float = [1, "dummy", True, None, [], (), {}]
not_numeric = ["dummy", True, None, [], (), {}]
not_str = [1, 1.0, True, None, [], (), {}]
not_bool = [1, 1.0, "dummy", None, [], (), {}]
not_tuple = [1, 1.0, "dummy", True, None, [], {}]
not_coordinates = not_str + [[value] for value in not_str]
not_dict = [1, 1.0, "dummy", True, None, [], ()]

# Example kernel parameters and distance dictionary
kernel_pars = {"radius": 0.6, "exponent": 3}
distance_dict = {(1.0, 1.0): 1.0}


# Test constructor types
@pytest.mark.parametrize("name", not_str)
def test_name_type(name):
    """Raise TypeError if `name` not a str."""
    with pytest.raises(TypeError):
        Dimension(name)


@pytest.mark.parametrize("coordinates", not_coordinates)
def test_coordinates_type(coordinates):
    """Raise TypeError if `coordinates` is not a str or list of str."""
    if coordinates is not None and coordinates != []:
        with pytest.raises(TypeError):
            Dimension("dummy", coordinates)


@pytest.mark.parametrize("kernel", not_str)
def test_kernel_type(kernel):
    """Raise TypeError if `kernel` is not a str."""
    with pytest.raises(TypeError):
        Dimension("dummy", kernel=kernel)


@pytest.mark.parametrize("radius", not_numeric)
def test_exponential_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    if radius is not None:
        with pytest.raises(TypeError):
            Dimension("dummy", kernel="exponential", radius=radius)


@pytest.mark.parametrize("exponent", not_numeric)
def test_tricubic_exponent_type(exponent):
    """Raise TypeError if `exponent` is not an int or float."""
    if exponent is not None:
        with pytest.raises(TypeError):
            Dimension("dummy", kernel="tricubic", exponent=exponent)


@pytest.mark.parametrize("radius", not_float)
def test_depth_radius_type(radius):
    """Raise TypeError if `radius` is not a float."""
    if radius is not None:
        with pytest.raises(TypeError):
            Dimension("dummy", kernel="depth", radius=radius)


@pytest.mark.parametrize("version", not_str)
def test_depth_version_type(version):
    """Raise TypeError if `version` is not a str."""
    if version is not None:
        with pytest.raises(TypeError):
            Dimension("dummy", kernel="depth", radius=0.6, version=version)


@pytest.mark.parametrize("radius", not_numeric)
def test_inverse_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    if radius is not None:
        with pytest.raises(TypeError):
            Dimension("dummy", kernel="inverse", radius=radius)


@pytest.mark.parametrize("distance", not_str)
def test_distance_type(distance):
    """Raise TypeError if `distance` is not a str."""
    if distance is not None:
        with pytest.raises(TypeError):
            Dimension("dummy", distance=distance)


@pytest.mark.parametrize("bad_dict", not_dict)
def test_distance_dict_type(bad_dict):
    """Raise TypeError if `distance_dict` not a dict."""
    if bad_dict is not None:
        with pytest.raises(TypeError):
            Dimension("dummy", distance="dictionary", distance_dict=bad_dict)


@pytest.mark.parametrize("key", not_tuple)
@pytest.mark.parametrize("value", [1, 1.0])
def test_distance_dict_key_type(key, value):
    """Raise TypeError if `distance_dict` keys are not tuples."""
    with pytest.raises(TypeError):
        bad_dict = {key: value}
        Dimension("dummy", distance="dictionary", distance_dict=bad_dict)


@pytest.mark.parametrize("key1", not_numeric)
@pytest.mark.parametrize("key2", not_numeric)
@pytest.mark.parametrize("value", [1, 1.0])
def test_distance_dict_key_element_type(key1, key2, value):
    """Raise TypeError if `distance_dict` keys contain invalid values."""
    with pytest.raises(TypeError):
        bad_dict = {(key1, key2): value}
        Dimension("dummy", distance="dictionary", distance_dict=bad_dict)


@pytest.mark.parametrize("key1", [1, 1.0])
@pytest.mark.parametrize("key2", [1, 1.0])
@pytest.mark.parametrize("value", not_numeric)
def test_distance_dict_value_type(key1, key2, value):
    """Raise TypeError if `distance_dict` values not all int or float."""
    with pytest.raises(TypeError):
        bad_dict = {(key1, key2): value}
        Dimension("dummy", distance="dictionary", distance_dict=bad_dict)


# Test constructor values
def test_coordinates_empty():
    """Raise ValueError if `coordinates` is an empty list."""
    with pytest.raises(ValueError):
        Dimension("dummy", [])


def test_coordinates_duplicates():
    """Raise ValueError if duplicates found in `coordinates`."""
    with pytest.raises(ValueError):
        Dimension("dummy", ["dummy", "dummy"])


def test_coordinates_default():
    """`coordinates` set to [`name`] if not supplied."""
    dim = Dimension("dummy")
    assert dim.coordinates == [dim.name]


def test_kernel_value():
    """Raise ValueError if `kernel` is not valid."""
    with pytest.raises(ValueError):
        Dimension("dummy", kernel="dummy")


def test_kernel_default():
    """`kernel` set to 'identity' if not supplied."""
    dim = Dimension("dummy")
    assert dim.kernel == "identity"


def test_exponential_radius_exist():
    """Raise AttributeError if `radius` is not passed."""
    with pytest.raises(AttributeError):
        Dimension("dummy", kernel="exponential")


@pytest.mark.parametrize("radius", [-1, -1.0, 0, 0.0])
def test_exponential_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension("dummy", kernel="exponential", radius=radius)


def test_tricubic_exponent_exist():
    """Raise AttributeError if `exponent` is not passed."""
    with pytest.raises(AttributeError):
        Dimension("dummy", kernel="tricubic")


@pytest.mark.parametrize("exponent", [-1, -1.0, 0, 0.0])
def test_tricubic_exponent_value(exponent):
    """Raise ValueError if `exponenent` is not valid."""
    with pytest.raises(ValueError):
        Dimension("dummy", kernel="tricubic", exponent=exponent)


def test_depth_radius_exist():
    """Raise AttributeError if `radius` is not passed."""
    with pytest.raises(AttributeError):
        Dimension("dummy", kernel="depth")


@pytest.mark.parametrize("radius", [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0])
def test_depth_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    with pytest.raises(ValueError):
        Dimension("dummy", kernel="depth", radius=radius)


def test_depth_version_value():
    """Raise ValueError if `version` is not valid."""
    with pytest.raises(ValueError):
        Dimension("dummy", kernel="depth", radius=0.6, version="dummy")


def test_depth_version_default():
    """`version` set to 'codem' if not passed."""
    dim = Dimension("dummy", kernel="depth", radius=0.6)
    assert dim.version == "codem"


def test_inverse_radius_exists():
    """Raise AttributeError if `radius` is not passed."""
    with pytest.raises(AttributeError):
        Dimension("dummy", kernel="inverse")


@pytest.mark.parametrize("radius", [-1, -1.0, 0, 0.0])
def test_inverse_radius_value(radius):
    """Raise ValueError if `radius` not valid."""
    with pytest.raises(ValueError):
        Dimension("dummy", kernel="inverse", radius=radius)


def test_distance_value():
    """Raise ValueError if `distance` is not valid."""
    with pytest.raises(ValueError):
        Dimension("dummy", distance="dummy")


@pytest.mark.parametrize("kernel", ["exponential", "tricubic", "identity"])
def test_euclidean_default(kernel):
    """`distance` set to 'euclidean' if not supplied."""
    dim = Dimension("dummy", kernel=kernel, **kernel_pars)
    assert dim.distance == "euclidean"


def test_tree_default():
    """`distance` is set to 'tree' if not supplied."""
    dim = Dimension("dummy", kernel="depth", radius=0.6)
    assert dim.distance == "tree"


def test_dictionary_distance_dict():
    """Raise ValueError if `distance_dict` if not passed."""
    with pytest.raises(ValueError):
        Dimension("dummy", distance="dictionary")


def test_dictionary_empty():
    """Raise ValueError if `distance_dict` is an empty dict."""
    with pytest.raises(ValueError):
        Dimension("dummy", kernel="dictionary", distance_dict={})


@pytest.mark.parametrize("key", [(), (1,), (1.0,), (1, 2, 3), (1.0, 2.0, 3.0)])
@pytest.mark.parametrize("value", [1, 1.0])
def test_distance_dict_key_length(key, value):
    """Raise ValueError if `distance_dict` keys not all length 2."""
    with pytest.raises(ValueError):
        bad_dict = {key: value}
        Dimension("dummy", distance="dictionary", distance_dict=bad_dict)


@pytest.mark.parametrize("key1", [1, 1.0])
@pytest.mark.parametrize("key2", [1, 1.0])
@pytest.mark.parametrize("value", [-1, -1.0])
def test_distance_dict_value_nonnegative(key1, key2, value):
    """Raise ValueError if `distance_dict` values not all nonnegative."""
    with pytest.raises(ValueError):
        bad_dict = {(key1, key2): value}
        Dimension("dummy", distance="dictionary", distance_dict=bad_dict)


# Test setter behavior
def test_name_immutable():
    """Raise AttributeError if attempt to reset `name`."""
    with pytest.raises(AttributeError):
        dim = Dimension("dummy")
        dim.name = "spam"


def test_coordinates_immutable():
    """Raise AttributeError if attempt to reset `coordinates`."""
    with pytest.raises(AttributeError):
        dim = Dimension("dummy")
        dim.coordinates = "spam"


def test_kernel_immutable():
    """Raise AttributeError if attempt to reset `kernel`."""
    with pytest.raises(AttributeError):
        dim = Dimension("dummy")
        dim.kernel = "exponential"


def test_distance_immutable():
    """Raise AttributeError if attempt to reset `distance`."""
    with pytest.raises(AttributeError):
        dim = Dimension("dummy")
        dim.distance = "tree"


def test_distance_dict_immutable():
    """Raise AttributeError if attempt to reset `distance_dict`."""
    with pytest.raises(AttributeError):
        dim = Dimension("dummy", distance="dictionary", distance_dict=distance_dict)
        dim.distance_dict = {(2.0, 2.0): 2.0}


@pytest.mark.parametrize("radius", not_numeric)
def test_exponential_reset_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    dim = Dimension("dummy", kernel="exponential", radius=0.6)
    if radius is not None:
        with pytest.raises(TypeError):
            dim.radius = radius


@pytest.mark.parametrize("exponent", not_numeric)
def test_tricubic_reset_exponent_type(exponent):
    """Raise TypeError if `exponent` is not an int or float."""
    dim = Dimension("dummy", kernel="tricubic", exponent=3)
    if exponent is not None:
        with pytest.raises(TypeError):
            dim.exponent = exponent


@pytest.mark.parametrize("radius", not_float)
def test_depth_reset_radius_type(radius):
    """Raise TypeError if `radius` is not a float."""
    dim = Dimension("dummy", kernel="depth", radius=0.6)
    if radius is not None:
        with pytest.raises(TypeError):
            dim.radius = radius


@pytest.mark.parametrize("version", not_str)
def test_depth_reset_version_type(version):
    """Raise TypeError if `version` not a str."""
    dim = Dimension("dummy", kernel="depth", radius=0.6)
    if version is not None:
        with pytest.raises(TypeError):
            dim.version = version


@pytest.mark.parametrize("radius", not_numeric)
def test_inverse_reset_radius_type(radius):
    """Raise TypeError if `radius` is not an int or float."""
    dim = Dimension("dummy", kernel="inverse", radius=0.6)
    if radius is not None:
        with pytest.raises(TypeError):
            dim.radius = radius


@pytest.mark.parametrize("radius", [-1, -1.0, 0, 0.0])
def test_exponential_reset_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    dim = Dimension("dummy", kernel="exponential", radius=0.6)
    with pytest.raises(ValueError):
        dim.radius = radius


@pytest.mark.parametrize("exponent", [-1, -1.0, 0, 0.0])
def test_tricubic_reset_exponent_value(exponent):
    """Raise ValueError if `exponenent` is not valid."""
    dim = Dimension("dummy", kernel="tricubic", exponent=3)
    with pytest.raises(ValueError):
        dim.exponent = exponent


@pytest.mark.parametrize("radius", [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0])
def test_depth_reset_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    dim = Dimension("dummy", kernel="depth", radius=0.6)
    with pytest.raises(ValueError):
        dim.radius = radius


def test_depth_reset_version_value():
    """Raise ValueError if `version` is not valid."""
    dim = Dimension("dummy", kernel="depth", radius=0.6)
    with pytest.raises(ValueError):
        dim.version = "dummy"


@pytest.mark.parametrize("radius", [-1, -1.0, 0, 0.0])
def test_inverse_reset_radius_value(radius):
    """Raise ValueError if `radius` is not valid."""
    dim = Dimension("dummy", kernel="inverse", radius=0.6)
    with pytest.raises(ValueError):
        dim.radius = radius
