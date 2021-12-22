"""Tests for Dimension class."""
import pytest

from weave.dimension import Dimension

from examples import levels, kernel_dict, test_dict


# Test constructor types
@pytest.mark.parametrize('dimension', test_dict['dimension'])
@pytest.mark.parametrize('kernel', kernel_dict.values())
def test_dimension_type(dimension, kernel):
    """Raise TypeError if `dimension` not a str or list of str."""
    with pytest.raises(TypeError):
        Dimension(dimension, kernel)


@pytest.mark.parametrize('kernel', test_dict['other'])
def test_kernel_type(kernel):
    """Raise TypeError if `kernel` not a kernel function."""
    with pytest.raises(TypeError):
        Dimension('dummy', kernel)


# Test duplicates in `dimension`
@pytest.mark.parametrize('kernel', kernel_dict.values())
def test_duplicate_dimension(kernel):
    """Raise ValueError if `dimension` contains duplicates."""
    with pytest.raises(ValueError):
        Dimension(['dummy', 'dummy'], kernel)


# Test equality
@pytest.mark.parametrize('dimension', ['dummy', levels])
@pytest.mark.parametrize('kernel1', kernel_dict.values())
@pytest.mark.parametrize('kernel2', kernel_dict.values())
def test_equal(dimension, kernel1, kernel2):
    """Return True if `dimension` equal."""
    dim1 = Dimension(dimension, kernel1)
    dim2 = Dimension(dimension, kernel2)
    assert dim1 == dim2


@pytest.mark.parametrize('kernel1', kernel_dict.values())
@pytest.mark.parametrize('kernel2', kernel_dict.values())
def test_not_equal(kernel1, kernel2):
    """Return False if `dimension` not equal."""
    dim1 = Dimension('dummy', kernel1)
    dim2 = Dimension(levels, kernel2)
    assert dim1 != dim2
