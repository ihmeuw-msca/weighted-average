"""Tests for Dimension class."""
import pytest

from weave.dimension import Dimension
from weave.distance import Hierarchical
from weave.kernels import Exponential, Tricubic, Depth

from examples import levels, test_dict

hier = Hierarchical()
kernel_list = [Exponential(0.5), Exponential(0.5, distance=hier),
               Tricubic(7.0), Tricubic(3, distance=hier), Depth(0.9)]
test_dict['dimension'] = test_dict['str'] + \
                         [[not_str] for not_str in test_dict['str']]


# Test constructor types
@pytest.mark.parametrize('dimension', test_dict['dimension'])
@pytest.mark.parametrize('kernel', kernel_list)
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
@pytest.mark.parametrize('kernel', kernel_list)
def test_duplicate_dimension(kernel):
    """Raise ValueError if `dimension` contains duplicates."""
    with pytest.raises(ValueError):
        Dimension(['dummy', 'dummy'], kernel)


# Test equality
@pytest.mark.parametrize('dimension', ['dummy', levels])
@pytest.mark.parametrize('kernel1', kernel_list)
@pytest.mark.parametrize('kernel2', kernel_list)
def test_equal(dimension, kernel1, kernel2):
    """Return True if `dimension` equal."""
    dim1 = Dimension(dimension, kernel1)
    dim2 = Dimension(dimension, kernel2)
    assert dim1 == dim2


@pytest.mark.parametrize('kernel1', kernel_list)
@pytest.mark.parametrize('kernel2', kernel_list)
def test_not_equal(kernel1, kernel2):
    """Return False if `dimension` not equal."""
    dim1 = Dimension('dummy', kernel1)
    dim2 = Dimension(levels, kernel2)
    assert dim1 != dim2
