"""Tests for smoother function."""
import pytest

from weave.dimension import Dimension
from weave.smoother import Smoother

from examples import kernel_dict, levels, test_dict


# Test constructor types
@pytest.mark.parametrize('dimensions', test_dict['dimensions'])
def test_dimensions_type(dimensions):
    """Raise TypeError if invalid type for `dimensions`."""
    with pytest.raises(TypeError):
        Smoother(dimensions)


# Test duplicates in `dimensions`
@pytest.mark.parametrize('dimension', ['dummy', levels])
@pytest.mark.parametrize('kernel1', kernel_dict.values())
@pytest.mark.parametrize('kernel2', kernel_dict.values())
def test_duplicate_dimensions(dimension, kernel1, kernel2):
    """Raise ValueError if duplicates in `dimensions`."""
    with pytest.raises(ValueError):
        dim1 = Dimension(dimension, kernel1)
        dim2 = Dimension(dimension, kernel2)
        Smoother([dim1, dim2])
