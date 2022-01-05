"""Tests for smoother function."""
import pytest

from weave.dimension import Dimension
from weave.smoother import Smoother

# Lists of wrong types to test exceptions
value_list = [1, 1.0, 'dummy', True, None, [], (), {}]
value_list += [[value] for value in value_list]
not_dimensions = value_list + [[value] for value in value_list]


# Test constructor types
@pytest.mark.parametrize('dimensions', not_dimensions)
def test_dimensions_type(dimensions):
    """Raise TypeError if invalid type for `dimensions`."""
    with pytest.raises(TypeError):
        Smoother(dimensions)


# Test duplicates in `dimensions`
@pytest.mark.parametrize('dimension1', ['dummy1', ['dummy1', 'dummy2']])
@pytest.mark.parametrize('dimension2', ['dummy1', ['dummy1', 'dummy2']])
@pytest.mark.parametrize('kernel1', ['exponential', 'tricubic', 'depth'])
@pytest.mark.parametrize('kernel2', ['exponential', 'tricubic', 'depth'])
def test_duplicate_dimensions(dimension1, dimension2, kernel1, kernel2):
    """Raise ValueError if duplicates in `dimensions`."""
    with pytest.raises(ValueError):
        dim1 = Dimension(dimension1, kernel1, radius=0.5, exponent=3)
        dim2 = Dimension(dimension2, kernel2, radius=0.5, exponent=3)
        Smoother([[dim1, dim2]])
