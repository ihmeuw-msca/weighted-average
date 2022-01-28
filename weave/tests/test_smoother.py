"""Tests for smoother function.

TODO:
* TypeError to ValueError for empty lists
* So many other things...

"""
import pytest

from weave.dimension import Dimension
from weave.smoother import Smoother

# Lists of wrong types to test exceptions
value_list = [1, 1.0, 'dummy', True, None, [], (), {}]
not_dimensions = value_list + [[value] for value in value_list]


# Test constructor types
@pytest.mark.parametrize('dimensions', not_dimensions)
def test_dimensions_type(dimensions):
    """Raise TypeError if invalid type for `dimensions`."""
    if dimensions != []:
        with pytest.raises(TypeError):
            Smoother(dimensions)


# Test constructor values
def test_dimensions_values():
    """Raise ValueError if `dimensions` is an empty list."""
    with pytest.raises(ValueError):
        Smoother([])


@pytest.mark.parametrize('kernel1', ['exponential', 'tricubic', 'depth'])
@pytest.mark.parametrize('kernel2', ['exponential', 'tricubic', 'depth'])
def test_duplicate_names(kernel1, kernel2):
    """Raise ValueError if duplicate names in `dimensions`."""
    with pytest.raises(ValueError):
        pars = {'radius': 0.5, 'exponent': 3}
        dim1 = Dimension('dummy', 'columns1', kernel1, pars)
        dim2 = Dimension('dummy', 'columns2', kernel2, pars)
        Smoother([dim1, dim2])


@pytest.mark.parametrize('columns1', ['dummy1', ['dummy1', 'dummy2']])
@pytest.mark.parametrize('columns2', ['dummy1', ['dummy1', 'dummy2']])
@pytest.mark.parametrize('kernel1', ['exponential', 'tricubic', 'depth'])
@pytest.mark.parametrize('kernel2', ['exponential', 'tricubic', 'depth'])
def test_duplicate_columns(columns1, columns2, kernel1, kernel2):
    """Raise ValueError if duplicate columns in `dimensions`."""
    with pytest.raises(ValueError):
        pars = {'radius': 0.5, 'exponent': 3}
        dim1 = Dimension('dummy1', columns1, kernel1, pars)
        dim2 = Dimension('dummy2', columns2, kernel2, pars)
        Smoother([dim1, dim2])
