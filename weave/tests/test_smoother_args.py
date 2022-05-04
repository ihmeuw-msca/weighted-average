"""Tests for smoother function.

TODO:
* Test behavior of functions within smoother module
  (create separate module)

"""
import pytest

import numpy as np
from pandas import DataFrame

from weave.dimension import Dimension
from weave.smoother import Smoother

# Lists of wrong types to test exceptions
value_list = [1, 1.0, 'dummy', True, None, [], (), {}]
not_str = [1, 1.0, True, None, [], (), {}]
not_bool = [1, 1.0, None, [], (), {}]
not_dimensions = value_list + [[value] for value in value_list]
not_columns = not_str + [[value] for value in not_str]

# Example smoother
age = Dimension('age_id', 'age_id', 'exponential', {'radius': 1}, 'euclidean')
year = Dimension('year_id', 'year_id', 'tricubic',
                 {'radius': 40, 'exponent': 0.5}, 'euclidean')
location = Dimension('location_id', ['level_1', 'level_2', 'level_3'], 'depth',
                     {'radius': 0.9}, 'hierarchical')
smoother = Smoother([age, year, location])

# Example data
data = DataFrame({
    'age_id': [1, 2, 3, 4, 4],
    'age_mean': [0.5, 1.5, 2.5, 3.5, 3.5],
    'year_id': [1980, 1990, 2000, 2010, 2020],
    'location_id': [5, 5, 6, 7, 9],
    'level_1': [1, 1, 1, 1, 2],
    'level_2': [3, 3, 3, 4, 8],
    'level_3': [5, 5, 6, 7, 9],
    'fit': [True, False, False, True, True],
    'predict': [False, True, True, False, False],
    'count': [1.0, 2.0, 3.0, 4.0, 5.0],
    'residual': [0.1, 0.2, 0.3, 0.4, 0.5],
    'name': ['a', 'b', 'c', 'd', 'e']
})


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


@pytest.mark.parametrize('coords1', ['dummy1', ['dummy1', 'dummy2']])
@pytest.mark.parametrize('coords2', ['dummy1', ['dummy1', 'dummy2']])
@pytest.mark.parametrize('kernel1', ['exponential', 'tricubic', 'depth'])
@pytest.mark.parametrize('kernel2', ['exponential', 'tricubic', 'depth'])
def test_duplicate_columns(coords1, coords2, kernel1, kernel2):
    """Raise ValueError if duplicate coordinates in `dimensions`."""
    with pytest.raises(ValueError):
        pars = {'radius': 0.5, 'exponent': 3}
        dim1 = Dimension('dummy1', coords1, kernel1, pars)
        dim2 = Dimension('dummy2', coords2, kernel2, pars)
        Smoother([dim1, dim2])


# Test input types
@pytest.mark.parametrize('bad_data', value_list)
def test_data_type(bad_data):
    """Raise TypeError if `data` is not a DataFrame."""
    with pytest.raises(TypeError):
        smoother(bad_data, 'residual')


@pytest.mark.parametrize('columns', not_columns)
def test_columns_type(columns):
    """Raise TypeError if `columns` is not a str or list of str."""
    if columns != []:
        with pytest.raises(TypeError):
            smoother(data, columns)


@pytest.mark.parametrize('fit', not_str)
def test_fit_type(fit):
    """Raise TypeError if `fit` is not a str."""
    if fit is not None:
        with pytest.raises(TypeError):
            smoother(data, 'residual', fit)


@pytest.mark.parametrize('predict', not_str)
def test_predict_type(predict):
    """Raise TypeError if `predict` is not a str."""
    if predict is not None:
        with pytest.raises(TypeError):
            smoother(data, 'residual', predict=predict)


@pytest.mark.parametrize('loop', not_bool)
def test_loop_type(loop):
    """Raise TypeError if `loop` is not a bool."""
    with pytest.raises(TypeError):
        smoother(data, 'residual', loop=loop)


@pytest.mark.parametrize('precompute', not_bool)
def test_precomptue_type(precompute):
    """Raise TypeError if `precompute` is not a bool."""
    with pytest.raises(TypeError):
        smoother(data, 'residual', precompute=precompute)


@pytest.mark.parametrize('parallel', not_bool)
def test_parallel_type(parallel):
    """Raise TypeError if `parallel` is not a bool."""
    with pytest.raises(TypeError):
        smoother(data, 'residual', parallel=parallel)


# Test input values
def test_columns_values():
    """Raise ValueError if `columns` is an empty list."""
    with pytest.raises(ValueError):
        smoother(data, [])


def test_columns_duplicates():
    """Raise ValueError if duplicate names in `columns`."""
    with pytest.raises(ValueError):
        smoother(data, ['residual', 'residual'])


# Test data keys
def test_names_in_data():
    """Raise KeyError if `dimension.name` not in `data`."""
    with pytest.raises(KeyError):
        dummy = Dimension('dummy', 'age_id', 'exponential', {'radius': 0.9})
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, 'residual')


@pytest.mark.parametrize('coords', ['dummy', ['age_id', 'dummy']])
def test_coordinates_in_data(coords):
    """Raise KeyError if `dimension.coordinates` not in `data`."""
    with pytest.raises(KeyError):
        dummy = Dimension('age_id', coords, 'exponential', {'radius': 0.9})
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, 'residual')


@pytest.mark.parametrize('columns', ['dummy', ['residual', 'dummy']])
def test_columns_in_data(columns):
    """Raise KeyError if `columns` not in `data`."""
    with pytest.raises(KeyError):
        smoother(data, columns)


def test_fit_in_data():
    """Raise KeyError if `fit` not in `data`."""
    with pytest.raises(KeyError):
        smoother(data, 'residual', 'dummy')


def test_predict_in_data():
    """Raise KeyError if `predict` not in `data`."""
    with pytest.raises(KeyError):
        smoother(data, 'residual', predict='dummy')


def test_coordinates_in_distance_dict():
    """Raise KeyError if not all `coordinates` in `distance_dict`."""
    with pytest.raises(KeyError):
        dummy = Dimension('age_id', distance='dictionary',
                          distance_dict={(1, 1): 0})
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, 'residual')


# Test data types
def test_data_name_type():
    """Raise TypeError if `dimension.name` not int or float."""
    with pytest.raises(TypeError):
        dummy = Dimension('name', 'age_id', 'exponential', {'radius': 0.9})
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, 'residual')


@pytest.mark.parametrize('coords', ['name', ['age_id', 'name']])
def test_data_coordinates_type(coords):
    """Raise TypeError if `dimension.coordinates` not int or float."""
    with pytest.raises(TypeError):
        dummy = Dimension('age_id', coords, 'exponential', {'radius': 0.9})
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, 'residual')


@pytest.mark.parametrize('columns', ['name', ['residual', 'name']])
def test_data_columns_type(columns):
    """Raise TypeError if `columns` are not int or float."""
    with pytest.raises(TypeError):
        smoother(data, columns)


@pytest.mark.parametrize('fit', ['age_id', 'count', 'name'])
def test_data_fit_type(fit):
    """Raise TypeError if `fit` column is not bool."""
    with pytest.raises(TypeError):
        smoother(data, 'residual', fit)


@pytest.mark.parametrize('predict', ['age_id', 'count', 'name'])
def test_data_predict_type(predict):
    """Raise TypeError if `predict` column is not bool."""
    with pytest.raises(TypeError):
        smoother(data, 'residual', predict=predict)


# Test data values
def test_data_nans():
    """Raise ValueError if NaNs in `data`."""
    with pytest.raises(ValueError):
        data2 = data.copy()
        data2['dummy'] = 5*[np.nan]
        smoother(data2, 'residual')


@pytest.mark.parametrize('value', [-np.inf, np.inf])
def test_data_infs(value):
    """Raise ValueError if Infs in `data`."""
    with pytest.raises(ValueError):
        data2 = data.copy()
        data2['residual'] = 5*[value]
        smoother(data2, 'residual')


def test_data_name2coord():
    """Raise ValueError if `name` maps to multiple `coordinates`."""
    with pytest.raises(ValueError):
        data2 = data.copy()
        data2.loc[2, 'location_id'] = 5
        smoother(data2, 'residual')


def test_data_coord2name():
    """Raise ValueError if `coordinates` maps to multiple `name`."""
    with pytest.raises(ValueError):
        data2 = data.copy()
        data2.loc[2, 'level_3'] = 5
        smoother(data2, 'residual')