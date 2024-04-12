"""Tests inputs for smoother function."""

import pytest

import numpy as np
from pandas import DataFrame

from weave.dimension import Dimension
from weave.smoother import Smoother, smooth, smooth_inverse

# Lists of wrong types to test exceptions
value_list = [1, 1.0, "dummy", True, None, [], (), {}]
not_str = [1, 1.0, True, None, [], (), {}]
not_bool = [1, 1.0, None, [], (), {}]
not_dimensions = value_list + [[value] for value in value_list]
not_columns = not_str + [[value] for value in not_str]
not_number = ["dummy", True, None, [], (), {}]

# Example smoother
age = Dimension(
    name="age_id",
    kernel="exponential",
    radius=1,
)
year = Dimension(
    name="year_id",
    kernel="tricubic",
    exponent=0.5,
)
location = Dimension(
    name="location_id",
    coordinates=["super_region", "region", "country"],
    kernel="depth",
    radius=0.9,
)
smoother = Smoother([age, year, location])

# Example inverse smoother
age_inverse = Dimension(
    name="age_id",
    kernel="inverse",
    radius=0.5,
)
year_inverse = Dimension(
    name="year_id",
    kernel="inverse",
    radius=0.5,
)
location_inverse = Dimension(
    name="location_id",
    coordinates=["super_region", "region", "country"],
    kernel="inverse",
    radius=0.5,
)
smoother_inverse = Smoother([age_inverse, year_inverse, location_inverse])


# Example data
data = DataFrame(
    {
        "age_id": [1, 2, 3, 4, 4],
        "age_mean": [0.5, 1.5, 2.5, 3.5, 3.5],
        "year_id": [1980, 1990, 2000, 2010, 2020],
        "location_id": [5, 5, 6, 7, 9],
        "super_region": [1, 1, 1, 1, 2],
        "region": [3, 3, 3, 4, 8],
        "country": [5, 5, 6, 7, 9],
        "fit": [True, False, False, True, True],
        "predict": [False, True, True, False, False],
        "count": [1.0, 2.0, 3.0, 4.0, 5.0],
        "fraction": [0.1, 0.2, 0.3, 0.4, 0.5],
        "residual": [0.2, 0.4, 0.6, 0.8, 1.0],
        "residual_sd": [0.01, 0.02, 0.03, 0.04, 0.05],
        "sd_zero": [0.0, 0.02, 0.03, 0.04, 0.05],
        "sd_negative": [-0.01, 0.02, 0.03, 0.04, 0.05],
        "name": ["a", "b", "c", "d", "e"],
    }
)


# Test constructor types
@pytest.mark.parametrize("dimensions", not_dimensions)
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


def test_duplicate_dimension_names():
    """Raise ValueError if duplicate names in `dimensions`."""
    with pytest.raises(ValueError):
        dim1 = Dimension("dummy", "columns1")
        dim2 = Dimension("dummy", "columns2")
        Smoother([dim1, dim2])


@pytest.mark.parametrize("coords1", ["dummy1", ["dummy1", "dummy2"]])
@pytest.mark.parametrize("coords2", ["dummy1", ["dummy1", "dummy2"]])
def test_duplicate_dimension_coords(coords1, coords2):
    """Raise ValueError if duplicate coordinates in `dimensions`."""
    with pytest.raises(ValueError):
        dim1 = Dimension("dummy1", coords1)
        dim2 = Dimension("dummy2", coords2)
        Smoother([dim1, dim2])


def test_dimension_immutable():
    """Raise AttributeError if attempt to reset `dimensions`."""
    with pytest.raises(AttributeError):
        dim = Dimension("dummy")
        smoother.dimensions = dim


def test_inverse_weights_value():
    """`inverse_weights` set to correct values."""
    assert smoother.inverse_weights is False
    assert smoother_inverse.inverse_weights is True


def test_inverse_weights_error():
    """Raise ValueError if dimensions have inverse and non-inverse kernels."""
    with pytest.raises(ValueError):
        Smoother([age, year_inverse])


def test_inverse_weights_immutable():
    """Raise AttributeError if attempt to reset `inverse_weights`."""
    with pytest.raises(AttributeError):
        smoother.inverse_weights = False


# Test input types
@pytest.mark.parametrize("bad_data", value_list)
def test_data_type(bad_data):
    """Raise TypeError if `data` is not a DataFrame."""
    with pytest.raises(TypeError):
        smoother(bad_data, "residual")


@pytest.mark.parametrize("observed", not_str)
def test_observed_type(observed):
    """Raise TypeError if `observed` is not a str."""
    with pytest.raises(TypeError):
        smoother(data, observed)


@pytest.mark.parametrize("stdev", not_str)
def test_stdev_type(stdev):
    """Raise TypeError if `stdev` is not a str."""
    if stdev is not None:
        with pytest.raises(TypeError):
            smoother(data, "residual", stdev)


@pytest.mark.parametrize("smoothed", not_str)
def test_smoothed_type(smoothed):
    """Raise TypeError if `smoothed` is not a str."""
    if smoothed is not None:
        with pytest.raises(TypeError):
            smoother(data, "residual", smoothed=smoothed)


@pytest.mark.parametrize("fit", not_str)
def test_fit_type(fit):
    """Raise TypeError if `fit` is not a str."""
    if fit is not None:
        with pytest.raises(TypeError):
            smoother(data, "residual", fit=fit)


@pytest.mark.parametrize("predict", not_str)
def test_predict_type(predict):
    """Raise TypeError if `predict` is not a str."""
    if predict is not None:
        with pytest.raises(TypeError):
            smoother(data, "residual", predict=predict)


@pytest.mark.parametrize("down_weight", not_number)
def test_down_weight_type(down_weight):
    """Raise TypeError if `down_weight` is not an int or float."""
    with pytest.raises(TypeError):
        smoother(data, "residual", down_weight=down_weight)


# Test input values
def test_observed_stdev_overlap():
    """Raise ValueError if `observed` == `stdev`."""
    with pytest.raises(ValueError):
        smoother(data, "residual", "residual")


def test_observed_smoothed_overlap():
    """Raise ValueError if `observed` == `smoothed`."""
    with pytest.raises(ValueError):
        smoother(data, "residual", smoothed="residual")


def test_stdev_smoothed_overlap():
    """Raise ValueError if `stdev` == `smoothed`."""
    with pytest.raises(ValueError):
        smoother(data, "residual", stdev="residual_sd", smoothed="residual_sd")


def test_stdev_passed_with_inverse_weights():
    """Raise ValueError if `stdev` not passed when `inverse_weights` is True."""
    with pytest.raises(ValueError):
        smoother_inverse(data, "residual")


def test_smoothed_warning():
    """Trigger UserWarning if `smoothed` already in `data`."""
    with pytest.warns(UserWarning):
        smoother(data, "count", smoothed="residual")


@pytest.mark.parametrize("down_weight", [-1, 2])
def test_down_weight_value(down_weight):
    """Raise ValueError if `down_weight` not in [0, 1]."""
    with pytest.raises(ValueError):
        smoother(data, "residual", stdev="residual_sd", down_weight=down_weight)


# Test data keys
def test_names_in_data():
    """Raise KeyError if `dimension.name` not in `data`."""
    with pytest.raises(KeyError):
        dummy = Dimension("dummy", "age_id")
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, "residual")


@pytest.mark.parametrize("coords", ["dummy", ["age_id", "dummy"]])
def test_coordinates_in_data(coords):
    """Raise KeyError if `dimension.coordinates` not in `data`."""
    with pytest.raises(KeyError):
        dummy = Dimension("age_id", coords)
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, "residual")


def test_observed_in_data():
    """Raise KeyError if `observed` not in `data`."""
    with pytest.raises(KeyError):
        smoother(data, "dummy")


def test_stdev_in_data():
    """Raise KeyError if `stdev` not in `data`."""
    with pytest.raises(KeyError):
        smoother(data, "residual", "dummy")


def test_fit_in_data():
    """Raise KeyError if `fit` not in `data`."""
    with pytest.raises(KeyError):
        smoother(data, "residual", fit="dummy")


def test_predict_in_data():
    """Raise KeyError if `predict` not in `data`."""
    with pytest.raises(KeyError):
        smoother(data, "residual", predict="dummy")


def test_names_in_distance_dict():
    """Raise KeyError if not all `names` in `distance_dict`."""
    with pytest.raises(KeyError):
        dummy = Dimension("age_id", distance="dictionary", distance_dict={(1, 1): 0})
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, "residual")


# Test data types
def test_data_name_type():
    """Raise TypeError if `dimension.name` not int or float."""
    with pytest.raises(TypeError):
        dummy = Dimension("name", "age_id")
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, "residual")


@pytest.mark.parametrize("coords", ["name", ["age_id", "name"]])
def test_data_coordinates_type(coords):
    """Raise TypeError if `dimension.coordinates` not int or float."""
    with pytest.raises(TypeError):
        dummy = Dimension("age_id", coords)
        smoother2 = Smoother([dummy, year, location])
        smoother2(data, "residual")


def test_data_observed_type():
    """Raise TypeError if `observed` not int or float."""
    with pytest.raises(TypeError):
        smoother(data, "name")


def test_data_stdev_type():
    """Raise TypeError if `stdev` not int or float."""
    with pytest.raises(TypeError):
        smoother(data, "residual", "name")


@pytest.mark.parametrize("fit", ["age_id", "count", "name"])
def test_data_fit_type(fit):
    """Raise TypeError if `fit` column is not bool."""
    with pytest.raises(TypeError):
        smoother(data, "residual", fit=fit)


@pytest.mark.parametrize("predict", ["age_id", "count", "name"])
def test_data_predict_type(predict):
    """Raise TypeError if `predict` column is not bool."""
    with pytest.raises(TypeError):
        smoother(data, "residual", predict=predict)


# Test data values
def test_data_name2coord():
    """Raise ValueError if `name` maps to multiple `coordinates`."""
    with pytest.raises(ValueError):
        data2 = data.copy()
        data2.loc[2, "location_id"] = 5
        smoother(data2, "residual")


def test_data_coord2name():
    """Raise ValueError if `coordinates` maps to multiple `name`."""
    with pytest.raises(ValueError):
        data2 = data.copy()
        data2.loc[2, "level_3"] = 5
        smoother(data2, "residual")


def test_data_nans():
    """Raise ValueError if NaNs in `data`."""
    with pytest.raises(ValueError):
        data2 = data.copy()
        data2["dummy"] = 5 * [np.nan]
        smoother(data2, "residual")


@pytest.mark.parametrize("value", [-np.inf, np.inf])
def test_data_infs(value):
    """Raise ValueError if Infs in `data`."""
    with pytest.raises(ValueError):
        data2 = data.copy()
        data2["residual"] = 5 * [value]
        smoother(data2, "residual")


@pytest.mark.parametrize("stdev", ["sd_zero", "sd_negative"])
def test_stdev_values(stdev):
    """Raise ValueError if `stdev` column contains zeros or negative values."""
    with pytest.raises(ValueError):
        smoother(data, "residual", stdev)


# Test smoother output
def test_idx_fit_len():
    """`get_indices` returns array of correct length."""
    idx_fit = smoother.get_indices(data, "fit")
    assert len(idx_fit) == data["fit"].sum()


def test_idx_predict_len():
    """`get_indices` returns array of correct length."""
    idx_pred = smoother.get_indices(data, "predict")
    assert len(idx_pred) == data["predict"].sum()


@pytest.mark.parametrize("fit", ["fit", "predict"])
def test_obs_shape(fit):
    """`get_observed` returns array of correct shape."""
    idx_fit = smoother.get_indices(data, fit)
    cols_obs = smoother.get_values(data, "residual", idx_fit)
    assert cols_obs.shape == (len(idx_fit),)


def test_points_shape():
    """`get_points` returns array of correct shape."""
    points = smoother.get_points(data)
    assert points.shape == (len(data), len(smoother.dimensions))


def test_typed_dimensions_len():
    """`get_typed_dimensions` returns dictionaries of correct length."""
    dim_list = smoother.get_typed_dimensions(data)
    assert len(dim_list) == len(smoother.dimensions)
    for dimension in dim_list:
        n_ids = len(data[dimension.name].unique())
        assert len(dimension.weight_dict) == n_ids**2


@pytest.mark.parametrize("predict", [None, "fit", "predict"])
def test_smooth_shape(predict):
    """`smooth` returns array of correct shape."""
    idx_fit = smoother.get_indices(data, None)
    idx_pred = smoother.get_indices(data, predict)
    col_obs = smoother.get_values(data, "residual", idx_fit)
    col_sd = smoother.get_values(data, None, idx_fit)
    points = smoother.get_points(data)
    dim_list = smoother.get_typed_dimensions(data)
    cols_smooth = smooth(dim_list, points, col_obs, col_sd, idx_fit, idx_pred, 1.0)
    if predict is None:
        assert cols_smooth.shape == (len(data),)
    else:
        assert cols_smooth.shape == (data[predict].sum(),)


@pytest.mark.parametrize("predict", [None, "fit", "predict"])
def test_smooth_inverse_shape(predict):
    """`smooth_inverse` returns array of correct shape."""
    idx_fit = smoother.get_indices(data, None)
    idx_pred = smoother.get_indices(data, predict)
    col_obs = smoother.get_values(data, "residual", idx_fit)
    col_sd = smoother.get_values(data, None, idx_fit)
    points = smoother.get_points(data)
    dim_list = smoother.get_typed_dimensions(data)
    cols_smooth = smooth_inverse(
        dim_list, points, col_obs, col_sd, idx_fit, idx_pred, 1.0
    )
    if predict is None:
        assert cols_smooth.shape == (len(data),)
    else:
        assert cols_smooth.shape == (data[predict].sum(),)


@pytest.mark.parametrize("predict", [None, "fit", "predict"])
def test_smoother_shape(predict):
    """Return data frame with correct shape."""
    result = smoother(data, "residual", predict=predict)
    if predict is None:
        assert len(result) == len(data)
    else:
        assert len(result) == data[predict].sum()
    assert len(result.columns) == len(data.columns) + 1


@pytest.mark.parametrize("smoothed", [None, "dummy"])
def test_smoother_columns(smoothed):
    """Return data frame with correct column names."""
    result = smoother(data, "residual", smoothed=smoothed)
    if smoothed is None:
        assert "residual_smooth" in result.columns
    else:
        assert smoothed in result.columns


@pytest.mark.parametrize("smoothed", [None, "dummy"])
def test_smoother_inverse_columns(smoothed):
    """Return data frame with correct column names."""
    result = smoother_inverse(data, "residual", "residual_sd", smoothed=smoothed)
    if smoothed is None:
        assert "residual_smooth" in result.columns
    else:
        assert smoothed in result.columns


def test_result():
    """Check output values."""
    result = smoother(data, "residual")
    vals = np.array([0.25019485, 0.41681382, 0.5839969, 0.79772836, 1.0])
    assert np.allclose(vals, result["residual_smooth"].values)
    result = smoother(data, "residual", "residual_sd")
    vals = np.array([0.20730056, 0.38848886, 0.5644261, 0.79567325, 1.0])
    assert np.allclose(vals, result["residual_smooth"].values)
    result = smoother(data, "residual", fit="fit")
    vals = np.array([0.20659341, 0.20659341, 0.26, 0.7934066, 1.0])
    assert np.allclose(vals, result["residual_smooth"].values)


def test_inverse_result():
    """Check output values."""
    result = smoother_inverse(data, "residual", "residual_sd")
    vals = np.array([0.20000343, 0.40000615, 0.59999865, 0.7999712, 0.99992466])
    assert np.allclose(vals, result["residual_smooth"].values)
    result = smoother_inverse(data, "residual", "residual_sd", fit="fit")
    vals = np.array([0.20000166, 0.4879758, 0.6842466, 0.7999973, 0.9999626])
    assert np.allclose(vals, result["residual_smooth"].values)
