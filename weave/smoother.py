# pylint: disable=C0103, E0611
"""Smooth data across multiple dimensions using weighted averages.

TODO
* Write checks and tests
* Fix mypy errors
* Type hints not consistent with numba types
* Check tests for Dimension vs. list of Dimension

Checks
* Check for duplicates in columns
* Check for NaNs and Infs in data frame
* Check columns, fit, predict in data frame
* Anything else?

"""
from typing import Dict, List, Optional, Tuple, Union

from numba import njit
from numba.typed import Dict as TypedDict, List as TypedList
from numba.types import DictType, float64, unicode_type, UniTuple
import numpy as np
from pandas import DataFrame

from weave.dimension import Dimension, TypedDimension
from weave.distance import dictionary, euclidean, hierarchical
from weave.kernels import exponential, depth, tricubic
from weave.utils import as_list, flatten

Numeric = Union[int, float]
Pars = Union[Numeric, bool]
DistanceDict = Dict[Tuple[Numeric, Numeric], Numeric]
TypedDistanceDict = DictType(UniTuple(float64, 2), float64)


class Smoother:
    """Smoother function.

    Attributes
    ----------
    dimensions : list of Dimension
        Smoothing dimensions.

    """

    def __init__(self, dimensions: Union[Dimension, List[Dimension]]) -> None:
        """Create smoother function.

        Parameters
        ----------
        dimensions : Dimension or list of Dimension
            Smoothing dimensions.

        """
        self.dimensions = dimensions

    @property
    def dimensions(self) -> List[Dimension]:
        """Get smoothing dimensions.

        Returns
        -------
        list of Dimension
            Smoothing dimensions.

        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: Union[Dimension, List[Dimension]]) \
            -> None:
        """Set smoothing dimensions.

        Parameters
        ----------
        dimensions : Dimension or list of Dimension
            Smoothing dimensions.

        Raises
        ------
        TypeError
            If `dimensions` is not a list of Dimension.
        ValueError
            If `dimensions` is an empty list, contains an empty list,
            or contains duplicate names or columns.

        """
        # Check types
        dimensions = as_list(dimensions)
        if not all(isinstance(dim, Dimension) for dim in dimensions):
            raise TypeError('`dimensions` contains invalid types.')

        # Check values
        if len(dimensions) == 0:
            raise ValueError('`dimensions` is an empty list.')
        name_list = [dim.name for dim in dimensions]
        if len(name_list) > len(set(name_list)):
            raise ValueError('Duplicate names found in `dimensions`.')
        col_list = flatten([dim.columns for dim in dimensions])
        if len(col_list) > len(set(col_list)):
            raise ValueError('Duplicate columns found in `dimensions`.')

        self._dimensions = dimensions

    def __call__(self, data: DataFrame, columns: Union[str, List[str]],
                 fit: str = None, predict: str = None) -> DataFrame:
        """Smooth data across dimensions with weighted averages.

        For each point in `predict`, calculate smoothed value of
        each column in `columns` using a weighted average of points in
        `fit`, where weights are calculated based on proximity across
        `dimensions`. Return a data frame of points in `predict` with
        added columns '{column}_smooth' for each column in `columns`
        containing smoothed values.

        Column `fit` should contain booleans indicating whether or not
        a given point should be used to compute weighted averages.
        Column `predict` should contain booleans indicating whether or
        not to calculate a smoothed value for a given point.

        Parameters
        ----------
        data : DataFrame
            Input data structure.
        columns : str or list of str
            Column names of values to smooth.
        fit : str, optional
            Column name indicating points to include in weighted
            averages. If None, all points in `data` are used.
        predict : str, optional
            Column name indicating points to predict smoothed values.
            If None, predictions are made for all points in `data`.

        Returns
        -------
        pandas.DataFrame
            Points in `predict` with smoothed `columns` values.

        Raises
        ------
        # Check input types
        # Check columns, fit, predict in data
        # Anything else?

        """
        # Run checks (TODO)

        # Extract data
        idx_fit = get_indices(data, fit)
        idx_pred = get_indices(data, predict)
        cols = get_columns(data, columns, idx_fit)
        point_list = self.get_points(data)
        dim_list = self.get_typed_dimensions()

        # Calculate smoothed values
        cols_smooth = smooth_data(dim_list, point_list, cols, idx_fit,
                                  idx_pred)

        # Construct smoothed data frame
        data_smooth = data.iloc[idx_pred].reset_index(drop=True)
        for idx_col, col in enumerate(as_list(columns)):
            data_smooth[f"{col}_smooth"] = cols_smooth[:, idx_col]
        return data_smooth

    def get_points(self, data: DataFrame) -> List[np.ndarray]:
        """Get point locations by dimension.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.

        Returns
        -------
        list of numpy.ndarray of float
            Point locations by dimension.

        """
        point_list = TypedList()
        for dim in self._dimensions:
            dim_array = np.atleast_2d(data[dim.columns].values)
            dim_array = np.ascontiguousarray(dim_array, dtype=float)
            point_list.append(dim_array)
        return point_list

    def get_typed_dimensions(self) -> List[TypedDimension]:
        """Get smoothing dimensions cast as jitclass objects.

        Returns
        -------
        list of TypedDimension
            Smoothing dimensions cast as jitclass objects.

        """
        dim_list = TypedList()
        for dim in self._dimensions:
            # Get typed version of attributes
            columns = TypedList(dim.columns)
            kernel_pars = get_typed_pars(dim.kernel_pars)
            if hasattr(dim, 'distance_dict'):
                distance_dict = get_typed_dict(dim.distance_dict)
            else:
                distance_dict = get_typed_dict()

            # Create typed dimension
            typed_dim = TypedDimension(dim.name, columns, dim.kernel,
                                       kernel_pars, dim.distance,
                                       distance_dict)
            dim_list.append(typed_dim)
        return dim_list


def get_indices(data: DataFrame, indicator: str = None) -> np.ndarray:
    """Get indices of `fit` or `predict` data.

    Parameters
    ----------
    data : DataFrame
        Input data structure.
    indicator : str, optional
        Column name indicating either `fit` or `predict` data.

    Returns
    -------
    numpy.ndarray of int
        Indices of `fit` or `predict` points.

    """
    if indicator is None:
        return np.arange(len(data))
    return np.where(data[indicator])[0]


def get_columns(data: DataFrame, columns: Union[str, List[str]],
                idx_fit: np.ndarray) -> np.ndarray:
    """Get values to smooth.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data structure.
    columns : str or list of str
        Column names of values to smooth.
    idx_fit : numpy.ndarray of int
        Indices of `fit` points.

    Returns
    -------
    numpy.ndarray of float
        Values to smooth.

    """
    return np.array([data[col].values[idx_fit] for col in as_list(columns)],
                    dtype=float)


def get_typed_pars(kernel_pars: Dict[str, Pars]) \
        -> DictType(unicode_type, float64):
    """Get typed version of `kernel_pars`.

    Parameters
    ----------
    kernel_pars : dict of {str: numeric or bool}
        Kernel function parameters.

    Returns
    -------
    dict of {str: float}
        Typed version of `kernel_pars`.

    """
    typed_pars = TypedDict()
    for key in kernel_pars:
        typed_pars[key] = float(kernel_pars[key])
    return typed_pars


def get_typed_dict(distance_dict: Optional[DistanceDict] = None) \
        -> TypedDistanceDict:
    """Get typed version of `distance_dict`.

    Parameters
    ----------
    distance_dict : dict of {(numeric, numeric): numeric}
        Dictionary of distances between points if `distance` is
        'dictionary'.

    Returns
    -------
    dict of {(float, float): float}
        Typed version of `distance_dict`.

    """
    typed_dict = TypedDict.empty(
        key_type=UniTuple(float64, 2),
        value_type=float64
    )
    if distance_dict is not None:
        for key in distance_dict:
            float_key = tuple(float(point) for point in key)
            typed_dict[float_key] = float(distance_dict[key])
    return typed_dict


@njit
def smooth_data(dim_list: List[TypedDimension], point_list: List[np.ndarray],
                cols: np.ndarray, idx_fit: np.ndarray, idx_pred: np.ndarray) \
        -> np.ndarray:
    """Smooth data across dimensions with weighted averages.

    Parameters
    ----------
    dim_list : list of TypedDimension
        Smoothing dimensions.
    point_list : list of numpy.ndarray of float
        Point locations by dimension.
    cols : numpy.ndarray of float
        Values to smooth.
    idx_fit : numpy.ndarray of int
        Indices of points to include in weighted averages.
    idx_pred: numpy.ndarray of int
        Indices of points to predict smoothed values.

    Returns
    -------
    numpy.ndarray of float
        Smoothed values.

    """
    # Initialize smoothed values
    n_cols = len(cols)
    n_pred = len(idx_pred)
    cols_smooth = np.empty((n_pred, n_cols))

    # Calculate smoothed values one point at a time
    for idx_x in range(n_pred):
        weights = get_weights(dim_list, point_list, idx_fit, idx_pred[idx_x])
        cols_smooth[idx_x, :] = cols.dot(weights)

    return cols_smooth


@njit
def get_weights(dim_list: List[TypedDimension], point_list: List[np.ndarray],
                idx_fit: np.ndarray, idx_x: int) -> np.ndarray:
    """Get smoothing weights for current point.

    Parameters
    ----------
    dim_list : list of TypedDimension
        Smoothing dimensions.
    point_list : list of numpy.ndarray
        Point locations by dimension.
    idx_fit : numpy.ndarray of int
        Indices of nearby points to include in weighted averages.
    idx_x : int
        Index of current point to predict smoothed values.

    Returns
    -------
    numpy.ndarray of nonnegative float
        Smoothing weights for current point.

    """
    # Initialize weight vector
    weights = np.ones(len(idx_fit))

    # Calculate weights one dimension at at a time
    for idx_dim, dim in enumerate(dim_list):
        dim_weights = get_dim_weights(dim, point_list[idx_dim], idx_fit, idx_x)

        # Optional normalize by subgroup
        if dim.kernel == 'depth' and dim.kernel_pars['normalize'] == 1.0:
            for weight in list(set(dim_weights)):
                idx_weight = np.where(dim_weights == weight)[0]
                weights[idx_weight] *= weight/weights[idx_weight].sum()
        else:
            weights *= dim_weights

    return weights/weights.sum()


@njit
def get_dim_weights(dimension: TypedDimension, points: np.ndarray,
                    idx_fit: np.ndarray, idx_x: int) -> np.ndarray:
    """Get smoothing weights for current point and dimension.

    Parameters
    ----------
    dimension : TypedDimension
        Current smoothing dimension.
    points : numpy.ndarray
        Point locations for current dimension.
    idx_fit : numpy.ndarray of int
        Indices of nearby points to include in weighted averages.
    idx_x : int
        Index of current point to predict smoothed values.

    Returns
    -------
    numpy.ndarray of nonnegative float
        Smoothing weights for current point and dimension.

    """
    # Initialize weight vector
    n_fit = len(idx_fit)
    weights = np.empty(n_fit)

    # Calculate weights one point at a time
    x = get_point(points, idx_x)
    for idx_y in range(n_fit):
        y = get_point(points, idx_fit[idx_y])
        dist = get_distance(x, y, dimension.distance, dimension.distance_dict)
        weights[idx_y] = get_weight(dist, dimension.kernel,
                                    dimension.kernel_pars)
    return weights


@njit
def get_point(dim: np.ndarray, idx_point: int) -> np.ndarray:
    """Get point `x` or `y` as a vector.

    Parameters
    ----------
    dim : 2D numpy.ndarray of float
        Point locations for a given dimension.
    idx_point : int
        Index of `x` or `y` in `dim`.

    Returns
    -------
    1D numpy.ndarray of float
        Point `x` or `y` as a vector.

    """
    if dim.shape[0] == 1:
        return np.atleast_1d(np.array(dim[0][idx_point]))
    return dim[idx_point]


@njit
def get_distance(x: np.ndarray, y: np.ndarray, distance: str,
                 distance_dict: TypedDistanceDict) -> float:
    """Get distance between `x` and `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    y : 1D numpy.ndarray of float
        Nearby point.
    distance : str
        Distance function name.
    distance_dict : dict of {(float, float): float}
        Dictionary of distances between points.

    Returns
    -------
    nonnegative float
        Distance between `x` and `y`.

    """
    if distance == 'dictionary':
        return dictionary(x, y, distance_dict)
    if distance == 'euclidean':
        return euclidean(x, y)
    return hierarchical(x, y)


@njit
def get_weight(distance: float, kernel: str, pars: Dict[str, float]) -> float:
    """Get smoothing weight.

    Parameters
    ----------
    distance : nonnegative float
        Distance between points.
    kernel : str
        Kernel function name.
    pars : dict of {str: float}
        Kernel function parameters.

    Returns
    -------
    nonnegative float
        Smoothing weight.

    """
    if kernel == 'exponential':
        return exponential(distance, pars)
    if kernel == 'tricubic':
        return tricubic(distance, pars)
    return depth(distance, pars)
