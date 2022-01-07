# pylint: disable=C0103, E0611, R0913, R0914
"""Smooth data across multiple dimensions using weighted averages.

TODO
* Write checks and tests

Checks
* Check for duplicates in columns
* Check for NaNs and Infs in data frame
* Check columns, fit, predict in data frame
* Anything else?

"""
from typing import Dict, List, Tuple, Union

from numba import njit
from numba.typed import List as TypedList
from numba.types import ListType, float64
import numpy as np
from pandas import DataFrame

from weave.dimension import Dimension
from weave.distance import euclidean, hierarchical
from weave.kernels import exponential, depth, tricubic
from weave.utils import as_list, flatten


class Smoother:
    """Smoother function.

    Attributes
    ----------
    dimensions : list of list of Dimension
        Smoothing dimension(s).

    """

    def __init__(self, dimensions: List[List[Dimension]]) -> None:
        """Create smoother function.

        Dimension weights are aggregated in the order and groupings
        present in `dimensions`. For example, if `dimensions` ==
        [['age', 'year'], ['location']], then age and year weights will
        be multiplied and normalized, then the result will be
        multiplied by location weights and normalized.

        Parameters
        ----------
        dimensions : list of list of Dimension
            Smoothing dimension(s).

        """
        self.dimensions = dimensions

    @property
    def dimensions(self) -> List[List[Dimension]]:
        """Get smoothing dimension(s).

        Returns
        -------
        list of list of Dimension
            Smoothing dimension(s).

        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: List[List[Dimension]]) -> None:
        """Set smoothing dimension(s).

        Parameters
        ----------
        dimensions : list of list of Dimension
            Smoothing dimension(s).

        Raises
        ------
        TypeError
            If `dimensions` is not a list of list of Dimension.
        ValueError
            If duplicates found in `dimensions`.

        """
        # Check types
        if not isinstance(dimensions, list):
            raise TypeError('`dimensions` is not a list.')
        if len(dimensions) == 0:
            raise TypeError('`dimensions` is an empty list.')
        if not all(isinstance(group, list) for group in dimensions):
            raise TypeError('`dimensions` contains invalid type(s).')
        for group in dimensions:
            if len(group) == 0:
                raise TypeError('`dimensions` contains an empty list.')
            if not all(isinstance(dim, Dimension) for dim in group):
                raise TypeError('`dimensions` contains invalid type(s).')

        # Check duplicates
        dim_list = flatten([dim.dimension for dim in flatten(dimensions)])
        if len(dim_list) > len(set(dim_list)):
            raise ValueError('Duplicates found in `dimensions`.')

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
            Column name(s) of values to smooth.
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

        # Extract data from data frame
        idx_fit = get_indices(data, fit)
        idx_pred = get_indices(data, predict)
        col_list = get_columns(data, columns, idx_fit)
        point_list = self.get_points(data)
        kernel_list = self.get_functions('kernel')
        pars_list = self.get_functions('pars')
        dist_list = self.get_functions('distance')

        # Calculate smoothed values
        cols_smooth = smooth_data(point_list, col_list, idx_fit, idx_pred,
                                  kernel_list, pars_list, dist_list)

        # Construct smoothed data frame
        data_smooth = data.iloc[idx_pred].reset_index(drop=True)
        for idx_col, col in enumerate(as_list(columns)):
            data_smooth[f"{col}_smooth"] = cols_smooth[:, idx_col]
        return data_smooth

    def get_points(self, data: DataFrame) -> List[List[np.ndarray]]:
        """Get points by dimension group.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.

        Returns
        -------
        list of list of numpy.ndarray of float
            Points by dimension group.

        """
        point_list = TypedList()
        for group in self._dimensions:
            group_list = TypedList(lsttype=ListType(float64[:, :]))
            for dim in group:
                dim_array = data[dim.dimension].values.astype(float)
                group_list.append(np.atleast_2d(dim_array))
            point_list.append(group_list)
        return point_list

    def get_functions(self, func: str) \
            -> List[List[Union[str, Dict[str, float]]]]:
        """Get kernel or distance function data by dimension group.

        Parameters
        ----------
        func : {'kernel', 'pars', 'distance'}
            Name of dimension attribute.

        Returns
        -------
        list of list of {str or dict of {str: float}}
            Kernel or distance function data by dimension group.

        """
        return TypedList(TypedList(getattr(dim, func) for dim in group)
                         for group in self._dimensions)


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
                idx_fit: np.ndarray) -> List[str]:
    """Get values to smooth.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data structure.
    columns : str or list of str
        Column name(s) of values to smooth.
    idx_fit : numpy.ndarray of int
        Indices of `fit` points.

    Returns
    -------
    list of str
        Values to smooth.

    """
    return TypedList(data[col].values[idx_fit] for col in as_list(columns))


@njit
def smooth_data(point_list: List[List[np.ndarray]], col_list: List[np.ndarray],
                idx_fit: np.ndarray, idx_pred: np.ndarray,
                kernel_list: List[List[str]],
                pars_list: List[List[Dict[str, float]]],
                dist_list: List[List[str]]) -> np.ndarray:
    """Smooth data across dimensions with weighted averages.

    Parameters
    ----------
    point_list : list of list of numpy.ndarray of float
        Point locations by dimension group.
    col_list : list of numpy.ndarray of float
        Values to smooth.
    idx_fit : numpy.ndarray of int
        Indices of points to include in weighted averages.
    idx_pred: numpy.ndarray of int
        Indices of points to predict smooth values.
    kernel_list : list of list of str
        Kernel function names by dimension group.
    pars_list : list of list of dict of {str: float}
        Kernel function parameters by dimension group.
    dist_lsit : list of list of str
        Distance function names by dimension group.

    Returns
    -------
    np.ndarray of float
        Smoothed values.

    """
    # Initialize smoothed values
    n_cols = len(col_list)
    n_pred = len(idx_pred)
    smooth_cols = np.empty((n_pred, n_cols))

    # Calculate smoothed values one point at a time
    for idx_x in range(n_pred):
        weights = get_weights(point_list, kernel_list, pars_list, dist_list,
                              idx_fit, idx_pred[idx_x])

        # Compute smoothed values one column at a time
        for idx_col in range(n_cols):
            smooth_cols[idx_x, idx_col] = weights.dot(col_list[idx_col])

    return smooth_cols


@njit
def get_weights(group_list: List[List[np.ndarray]],
                kernel_list: List[List[str]],
                pars_list: List[List[Dict[str, float]]],
                dist_list: List[List[str]], idx_fit: np.ndarray, idx_x: int) \
        -> np.ndarray:
    """Get smoothing weights for current point.

    Parameters
    ----------
    group_list : list of list of numpy.ndarray
        Point locations across dimension groups.
    kernel_list : list of list of str
        Kernel function names for dimension groups.
    pars_list : list of list of dict of {str: float}
        Kernel function parameters for dimension groups.
    dist_list : list of list of str
        Distance function names for dimension groups.
    idx_fit : numpy.ndarray of int
        Indices of nearby points in `group_list`.
    idx_x : int
        Index of current point in `group_list`.

    Returns
    -------
    numpy.ndarray of nonnegative float
        Smoothing weights for current point.

    """
    # Initialize weight vector
    weights = np.ones(len(idx_fit))

    # Calculate weights one group at a time
    for idx_group, dim_list in enumerate(group_list):
        group_weights = get_group_weights(dim_list, kernel_list[idx_group],
                                          pars_list[idx_group],
                                          dist_list[idx_group], idx_fit, idx_x)
        weights *= group_weights
        weights /= weights.sum()

    return weights


@njit
def get_group_weights(dim_list: List[np.ndarray], kernel_list: List[str],
                      pars_list: List[Tuple[Union[int, float]]],
                      dist_list: List[str], idx_fit: np.ndarray, idx_x: int) \
        -> np.ndarray:
    """Get smoothing weights for current point and dimension group.

    Parameters
    ----------
    dim_list : list of numpy.npdarray
        Point locations across group dimension(s).
    kernel_list : list of str
        Kernel function names for group dimension(s).
    pars_list : list of tuple of int or float
        Kernel function parameters for group dimension(s).
    dist_list : list of str
        Distance function names for group dimension(s).
    idx_fit : numpy.ndarray of int
        Indices of nearby points in `group_list`.
    idx_x : int
        Index of current point in `group_list`.

    Returns
    -------
    numpy.ndarray of nonnegative float
        Smoothing weights for current point and dimension group.

    """
    # Initialize weight vector
    n_fit = len(idx_fit)
    weights = np.ones(n_fit)

    # Calculate weights one dimension at a time
    for idx_dim, dim in enumerate(dim_list):
        x = get_point(dim, idx_x)
        pars = pars_list[idx_dim]

        # Calculate weights one point at a time
        dim_weights = np.empty_like(weights)
        for idx_y in range(n_fit):
            y = get_point(dim, idx_fit[idx_y])
            dist = get_distance(x, y, dist_list[idx_dim])
            dim_weights[idx_y] = get_weight(dist, kernel_list[idx_dim], pars)
        weights *= dim_weights

    return weights/weights.sum()


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
def get_distance(x: np.ndarray, y: np.ndarray, distance: str) -> float:
    """Get distance between `x` and `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    y : 1D numpy.ndarray of float
        Nearby point.
    distance : str
        Distance function name.

    Returns
    -------
    nonnegative float
        Distance between `x` and `y`.

    """
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
