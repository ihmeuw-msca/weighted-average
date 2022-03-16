# pylint: disable=C0103, E0611, R0913
"""Smooth data across multiple dimensions using weighted averages."""
from typing import Dict, List, Optional, Tuple, Union

from numba import njit  # type: ignore
from numba.typed import List as TypedList  # type: ignore
import numpy as np
from pandas import DataFrame  # type: ignore
from pandas.api.types import is_bool_dtype, is_numeric_dtype  # type: ignore

from weave.dimension import Dimension, TypedDimension, get_typed_dimension
from weave.distance import dictionary, euclidean, hierarchical
from weave.kernels import exponential, depth, tricubic
from weave.utils import as_list, flatten


class Smoother:
    """Smoother function.

    Attributes
    ----------
    dimensions : list of Dimension
        Smoothing dimensions.

    See Also
    --------
    weave.dimension.Dimension

    """

    def __init__(self, dimensions: Union[Dimension, List[Dimension]]) -> None:
        """Create smoother function.

        Parameters
        ----------
        dimensions : Dimension or list of Dimension
            Smoothing dimensions.

        Examples
        --------
        Create a space-time smoother to smooth data across age, year,
        and location.

        >>> from weave.dimension import Dimension
        >>> from weave.smoother import Smoother
        >>> age = Dimension(
                name='age',
                columns='age_mid',
                kernel='exponential',
                kernel_pars={'radius': 1}
            )
        >>> year = Dimension(
                name='year',
                columns='year_id',
                kernel='tricubic',
                kernel_pars={'radius': 40, 'exponent': 0.5}
            )
        >>> location = Dimension(
                name='location',
                columns=['super_region', 'region', 'country'],
                kernel='depth',
                kernel_pars={'radius': 0.9}
            )
        >>> dimensions = [age, year, location]
        >>> smoother = Smoother(dimensions)

        """
        self.dimensions = dimensions  # type: ignore

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
                 fit: str = None, predict: str = None, loop: bool = False) \
            -> DataFrame:
        """Smooth data across dimensions with weighted averages.

        For each point in `predict`, calculate a smoothed value of each
        column in `columns` using a weighted average of points in
        `fit`, where weights are calculated based on proximity across
        `dimensions`. Return a data frame of points in `predict` with
        added columns '{column}_smooth' containing smoothed values for
        each for each column in `columns`.

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
        loop : bool, optional
            If True, smooth values for each point in `predict`
            separately in a loop. Requires less memory, but is slower.
            Otherwise, populate a matrix of weights for all points in
            `predict` and smooth values using matrix--vector
            multiplication. Requires more memory, but is faster.
            Default is False.

        Returns
        -------
        pandas.DataFrame
            Points in `predict` with smoothed `columns` values.

        Examples
        --------
        Using the smoother created in the previous example, smooth data
        across age, year, and location. Create smoothed version of
        multiple columns for all points using all points.

        >>> from pandas import DataFrame
        >>> data = DataFrame({
                'age_mid': [0.5, 1.5, 2.5, 3.5, 3.5],
                'year_id': [1980, 1990, 2000, 2010, 2020],
                'super_region': [1, 1, 1, 1, 2],
                'region': [3, 3, 3, 4, 8],
                'country': [5, 5, 6, 7, 9],
                'count': [1.0, 2.0, 3.0, 4.0, 5.0],
                'fraction': [0.1, 0.2, 0.3, 0.4, 0.5]
            })
        >>> smoother(data, ['count', 'fraction'])
           age_mid  ...  count  fraction  count_smooth  fraction_smooth
        0      0.5  ...    1.0       0.1      1.249567         0.124957
        1      1.5  ...    2.0       0.2      2.070433         0.207043
        2      2.5  ...    3.0       0.3      2.913803         0.291380
        3      3.5  ...    4.0       0.4      3.988203         0.398820
        4      3.5  ...    5.0       0.5      5.000000         0.500000

        Create smoothed version of one column for all points using a
        subset of points.

        >>> data['fit'] = [True, False, False, True, True]
        >>> smoother(data, 'count', fit='fit')
           age_mid  ...  count  fraction    fit  count_smooth
        0      0.5  ...    1.0       0.1   True      1.032967
        1      1.5  ...    2.0       0.2  False      1.032967
        2      2.5  ...    3.0       0.3  False      1.300000
        3      3.5  ...    4.0       0.4   True      3.967033
        4      3.5  ...    5.0       0.5   True      5.000000

        Create a smoothed version of one column for a subset of points
        using all points.

        >>> data['predict'] = [False, True, True, False, False]
        >>> smoother(data, 'fraction', predict='predict')
           age_mid  ...  count  fraction  predict  fraction_smooth
        0      1.5  ...    2.0       0.2     True         0.207043
        1      2.5  ...    3.0       0.3     True         0.291380

        """
        # Check input
        self.check_args(data, columns, fit, predict, loop)
        self.check_data(data, columns, fit, predict)

        # Extract data
        idx_fit = self.get_indices(data, fit)
        idx_pred = self.get_indices(data, predict)
        cols = self.get_columns(data, columns, idx_fit)
        point_list = self.get_points(data)
        dim_list = self.get_typed_dimensions()

        # Calculate smoothed values
        cols_smooth = smooth_data(dim_list, point_list, cols, idx_fit,
                                  idx_pred, loop)

        # Construct smoothed data frame
        data_smooth = data.iloc[idx_pred].reset_index(drop=True)
        for idx_col, col in enumerate(as_list(columns)):
            data_smooth[f"{col}_smooth"] = cols_smooth[:, idx_col]

        return data_smooth

    @staticmethod
    def check_args(data: DataFrame, columns: Union[str, List[str]],
                   fit: Optional[str], predict: Optional[str], loop: bool) \
            -> None:
        """Check `smoother` argument types and values.

        Parameters
        ----------
        data : DataFrame
            Input data structure.
        columns : str or list of str
            Column names of values to smooth.
        fit : str or None
            Column name indicating points to include in weighted
            averages.
        predict : str or None
            Column name indicating points to predict smoothed values.
        loop : bool
            If True, smooth values for each point in `predict`
            separately in a loop. Otherwise, populate a matrix of weights
            for all points in `predict` and smooth values together.

        Raises
        ------
        TypeError
            If `smoother` arguments contain invalid types.
        ValueError
            If `columns` is an empty list or contains duplicates.

        """
        # Check types
        if not isinstance(data, DataFrame):
            raise TypeError('`data` is not a DataFrame.')
        columns = as_list(columns)
        if not all(isinstance(col, str) for col in columns):
            raise TypeError('`columns` contains invalid types.')
        if fit is not None and not isinstance(fit, str):
            raise TypeError('`fit` is not a str.')
        if predict is not None and not isinstance(predict, str):
            raise TypeError('`predict` is not a str.')
        if not isinstance(loop, bool):
            raise TypeError('`loop` is not a bool.')

        # Check values
        if len(columns) == 0:
            raise ValueError('`columns` is an empty list.')
        if len(columns) > len(set(columns)):
            raise ValueError('`columns` contains duplicates.')

    def check_data(self, data: DataFrame, columns: Union[str, List[str]],
                   fit: Optional[str], predict: Optional[str]) -> None:
        """Check input data.

        Parameters
        ----------
        data : DataFrame
            Input data structure.
        columns : str or list of str
            Column names of values to smooth.
        fit : str or None
            Column name indicating points to include in weighted
            averages.
        predict : str or None
            Column name indicating points to predict smoothed values.

        Raises
        ------
        KeyError
            If `dimensions.columns`, `columns`, `fit`, or `predict` not
            in `data`.
        TypeError
            If `dimensions.columns`, `columns`, `fit`, or `predict` in
            `data` contain invalid types.
        ValueError
            If `data` contains NaNs or Infs.

        """
        # Get column names
        dim_cols = [col for dim in self._dimensions for col in dim.columns]
        cols = as_list(columns)

        # Check keys
        if not all(col in data for col in dim_cols):
            raise KeyError('Not all `dimensions.columns` in `data`.')
        if not all(col in data for col in cols):
            raise KeyError('Not all `columns` in `data`.')
        if fit is not None and fit not in data:
            raise KeyError('`fit` not in `data`.')
        if predict is not None and predict not in data:
            raise KeyError('`predict` not in `data`.')

        # Check types
        if not all(is_numeric_dtype(data[col]) for col in dim_cols):
            raise TypeError('Not all `dimensions.columns` data int or float.')
        if not all(is_numeric_dtype(data[col]) for col in cols):
            raise TypeError('Not all `columns` data int or float.')
        if fit is not None:
            if not is_bool_dtype(data[fit]):
                raise TypeError('`fit` data is not bool.')
        if predict is not None:
            if not is_bool_dtype(data[predict]):
                raise TypeError('`predict` data is not bool.')

        # Check values
        if data.isna().any(None):
            raise ValueError('`data` contains NaNs.')
        if np.isinf(data[dim_cols + cols]).any(None):
            raise ValueError('`data` contains Infs.')

    @staticmethod
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
        1D numpy.ndarray of int
            Indices of `fit` or `predict` points.

        """
        if indicator is None:
            return np.arange(len(data))
        return np.where(data[indicator])[0]

    @staticmethod
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
        2D numpy.ndarray of float
            Values to smooth.

        """
        return np.array([data[col].values[idx_fit]
                         for col in as_list(columns)], dtype=float).T

    def get_points(self, data: DataFrame) -> List[np.ndarray]:
        """Get point locations by dimension.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.

        Returns
        -------
        numba.typed.List of 2D numpy.ndarray of float
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
        numba.typed.List of TypedDimension
            Smoothing dimensions cast as jitclass objects.

        """
        dim_list = TypedList()
        for dim in self._dimensions:
            dim_list.append(get_typed_dimension(dim))
        return dim_list


@njit(parallel=True)
def smooth_data(dim_list: List[TypedDimension], point_list: List[np.ndarray],
                cols: np.ndarray, idx_fit: np.ndarray, idx_pred: np.ndarray,
                loop: bool = False) -> np.ndarray:
    """Smooth data across dimensions with weighted averages.

    Parameters
    ----------
    dim_list : list of TypedDimension
        Smoothing dimensions.
    point_list : list of 2D numpy.ndarray of float
        Point locations by dimension.
    cols : 2D numpy.ndarray of float
        Values to smooth.
    idx_fit : 1D numpy.ndarray of int
        Indices of points to include in weighted averages.
    idx_pred: 1D numpy.ndarray of int
        Indices of points to predict smoothed values.
    loop : bool, optional
        If True, smooth values for each point in `predict`
        separately in a loop. Requires less memory, but is slower.
        Otherwise, populate a matrix of weights for all points in
        `predict` and smooth values together. Requires more
        memory, but is faster. Default is False.

    Returns
    -------
    2D numpy.ndarray of float
        Smoothed values.

    """
    # Initialize smoothed values
    n_cols = cols.shape[1]
    n_fit = len(idx_fit)
    n_pred = len(idx_pred)

    if loop:  # Calculate smoothed values one point at a time
        cols_smooth = np.empty((n_pred, n_cols))
        for idx_x in prange(n_pred):
            weights = get_weights(dim_list, point_list, idx_fit,
                                  idx_pred[idx_x])
            cols_smooth[idx_x, :] = weights.dot(cols)
    else:  # Calculate smoothed values together
        weights = np.empty((n_pred, n_fit))
        for idx_x in prange(n_pred):
            weights[idx_x, :] = get_weights(dim_list, point_list, idx_fit,
                                            idx_pred[idx_x])
        cols_smooth = weights.dot(cols)

    return cols_smooth


@njit
def get_weights(dim_list: List[TypedDimension], point_list: List[np.ndarray],
                idx_fit: np.ndarray, idx_x: int) -> np.ndarray:
    """Get smoothing weights for current point.

    Parameters
    ----------
    dim_list : list of TypedDimension
        Smoothing dimensions.
    point_list : list of 2D numpy.ndarray of float
        Point locations by dimension.
    idx_fit : 1D numpy.ndarray of int
        Indices of nearby points to include in weighted averages.
    idx_x : int
        Index of current point to predict smoothed values.

    Returns
    -------
    1D numpy.ndarray of nonnegative float
        Smoothing weights for current point.

    """
    # Initialize weight vector
    weights = np.ones(len(idx_fit))

    # Calculate weights one dimension at at a time
    for idx_dim, dim in enumerate(dim_list):
        dim_points = point_list[idx_dim]
        dim_dists = get_dim_distances(dim_points[idx_x], dim_points[idx_fit],
                                      dim.distance, dim.distance_dict)
        if dim.kernel == 'identity':
            dim_weights = dim_dists
        else:
            dim_weights = get_dim_weights(dim_dists, dim.kernel,
                                          dim.kernel_pars)

        # Optional normalize by subgroup
        if dim.kernel_pars['normalize'] == 1.0:
            for weight in list(set(dim_weights)):
                idx_weight = np.where(dim_weights == weight)[0]
                if weights[idx_weight].sum() != 0.0:
                    weights[idx_weight] *= weight/weights[idx_weight].sum()
        else:
            weights *= dim_weights

    return weights/weights.sum()


@njit
def get_dim_distances(x: np.ndarray, y: np.ndarray, distance: str,
                      distance_dict: Dict[Tuple[float, float], float]) \
        -> np.ndarray:
    """Get distances between `x` and `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    y : 2D numpy.ndarray of float
        Nearby points.
    distance : str
        Distance function name.
    distance_dict : dict of {(float, float): float}
        Dictionary of distances between points.

    Returns
    -------
    1D numpy.ndarray of nonnegative float
        Distances between `x` and `y`.

    """
    if distance == 'dictionary':
        return dictionary(x, y, distance_dict)
    if distance == 'euclidean':
        return euclidean(x, y)
    return hierarchical(x, y)


@njit
def get_dim_weights(distance: np.ndarray, kernel: str,
                    kernel_pars: Dict[str, float]) -> np.ndarray:
    """Get smoothing weights.

    Parameters
    ----------
    distance : 1D numpy.ndarray of nonnegative float
        Distances between points.
    kernel : str
        Kernel function name.
    kernel_pars : dict of {str: float}
        Kernel function parameters.

    Returns
    -------
    1D numpy.ndarray of nonnegative float
        Smoothing weights.

    """
    if kernel == 'exponential':
        return exponential(distance, kernel_pars['radius'])
    if kernel == 'tricubic':
        return tricubic(distance, kernel_pars['radius'],
                        kernel_pars['exponent'])
    return depth(distance, kernel_pars['radius'])
