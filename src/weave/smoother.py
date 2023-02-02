# pylint: disable=C0103, E0611, E1133, R0912, R0913, R0914
"""Smooth data across multiple dimensions using weighted averages."""
from itertools import product
from typing import List, Optional, Union
import warnings

from numba import njit, prange  # type: ignore
from numba.typed import List as TypedList  # type: ignore
import numpy as np
from pandas import DataFrame  # type: ignore
from pandas.api.types import is_bool_dtype, is_numeric_dtype  # type: ignore

from weave.dimension import Dimension, TypedDimension
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
                name='age_id',
                coordinates='age_mean',
                kernel='exponential',
                radius=1
            )
        >>> year = Dimension(
                name='year_id',
                kernel='tricubic',
                exponent=0.5
            )
        >>> location = Dimension(
                name='location_id',
                coordinates=['super_region', 'region', 'country'],
                kernel='depth',
                radius=0.9
            )
        >>> dimensions = [age, year, location]
        >>> smoother = Smoother(dimensions)

        """
        self.dimensions = as_list(dimensions)

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
    def dimensions(self, dimensions: List[Dimension]) -> None:
        """Set smoothing dimensions.

        Parameters
        ----------
        dimensions : Dimension or list of Dimension
            Smoothing dimensions.

        Raises
        ------
        AttributeError
            If `dimensions` has already been set.
        TypeError
            If `dimensions` is not a list of Dimension.
        ValueError
            If `dimensions` is an empty list, contains an empty list, or
            contains duplicate names or columns.

        """
        # Once set, `name` cannot be changed
        if hasattr(self, 'dimensions'):
            raise AttributeError('`dimensions` cannot be changed')

        # Check types
        if not all(isinstance(dim, Dimension) for dim in dimensions):
            raise TypeError('`dimensions` contains invalid types')

        # Check values
        if len(dimensions) == 0:
            raise ValueError('`dimensions` is an empty list')
        name_list = [dim.name for dim in dimensions]
        if len(name_list) > len(set(name_list)):
            raise ValueError('Duplicate names found in `dimensions`')
        coord_list = flatten([dim.coordinates for dim in dimensions])
        if len(coord_list) > len(set(coord_list)):
            raise ValueError('Duplicate coordinates found in `dimensions`')

        self._dimensions = dimensions

    def __call__(self, data: DataFrame, observed: str,
                 stdev: Optional[str] = None, smoothed: Optional[str] = None,
                 fit: Optional[str] = None, predict: Optional[str] = None) \
            -> DataFrame:
        """Smooth data across dimensions with weighted averages.

        For each point in `predict`, smooth values in `observed` using
        a weighted average of points in `fit`, where weights are
        calculated based on proximity across `dimensions`. Return a
        data frame of points in `predict` with column `smoothed`
        containing smoothed values.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.
        observed : str
            Column name of values to smooth.
        stdev: str, optional
            Column name of standard deviations.
        smoothed : str, optional
            Column name of smoothed values. If None, append '_smooth'
            to  `observed`.
        fit : str, optional
            Column name indicating points to include in weighted
            averages. If None, all points in `data` are used.
        predict : str, optional
            Column name indicating where to predict smoothed values.
            If None, predictions are made for all points in `data`.

        Returns
        -------
        pandas.DataFrame
            Points in `predict` with smoothed values `smoothed`.

        Examples
        --------
        Using the smoother created in the previous example, smooth data
        across age, year, and location. Create smoothed version of
        column `count` for all points using all points.

        >>> from pandas import DataFrame
        >>> data = DataFrame({
                'age_id': [1, 2, 3, 4, 4],
                'age_mean': [0.5, 1.5, 2.5, 3.5, 3.5],
                'year_id': [1980, 1990, 2000, 2010, 2020],
                'location_id': [5, 5, 6, 7, 9],
                'super_region': [1, 1, 1, 1, 2],
                'region': [3, 3, 3, 4, 8],
                'country': [5, 5, 6, 7, 9],
                'count': [1.0, 2.0, 3.0, 4.0, 5.0]
            })
        >>> smoother(data, 'count')
           age_id  ...  count  count_smooth
        0       1  ...    1.0      1.250974
        1       2  ...    2.0      2.084069
        2       3  ...    3.0      2.919984
        3       4  ...    4.0      3.988642
        4       4  ...    5.0      5.000000

        Create smoothed version of one column for all points using a
        subset of points.

        >>> data['train'] = [True, False, False, True, True]
        >>> smoother(data, 'count', fit='train')
           age_id  ...  count  train  count_smooth
        0       1  ...    1.0   True      1.032967
        1       2  ...    2.0  False      1.032967
        2       3  ...    3.0  False      1.300000
        3       4  ...    4.0   True      3.967033
        4       4  ...    5.0   True      5.000000

        Create a smoothed version of one column for a subset of points
        using all points.

        >>> data['test'] = [False, True, True, False, False]
        >>> smoother(data, 'count', predict='test')
           age_id  ...  count  test  count_smooth
        0       2  ...    2.0  True      2.084069
        1       3  ...    3.0  True      2.919984

        """
        # Check input
        self.check_args(data, observed, stdev, smoothed, fit, predict)
        self.check_data(data, observed, stdev, smoothed, fit, predict)
        smoothed = f"{observed}_smooth" if smoothed is None else smoothed

        # Extract data
        idx_fit = self.get_indices(data, fit)
        idx_pred = self.get_indices(data, predict)
        col_obs = self.get_values(data, observed, idx_fit)
        col_sd = self.get_values(data, stdev, idx_fit)
        points = self.get_points(data)
        dim_list = self.get_typed_dimensions(data)

        # Calculate smoothed values
        col_smooth = smooth(dim_list, points, col_obs, col_sd, idx_fit,
                            idx_pred)

        # Construct smoothed data frame
        data_smooth = data.iloc[idx_pred].reset_index(drop=True)
        data_smooth[smoothed] = col_smooth

        return data_smooth

    @staticmethod
    def check_args(data: DataFrame, observed: str, stdev: Optional[str],
                   smoothed: Optional[str], fit: Optional[str],
                   predict: Optional[str]) -> None:
        """Check `smoother` argument types and values.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.
        observed : str
            Column name of values to smooth.
        stdev : str, optional
            Column name of standard deviations.
        smoothed : str, optional
            Column name of smoothed values.
        fit : str, optional
            Column name indicating points to include in weighted
            averages.
        predict : str, optional
            Column name indicating where to predict smoothed values.

        Raises
        ------
        TypeError
            If `smoother` arguments contain invalid types.
        ValueError
            If `observed`, `stdev`, or `smoothed` overlap.

        """
        # Check types
        if not isinstance(data, DataFrame):
            raise TypeError('`data` is not a DataFrame')
        if not isinstance(observed, str):
            raise TypeError('`observed` is not a str')
        if stdev is not None and not isinstance(stdev, str):
            raise TypeError('`stdev` is not a str')
        if smoothed is not None and not isinstance(smoothed, str):
            raise TypeError('`smoothed` is not a str')
        if fit is not None and not isinstance(fit, str):
            raise TypeError('`fit` is not a str')
        if predict is not None and not isinstance(predict, str):
            raise TypeError('`predict` is not a str')

        # Check values
        col_set = set([observed, stdev, smoothed])
        if not (stdev is None and smoothed is None) and len(col_set) < 3:
            raise ValueError('Duplicates in `observed`, `stdev`, `smoothed`')

    def check_data(self, data: DataFrame, observed: str, stdev: Optional[str],
                   smoothed: Optional[str], fit: Optional[str],
                   predict: Optional[str]) -> None:
        """Check input data.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.
        observed : str
            Column name of values to smooth.
        stdev : str, optional
            Column name of standard deviations.
        smoothed : str, optional
            Column name of smoothed values.
        fit : str, optional
            Column name indicating points to include in weighted
            averages.
        predict : str, optional
            Column name indicating where to predict smoothed values.

        Raises
        ------
        KeyError
            If columns `dimension.name`, `dimensions.coordinates`,
            `observed`, `stdev`, `fit`, or `predict` not in `data`.
            If `dimension.distance` is 'dictionary', but not all keys
            `dimension.name` in `dimension.distance_dict`.
        TypeError
            If columns `dimension.name`, `dimensions.coordinates`,
            `observed`, `stdev`, `fit`, or `predict` in `data` contain
            invalid types.
        ValueError
            If columns `dimension.name` and `dimension.coordinates` not
            one-to-one in `data`.
            If `data` contains NaNs or Infs.

        Warns
        -----
        If columns in `smoothed` in `data`.

        """
        # Get column names
        names = [dim.name for dim in self._dimensions]
        coords = flatten([dim.coordinates for dim in self._dimensions])

        # Check data frame columns
        if not all(name in data for name in names):
            raise KeyError('Not all `dimension.name` in data')
        if not all(coord in data for coord in coords):
            raise KeyError('Not all `dimension.coordinates` in data')
        if observed not in data:
            raise KeyError(f"`observed` column {observed} not in data")
        if stdev is not None and stdev not in data:
            raise KeyError(f"`stdev` column {stdev} not in data")
        if smoothed in data:
            msg = f"`smoothed` column {smoothed} will be overwritten"
            warnings.warn(msg)
        if fit is not None and fit not in data:
            raise KeyError(f"`fit` column {fit} not in data")
        if predict is not None and predict not in data:
            raise KeyError(f"`predict` column {predict} not in data")

        # Check dictionary keys
        for dim in self._dimensions:
            if dim.distance == 'dictionary':
                names = data[dim.name].unique()
                for key in product(names, repeat=2):
                    if key[0] <= key[1] and key not in dim.distance_dict:
                        msg = 'Not all `dimension.name` in '
                        msg += '`dimension.distance_dict`'
                        raise KeyError(msg)

        # Check types
        if not all(is_numeric_dtype(data[name]) for name in names):
            raise TypeError('Not all `dimension.name` data int or float')
        if not all(is_numeric_dtype(data[coord]) for coord in coords):
            msg = 'Not all `dimension.coordinates` data int or float'
            raise TypeError(msg)
        if not is_numeric_dtype(data[observed]):
            raise TypeError(f"`observed` data {observed} not int or float")
        if stdev is not None:
            if not is_numeric_dtype(data[stdev]):
                raise TypeError(f"`stdev` data {stdev} is not int or float")
        if fit is not None:
            if not is_bool_dtype(data[fit]):
                raise TypeError(f"`fit` data {fit} is not bool")
        if predict is not None:
            if not is_bool_dtype(data[predict]):
                raise TypeError(f"`predict` data {predict} is not bool")

        # Check `name` and `coordinates` one-to-one
        for dim in self._dimensions:
            if [dim.name] != dim.coordinates:
                points = data[[dim.name] + dim.coordinates].drop_duplicates()
                points = points.loc[:, ~points.columns.duplicated()]
                if any(points.groupby(dim.name).size() != 1):
                    raise ValueError('`name` maps to multiple `coordinates`')
                if any(points.groupby(dim.coordinates).size() != 1):
                    raise ValueError('`coordinates` maps to multiple `name`')

        # Check values
        if data.isna().any(None):
            raise ValueError('`data` contains NaNs')
        cols_in = [observed] if stdev is None else [observed, stdev]
        if np.isinf(data[names + coords + cols_in]).any(None):
            raise ValueError('`data` contains Infs')

    @staticmethod
    def get_indices(data: DataFrame, indicator: str = None) -> np.ndarray:
        """Get indices of `fit` or `predict` data.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.
        indicator : str, optional
            Column name indicating either `fit` or `predict` data.

        Returns
        -------
        1D numpy.ndarray of int32
            Indices of `fit` or `predict` points.

        """
        if indicator is None:
            return np.arange(len(data)).astype(np.int32)
        return np.where(data[indicator])[0].astype(np.int32)

    @staticmethod
    def get_values(data: DataFrame, values: Optional[str],
                   idx_fit: np.ndarray) -> np.ndarray:
        """Get input values.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.
        values : str, optional
            Column names of values.
        idx_fit : numpy.ndarray of int
            Indices of `fit` points.

        Returns
        -------
        numpy.ndarray of float32
            Input values.

        """
        if values is None:
            return np.nan*np.ones(len(idx_fit)).astype(np.float32)
        return np.array(data[values].values[idx_fit], dtype=np.float32)

    def get_points(self, data: DataFrame) -> np.ndarray:
        """Get point IDs.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.

        Returns
        -------
        2D numpy.ndarray of float32
            Point IDs.

        """
        points = [dim.name for dim in self._dimensions]
        return np.ascontiguousarray(data[points].values, dtype=np.float32)

    def get_typed_dimensions(self, data: DataFrame) \
            -> TypedList[TypedDimension]:
        """Get smoothing dimensions cast as jitclass objects.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.

        Returns
        -------
        numba.typed.List of TypedDimension
            Smoothing dimensions cast as jitclass objects.

        """
        return TypedList([dimension.get_typed_dimension(data)
                          for dimension in self._dimensions])


@njit(parallel=True)
def smooth(dim_list: List[TypedDimension], points: np.ndarray,
           col_obs: np.ndarray, col_sd: np.ndarray, idx_fit: np.ndarray,
           idx_pred: np.ndarray) -> np.ndarray:
    """Smooth data across dimensions with weighted averages.

    Parameters
    ----------
    dim_list : list of TypedDimension
        Smoothing dimensions.
    points : 2D numpy.ndarray of float
        Point IDs.
    col_obs : 1D numpy.ndarray of float
        Values to smooth.
    col_sd: 1D numpy.ndarray of float
        Standard deviations.
    idx_fit : 1D numpy.ndarray of int
        Indices of points to include in weighted averages.
    idx_pred: 1D numpy.ndarray of int
        Indices of points to predict smoothed values.

    Returns
    -------
    1D numpy.ndarray of float32
        Smoothed values.

    """
    # Initialize weight matrix
    n_fit = len(idx_fit)
    n_pred = len(idx_pred)
    weights = np.ones((n_pred, n_fit), dtype=np.float32)

    # Calculate weights one dimension at a time
    for idx_dim, dim in enumerate(dim_list):
        dim_weights = np.zeros((n_pred, n_fit), dtype=np.float32)
        for ii in prange(n_pred):
            pred = points[idx_pred[ii], idx_dim]
            for jj in range(n_fit):
                fit = points[idx_fit[jj], idx_dim]
                dim_weights[ii, jj] = dim.weight_dict[(pred, fit)]

            # Normalize by depth subgroup
            if dim.kernel == 'depth':
                for weight in list(set(dim_weights[ii, :])):
                    cond = dim_weights[ii, :] == weight
                    scale = np.where(cond, weights[ii, :], 0).sum()
                    if scale != 0:
                        weights[ii, :] = np.where(cond, weights[ii, :]/scale,
                                                  weights[ii, :])

        # Update weight matrix
        weights *= dim_weights

    # Scale by standard deviation
    if not np.isnan(col_sd).any():
        weights = weights/(col_sd**2)

    # Compute smoothed values
    return weights.dot(col_obs)/weights.sum(axis=1)
