# pylint: disable=C0103, E0611, E1133, R0912, R0913, R0914
"""Smooth data across multiple dimensions using weighted averages."""
from itertools import product
from typing import List, Optional, Union
import warnings

from numba import njit  # type: ignore
from numba.typed import List as TypedList  # type: ignore
import numpy as np
from pandas import DataFrame  # type: ignore
from pandas.api.types import is_bool_dtype, is_numeric_dtype  # type: ignore

from weave.dimension import Dimension, TypedDimension
from weave.utils import as_list, flatten, is_number

number = Union[int, float]


class Smoother:
    """Smoother function.

    Attributes
    ----------
    dimensions : list of Dimension
        Smoothing dimensions.
    inverse_weights: bool
        Whether or not to use inverse-distance weights.

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
        self.inverse_weights = all(dim.kernel == "inverse" for dim in self._dimensions)

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
        # Once set, `dimensions` cannot be changed
        if hasattr(self, "dimensions"):
            raise AttributeError("`dimensions` cannot be changed")

        # Check types
        if not all(isinstance(dim, Dimension) for dim in dimensions):
            raise TypeError("`dimensions` contains invalid types")

        # Check values
        if len(dimensions) == 0:
            raise ValueError("`dimensions` is an empty list")
        name_list = [dim.name for dim in dimensions]
        if len(name_list) > len(set(name_list)):
            raise ValueError("Duplicate names found in `dimensions`")
        coord_list = flatten([dim.coordinates for dim in dimensions])
        if len(coord_list) > len(set(coord_list)):
            raise ValueError("Duplicate coordinates found in `dimensions`")

        self._dimensions = dimensions

    @property
    def inverse_weights(self) -> bool:
        """Get inverse-distance weights flag.

        Returns
        -------
        bool
            Whether or not to use inverse-distance weights.

        """
        return self._inverse_weights

    @inverse_weights.setter
    def inverse_weights(self, inverse_weights: bool) -> None:
        """Set inverse-distance weights flag.

        Parameters
        ----------
        inverse_weights : bool
            Whether or not to use inverse-distance weights.

        Raises
        ------
        AttributeError
            If `inverse_weights` has already been set.
        ValueError
            If dimensions have both inverse and non-inverse kernels.

        """
        # Once set, `inverse_weights` cannot be changed
        if hasattr(self, "inverse_weights"):
            raise AttributeError("`inverse_weights` cannot be changed")

        # Check values
        if inverse_weights:
            self._inverse_weights = True
        else:
            if any(dim.kernel == "inverse" for dim in self._dimensions):
                raise ValueError("Cannot mix inverse and non-inverse kernels")
            self._inverse_weights = False

    def __call__(
        self,
        data: DataFrame,
        observed: str,
        stdev: Optional[str] = None,
        smoothed: Optional[str] = None,
        fit: Optional[str] = None,
        predict: Optional[str] = None,
        down_weight: Optional[number] = 1,
    ) -> DataFrame:
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
            Column name of standard deviations. Required for
            inverse-distance kernels.
        smoothed : str, optional
            Column name of smoothed values. If None, append '_smooth'
            to  `observed`.
        fit : str, optional
            Column name indicating points to include in weighted
            averages. If None, all points in `data` are used.
        predict : str, optional
            Column name indicating where to predict smoothed values.
            If None, predictions are made for all points in `data`.
        down_weight : int or float in [0, 1], optional
            Down-weight neighbors for in-sample points. Default is 1,
            which corresponds to no down-weighting. If 0, in-sample
            points are not smoothed.

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
        self.check_input(data, observed, stdev, smoothed, fit, predict, down_weight)
        smoothed = f"{observed}_smooth" if smoothed is None else smoothed
        down_weight = np.float32(down_weight)

        # Extract data
        idx_fit = self.get_indices(data, fit)
        idx_pred = self.get_indices(data, predict)
        col_obs = self.get_values(data, observed, idx_fit)
        col_sd = self.get_values(data, stdev, idx_fit)
        points = self.get_points(data)
        dim_list = self.get_typed_dimensions(data)

        # Calculate smoothed values
        if self.inverse_weights:
            col_smooth = smooth_inverse(
                dim_list, points, col_obs, col_sd, idx_fit, idx_pred, down_weight
            )
        else:
            col_smooth = smooth(
                dim_list, points, col_obs, col_sd, idx_fit, idx_pred, down_weight
            )

        # Construct smoothed data frame
        data_smooth = data.iloc[idx_pred].reset_index(drop=True)
        data_smooth[smoothed] = col_smooth

        return data_smooth

    def check_input(
        self,
        data: DataFrame,
        observed: str,
        stdev: Optional[str],
        smoothed: Optional[str],
        fit: Optional[str],
        predict: Optional[str],
        down_weight: float,
    ) -> None:
        """Check `smoother` arguments and data.

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
        down_weight : float in [0, 1]
            Down-weight neighbors for in-sample points.

        """
        # Check argument types and values
        self.check_arg_types(data, observed, stdev, smoothed, fit, predict, down_weight)
        self.check_arg_values(observed, stdev, smoothed, down_weight)

        # Check data and dictionary keys
        names = [dim.name for dim in self._dimensions]
        coords = flatten([dim.coordinates for dim in self._dimensions])
        self.check_data_columns(
            names, coords, data, observed, stdev, smoothed, fit, predict
        )
        for dim in self._dimensions:
            if dim.distance == "dictionary":
                self.check_dist_dict(dim, data)

        # Check data types and values
        self.check_data_types(names, coords, data, observed, stdev, fit, predict)
        self.check_data_values(names, coords, data, observed, stdev)
        self.check_dim_values(data)

    @staticmethod
    def check_arg_types(
        data: DataFrame,
        observed: str,
        stdev: Optional[str],
        smoothed: Optional[str],
        fit: Optional[str],
        predict: Optional[str],
        down_weight: float,
    ) -> None:
        """Check `smoother` argument types.

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
        down_weight : float in [0, 1], optional
            Down-weight neighbors for in-sample points.

        Raises
        ------
        TypeError
            If `smoother` arguments contain invalid types.

        """
        if not isinstance(data, DataFrame):
            raise TypeError("`data` is not a DataFrame")
        if not isinstance(observed, str):
            raise TypeError("`observed` is not a str")
        if stdev is not None and not isinstance(stdev, str):
            raise TypeError("`stdev` is not a str")
        if smoothed is not None and not isinstance(smoothed, str):
            raise TypeError("`smoothed` is not a str")
        if fit is not None and not isinstance(fit, str):
            raise TypeError("`fit` is not a str")
        if predict is not None and not isinstance(predict, str):
            raise TypeError("`predict` is not a str")
        if not is_number(down_weight):
            raise TypeError("`down_weight` is not an int or float")

    def check_arg_values(
        self,
        observed: str,
        stdev: Optional[str],
        smoothed: Optional[str],
        down_weight: float,
    ) -> None:
        """Check `smoother` argument values.

        Parameters
        ----------
        observed : str
            Column name of values to smooth.
        stdev : str, optional
            Column name of standard deviations.
        smoothed : str, optional
            Column name of smoothed values.
        down_weight : float in [0, 1], optional
            Down-weight neighbors for in-sample points.

        Raises
        ------
        ValueError
            If `observed`, `stdev`, or `smoothed` overlap.
            If `stdev` not passed when `self.inverse_weights` is True.
            If `down_weight` is not in [0, 1].

        """
        col_set = set([observed, stdev, smoothed])
        if not (stdev is None and smoothed is None) and len(col_set) < 3:
            raise ValueError("Duplicates in `observed`, `stdev`, `smoothed`")
        if self.inverse_weights and stdev is None:
            raise ValueError("`stdev` required for inverse-distance weighting")
        if not 0 <= down_weight <= 1:
            raise ValueError("`down_weight` must be in [0, 1]")

    @staticmethod
    def check_data_columns(
        names: List[str],
        coords: List[str],
        data: DataFrame,
        observed: str,
        stdev: Optional[str],
        smoothed: Optional[str],
        fit: Optional[str],
        predict: Optional[str],
    ) -> None:
        """Check data frame column names.

        Parameters
        ----------
        names : list of str
            Smoothing dimension names.
        coords : list of str
            Smoothing dimension coordinates.
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

        Warns
        -----
        If columns in `smoothed` in `data`.

        """
        if not all(name in data for name in names):
            raise KeyError("Not all `dimension.name` in data")
        if not all(coord in data for coord in coords):
            raise KeyError("Not all `dimension.coordinates` in data")
        if observed not in data:
            raise KeyError(f"`observed` column {observed} not in data")
        if stdev is not None and stdev not in data:
            raise KeyError(f"`stdev` column {stdev} not in data")
        if smoothed in data:
            warnings.warn(f"`smoothed` column {smoothed} will be overwritten")
        if fit is not None and fit not in data:
            raise KeyError(f"`fit` column {fit} not in data")
        if predict is not None and predict not in data:
            raise KeyError(f"`predict` column {predict} not in data")

    @staticmethod
    def check_dist_dict(dimension: Dimension, data: DataFrame) -> None:
        """Check distance dictionary keys.

        Parameters
        ----------
        dimension : Dimension
            Smoothing dimension.
        data : pandas.DataFrame
            Input data structure.

        Raises
        ------
        KeyError
            If `dimension.distance` is 'dictionary', but not all keys
            `dimension.name` in `dimension.distance_dict`.

        """
        dim_names = data[dimension.name].unique()
        for key in product(dim_names, repeat=2):
            if key not in dimension.distance_dict:
                raise KeyError("Not all `dimension.name` in `dimension.distance_dict`")

    @staticmethod
    def check_data_types(
        names: List[str],
        coords: List[str],
        data: DataFrame,
        observed: str,
        stdev: Optional[str],
        fit: Optional[str],
        predict: Optional[str],
    ) -> None:
        """Check input data types.

        Parameters
        ----------
        names : list of str
            Smoothing dimension names.
        coords : list of str
            Smoothing dimension coordinates.
        data : pandas.DataFrame
            Input data structure.
        observed : str
            Column name of values to smooth.
        stdev : str, optional
            Column name of standard deviations.
        fit : str, optional
            Column name indicating points to include in weighted
            averages.
        predict : str, optional
            Column name indicating where to predict smoothed values.

        Raises
        ------
        TypeError
            If columns `dimension.name`, `dimensions.coordinates`,
            `observed`, `stdev`, `fit`, or `predict` in `data` contain
            invalid types.

        """
        if not all(is_numeric_dtype(data[name]) for name in names):
            raise TypeError("Not all `dimension.name` data int or float")
        if not all(is_numeric_dtype(data[coord]) for coord in coords):
            raise TypeError("Not all `dimension.coordinates` data int or float")
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

    @staticmethod
    def check_data_values(
        names: List[str],
        coords: List[str],
        data: DataFrame,
        observed: str,
        stdev: Optional[str],
    ) -> None:
        """Check input data.

        Parameters
        ----------
        names : list of str
            Smoothing dimension names.
        coords : list of str
            Smoothing dimension coordinates.
        data : pandas.DataFrame
            Input data structure.
        observed : str
            Column name of values to smooth.
        stdev : str, optional
            Column name of standard deviations.

        Raises
        ------
        ValueError
            If `data` contains NaNs or Infs.
            If `stdev` contains zeros or negative values.

        """
        if data.isna().any(axis=None):
            raise ValueError("`data` contains NaNs")
        cols_in = [observed] if stdev is None else [observed, stdev]
        if np.isinf(data[names + coords + cols_in]).any(axis=None):
            raise ValueError("`data` contains Infs")
        if stdev is not None:
            if np.any(data[stdev] <= 0):
                raise ValueError("`stdev` values must be positive")

    def check_dim_values(
        self,
        data: DataFrame,
    ) -> None:
        """Check dimension names and coordinates one-to-one in data.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.

        Raises
        ------
        ValueError
            If columns `dimension.name` and `dimension.coordinates` not
            one-to-one in `data`.

        """
        for dim in self._dimensions:
            if [dim.name] != dim.coordinates:
                points = data[[dim.name] + dim.coordinates].drop_duplicates()
                points = points.loc[:, ~points.columns.duplicated()]
                if any(points.groupby(dim.name).size() != 1):
                    raise ValueError("`name` maps to multiple `coordinates`")
                if any(points.groupby(dim.coordinates).size() != 1):
                    raise ValueError("`coordinates` maps to multiple `name`")

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
    def get_values(
        data: DataFrame, values: Optional[str], idx_fit: np.ndarray
    ) -> np.ndarray:
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
            return np.nan * np.ones(len(idx_fit)).astype(np.float32)
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

    def get_typed_dimensions(self, data: DataFrame) -> TypedList[TypedDimension]:
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
        return TypedList(
            [dimension.get_typed_dimension(data) for dimension in self._dimensions]
        )


@njit
def smooth(
    dim_list: List[TypedDimension],
    points: np.ndarray,
    col_obs: np.ndarray,
    col_sd: np.ndarray,
    idx_fit: np.ndarray,
    idx_pred: np.ndarray,
    down_weight: float,
) -> np.ndarray:
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
    down_weight: float in [0, 1]
        Down-weight neighbors for in-sample points.

    Returns
    -------
    1D numpy.ndarray of float32
        Smoothed values.

    """
    # Initialize weight matrix
    n_fit = len(idx_fit)
    n_pred = len(idx_pred)
    weights = np.ones((n_pred, n_fit), dtype=np.float32)

    # Calculate weights one prediction at a time
    for ii in range(n_pred):
        for idx_dim, dim in enumerate(dim_list):
            pred = points[idx_pred[ii], idx_dim]
            dim_weights = np.zeros(n_fit, dtype=np.float32)
            for jj in range(n_fit):
                fit = points[idx_fit[jj], idx_dim]
                dim_weights[jj] = dim.weight_dict[(pred, fit)]

            # Normalize by depth subgroup
            if dim.kernel == "depth":
                for weight in list(set(dim_weights)):
                    cond = dim_weights == weight
                    scale = np.where(cond, weights[ii], 0).sum()
                    if scale != 0:
                        weights[ii] = np.where(cond, weights[ii] / scale, weights[ii])

            # Update weight matrix
            weights[ii] *= dim_weights

        # Down-weight neighbors for in-sample points
        if idx_pred[ii] in idx_fit and down_weight < 1:
            neighbors = idx_pred[ii] != idx_fit
            weights[ii] = np.where(neighbors, weights[ii] * down_weight, weights[ii])

    # Scale by standard deviation
    if not np.isnan(col_sd).any():
        weights = weights / (col_sd**2)

    # Compute smoothed values
    return weights.dot(col_obs) / weights.sum(axis=1)


@njit
def smooth_inverse(
    dim_list: List[TypedDimension],
    points: np.ndarray,
    col_obs: np.ndarray,
    col_sd: np.ndarray,
    idx_fit: np.ndarray,
    idx_pred: np.ndarray,
    down_weight: float,
) -> np.ndarray:
    """Smooth data across dimensions with inverse-distance weighted averages.

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
    down_weight: float in [0, 1]
        Down-weight neighbors for in-sample points.

    Returns
    -------
    1D numpy.ndarray of float32
        Smoothed values.

    """
    # Initialize distance matrix
    n_fit = len(idx_fit)
    n_pred = len(idx_pred)
    weights = np.zeros((n_pred, n_fit), dtype=np.float32)

    # Calculate distance weights one prediction at a time
    for ii in range(n_pred):
        distance = col_sd**2
        for idx_dim, dim in enumerate(dim_list):
            pred = points[idx_pred[ii], idx_dim]
            dim_distance = np.zeros(n_fit, dtype=np.float32)
            for jj in range(n_fit):
                fit = points[idx_fit[jj], idx_dim]
                dim_distance[jj] = dim.weight_dict[(pred, fit)]
            distance += dim_distance
        weights[ii] = 1 / distance
        if idx_pred[ii] in idx_fit and down_weight < 1:
            neighbors = idx_pred[ii] != idx_fit
            weights[ii] = np.where(neighbors, weights[ii] * down_weight, weights[ii])

    # Compute smoothed values with inverse-distance weights
    return weights.dot(col_obs) / weights.sum(axis=1)
