"""Smooth data across multiple dimensions using weighted averages.

TODO
* Modify __init__ to import Smoother
* Write checks and tests
* Numba-fy
* Specify order of weight aggregation (e.g., location last)

Checks
* Check for duplicates in columns
* Check for NaNs and Infs in data frame
* Check columns, fit, predict in data frame
* Anything else?

"""
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from weave.dimension import Dimension
from weave.utils import as_list


class Smoother:
    """Smoother function.

    Attributes
    ----------
    dimensions : Dimension or list of Dimension
        Smoothing dimension(s).

    """

    def __init__(self, dimensions: Union[Dimension, List[Dimension]]) -> None:
        """Create smoother function.

        Parameters
        ----------
        dimensions : Dimension or list of Dimension
            Smoothing dimension(s).

        """
        self.dimensions = dimensions

    @property
    def dimensions(self) -> Union[Dimension, List[Dimension]]:
        """Get smoothing dimension(s).

        Returns
        -------
        Dimension or list of Dimension
            Smoothing dimension(s).

        """
        if len(self._dimensions) == 1:
            return self._dimensions[0]
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: Union[Dimension, List[Dimension]]) \
            -> None:
        """Set smoothing dimension(s).

        Parameters
        ----------
        dimensions : Dimension or list of Dimension
            Smoothing dimension(s).

        Raises
        ------
        TypeError
            If `dimensions` is not a Dimension or list of Dimension.
        ValueError
            If duplicates found in `dimensions`.

        """
        # Check types
        dimensions = as_list(dimensions)
        if len(dimensions) == 0:
            raise TypeError('`dimensions` is an empty list.')
        if not all(isinstance(dim, Dimension) for dim in dimensions):
            raise TypeError('Invalid type(s) in `dimensions`.')

        # Check duplicates
        dim_list = [tuple(dim.dimension) for dim in dimensions]
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
        # Run checks
        columns = as_list(columns)

        # Extract data from data frame
        dim_list = [data[dim.dimension].values for dim in self._dimensions]
        idx_fit, col_list = self.get_data(data, fit, columns)
        idx_pred, _ = self.get_data(data, predict)

        # Calculate smoothed values
        smooth_cols = self.smooth_data(dim_list, col_list, idx_fit, idx_pred)

        # Construct smoothed data frame
        smooth_data = data[data[predict]]
        for idx_col, col in enumerate(columns):
            data[f"{col}_smooth"] = smooth_cols[:, idx_col]
        return smooth_data

    @staticmethod
    def get_data(data: DataFrame, indicator: str, columns: List[str] = None) \
            -> Tuple[NDArray, Union[None, List[NDArray]]]:
        """Get `fit` or `predict` data as numpy arrays.

        Parameters
        ----------
        data : DataFrame
            Input data structure.
        indicator : str
            Column name indicating either `fit` or `predict` data.
        columns : list of str, optional
            Column name(s) of values to smooth.

        Returns
        -------
        numpy.ndarray
            Indices of `fit` or `predict` points.
        None or list of np.ndarray
            None if `columns` is None. Otherwise, values to smooth.

        """
        if indicator is None:
            idx_ind = np.arange(len(data))
            col_list = None if columns is None \
                else [data[col].values for col in columns]
        else:
            data_ind = data[indicator]
            idx_ind = np.where(data_ind)[0]
            col_list = None if columns is None \
                else [data[data_ind][col].values for col in columns]
        return idx_ind, col_list

    def smooth_data(self, dim_list: List[NDArray], col_list: List[NDArray],
                    idx_fit: NDArray, idx_pred: NDArray) -> NDArray:
        """Smooth data across dimensions with weighted averages.

        Parameters
        ----------
        dim_list : list of numpy.ndarray
            Point locations across dimensions.
        col_list : list of numpy.ndarray
            Values to smooth.
        idx_fit : numpy.ndarray
            Indices of points to include in weighted averages.
        idx_pred : numpy.ndarray
            Indices of points to predict smoothed values.

        Returns
        -------
        numpy.ndarray
            Smoothed values.

        """
        # Initialize smoothed values
        n_cols = len(col_list)
        n_pred = len(idx_pred)
        smooth_cols = np.empty((n_pred, n_cols))

        # Calculate smoothed values one point at a time
        for idx_x in range(n_pred):
            weights = self.get_weights(dim_list, idx_fit, idx_pred[idx_x])

            # Compute smoothed values one column at a time
            for idx_col in range(n_cols):
                smooth_cols[idx_x, idx_col] = weights.dot(col_list[idx_col])

        return smooth_cols

    def get_weights(self, dim_list, idx_fit, idx_x):
        """Get smoothing weights for current point.

        Parameters
        ----------
        dim_list : list of numpy.ndarray
            Point locations across dimensions.
        idx_fit : numpy.ndarray
            Indices of nearby points in `dim_list`.
        idx_x : int
            Index of current point in `dim_list`.

        Returns
        -------
        numpy.ndarray
            Smoothing weights for current point.

        """
        # Initialize weight vectors
        n_dims = len(dim_list)
        n_fit = len(idx_fit)
        dim_weights = np.empty((n_dims, n_fit))

        # Calculate weights one dimension at a time
        for idx_dim in range(n_dims):
            x = dim_list[idx_dim][idx_x]
            kernel = self._dimensions[idx_dim].kernel

            # Calculate weights one point at a time
            for idx_y in range(n_fit):
                y = dim_list[idx_dim][idx_fit[idx_y]]
                dim_weights[idx_dim, idx_y] = kernel(x, y)

        # Aggregate dimension weights
        weights = dim_weights.prod(axis=0)
        return weights/weights.sum()
