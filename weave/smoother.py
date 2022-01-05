# pylint: disable=C0103, R0913, R0914
"""Smooth data across multiple dimensions using weighted averages.

TODO
* Write checks and tests
* Numba-fy any methods where possible
* Check meeting notes for other changes

Checks
* Check for duplicates in columns
* Check for NaNs and Infs in data frame
* Check columns, fit, predict in data frame
* Anything else?

"""
from typing import List, Tuple, Union

from numba import jit
import numpy as np
from pandas import DataFrame

from weave.dimension import Dimension
from weave.distance import continuous, euclidean, hierarchical
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
        # Run checks
        columns = as_list(columns)

        # Extract data from data frame
        group_list = [[data[dim.dimension].values for dim in group]
                      for group in self._dimensions]
        idx_fit, col_list = self.get_data(data, fit, columns)
        idx_pred, _ = self.get_data(data, predict)

        # Calculate smoothed values
        smooth_cols = self.smooth_data(group_list, col_list, idx_fit, idx_pred)

        # Construct smoothed data frame
        smooth_data = data.iloc[idx_pred].reset_index(drop=True)
        for idx_col, col in enumerate(columns):
            smooth_data[f"{col}_smooth"] = smooth_cols[:, idx_col]
        return smooth_data

    @staticmethod
    def get_data(data: DataFrame, indicator: str, columns: List[str] = None) \
            -> Tuple[np.ndarray, Union[None, List[np.ndarray]]]:
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

    @jit
    def smooth_data(self, group_list: List[List[np.ndarray]],
                    col_list: List[np.ndarray], idx_fit: np.ndarray,
                    idx_pred: np.ndarray) -> np.ndarray:
        """Smooth data across dimensions with weighted averages.

        Parameters
        ----------
        group_list : list of list of numpy.ndarray
            Point locations across dimension groups.
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
            weights = self.get_weights(group_list, idx_fit, idx_pred[idx_x])

            # Compute smoothed values one column at a time
            for idx_col in range(n_cols):
                smooth_cols[idx_x, idx_col] = weights.dot(col_list[idx_col])

        return smooth_cols

    @jit
    def get_weights(self, group_list: List[List[np.ndarray]],
                    idx_fit: np.ndarray, idx_x: int) -> np.ndarray:
        """Get smoothing weights for current point.

        Parameters
        ----------
        group_list : list of list of numpy.ndarray
            Point locations across dimension groups.
        idx_fit : numpy.ndarray
            Indices of nearby points in `dim_list`.
        idx_x : int
            Index of current point in `dim_list`.

        Returns
        -------
        numpy.ndarray
            Smoothing weights for current point.

        """
        # Initialize weight vector
        weights = np.ones(len(idx_fit))

        # Calculate weights one group at a time
        for idx_group, dim_list in enumerate(group_list):
            group_weights = self.get_group_weights(dim_list, idx_group,
                                                   idx_fit, idx_x)
            weights *= group_weights
            weights /= weights.sum()

        return weights

    @jit
    def get_group_weights(self, dim_list: List[np.ndarray], idx_group: int,
                          idx_fit: np.ndarray, idx_x: int) -> np.ndarray:
        """Get smoothing weights for current point and dimension group.

        Parameters
        ----------
        dim_list : list of numpy.npdarray
            Point locations across group dimension(s).
        idx_group : int
            Index of current dimension group.
        idx_fit : numpy.ndarray
            Indices of nearby points in `group_list`.
        idx_x : int
            Index of current point in `group_list`.

        Returns
        -------
        numpy.ndarray
            Smoothing weights for current point and dimension group.

        """
        # Initialize weight vector
        n_fit = len(idx_fit)
        weights = np.ones(n_fit)

        # Calculate weights one dimension at a time
        for idx_dim, dim in enumerate(dim_list):
            x = dim[idx_x]
            pars = tuple(self._dimensions[idx_group][idx_dim].pars.values())

            # Calculate weights one point at a time
            dim_weights = np.empty(n_fit)
            for idx_y in range(n_fit):
                y = dim[idx_fit[idx_y]]

                # Get distance
                dist_name = self._dimensions[idx_group][idx_dim].distance
                if dist_name == 'continuous':
                    distance = continuous(x, y)
                elif dist_name == 'euclidean':
                    distance = euclidean(x, y)
                else:  # hierarchical
                    distance = hierarchical(x, y)

                # Get weight
                kernel_name = self._dimensions[idx_group][idx_dim].kernel
                if kernel_name == 'exponential':
                    dim_weights[idx_y] = exponential(distance, *pars)
                elif kernel_name == 'tricubic':
                    dim_weights[idx_y] = tricubic(distance, *pars)
                else:  # depth
                    dim_weights[idx_y] = depth(distance, *pars)

            weights *= dim_weights

        return weights/weights.sum()
