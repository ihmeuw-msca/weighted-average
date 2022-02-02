# pylint: disable=C0103
"""Calculate the distance between two points.

Distance functions to calculate the distance between two points, where
points are given as scalars or vectors.

In general, distance functions should satisfy the following properties:
1. d(x, y) is real-valued, finite, and nonnegative
2. d(x, y) == 0 if and only if x == y
3. d(x, y) == d(y, x) (symmetry)
4. d(x, y) <= d(x, z) + d(z, y) (triangle inequality)

TODO
* Change docstrings based on vectorization
* Didn't know how to vectorize hierarchical or dictionary

"""
from typing import Dict, Tuple, Union

from numba import njit
import numpy as np

from weave.utils import is_numeric

Numeric = Union[int, float]


@njit
def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    """Get Euclidean distance between `x` and `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    y : 1D numpy.ndarray of float
        Nearby point.

    Returns
    -------
    nonnegative float
        Euclidean distance between `x` and `y`.

    """
    return 1.0*np.linalg.norm(x - y, axis=0)


@njit
def hierarchical(x: np.ndarray, y: np.ndarray) -> float:
    """Get hierarchical distance between `x` and `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    y : 1D numpy.ndarray of float
        Nearby point.

    Returns
    -------
    nonnegative float
        Hierarchical distance between `x` and `y`.

    """
    # Distance from one nearby point
    def get_distance(x, y):
        if (x == y).all():
            return 0.0
        for ii in range(1, len(x)):
            if (x[:-ii] == y[:-ii]).all():
                return 1.0*ii
        return 1.0*len(x)

    # Distance from all nearby points
    distance = np.empty(len(y))
    for ii, yi in enumerate(y):
        distance[ii] = get_distance(x, yi)

    return distance


@njit
def dictionary(x: np.ndarray, y: np.ndarray,
               distance_dict: Dict[Tuple[float, float], float]) -> float:
    """Get dictionary distance between `x` and `y`.

    Dictionary `distance_dict` contains the distance between points `x`
    and `y`. For type consistency among distance functions, `x` and `y`
    are 1D numpy arrays of float, but they should contain a single
    value (e.g., location ID). Dictionary keys are tuples of point
    pairs `(x[0], y[0])`, and dictionary values are the corresponding
    distances. Because distances are assumed to be symmetric, point
    pairs are listed from smallest to largest, e.g., `x` <= `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    y : 1D numpy.ndarray of float
        Nearby point.
    distance_dict : dict of {(float, float): float}
        Dictionary of distances between points.

    Returns
    -------
    nonnegative float
        Dictionary distance between `x` and `y`.

    """
    # Distance from one nearby point
    def get_distance(x, y):
        x0 = float(x[0])
        y0 = float(y[0])
        if x0 <= y0:
            return distance_dict[(x0, y0)]
        return distance_dict[(y0, x0)]

    # Distance from all nearby points
    distance = np.empty(len(y))
    for ii, yi in enumerate(y):
        distance[ii] = get_distance(x, yi)
    return distance


def check_dict(distance_dict: Dict[Tuple[Numeric, Numeric], Numeric]) -> None:
    """Check dictionary keys and values.

    Parameters
    ----------
    distance_dict : dict of {(numeric, numeric): numeric}
        Dictionary of distances between points.

    Raises
    ------
    TypeError
        If `distance_dict`, keys, or values are an invalid type.
    ValueError
        If `dictionary_dict` is empty, dictionary keys are not all
        length 2, or dictionary values are not all nonnegative.

    """
    # Check types
    if not isinstance(distance_dict, dict):
        raise TypeError('`distance_dict` is not a dict.')
    if not all(isinstance(key, tuple) for key in distance_dict):
        raise TypeError('`distance_dict` keys not all tuple.')
    if not all(is_numeric(point) for key in distance_dict for point in key):
        raise TypeError('`distance_dict` key entries not all int or float.')
    if not all(is_numeric(value) for value in distance_dict.values()):
        raise TypeError('`distance_dict` values not all int or float.')

    # Check values
    if len(distance_dict) == 0:
        raise ValueError('`distance_dict` is an empty dict.')
    if any(len(key) != 2 for key in distance_dict):
        raise ValueError('`distance_dict` keys are not all length 2.')
    if any(value < 0.0 for value in distance_dict.values()):
        raise ValueError('`distance_dict` contains negative values.')
