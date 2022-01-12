# pylint: disable=C0103
"""Calculate the distance between two points.

Distance functions to calculate the distance between two points, where
points are given as scalars or vectors.

In general, distance functions should satisfy the following properties:
1. d(x, y) is real-valued, finite, and nonnegative
2. d(x, y) == 0 if and only if x == y
3. d(x, y) == d(y, x) (symmetry)
4. d(x, y) <= d(x, z) + d(z, y) (triangle inequality)

"""
from typing import Dict, Tuple

from numba import njit
import numpy as np

DistanceDict = Dict[Tuple[np.ndarray, np.ndarray], float]


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
    return 1.0*np.linalg.norm(x - y)


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
    if (x == y).all():
        return 0.0
    for ii in range(1, len(x)):
        if (x[:-ii] == y[:-ii]).all():
            return 1.0*ii
    return 1.0*len(x)


@njit
def dictionary(x: np.ndarray, y: np.ndarray, distance_dict: DistanceDict) \
        -> float:
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
    distance_dict : dict of {tuple of float: float}
        Dictionary of distances between points.

    Returns
    -------
    nonnegative float
        Dictionary distance between `x` and `y`.

    """
    x = x[0]
    y = y[0]
    if x <= y:
        return distance_dict[(x, y)]
    return distance_dict[(y, x)]
