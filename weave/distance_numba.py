# pylint: disable=C0103
"""Calculate the distance between two points.

Distance functions to calculate the distance between two points, where
points are given as scalars or vectors. In general, distance functions
should satisfy the following properties:
1. d(x, y) is real-valued, finite, and nonnegative
2. d(x, y) == 0 if and only if x == y
3. d(x, y) == d(y, x) (symmetry)
4. d(x, y) <= d(x, z) + d(z, y) (triangle inequality)

"""
from typing import Union

from numba import njit
import numpy as np


@njit
def continuous(x: Union[int, float], y: Union[int, float]) \
        -> Union[int, float]:
    """Get continuous distance between `x` and `y`.

    Parameters
    ----------
    x : int or float
        Current point.
    y : int or float
        Nearby point.

    Returns
    -------
    int or float
        Continuous distance between `x` and `y`.

    """
    return np.abs(x - y)


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
    float
        Euclidean distance between `x` and `y`.

    """
    return np.linalg.norm(x - y)


@njit
def hierarchical(x: np.ndarray, y: np.ndarray) -> int:
    """Get hierarchical distance between `x` and `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray
        Current point.
    y : 1D numpy.ndarray
        Nearby point.

    Returns
    -------
    int
        Hierarchical distance between `x` and `y`.

    """
    temp = np.ones(x.shape[0] + 1)
    temp[1:] = x == y
    return np.where(temp[::-1])[0][0]
