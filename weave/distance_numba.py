"""Calculate distance between points.

Callable distance function classes that calculate the distance between
two points, where points are given as scalars or vectors. In general,
distance functions should satisfy the following properties:
1. d(x, y) is real-valued, finite, and nonnegative
2. d(x, y) == 0 if and only if x == y
3. d(x, y) == d(y, x) (symmetry)
4. d(x, y) <= d(x, z) + d(z, y) (triangle inequality)

"""
from abc import ABC, abstractmethod
from typing import Union

from numba import njit
import numpy as np
from numpy.typing import ArrayLike


class Distance(ABC):
    """Abstract distance function."""

    @staticmethod
    @abstractmethod
    def __call__(x: ArrayLike, y: ArrayLike) -> Union[int, float]:
        """Get distance between `x` and `y`.

        Parameters
        ----------
        x : array_like
            Current point.
        y: array_like
            Nearby point.

        Returns
        -------
        int or float
            Distance between `x` and `y`.

        """
        raise NotImplementedError


class Continuous(Distance):
    """Continuous distance function."""

    @staticmethod
    @njit
    def __call__(x: np.ndarray, y: np.ndarray) -> float:
        """Get continuous distance between `x` and `y`.

        Parameters
        ----------
        x : 1D numpy.ndarray of float
            Current point.
        y : 1D numpy.ndarray of float
            Nearby point.

        Returns
        -------
        float
            Continuous distance between `x` and `y`.

        """
        return np.linalg.norm(x - y)


class Hierarchical(Distance):
    """Hierarchical distance function."""

    @staticmethod
    @njit
    def __call__(x: np.ndarray, y: np.ndarray) -> int:
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


@njit
def check_shapes(x: np.ndarray, y: np.ndarray) -> None:
    """Check input shapes.

    Parameters
    ----------
    x : 1D numpy.ndarray
        Current point.
    y : 1D numpy.ndarray
        Nearby point.

    Raises
    ------
    ValueError
        If input shapes do not match.
    ValueError
        If input has wrong number of dimensions.

    """
    if x.shape != y.shape:
        raise ValueError('Input shapes do not match.')
    if len(x.shape) != 1:
        raise ValueError('Input has wrong number of dimensions.')
