"""Calculate distance between points.

Calculate the distance between two points, where points are given as
scalars or vectors. In general, distance functions should satisfy the
following properties:
1. d(x, y) is real-valued, finite, and nonnegative
2. d(x, y) == 0 if and only if x == y
3. d(x, y) == d(y, x) (symmetry)
4. d(x, y) <= d(x, z) + d(z, y) (triangle inequality)

NOTE:
* To get numba to work, had to install scipy and revert numpy to
  older version (1.20?)
* I think scipy was to use np.linalg.norm
* Also, np.linalg.norm wouldn't work with scalars or integers
* Will need to be way more specific about types in general, and not
  sure could use same helper functions for different functions that
  have arguments of different types or shapes (esp. with scalars vs.
  vectors and calls to shape)
* jitclass might not work for distance functions, because they don't
  have an __init__ or attributes
* numba will already do lots of checks, do we need to explicitly add
  them to our functions?
* if we are prioritizing speed, and we know how our functions are
  being called by the kernels, do we need to check for types and
  shapes? Won't that slow us down if we do it every time?

"""
from abc import ABC, abstractmethod
from typing import Union

from numba import njit
from numba.experimental import jitclass
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


@jitclass()
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
        check_shapes(x, y)
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
        check_shapes(x, y)
        temp = np.ones(x.shape[0] + 1)
        temp[1:] = x == y
        return np.where(temp[::-1])[0][0]


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
    check_shapes(x, y)
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
    check_shapes(x, y)
    temp = np.ones(x.shape[0] + 1)
    temp[1:] = x == y
    return np.where(temp[::-1])[0][0]
    

@njit
def check_scalars(x: Union[int, float], y: Union[int, float]) -> None:
    """Check scalar input types.

    Parameters
    ----------
    x : int or float
        Current point.
    y : int or float
        Nearby point.

    Raises
    ------
    TypeError
        If input is incorrect type.

    """
    x_is_scalar = isinstance(x, (int, np.integer, float, np.floating))
    y_is_scalar = isinstance(y, (int, np.integer, float, np.floating))
    if not (x_is_scalar and y_is_scalar):
        raise TypeError('Input has incorrect type.')


@njit
def check_vectors(x: np.ndarray, y: np.ndarray) -> None:
    """Check vector input types and shapes.

    Parameters
    ----------
    x : 1D numpy.ndarray
        Current point.
    y : 1D numpy.ndarray
        Nearby point.

    Raises
    ------
    TypeError
        If input is incorrect type.
    ValueError
        If input shapes do not match.
    ValueError
        If input has incorrect number of dimensions.

    """
    x_is_vec = isinstance(x, np.ndarray)
    y_is_vec = isinstance(y, np.ndarray)
    if not (x_is_vec and y_is_vec):
        raise TypeError('Input is incorrect type.')
    if x.shape != y.shape:
        raise ValueError('Input shapes do not match.')
    if len(x.shape) != 1:
        raise ValueError('Input has incorrect number of dimensions.')
