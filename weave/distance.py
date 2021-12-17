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

import numpy as np
from numpy.typing import ArrayLike


class Distance(ABC):
    """Abstract distance function."""

    @abstractmethod
    def __call__(self, x: ArrayLike, y: ArrayLike) -> Union[int, float]:
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

    @staticmethod
    def _check_length(x: ArrayLike, y: ArrayLike) -> None:
        """Check input lengths match.

        Parameters
        ----------
        x : array_like
            Current point.
        y : array_like
            Nearby point.

        Raises
        ------
        ValueError
            If input lengths do not match.

        """
        if np.array(x).shape != np.array(y):
            raise ValueError('Input lengths do not match.')


class Continuous(Distance):
    """Continuous distance function."""

    def __call__(self, x: ArrayLike, y: ArrayLike) -> float:
        """Get continuous distance between `x` and `y`.

        Parameters
        ----------
        x : array_like
            Current point.
        y : array_like
            Nearby point.

        Returns
        -------
        float
            Continuous distance between `x` and `y`.

        """
        self._check_length(x, y)
        return np.linalg.norm(np.array(x) - np.array(y))


class Hierarchical(Distance):
    """Hierarchical distance function."""

    def __call__(self, x: ArrayLike, y: ArrayLike) -> int:
        """Get hierarchical distance between `x` and `y`.

        Parameters
        ----------
        x : array_like
            Current point.
        y : array_like
            Nearby point.

        Returns
        -------
        int
            Hierarchical distance between `x` and `y`.

        """
        self._check_length(x, y)
        try:
            temp = np.ones(x.shape[0] + 1)
        except IndexError:
            temp = np.ones(2)
        temp[1:] = np.array(x) == np.array(y)
        return np.where(temp[::-1])[0][0]
