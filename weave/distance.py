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
        x = np.array(x)
        y = np.array(y)
        try:
            temp = np.ones(len(x) + 1)
        except TypeError:
            temp = np.ones(2)
        temp[1:] = x == y
        return np.where(temp[::-1])[0][0]
