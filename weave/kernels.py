"""Kernel functions to calculate smoothing weights.

Callable kernel function classes that calculate the smoothing weight
between two points, where points are given as scalars or vectors. In
general, kernel functions should have the following form:
* k_r(x, y) = f(d(x, y)/r)
* f: nonnegative real-valued function whose value is decreasing (or
     non-increasing) for increasing distance between `x` and `y`
* d: distance function
* r: kernel radius

TODO:
* Revisit tricubic function (radius might not be correct)
* Generalize depth function to include more levels (e.g., sub-national)
* STGPR has a different depth function than CODEm
* Some parameters may depend on whether or not there is country-level
  data (should we add an argument for that? an alternate radius?)

"""
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from weave.distance import Distance


class Kernel(ABC):
    """Abstract kernel function.

    Attributes
    ----------
    distance : weave.distance.Distance
        Distance function.
    radius : int or float
        Kernel radius.

    """

    def __init__(self, distance: Distance, radius: Union[int, float]) -> None:
        """Create kernel function.

        Parameters
        ----------
        distance : weave.distance.Distance
            Distance function.
        radius : int or float
            Kernel radius.

        """

    @property
    def distance(self) -> Distance:
        """Get distance function.

        Returns
        -------
        weave.distance.Distance
            Distance function.

        """
        return self._distance

    @distance.setter
    def distance(self, distance: Distance) -> None:
        """Set distance function.

        Parameters
        ----------
        distance : weave.distance.Distance
            Distance function.

        Raises
        ------
        TypeError
            If `distance` is not a distance function.

        """
        if not isinstance(distance, Distance):
            raise TypeError(f"Invalid type for `distance`: {type(distance)}.")
        self._distance = distance

    @property
    def radius(self) -> Union[int, float]:
        """Get kernel radius.

        Returns
        -------
        int or float
            Kernel radius.

        """
        return self._radius

    @radius.setter
    def radius(self, radius: Union[int, float]) -> None:
        """Set kernel radius.

        Parameters
        ----------
        radius : int or float
            Kernel radius.

        Raises
        ------
        TypeError
            If `radius` not an int or float.
        ValueError
            If `radius` is not positive.

        """
        is_bool = isinstance(radius, bool)
        is_num = isinstance(radius, (int, np.integer, float, np.floating))
        if is_bool or not is_num:
            raise TypeError(f"Invalid type for `radius`: {type(radius)}.")
        if radius <= 0.0:
            raise ValueError(f"`radius` is not positive: {radius}.")
        self._radius = radius

    @abstractmethod
    def __call__(self, x: ArrayLike, y: ArrayLike) -> Union[int, float]:
        """Get smoothing weight between `x` and `y`.

        Weight is used to compute smoothed version of `x` using
        weighted average of nearby points `y`.

        Parameters
        ----------
        x : array_like
            Current point.
        y : array_like
            Nearby point.

        Returns
        -------
        int or float
            Smoothing weight between `x` and `y`.

        """
        raise NotImplementedError
