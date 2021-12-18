"""Kernel functions to calculate smoothing weights.

Callable kernel function classes that calculate the smoothing weight
between two points, where points are given as scalars or vectors. In
general, kernel functions should have the following form:
* k_r(x, y) = f(d(x, y)/r)
* f: nonnegative real-valued function whose value is decreasing (or
     non-increasing) for increasing distance between `x` and `y`
* r: kernel radius
* d: distance function

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

from weave.distance import Distance, Continuous


class Kernel(ABC):
    """Abstract kernel function.

    Attributes
    ----------
    radius : int or float
        Kernel radius.
    distance : weave.distance.Distance
        Distance function.

    """

    def __init__(self, radius: Union[int, float], distance: Distance) -> None:
        """Create kernel function.

        Parameters
        ----------
        radius : int or float
            Kernel radius.
        distance : weave.distance.Distance
            Distance function.

        """

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


class Exponential(Kernel):
    """Exponential kernel function.

    k_r(x, y) = 1/exp(d(x, y)/r)
    In CODEm, r = 1/omega

    Attributes
    ----------
    radius : int or float
        Kernel radius.
    distance : weave.distance.Distance
        Distance function.

    """

    def __init__(self, radius: Union[int, float], distance: Distance = None) \
            -> None:
        """Create exponential kernel function.

        Parameters
        ----------
        radius : int or float
            Kernel radius.
        distance : weave.distance.Distance, optional
            Distance function; default is Continuous.

        """
        if distance is None:
            distance = Continuous()
        super().__init__(radius, distance)

    def __call__(self, x: ArrayLike, y: ArrayLike) -> float:
        """Get exponential smoothing weight between `x` and `y`.

        Parameters
        ----------
        x : array_like
            Current point.
        y : array_like
            Nearby point.

        Returns
        -------
        float
            Exponential smoothing weight between `x` and `y`.

        """
        return 1.0/np.exp(self._distance(x, y)/self._radius)


class Tricubic(Kernel):
    """Tricubic kernel function.

    k_r(x, y) = (1 - (d(x, y)/r)^2)^3
    In CODEm, s = lambda and r = aMax + 1,
    where aMax is max(x - x_min, x_max - x)

    There may be some confusion about aMax. Specifically, should it
    instead by x_max - x_min? For example, the tri-cube weight function
    is often given as (1 - |u|^3)^3, where |u| <= 1.

    I've implemented my interpretation here, which is slightly
    different from what's in the CODEm function.

    Attributes
    ----------
    radius : int or float
        Kernel radius.
    lam : int or float
        Exponenent value.
    distance : weave.distance.Distance
        Distance function.

    """

    def __init__(self, radius: Union[int, float], lam: Union[int, float] = 3,
                 distance: Distance = None) -> None:
        """Create tricubic kernel function.

        Parameters
        ----------
        radius : int or float
            Kernel radius.
        lam : int or float, optional
            Exponent value; default is 3.
        distance : weave.distance.Distance, optional
            Distance function; default is Continuous.

        """
        if distance is None:
            distance = Continuous()
        super().__init__(radius, distance)
        self.lam = lam

    @property
    def lam(self) -> Union[int, float]:
        """Get exponent value.

        Returns
        -------
        int or float
            Exponent value.

        """
        return self._lam

    @lam.setter
    def lam(self, lam: Union[int, float]) -> None:
        """Set exponent value.

        Parameters
        ----------
        lam : int or float
            Exponent value.

        Raises
        ------
        TypeError
            If `lam` not an int or float.
        ValueError
            If `lam` not positive.

        """
        is_bool = isinstance(lam, bool)
        is_num = isinstance(lam, (int, np.integer, float, np.floating))
        if is_bool or not is_num:
            raise TypeError(f"Invalid type for `lam`: {type(lam)}.")
        if lam <= 0.0:
            raise ValueError(f"`lam` is not positive: {lam}.")
        self._lam = lam

    def __cal__(self, x: ArrayLike, y: ArrayLike) -> float:
        """Get tricubic smoothing weight between `x` and `y`.

        Parameters
        ----------
        x : array_like
            Current point.
        y : array_like
            Nearby point.

        Returns
        -------
        float
            Tricubic smoothing weight between `x` and `y`.

        Raises
        ------
        ValueError
            If distance between `x` and `y` is greater than `radius`.

        """
        distance = self._distance(x, y)
        if distance > self._radius:
            msg = 'Distance between `x` and `y` is greater than `radius`: '
            msg += f"{distance}, {self._radius}."
            raise ValueError(msg)
        return (1.0 - (distance/self._radius)**self._lam)**3
