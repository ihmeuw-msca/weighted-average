"""Calculate the smoothing weight for nearby point given current point.

Kernel functions to calculate the smoothing weight for a nearby point
given the current point, where points are given as scalars or vectors.
In general, kernel functions should have the following form:
* k_r(x, y) = f(d(x, y)/r)
* f: function whose value is decreasing (or non-increasing) for
     increasing distance between `x` and `y`
* d: distance function
* r: kernel radius

TODO:
* Generalize depth function to include more levels (e.g., sub-national)
* STGPR has a different depth function than CODEm
* Some parameters may depend on whether or not there is country-level
  data (should we add an argument for that? an alternate radius?)
* Tricubic should have conditions on arguments to guarantee that output
  is nonnegative

"""
from typing import Union

from numba import njit
import numpy as np


@njit
def exponential(distance: float, radius: Union[int, float]) -> float:
    """Get exponential smoothing weight.

    k_r(x, y) = 1/exp(d(x, y)/r)
    CODEm: r = 1/omega

    Parameters
    ----------
    distance : float
        Distance between points.
    radius : int or float
        Kernel radius.

    Returns
    -------
    float
        Exponential smoothing weight.

    """
    return 1.0/np.exp(distance/radius)


@njit
def tricubic(distance: float, radius: Union[int, float],
             exponent: Union[int, float]) -> float:
    """Get tricubic smoothing weight.

    k_r(x, y) = (1 - (d(x, y)/r)^s)^3
    CODEm: s = lambda, r = max(x - x_min, x_max - x) + 1

    Parameters
    ----------
    distance : float
        Distance between points.
    radius : int or float
        Kernel radius.
    exponent : int or float
        Exponent value.

    Returns
    -------
    float
        Tricubic smoothing weight.

    """
    return (1.0 - (distance/radius)**exponent)**3


@njit
def depth(distance: float, radius: float):
    """Get depth smoothing weight.

    If distance == 0 (same country):
        weight = radius
    If distance == 1 (same region):
        weight = radius*(1 - radius)
    If distance == 2 (same super-region):
        weight = (1 - radius)^2
    If distance >= 3 (different super-region):
        weight = 0

    Need to generalize for more levels (e.g., sub-national).
    STGPR and CODEm have differnet versions. This function uses the
    CODEm version.

    Parameters
    ----------
    distance : float
        Distance between points.
    radius : float
        Kernel radius.

    Returns
    -------
    float
        Depth smoothing weight.

    """
    if distance == 0.0:
        return radius
    if distance == 1.0:
        return radius*(1.0 - radius)
    if distance == 2.0:
        return (1.0 - radius)**2
    return 0.0
