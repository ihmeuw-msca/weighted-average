"""Calculate the smoothing weight for nearby point given current point.

Kernel functions to calculate the smoothing weight for a nearby point
given the current point, where points are given as scalars or vectors.

In general, kernel functions should have the following form:
* k_r(x, y) = f(d(x, y)/r)
* f: function whose value is decreasing (or non-increasing) for
     increasing distance between `x` and `y`
* d: distance function
* r: kernel radius

In general, kernel functions should satisfy the following properties:
1. k(x, y) is real-valued, finite, and nonnegative
2. k(x, y) <= k(x', y') if d(x, y) > d(x', y')
   k(x, y) >= k(x', y') if d(x, y) < d(x', y')

TODO:
* Generalize depth function to include more levels (e.g., sub-national)
* STGPR has a different depth function than CODEm

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
    distance : nonnegative float
        Distance between points.
    radius : positive int or float
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

    k_r(x, y) = max(0, (1 - (d(x, y)/r)^s)^3)
    CODEm: s = lambda, r = max(x - x_min, x_max - x) + 1

    Parameters
    ----------
    distance : nonnegative float
        Distance between points.
    radius : positive int or float
        Kernel radius.
    exponent : positive int or float
        Exponent value.

    Returns
    -------
    float
        Tricubic smoothing weight.

    """
    return max(0.0, (1.0 - (distance/radius)**exponent)**3)


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
    distance : nonnegative float
        Distance between points.
    radius : float in (0, 1)
        Kernel radius.

    Returns
    -------
    float
        Depth smoothing weight.

    """
    if distance == 0.0:
        return radius
    if distance ==1.0:
        return radius*(1.0 - radius)
    if distance == 2.0:
        return (1.0 - radius)**2
    return 0.0
