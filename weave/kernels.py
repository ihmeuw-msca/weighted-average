# pylint: disable=E0611
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
from numba import njit
from numba.typed import Dict
import numpy as np


@njit
def exponential(distance: float, pars: Dict[str, float]) -> float:
    """Get exponential smoothing weight.

    k_r(x, y) = 1/exp(d(x, y)/r)
    CODEm: r = 1/omega

    Parameters
    ----------
    distance : nonnegative float
        Distance between points.
    pars : dict of {str: float}
        Kernel function parameters.

    Kernel function parameters
    --------------------------
    radius : positive float
        Kernel radius.

    Returns
    -------
    float
        Exponential smoothing weight.

    """
    return 1.0/np.exp(distance/pars['radius'])


@njit
def tricubic(distance: float, pars: Dict[str, float]) -> float:
    """Get tricubic smoothing weight.

    k_r(x, y) = max(0, (1 - (d(x, y)/r)^s)^3)
    CODEm: s = lambda, r = max(x - x_min, x_max - x) + 1

    Parameters
    ----------
    distance : nonnegative float
        Distance between points.
    pars : dict of {str: float}
        Kernel function parameters.

    Kernel function parameters
    --------------------------
    radius : positive float
        Kernel radius.
    exponent : positive float
        Exponent value.

    Returns
    -------
    float
        Tricubic smoothing weight.

    """
    return max(0.0, (1.0 - (distance/pars['radius'])**pars['exponent'])**3)


@njit
def depth(distance: float, pars: Dict[str, float]) -> float:
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
    pars : dict of {str: float}
        Kernel function parameters.

    Kernel function parameters
    --------------------------
    radius : float in (0, 1)
        Kernel radius.

    Returns
    -------
    float
        Depth smoothing weight.

    """
    if distance == 0.0:
        return pars['radius']
    if distance == 1.0:
        return pars['radius']*(1.0 - pars['radius'])
    if distance == 2.0:
        return (1.0 - pars['radius'])**2
    return 0.0
