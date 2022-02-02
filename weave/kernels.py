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
* Change docstrings based on vectorization
* Change tests based on vectorization

"""
from typing import Dict, List, Union

from numba import njit, vectorize
import numpy as np

from weave.utils import as_list, is_numeric


@njit
def exponential(distance: float, radius: float) -> float:
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
    nonnegative float
        Exponential smoothing weight.

    """
    return 1.0/np.exp(distance/radius)


@njit
def tricubic(distance: float, radius: float, exponent: float) -> float:
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
    nonnegative float
        Tricubic smoothing weight.

    """
    return np.maximum(0.0, (1.0 - (distance/radius)**exponent)**3)


@vectorize(['float64(float64, float64)'])
def depth(distance: float, radius: float) -> float:
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
    nonnegative float
        Depth smoothing weight.

    """
    if distance == 0.0:
        return radius
    if distance <= 1.0:
        return radius*(1.0 - radius)
    if distance <= 2.0:
        return (1.0 - radius)**2
    return 0.0


def check_pars(pars: Dict[str, Union[int, float]],
               names: Union[str, List[str]], types: Union[str, List[str]]) \
        -> None:
    """Check kernel parameter types and values.

    Parameters
    ----------
    pars : dict of {str: int or float}
        Kernel parameters
    names : str or list of str
        Parameter names.
    types : str or list of str
        Parameter types. Valid types are 'pos_num', 'pos_frac', 'bool'.

    Raises
    ------
    KeyError
        If `pars` is missing a kernel parameter.
    TypeError
        If a kernel parameter is an invalid type.
    ValueError
        If a kernel parameter is an invalid value.

    """
    names = as_list(names)
    if isinstance(types, str):
        types = [types]*len(names)

    for idx_par, par_name in enumerate(names):
        # Check key
        if par_name not in pars:
            raise KeyError(f"`{par_name}` is not in `pars`.")
        par_val = pars[par_name]

        if types[idx_par] == 'pos_num':
            # Check type
            if not is_numeric(par_val):
                raise TypeError(f"`{par_name}` is not an int or float.")

            # Check value
            if par_val <= 0.0:
                raise ValueError(f"`{par_name}` is not positive.")

        elif types[idx_par] == 'pos_frac':
            # Check type
            if not isinstance(par_val, (float, np.floating)):
                raise TypeError(f"`{par_name}` is not a float.")

            # Check value
            if par_val <= 0.0 or par_val >= 1.0:
                raise ValueError(f"`{par_name}` is not in (0, 1).")

        else:  # 'bool'
            # Check type
            if not isinstance(par_val, bool):
                raise TypeError(f"`{par_name}` is not a bool.")
