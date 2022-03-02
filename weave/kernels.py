# pylint: disable=C0103, E0611
"""Calculate the smoothing weight for nearby point given current point.

Kernel functions to calculate the smoothing weight for a nearby point
or a vector of nearby points given the current point using the distance
between points as input.

Notes
-----
In general, kernel functions should have the form [1]_

.. math:: k(x, y; r) = f\\left(\\frac{d(x, y)}{r}\\right)

with components:

* :math:`d(x, y)`: distance function
* :math:`r`: kernel radius

Kernel functions should also satisfy the following properties:

1. :math:`k(x, y; r)` is real-valued, finite, and nonnegative
2. :math:`k(x, y; r)` is decreasing (or non-increasing) for increasing
   distances between :math:`x` and :math:`y`:

   - :math:`k(x, y; r) \\leq k(x', y'; r)` if :math:`d(x, y) > d(x', y')`
   - :math:`k(x, y; r) \\geq k(x', y'; r)` if :math:`d(x, y) < d(x', y')`

The :func:`exponential`, :func:`tricubic`, and :func:`depth` kernel
functions are modeled after the age, time, and location weights CODEm
[2]_ (see the spatial-temporal models sub-section within the methods
section). There are many other kernel functions in common use [3]_.

The kernel functions in this module compute weights using the distance
between points as input rather than the points themselves. They are
also universal functions in the sense that the input can be either a
scalar distance or a vector of distances, as opposed to the functions
in :mod:`weave.distance`, which are stricter in terms of input
structure.

References
----------
.. [1] `Kernel smoother
       <https://en.wikipedia.org/wiki/Kernel_smoother>`_
.. [2] `Cause of Death Ensemble model
       <https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-10-1>`_
.. [3] `Kernel (statistics)
       <https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use>`_

"""
from typing import Dict, List, Optional, Union

from numba import njit, vectorize  # type: ignore
from numba.typed import Dict as TypedDict  # type: ignore
from numba.types import float32, unicode_type  # type: ignore
import numpy as np

from weave.utils import as_list, is_number

pars = Union[int, float, bool]


@njit
def exponential(distance: float, radius: float) -> float:
    """Get exponential smoothing weight.

    Parameters
    ----------
    distance : nonnegative float
        Distance between points.
    radius : positive float
        Kernel radius.

    Returns
    -------
    nonnegative float
        Exponential smoothing weight.

    Notes
    -----
    The exponential kernel function is defined as

    .. math:: k(d; r) = \\frac{1}{\\exp\\left(\\frac{d}{r}\\right)},

    which is equivalent to the CODEm age weight

    .. math:: w_{a_{i, j}} = \\frac{1}{\\exp(\\omega \\cdot d_{i, j})}

    with :math:`r = \\frac{1}{\\omega}` and :math:`d_{i, j} =`
    :mod:`weave.distance.euclidean`:math:`(a_i, a_j)`.

    Examples
    --------
    Get weight for a pair of points.

    >>> from weave.kernels import exponential
    >>> radius = 0.5
    >>> distance = 1.
    >>> exponential(distance, radius)
    0.1353352832366127

    Get weights for a vector of point pairs.

    >>> import numpy as np
    >>> from weave.kernels import exponential
    >>> radius = 0.5
    >>> distance = np.array([0., 1., 2.])
    >>> exponential(distance, radius)
    array([1., 0.13533528, 0.01831564])

    """
    return 1/np.exp(distance/radius)


@njit
def tricubic(distance: float, radius: float, exponent: float) -> float:
    """Get tricubic smoothing weight.

    Parameters
    ----------
    distance : nonnegative float
        Distance between points.
    radius : positive float
        Kernel radius.
    exponent : positive float
        Exponent value.

    Returns
    -------
    nonnegative float
        Tricubic smoothing weight.

    Notes
    -----
    The tricubic kernel function is defined as

    .. math:: k(d; r, s) = \\left(1 -
              \\left(\\frac{d}{r}\\right)^s\\right)^3_+,

    which is similar to the CODEm time weight

    ..  math:: w_{t_{i, j}} = \\left(1 - \\left(\\frac{d_{i,
               j}}{\\max_k|t_i - t_k| + 1}\\right)^\\lambda\\right)^3

    with :math:`s = \\lambda` and :math:`d_{i, j} =`
    :mod:`weave.distance.euclidean`:math:`(t_i, t_j)`. However, the
    denominator in the CODEm weight varies by input :math:`t_i`, while
    the kernel radius :math:`r` does not depend on the input :math:`d`.

    Examples
    --------
    Get weight for a pair of points.

    >>> from weave.kernels import tricubic
    >>> radius = 2.
    >>> exponent = 3.
    >>> distance = 1.
    >>> tricubic(distance, radius, exponent)
    0.669921875

    Get weights for a vector of point pairs.

    >>> import numpy as np
    >>> from weave.kernels import tricubic
    >>> radius = 2.
    >>> exponent = 3.
    >>> distance = np.array([0., 1., 2.])
    >>> tricubic(distance, radius, exponent)
    array([1., 0.66992188, 0.])

    """
    return np.maximum(0, (1 - (distance/radius)**exponent)**3)


@vectorize(['float32(float32,float32)'])
def depth(distance: float, radius: float) -> float:
    """Get depth smoothing weight.

    Parameters
    ----------
    distance : nonnegative float
        Distance between points.
    radius : float in (0, 1)
        Kernel radius.

    Returns
    -------
    nonnegative float
        Depth smoothing weight.

    Notes
    -----
    The depth kernel function is defined as

    .. math:: k(d; r) = \\begin{cases} r & \\text{if } d = 0, \\\\
              r(1 - r) & \\text{if } 0 < d \\leq 1, \\\\ (1 - r)^2 &
              \\text{if } 1 < d \\leq 2, \\\\ 0 & \\text{otherwise},
              \\end{cases}

    which is the same as CODEm's location scale factors with
    :math:`r = \\zeta` and :math:`d =`
    :mod:`weave.distance.hierarchical`:math:`(\\ell_i, \\ell_j)`. This
    corresponds to points that have the same country, region, or super
    region, respectively, but the kernel function has not yet been
    generalized to consider further location divisions (e.g., state or
    county).

    Examples
    --------
    Get weight for a pair of points.

    >>> from weave.kernels import depth
    >>> radius = 0.9
    >>> distance = 1.
    >>> depth(distance, radius)
    0.08999999999999998

    Get weights for a vector of point pairs.

    >>> import numpy as np
    >>> from weave.kernels import depth
    >>> radius = 0.9
    >>> distance = np.array([0., 1., 2., 3.])
    >>> depth(distance, radius)
    array([0.9, 0.09, 0.01, 0.])

    """
    if distance == 0:
        return radius
    if distance <= 1:
        return radius*(1 - radius)
    if distance <= 2:
        return (1 - radius)**2
    return 0


def get_typed_pars(kernel_pars: Optional[Dict[str, pars]] = None) \
        -> Dict[str, float]:
    """Get typed version of `kernel_pars`.

    Parameters
    ----------
    kernel_pars : dict of {str: int, float, or bool}, optional
        Kernel function parameters.

    Returns
    -------
    numba.typed.Dict of {unicode_type: float32}
        Typed version of `kernel_pars`.

    """
    typed_pars = TypedDict.empty(
        key_type=unicode_type,
        value_type=float32
    )
    if kernel_pars is not None:
        for key in kernel_pars:
            typed_pars[key] = np.float32(kernel_pars[key])
    return typed_pars


def _check_pars(kernel_pars: Dict[str, pars], names: Union[str, List[str]],
                types: Union[str, List[str]]) -> None:
    """Check kernel parameter types and values.

    Parameters
    ----------
    pars : dict of {str: int, float, or bool}
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
    # Check type
    if not isinstance(kernel_pars, dict):
        raise TypeError('`kernel_pars` is not a dict.')

    # Get parameter names
    names = as_list(names)
    if isinstance(types, str):
        types = [types]*len(names)

    for idx_par, par_name in enumerate(names):
        # Check key
        if par_name not in kernel_pars:
            raise KeyError(f"`{par_name}` is not in `pars`.")
        par_val = kernel_pars[par_name]

        if types[idx_par] == 'pos_num':
            # Check type
            if not is_number(par_val):
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
