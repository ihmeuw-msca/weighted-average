# pylint: disable=C0103, E0611, R0912
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

from weave.utils import as_list, is_int, is_float, is_number

pars = Union[int, float, bool]


@vectorize(['float32(float32,float32,float32)'])
def depth(distance: float, radius: float, levels: float) -> float:
    """Get depth smoothing weight.

    Parameters
    ----------
    distance : nonnegative float32
        Distance between points.
    radius : float32 in (0, 1)
        Kernel radius.
    levels : positive float
        Number of levels. If `dimension.distance` is 'tree', this is
        equal to the length of `dimension.coordinates`.

    Returns
    -------
    nonnegative float32
        Depth smoothing weight.

    Notes
    -----
    The depth kernel function is defined as

    .. math:: k(d; r, s) = \\begin{cases} r & \\text{if } d = 0, \\\\
              r(1 - r)^{\\lceil d \\rceil} & \\text{if } 0 < d \\leq
              s - 2, \\\\ (1 - r)^{\\lceil d \\rceil} & \\text{if }
              s - 2 < d \\leq s - 1, \\\\ 0 & \\text{if } d > s - 1,
              \\end{cases}

    which is the same as CODEm's location scale factors with
    :math:`d =`:mod:`weave.distance.tree`:math:`(\\ell_i, \\ell_j)`,
    :math:`r = \\zeta`, and :math:`s =` the number of levels in the
    location hierarchy (e.g., locations with coordinates
    'super_region', 'region', and 'country' would have :math:`s = 3`).
    If :math:`s = 1`, the possible weight values are 1 and 0.

    Examples
    --------
    Get weight for a pair of points.

    >>> import numpy as np
    >>> from weave.kernels import depth
    >>> radius = np.float32(0.9)
    >>> distance = np.float32(1.)
    >>> levels = np.float32(3)
    >>> depth(distance, radius, levels)
    0.09000002

    Get weights for a vector of point pairs.

    >>> import numpy as np
    >>> from weave.kernels import depth
    >>> radius = np.float32(0.9)
    >>> distance = np.array([0., 1., 2., 3.]).astype(np.float32)
    >>> levels = np.float32(3)
    >>> depth(distance, radius, levels)
    array([0.9, 0.09000002, 0.01, 0.], dtype=float32)

    """
    same_tree = distance <= levels - 1
    not_root = levels > 1 and distance <= levels - 2
    return same_tree*radius**not_root*(1 - radius)**np.ceil(distance)


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

    with :math:`d_{i, j} =`:mod:`weave.distance.euclidean`
    :math:`(a_i, a_j)` and :math:`r = \\frac{1}{\\omega}`.

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

    with :math:`d_{i, j} =`:mod:`weave.distance.euclidean`
    :math:`(t_i, t_j)` and :math:`s = \\lambda`. However, the
    denominator in the CODEm weight varies based on the coordinate
    :math:`t_i`, while the kernel radius :math:`r` is fixed.

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
        Parameter types. Valid types are 'pos_num', 'pos_int', 'pos_frac',
        and 'bool'.

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
        msg = f"`{par_name}` is not "

        # Check key
        if par_name not in kernel_pars:
            raise KeyError(msg + 'in `pars`.')
        par_val = kernel_pars[par_name]

        # Check type and value
        if types[idx_par] == 'bool':
            if not isinstance(par_val, bool):
                raise TypeError(msg + 'a bool.')
        else:
            if types[idx_par] == 'pos_frac':
                if not is_float(par_val):
                    raise TypeError(msg + 'a float.')
                if par_val <= 0.0 or par_val >= 1.0:
                    raise ValueError(msg + 'in (0, 1).')
            else:
                if types[idx_par] == 'pos_num':
                    if not is_number(par_val):
                        raise TypeError(msg + 'an int or float.')
                else:  # 'pos_int'
                    if not is_int(par_val):
                        raise TypeError(msg + 'an int.')
                if par_val <= 0.0:
                    raise ValueError(msg + 'positive.')
