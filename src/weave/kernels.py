# pylint: disable=C0103
"""Calculate the smoothing weight for nearby point given current point.

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

   - :math:`k(x, y; r) \\leq k(x, y'; r)` if :math:`d(x, y) > d(x, y')`
   - :math:`k(x, y; r) \\geq k(x, y'; r)` if :math:`d(x, y) < d(x, y')`

The :func:`exponential`, :func:`tricubic`, and :func:`depth` kernel
functions are modeled after the age, time, and location weights used in
CODEm [2]_ (see the spatial-temporal models sub-section within the
methods section). There are many other kernel functions in common use
[3]_.

The kernel functions in this module compute weights using the distance
between points as input rather than the points themselves.

References
----------
.. [1] `Kernel smoother
       <https://en.wikipedia.org/wiki/Kernel_smoother>`_
.. [2] `Cause of Death Ensemble model
       <https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-10-1>`_
.. [3] `Kernel (statistics)
       <https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use>`_

"""
# TODO: add note about inverse-distance kernel to kernel module documentation
from typing import Union

import numpy as np

number = Union[int, float]


def exponential(distance: number, radius: number) -> np.float32:
    """Get exponential smoothing weight.

    Parameters
    ----------
    distance : nonnegative int or float
        Distance between points.
    radius : positive int or float
        Kernel radius.

    Returns
    -------
    nonnegative numpy.float32
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
    Get exponential smoothing weights.

    >>> import numpy as np
    >>> from weave.kernels import exponential
    >>> exponential(0, 0.5)
    1.0
    >>> exponential(1, 0.5)
    0.13533528
    >>> exponential(2, 0.5)
    0.01831564

    """
    return np.float32(1 / np.exp(distance / radius))


def tricubic(distance: number, radius: number, exponent: number) -> np.float32:
    """Get tricubic smoothing weight.

    Parameters
    ----------
    distance : nonnegative int or float
        Distances between points.
    radius : positive int or float
        Kernel radius.
    exponent : positive int or float
        Exponent value.

    Returns
    -------
    nonnegative numpy.float32
        Tricubic smoothing weight.

    Notes
    -----
    The tricubic kernel function is defined as

    .. math:: k(d; r, s) = \\left(1 -
              \\left(\\frac{d}{r}\\right)^s\\right)^3_+,

    which is equivalent to the CODEm time weight

    ..  math:: w_{t_{i, j}} = \\left(1 - \\left(\\frac{d_{i,
               j}}{\\max_k|t_i - t_k| + 1}\\right)^\\lambda\\right)^3

    with :math:`r = \\max_k|t_i - t_k| + 1`, :math:`s = \\lambda`, and
    :math:`d_{i, j} =`:mod:`weave.distance.euclidean`:math:`(t_i, t_j)`.

    The parameter `radius` is not assigned in `dimension.kernel_pars`
    because it is automatically set to :math:`\\max_k d_{i, k} + 1`.
    Since the radius depends on :math:`t_i`, this kernel is not
    symmetric.

    Examples
    --------
    Get tricubic smoothing weights.

    >>> import numpy as np
    >>> from weave.kernels import tricubic
    >>> tricubic(0, 2, 3)
    1.0
    >>> tricubic(1, 2, 3)
    0.6699219
    >>> tricubic(2, 2, 3)
    0.0

    """
    return np.float32(np.maximum(0, (1 - (distance / radius) ** exponent) ** 3))


def depth(distance: number, levels: int, radius: float, version: str) -> np.float32:
    """Get depth smoothing weight.

    Parameters
    ----------
    distance : nonnegative int or float
        Distance between points.
    levels : positive int
        Number of levels in `distance.tree`.
    radius : float in (0.5, 1)
        Kernel radius.
    version : {'codem', 'stgpr'}
        Depth kernel version corresponding to CODEm's location scale
        factors or ST-GPR's location scale factors.

    Returns
    -------
    nonnegative numpy.float32
        Depth smoothing weight.

    Notes
    -----
    When `version` = 'codem', the depth kernel is defined as

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

    When `version` = 'stgpr', the depth kernel function is defined as

    .. math:: k(d; r, s) = \\begin{cases} 1 & \\text{if } d = 0, \\\\
              r^{\\lceil d \\rceil} & \\text{if } 0 < d \\leq s - 1,
              \\\\ 0 & \\text{if } d > s - 1, \\end{cases}

    which is the same as ST-GPR's location scale factors with
    :math:`d =`:mod:`weave.distance.tree`:math:`(\\ell_i, \\ell_j)`,
    :math:`r = \\zeta`, and :math:`s =` the number of levels in the
    location hierarchy (e.g., locations with coordinates
    'super_region', 'region', and 'country' would have :math:`s = 3`).
    If :math:`s = 1`, the possible weight values are 1 and 0.

    The parameter `levels` is not assigned in `dimension.kernel_pars`
    because it is automatically set to the length of
    `dimension.coordinates`.

    Examples
    --------
    Get weight for a pair of points (CODEm version).

    >>> import numpy as np
    >>> from weave.kernels import depth
    >>> depth(0, 3, 0.9, 'codem')
    0.9
    >>> depth(1, 3, 0.9, 'codem')
    0.09
    >>> depth(2, 3, 0.9, 'codem')
    0.01
    >>> depth(3, 3, 0.9, 'codem')
    0.0

    Get weight for a pair of points (ST-GPR version).

    >>> import numpy as np
    >>> from weave.kernels import depth
    >>> depth(0, 3, 0.9, 'stgpr')
    1.0
    >>> depth(1, 3, 0.9, 'stgpr')
    0.9
    >>> depth(2, 3, 0.9, 'stgpr')
    0.81
    >>> depth(3, 3, 0.9, 'stgpr')
    0.0

    """
    same_tree = distance <= levels - 1
    if version == "stgpr":
        return np.float32(same_tree * radius ** np.ceil(distance))
    not_root = levels > 1 and distance <= levels - 2
    weight = same_tree * radius**not_root * (1 - radius) ** np.ceil(distance)
    return np.float32(weight)


def inverse(distance: number, radius: float) -> np.float32:
    """Get inverse-distance smoothing weight.

    Parameters
    ----------
    distance : nonnegative int or float
        Distance between points.
    radius : positive int or float
        Kernel radius.

    Returns
    -------
    nonnegative numpy.float32
        Inverse-distance smoothing weight.

    Notes
    -----
    The inverse-distance kernel function for a single dimension is
    defined as

    .. math:: k(d; r) = \\frac{d}{r},

    which is combined over all dimensions :math:`m \in \mathcal{M}` to
    create intermediate weights

    .. math:: \\tilde{w}_{i,j} = \\frac{1}
              {\\sum_{m \\in \\mathcal{M}} k(d_{i,j}^m; r^m) + \\sigma_i^2}.

    When using inverse-distance weights, all dimension kernels must be
    set to 'inverse', and the `stdev` argument is required for
    :mod:`weave.smoother.Smoother()`.

    """
    return np.float32(distance / radius)
