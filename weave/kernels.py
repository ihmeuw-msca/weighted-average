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

   - :math:`k(x, y; r) \\leq k(x', y'; r)` if :math:`d(x, y) > d(x', y')`
   - :math:`k(x, y; r) \\geq k(x', y'; r)` if :math:`d(x, y) < d(x', y')`

The :func:`exponential`, :func:`tricubic`, and :func:`depth` kernel
functions are modeled after the age, time, and location weights CODEm
[2]_ (see the spatial-temporal models sub-section within the methods
section). There are many other kernel functions in common use [3]_.

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
import numpy as np


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
    nonnegative float32
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
    >>> radius = 0.5
    >>> exponential(0., radius)
    1.0
    >>> exponential(1., radius)
    0.13533528
    >>> exponential(2., radius)
    0.01831564

    """
    return np.float32(1/np.exp(distance/radius))


def depth(distance: float, radius: float) -> float:
    """Get depth smoothing weight.

    Parameters
    ----------
    distance : nonnegative float
        Distances between points.
    radius : float in (0, 1)
        Kernel radius.

    Returns
    -------
    nonnegative float32
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
    :mod:`weave.distance.tree`:math:`(\\ell_i, \\ell_j)`. This
    corresponds to points that have the same country, region, or super
    region, respectively, but the kernel function has not yet been
    generalized to consider further location divisions (e.g., state or
    county).

    Examples
    --------
    Get depth smoothing weights.

    >>> import numpy as np
    >>> from weave.kernels import depth
    >>> radius = 0.9
    >>> depth(0., radius)
    0.9
    >>> depth(1., radius)
    0.09
    >>> depth(2., radius)
    0.01
    >>> depth(3., radius)
    0.0

    """
    if distance == 0:
        return np.float32(radius)
    if distance <= 1:
        return np.float32(radius*(1 - radius))
    if distance <= 2:
        return np.float32((1 - radius)**2)
    return np.float32(0)


def tricubic(distance: float, radius: float, exponent: float) -> float:
    """Get tricubic smoothing weight.

    Parameters
    ----------
    distance : nonnegative float
        Distances between points.
    radius : positive float
        Kernel radius.
    exponent : positive float
        Exponent value.

    Returns
    -------
    nonnegative float32
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
    Get tricubic smoothing weights.

    >>> import numpy as np
    >>> from weave.kernels import tricubic
    >>> distance = np.array([0., 1., 2.])
    >>> radius = 2.
    >>> exponent = 3.
    >>> tricubic(0., radius, exponent)
    1.0
    >>> tricubic(1., radius, exponent)
    0.6699219
    >>> tricubic(2., radius, exponent)
    0.0

    """
    return np.float32(np.maximum(0, (1 - (distance/radius)**exponent)**3))
