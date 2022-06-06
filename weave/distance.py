# pylint: disable=C0103
"""Calculate the distance between points.

Notes
-----
In general, distance functions should satisfy the following properties
[1]_:

1. :math:`d(x, y)` is real-valued, finite, and nonnegative
2. :math:`d(x, y) = 0` if and only if :math:`x = y`
3. :math:`d(x, y) = d(y, x)` (symmetry)
4. :math:`d(x, y) \\leq d(x, z) + d(z, y)` (triangle inequality)

References
----------
.. [1] `Metric (mathematics)
       <https://en.wikipedia.org/wiki/Metric_(mathematics)>`_

"""
import numpy as np


def euclidean(x: np.ndarray, y: np.ndarray) -> np.float32:
    """Get Euclidean distance between `x` and `y`.

    Points `x` and `y` are specified as vectors of coordinate values.

    Parameters
    ----------
    x : 1D numpy.ndarray
        Current point.
    y : 1D numpy.ndarray
        Nearby point.

    Returns
    -------
    nonnegative numpy.float32
        Euclidean distance between `x` and `y`.

    Notes
    -----
    For a pair of points with *n* coordinates, this function computes
    the *n*-dimensional Euclidean distance [2]_:

    .. math:: d(x, y) = \\sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \\dots +
              (x_n - y_n)^2}

    If :math:`n = 1`, this is equivalent to the absolute value of the
    difference between points:

    .. math:: d(x, y) = |x - y|

    References
    ----------
    .. [2] `Euclidean distance
           <https://en.wikipedia.org/wiki/Euclidean_distance>`_

    Examples
    --------
    Get Euclidean distances between points.

    >>> import numpy as np
    >>> from weave.distance import euclidean
    >>> euclidean(np.array([0]), np.array([0]))
    0.0
    >>> euclidean(np.array([0, 0]), np.array([1, 1]))
    1.4142135
    >>> euclidean(np.array([0, 0, 0]), np.array([1, 2, 3]))
    3.7416575

    """
    _check_input(x, y)
    return np.linalg.norm(x - y).astype(np.float32)


def tree(x: np.ndarray, y: np.ndarray) -> np.float32:
    """Get tree distance between `x` and `y`.

    Points `x` and `y` are specified as vectors of IDs corresponding to
    nodes in a tree, starting with the root node and ending at the leaf
    node. The distance between two points is defined as the number of
    edges between the leaf nodes and their nearest common ancestor
    node.

    Parameters
    ----------
    x : 1D numpy.ndarray
        Current point.
    y : 1D numpy.ndarray
        Nearby point.

    Returns
    -------
    nonnegative numpy.float32
        Tree distance between `x` and `y`.

    Examples
    --------
    Get tree distances between leaf nodes from the following tree.

    .. image:: images/hierarchy.png

    >>> import numpy as np
    >>> from weave.distance import tree
    >>> tree(np.array([1, 2, 4]), np.array([1, 2, 4]))
    0.0
    >>> tree(np.array([1, 2, 4]), np.array([1, 2, 5]))
    1.0
    >>> tree(np.array([1, 2, 4]), np.array([1, 3, 6]))
    2.0
    >>> tree(np.array([1, 2, 4]), np.array([7, 8, 9]))
    3.0

    """
    _check_input(x, y)
    return _tree(x, y, 0)


def _tree(x: np.ndarray, y: np.ndarray, n: int) -> np.float32:
    """Get tree distance between `x` and `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray
        Current point.
    y : 1D numpy.ndarray
        Nearby point.
    n : int
        Recursion parameter.

    Returns
    -------
    nonnegative numpy.float32
        Tree distance between `x` and `y`.

    """
    if (x == y).all():
        return np.float32(n)
    return _tree(x[:-1], y[:-1], n + 1)


def _check_input(x: np.ndarray, y: np.ndarray) -> None:
    """Check distance function input.

    Parameters
    ----------
    x : 1D numpy.ndarray
        Current point.
    y : 1D numpy.ndarray
        Nearby point.

    Raises
    ------
    TypeError
        If inputs are not numpy.ndarray.
    ValueError
        If inputs are not 1D.
        If inputs have different lengths.

    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise TypeError('Inputs are not numpy.ndarray.')
    if not (x.ndim == 1 and y.ndim == 1):
        raise ValueError('Inputs are not 1D.')
    if not len(x) == len(y):
        raise ValueError('Inputs have different lengths.')
