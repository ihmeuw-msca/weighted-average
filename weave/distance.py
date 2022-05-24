# pylint: disable=C0103, E0611, W1401
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
from typing import Dict, Optional, Tuple, Union

from numba import njit  # type: ignore
from numba.typed import Dict as TypedDict  # type: ignore
from numba.types import float32, UniTuple  # type: ignore
import numpy as np

from weave.utils import is_number

number = Union[int, float]
DistanceDict = Dict[Tuple[number, number], number]


def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    """Get Euclidean distance between `x` and `y`.

    Points `x` and `y` should have the same length.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    y : 1D numpy.ndarray of float
        Nearby point.

    Returns
    -------
    nonnegative float32
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
    Get Euclidean distance between points.

    >>> import numpy as np
    >>> from weave.distance import euclidean
    >>> euclidean(np.array([0., 0.]), np.array([1., 1.]))
    1.4142135

    """
    return np.linalg.norm(x - y).astype(np.float32)


def dictionary(x: np.ndarray, Y: np.ndarray,
               distance_dict: Dict[Tuple[float, float], float]) -> np.ndarray:
    """Get dictionary distances between `x` and `Y`.

    Returns user-defined distances between points `x` and `Y`,
    specified in the dictionary `distance_dict`. Dictionary keys are
    tuples of point ID pairs `(x, y)`, and dictionary values are the
    corresponding distances. Because distances are assumed to be
    symmetric, point IDs are listed from smallest to largest, e.g.
    `x` :math:`\\leq` `y`.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    Y : 2D numpy.ndarray of float
        Matrix of nearby points.
    distance_dict : numba.typed.Dict of {(float32, float32): float32}
        Typed dictionary of distances between points.

    Returns
    -------
    1D numpy.ndarray of nonnegative float32
        Dictionary distances between `x` and `Y`.

    Notes
    -----
    Because this is a Numba just-in-time function, the parameter
    `distance_dict` must be of type `numba.typed.Dict
    <https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict>`_.
    Dictionaries can be cast as a typed dictionary using the function
    :func:`get_typed_dict`, but values are not checked for validity.

    Examples
    --------
    Get user-defined distances between points based on scalar point
    IDs.

    >>> import numpy as np
    >>> from weave.distance import dictionary, get_typed_dict
    >>> distance_dict = {
            (4, 4): 0,
            (4, 5): 1,
            (4, 6): 2,
            (5, 6): 2
        }
    >>> typed_dict = get_typed_dict(distance_dict)
    >>> x = np.array([4.])
    >>> y = np.array([[4.], [5.], [6.]])
    >>> dictionary(x, y, typed_dict)
    array([0., 1., 2.], dtype=float32)

    """
    # Get distance from one nearby point
    def get_distance(x, y):
        x0 = np.float32(x[0])
        y0 = np.float32(y[0])
        if x0 <= y0:
            return distance_dict[(x0, y0)]
        return distance_dict[(y0, x0)]

    # Get distance from all nearby points
    return np.array([get_distance(x, y) for y in Y], dtype=np.float32)


def tree(x: np.ndarray, y: np.ndarray) -> float:
    """Get tree distance between `x` and `y`.

    Points are specified as a vector of IDs corresponding to nodes in a
    tree, starting from the root node and ending at a leaf node.
    The distance between two points is defined as the number of edges
    between the leaf nodes and their nearest common parent node.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    Y : 1D numpy.ndarray of float
        Nearby point.

    Returns
    -------
    nonnegative float32
        Tree distance between `x` and `y`.

    Examples
    --------
    Get tree distances between leaf nodes from the following tree.

    .. image:: images/hierarchy.png

    >>> import numpy as np
    >>> from weave.distance import tree
    >>> tree(np.array([1., 2., 4.]), np.array([1., 2., 4.]))
    0.
    >>> tree(np.array([1., 2., 4.]), np.array([1., 2., 5.]))
    1.
    >>> tree(np.array([1., 2., 4.]), np.array([1., 3., 6.]))
    2.
    >>> tree(np.array([1., 2., 4.]), np.array([7., 8., 9.]))
    3.

    """
    if (x == y).all():
        return np.float32(0)
    for ii in range(1, len(x)):
        if (x[:-ii] == y[:-ii]).all():
            return np.float32(ii)
    return np.float32(len(x))


def get_typed_dict(distance_dict: Optional[DistanceDict] = None) \
        -> Dict[Tuple[float, float], float]:
    """Get typed version of `distance_dict`.

    Parameters
    ----------
    distance_dict : dict of {(number, number): number}, optional
        Dictionary of distances between points.

    Returns
    -------
    : numba.typed.Dict of {(float32, float32): float32}
        Typed version of `distance_dict`.

    Examples
    --------
    Cast a dictionary as an instance of `numba.typed.Dict
    <https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict>`_.

    >>> from weave.distance import get_typed_dict
    >>> distance_dict = {(1, 1): 0}
    >>> get_typed_dict(distance_dict)
    DictType[UniTuple(float32 x 2),float32]<iv=None>({(1.0, 1.0): 0.0})

    """
    typed_dict = TypedDict.empty(
        key_type=UniTuple(float32, 2),
        value_type=float32
    )
    if distance_dict is not None:
        for key in distance_dict:
            float_key = tuple(np.float32(point) for point in key)
            typed_dict[float_key] = np.float32(distance_dict[key])
    return typed_dict


def _check_dict(distance_dict: Dict[Tuple[number, number], number]) -> None:
    """Check dictionary keys and values.

    Parameters
    ----------
    distance_dict : dict of {(number, number): number}
        Dictionary of distances between points.

    Raises
    ------
    TypeError
        If `distance_dict`, keys, or values are an invalid type.
    ValueError
        If `dictionary_dict` is empty, dictionary keys are not all
        length 2, or dictionary values are not all nonnegative.

    """
    # Check types
    if not isinstance(distance_dict, dict):
        raise TypeError('`distance_dict` is not a dict.')
    if not all(isinstance(key, tuple) for key in distance_dict):
        raise TypeError('`distance_dict` keys not all tuple.')
    if not all(is_number(point) for key in distance_dict for point in key):
        raise TypeError('`distance_dict` key entries not all int or float.')
    if not all(is_number(value) for value in distance_dict.values()):
        raise TypeError('`distance_dict` values not all int or float.')

    # Check values
    if len(distance_dict) == 0:
        raise ValueError('`distance_dict` is an empty dict.')
    if any(len(key) != 2 for key in distance_dict):
        raise ValueError('`distance_dict` keys are not all length 2.')
    if any(value < 0.0 for value in distance_dict.values()):
        raise ValueError('`distance_dict` contains negative values.')
