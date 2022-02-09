# pylint: disable=C0103, E0611, W1401
"""Calculate the distance between points.

Distance functions to calculate the distances between the current point
and a vector of nearby points. While points can be either scalars or
vectors, scalars must be cast as 1D vectors to comply with `Numba
<https://numba.readthedocs.io/en/stable/index.html>`_ just-in-time
compilation.

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
from numba.types import float64, UniTuple  # type: ignore
import numpy as np

from weave.utils import is_number

number = Union[int, float]
DistanceDict = Dict[Tuple[number, number], number]


@njit
def euclidean(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Get Euclidean distances between `x` and `Y`.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    Y : 2D numpy.ndarray of float
        Matrix of nearby points.

    Returns
    -------
    1D numpy.ndarray of nonnegative float
        Euclidean distances between `x` and `Y`.

    Notes
    -----
    For a pair of 1D points, this function computes the absolute value
    of their difference:

    .. math:: d(x, y) = |x - y|

    For a pair of ND points, this function computes the *n*-dimensional
    Euclidean distance [2]_:

    .. math:: d(x, y) = \\sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \\dots +
              (x_n - y_n)^2}

    References
    ----------
    .. [2] `Euclidean distance
           <https://en.wikipedia.org/wiki/Euclidean_distance>`_

    Examples
    --------
    >>> import numpy as np
    >>> from weave.distance import euclidean
    >>> x = np.array([0.])
    >>> Y = np.array([[-1.], [0.], [1.]])
    >>> euclidean(x, Y)
    array([1., 0., 1.])

    >>> import numpy as np
    >>> from weave.distance import euclidean
    >>> x = np.array([0., 0.])
    >>> Y = np.array([[-1., -1.], [0., 0.], [1., 1.]])
    >>> euclidean(x, Y)
    array([1.41421356, 0., 1.41421356])

    """
    # Scalars
    if len(x) == 1:
        return 1.0*np.abs(x - Y).flatten()

    # Vectors
    distance = np.empty(len(Y))
    for ii, y in enumerate(Y):
        distance[ii] = 1.0*np.linalg.norm(x - y)

    return distance


@njit
def hierarchical(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Get hierarchical distances between `x` and `Y`.

    Points are specified as a vector of IDs corresponding to nodes in a
    tree, starting from the root node and ending at a leaf node.
    The distance between two points is defined as the number of edges
    between the leaf nodes and their nearest common parent node.

    Parameters
    ----------
    x : 1D numpy.ndarray of float
        Current point.
    Y : 2D numpy.ndarray of float
        Matrix of nearby points.

    Returns
    -------
    1D numpy.ndarray of nonnegative float
        Hierarchical distances between `x` and `Y`.

    Examples
    --------
    .. image:: images/hierarchy.png

    >>> import numpy as np
    >>> from weave.distance import hierarchical
    >>> x = np.array([1., 2., 4.])
    >>> Y = np.array([[1., 2., 4.], [1., 2., 5.], [1., 3., 6.]])
    >>> hierarchical(x, Y)
    array([0., 1., 2.])

    """
    # Get distance from one nearby point
    def get_distance(x, y):
        if (x == y).all():
            return 0.0
        for ii in range(1, len(x)):
            if (x[:-ii] == y[:-ii]).all():
                return 1.0*ii
        return 1.0*len(x)

    # Get distance between all nearby points
    distance = np.empty(len(Y))
    for ii, y in enumerate(Y):
        distance[ii] = get_distance(x, y)

    return distance


@njit
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
    distance_dict : numba.typed.Dict of {(float64, float64): float64}
        Typed dictionary of distances between points.

    Returns
    -------
    1D numpy.ndarray of nonnegative float
        Dictionary distances between `x` and `Y`.

    See Also
    --------
    get_typed_dict

    Notes
    -----
    Because this is a Numba just-in-time function, the parameter
    `distance_dict` must be of type `numba.typed.Dict
    <https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict>`_.
    Dictionaries can be cast as a typed dictionary using the function
    :func:`get_typed_dict`, but values are not checked for validity.

    Examples
    --------
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
    array([0., 1., 2.])

    """
    # Get distance from one nearby point
    def get_distance(x, y):
        x0 = float(x[0])
        y0 = float(y[0])
        if x0 <= y0:
            return distance_dict[(x0, y0)]
        return distance_dict[(y0, x0)]

    # Get distance from all nearby points
    distance = np.empty(len(Y))
    for ii, y in enumerate(Y):
        distance[ii] = get_distance(x, y)

    return distance


def get_typed_dict(distance_dict: Optional[DistanceDict] = None) \
        -> Dict[Tuple[float, float], float]:
    """Get typed version of `distance_dict`.

    Parameters
    ----------
    distance_dict : dict of {(number, number): number}, optional
        Dictionary of distances between points.

    Returns
    -------
    : numba.typed.Dict of {(float64, float64): float64}
        Typed version of `distance_dict`.

    See Also
    --------
    `numba.typed.Dict
    <https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict>`_

    Examples
    --------
    >>> from weave.distance import get_typed_dict
    >>> distance_dict = {(1, 1): 0}
    >>> get_typed_dict(distance_dict)
    DictType[UniTuple(float64 x 2),float64]<iv=None>({(1.0, 1.0): 0.0})

    """
    typed_dict = TypedDict.empty(
        key_type=UniTuple(float64, 2),
        value_type=float64
    )
    if distance_dict is not None:
        for key in distance_dict:
            float_key = tuple(float(point) for point in key)
            typed_dict[float_key] = float(distance_dict[key])
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
