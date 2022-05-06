# pylint: disable=E0611
"""General utility functions."""
from typing import Any, List, Union

from numba.typed import List as TypedList
import numpy as np


def as_list(values: Union[Any, List[Any]]) -> List[Any]:
    """Cast `values` as list if not already.

    Parameters
    ----------
    values : Any
        Either single value or list of values.

    Returns
    -------
    list of Any
        Input `values` as a list.

    Examples
    --------
    Single values are cast as a list, while lists remain unchanged.

    >>> from weave.utils import as_list
    >>> as_list('single_value')
    ['single_value']
    >>> as_list(['list', 'of', 'values'])
    ['list', 'of', 'values']

    """
    if isinstance(values, list):
        return values
    return [values]


def flatten(values: List[Union[Any, List[Any]]]) -> List[Any]:
    """Flatten a list of lists.

    Parameters
    ----------
    values : list of {Any or list of Any}
        List of values.

    Returns
    -------
    list of Any
        Flattened list.

    Examples
    --------
    Returns a flattened version of a nested list.

    >>> from weave.utils import flatten
    >>> flatten([1, [2, [3, [4]]]])
    [1, 2, 3, 4]
    >>> flatten(['age', 'year', ['super_region', 'region', 'country']])
    ['age', 'year', 'super_region', 'region', 'country']

    """
    if not isinstance(values, (list, TypedList)):
        raise TypeError('`values` is not a list.')
    if len(values) == 0:
        return list(values)
    if isinstance(values[0], (list, TypedList)):
        return flatten(values[0]) + flatten(values[1:])
    return list(values[:1]) + flatten(values[1:])


def is_int(value: Any) -> bool:
    """Determine if `value` is an int.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        If `value` is not an int.

    Examples
    --------
    Returns ``True`` for ints but ``False`` otherwise.

    >>> from weave.utils import is_int
    >>> is_int(1)
    True
    >>> is_int(1.0)
    False
    >>> is_int(True)
    False

    """
    is_ints = isinstance(value, (int, np.integer))
    is_bool = isinstance(value, bool)
    return is_ints and not is_bool


def is_float(value: Any) -> bool:
    """Determine if `value` is a float.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        If `value` is a float.

    Examples
    --------
    Returns ``True`` for floats, but ``False`` otherwise.

    >>> from weave.utils import is_float
    >>> is_float(1.0)
    True
    >>> is_float(1)
    False
    >>> is_float(np.nan)
    False
    >>> is_float(np.inf)
    False

    """
    if isinstance(value, (float, np.floating)):
        if not (np.isnan(value) or np.isinf(value)):
            return True
    return False


def is_number(value: Any) -> bool:
    """Determine if `value` is an int or float.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        If `value` is an int or float.

    Examples
    --------
    Returns ``True`` for ints and floats, but ``False`` otherwise.

    >>> from weave.utils import is_number
    >>> is_number(1)
    True
    >>> is_number(1.0)
    True
    >>> is_number(True)
    False
    >>> is_number(np.nan)
    False
    >>> is_number(np.inf)
    False

    """
    return is_int(value) or is_float(value)
