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

    """
    if not isinstance(values, (list, TypedList)):
        raise TypeError('`values` is not a list.')
    if len(values) == 0:
        return list(values)
    if isinstance(values[0], (list, TypedList)):
        return flatten(values[0]) + flatten(values[1:])
    return list(values[:1]) + flatten(values[1:])


def is_numeric(value: Any) -> bool:
    """Determine if `value` is an int or float.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        If `value` is an int or float.

    """
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return True
    if isinstance(value, (float, np.floating)):
        if not (np.isnan(value) or np.isinf(value)):
            return True
    return False
