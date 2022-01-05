"""General utility functions."""
from typing import Any, List, Union


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
    if not isinstance(values, list):
        raise TypeError('`values` is not a list.')
    if len(values) == 0:
        return values
    if isinstance(values[0], list):
        return flatten(values[0]) + flatten(values[1:])
    return values[:1] + flatten(values[1:])
