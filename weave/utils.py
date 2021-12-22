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
