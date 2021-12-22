"""Smooth data across multiple dimensions using weighted averages."""
from typing import List, Union

from weave.dimension import Dimension
from weave.utils import as_list


class Smoother:
    """Smoother function.

    Attributes
    ----------
    dimensions : Dimension or list of Dimension
        Smoothing dimension(s).

    """

    def __init__(self, dimensions: Union[Dimension, List[Dimension]]) -> None:
        """Create smoother function.

        Parameters
        ----------
        dimensions : Dimension or list of Dimension
            Smoothing dimension(s).

        """
        self.dimensions = dimensions

    @property
    def dimensions(self) -> Union[Dimension, List[Dimension]]:
        """Get smoothing dimension(s).

        Returns
        -------
        Dimension or list of Dimension
            Smoothing dimension(s).

        """
        if len(self._dimensions) == 1:
            return self._dimensions[0]
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: Union[Dimension, List[Dimension]]) \
            -> None:
        """Set smoothing dimension(s).

        Parameters
        ----------
        dimensions : Dimension or list of Dimension
            Smoothing dimension(s).

        Raises
        ------
        TypeError
            If `dimensions` is not a Dimension or list of Dimension.
        ValueError
            If duplicates found in `dimensions`.

        """
        # Check types
        dimensions = as_list(dimensions)
        if len(dimensions) == 0:
            raise TypeError('`dimensions` is an empty list.')
        if not all(isinstance(dim, Dimension) for dim in dimensions):
            raise TypeError('Invalid type(s) in `dimensions`.')

        # Check duplicates
        dim_list = [tuple(dim.dimension) for dim in dimensions]
        if len(dim_list) > len(set(dim_list)):
            raise ValueError('Duplicates found in `dimensions`.')
        self._dimensions = dimensions
