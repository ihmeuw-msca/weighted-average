"""Smoothing dimension specifications.

Dimension class to specify smoothing dimension column name(s) and
kernel function.

TODO:
* Equality is currently based on self._dimension == other.dimension,
  but there could also be instances where their `dimension` attributes
  intersect (e.g., self._dimension = ['age_mid', 'year_id'] and
  other.dimension = ['age_mid', 'location_id']). Do we want to add a
  check for intersections?

"""
from typing import Any, List, Union

from weave.kernels import Kernel

from weave.utils import as_list


class Dimension:
    """Smoothing dimension specifications.

    Attributes
    ----------
    dimension : str or list of str
        Dimension column name(s).
    kernel : weave.kernels.Kernel
        Kernel function.

    """

    def __init__(self, dimension: Union[str, List[str]], kernel: Kernel) \
            -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        dimension : str or list of str
            Dimension column name(s).
        kernel : weave.kernels.Kernel
            Kernel function.

        """
        self.dimension = dimension
        self.kernel = kernel

    @property
    def dimension(self) -> Union[str, List[str]]:
        """Get dimension column name(s).

        Returns
        -------
        str or list of str
            Dimension column name(s).

        """
        if len(self._dimension) == 1:
            return self._dimension[0]
        return self._dimension

    @dimension.setter
    def dimension(self, dimension: Union[str, List[str]]) -> None:
        """Set dimension column name(s).

        Parameters
        ----------
        dimension : str or list of str
            Dimension column name(s).

        Raises
        ------
        TypeError
            If `dimension` not a str or list of str.
        ValueError
            If duplicates found in `dimension`.

        """
        # Check types
        dimension = as_list(dimension)
        if len(dimension) == 0:
            raise TypeError('`dimension` is an empty list.')
        if not all(isinstance(dim, str) for dim in dimension):
            raise TypeError('Invalid type(s) in `dimension`.')

        # Check duplicates
        if len(dimension) > len(set(dimension)):
            raise ValueError('Duplicates found in `dimension`.')
        self._dimension = dimension

    @property
    def kernel(self) -> Kernel:
        """Get kernel function.

        Returns
        -------
        weave.kernels.Kernel
            Kernel function.

        """
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: Kernel) -> None:
        """Set kernel function.

        Parameters
        ----------
        kernel : weave.kernels.Kernel
            Kernel function.

        Raises
        ------
        TypeError
            If `kernel` is not a kernel function.

        """
        if not isinstance(kernel, Kernel):
            raise TypeError(f"Invalid type for `kernel`: {type(kernel)}.")
        self._kernel = kernel

    def __eq__(self, other: Any) -> bool:
        """Check dimension equality.

        Equality determined by `dimension` attribute only; `kernel`
        is ignored. Used by Smoother class to check for duplicate
        dimensions.

        Parameters
        ----------
        other : Any
            Variable to be compared against.

        Returns
        -------
        bool
            Whether or not `self.dimension` equals `other.dimension`.

        """
        if not isinstance(other, Dimension):
            return False
        return other.dimension == self.dimension
