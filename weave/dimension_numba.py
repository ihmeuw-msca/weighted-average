"""Smoothing dimension specifications.

Dimension class to specify smoothing dimension column name(s), distance
function, and kernel function.

TODO:
* Add checks to kernel parameters based on kernel

"""
from typing import Dict, List, Union
import warnings

from weave.utils import as_list


class Dimension:
    """Smoothing dimension specifications.

    Attributes
    ----------
    dimension : str or list of str
        Dimension column name(s).
    kernel : {'exponential', 'tricubic', 'depth'}
        Kernel function name.
    kernel_pars : dict of {str: int or float}
        Kernel function parameters.
    distance : {'continuous', 'euclidean', 'hierarchical'}
        Distance function name.

    """

    def __init__(self, dimension: Union[str, List[str]], kernel: str,
                 kernel_pars: Dict[str, Union[str, float]],
                 distance: str = None) -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        dimension : str or list of str
            Dimension column name(s).
        kernel : {'exponential', 'tricubic', 'depth'}
            Kernel function name.
        kernel_pars : dict of {str: int or float}
            Kernel function parameters.
        distance : {'continuous', 'euclidean', 'hierarchical'}, optional
            Distance function name.

        """
        self._dimension = dimension
        self._kernel = kernel
        self._kernel_pars = kernel_pars
        if distance is None:
            distance = 'hierarchical' if kernel == 'depth' else 'continuous'
        self._distance = distance

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
            If `dimension` contains duplicates.

        """
        # Check types
        dimension = as_list(dimension)
        empty_list = len(dimension) == 0
        not_all_str = not all(isinstance(dim, str) for dim in dimension)
        if empty_list or not_all_str:
            raise TypeError('`dimension` contains invalid type(s).')

        # Check duplicates
        if len(dimension) > len(set(dimension)):
            raise ValueError('`dimension` contains duplicates.')
        self._dimension = dimension

    @property
    def kernel(self) -> str:
        """Get kernel function name.

        Returns
        -------
        str
            Kernel function name.

        """
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: str) -> None:
        """Set kernel function name.

        Parameters
        ----------
        kernel : str
            Kernel function name.

        Raises
        ------
        TypeError
            If `kernel` not a str.
        ValueError
            If `kernel` is not a valid kernel function.

        """
        if not isinstance(kernel, str):
            raise TypeError('`kernel` is not a str.')
        if kernel not in ('exponential', 'tricubic', 'depth'):
            raise ValueError('`kernel` is not a valid kernel function.')
        if hasattr(self, 'kernel_pars'):
            warnings.warn('`kernel` has changed; must reset `kernel_pars`.')
            del self.kernel_pars
        if hasattr(self, 'distance'):
            warnings.warn('ChEcK dIsTaNcE!')
        self._kernel = kernel

    @property
    def kernel_pars(self) -> Dict[str, Union[int, float]]:
        """Get kernel function parameters.

        Returns
        -------
        dict of {str: int or float}
            Kernel function parameters.

        """
        return self._kernel_pars

    @kernel_pars.setter
    def kernel_pars(self, kernel_pars: Dict[str, Union[int, float]]) -> None:
        """Set kernel function parameters.

        Parameters
        ----------
        kernel_pars : dict of {str: int or float}
            Kernel function parameters.

        Raises
        ------
        All sorts of things!

        """
        # checks
        self._kernel_pars = kernel_pars

    @kernel_pars.deleter
    def kernel_pars(self) -> None:
        """Delete kernel function."""
        del self._kernel_pars

    @property
    def distance(self) -> str:
        """Get distance function name.

        Returns
        -------
        str
            Distance function name.

        """
        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        """Set distance function name.

        Parameters
        ----------
        distance : {'continuous', 'euclidean', 'hierarchical'}
            Distance function name.

        Raises
        ------
        TypeError
            If `distance` is not a str.
        ValueError
            If `distance` is not a valid distance function.

        """
        if not isinstance(distance, str):
            raise TypeError('`distance` is not a str.')
        if distance not in ('continuous', 'euclidean', 'hierarchical'):
            msg = '`distance` is not a valid distance function.'
            raise ValueError(msg)
        if self.kernel == 'depth' and distance != 'hierarchical':
            msg = "`kernel` == 'depth' but `distance` != 'hierarchical'. "
            msg += "Using 'hierarchical' instead."
            warnings.warn(msg)
            distance = 'hierarchical'
        self._distance = distance
