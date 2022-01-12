# pylint: disable=E0611, R0902, R0913
"""Smoothing dimension specifications.

Dimension class to specify smoothing dimension column name(s), distance
function, and kernel function.

TODO:
* Update methods and functions for new `dictionary` distance function

"""
from typing import Dict, List, Optional, Union
import warnings

from numba.typed import Dict as TypedDict

from weave.kernels import check_pars
from weave.utils import as_list


class Dimension:
    """Smoothing dimension specifications.

    Attributes
    ----------
    name : str
        Dimension name.
    columns : list of str
        Dimension column name(s).
    kernel : {'exponential', 'tricubic', 'depth'}
        Kernel function name.
    kernel_pars : dict of {str: float}
        Kernel function parameters.
    distance : {'euclidean', 'hierarchical'}
        Distance function name.

    """

    def __init__(self, name: str, columns: Union[str, List[str]], kernel: str,
                 kernel_pars: Dict[str, Union[int, float]],
                 distance: Optional[str] = None) -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        name : str
            Dimension name.
        columns : str or list of str
            Dimension column name(s).
        kernel : {'exponential', 'tricubic', 'depth'}
            Kernel function name.
        kernel_pars : dict of {str: int or float}
            Kernel function parameters.
        distance : {'euclidean', 'hierarchical'}, optional
            Distance function name.

        Distance function defaults
        --------------------------
        `kernel` : {'exponential', 'tricubic'}
            `distance` : 'euclidean'
        `kernel` : 'depth'
            `distance` : 'hierarchical'

        Kernel function parameters
        --------------------------
        `kernel` : 'exponential'
            `radius` : positive int or float
        `kernel` : 'tricubic'
            `radius` : positive int or float
            `exponent` : positive int or float
        `kernel` : 'depth'
            `radius` : float in (0, 1)

        """
        self.name = name
        self.columns = columns
        self.kernel = kernel
        self.kernel_pars = kernel_pars
        self.distance = distance

    @property
    def name(self) -> str:
        """Get dimension name.

        Returns
        -------
        str
            Dimension name.

        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set dimension name.

        Parameters
        ----------
        name : str
            Dimension name.

        Raises
        ------
        AttributeError
            If `name` has already been set.
        TypeError
            If `name` is not a str.

        """
        # Once set, `name` cannot be changed
        if hasattr(self, 'name'):
            raise AttributeError('`name` cannot be changed.')

        # Check type
        if not isinstance(name, str):
            raise TypeError('`name` is not a str.')

        self._name = name

    @property
    def columns(self) -> List[str]:
        """Get dimension column name(s).

        Returns
        -------
        list of str
            Dimension column name(s).

        """
        return self._columns

    @columns.setter
    def columns(self, columns: Union[str, List[str]]) -> None:
        """Set dimension column name(s).

        Parameters
        ----------
        columns : str or list of str
            Dimension column name(s).

        Raises
        ------
        AttributeError
            If `columns` has already been set.
        TypeError
            If `columns` not a str or list of str.
        ValueError
            If `columns` contains duplicates.

        """
        # Once set, `columns` cannot be changed
        if hasattr(self, 'columns'):
            raise AttributeError('`columns` cannot be changed.')

        # Check types
        columns = as_list(columns)
        if len(columns) == 0:
            raise TypeError('`columns` is an empty list.')
        if not all(isinstance(dim, str) for dim in columns):
            raise TypeError('`columns` contains invalid type(s).')

        # Check duplicates
        if len(columns) > len(set(columns)):
            raise ValueError('`columns` contains duplicates.')

        self._columns = columns

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
        kernel : {'exponential', 'tricubic', 'depth'}
            Kernel function name.

        Raises
        ------
        TypeError
            If `kernel` not a str.
        ValueError
            If `kernel` is not a valid kernel function.

        Warns
        -----
        UserWarning
            Attribute `kernel_pars` is deleted when `kernel` is reset,
            so `kernel_pars` must also be reset.

        """
        # Check type
        if not isinstance(kernel, str):
            raise TypeError('`kernel` is not a str.')

        # Check value
        if kernel not in ('exponential', 'tricubic', 'depth'):
            raise ValueError('`kernel` is not a valid kernel function.')

        # Delete kernel parameters
        if hasattr(self, 'kernel_pars'):
            warnings.warn('`kernel` has changed; must reset `kernel_pars`.')
            del self._kernel_pars

        self._kernel = kernel

    @property
    def kernel_pars(self) -> Dict[str, float]:
        """Get kernel function parameters.

        Returns
        -------
        dict of {str: float}
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

        """
        # Check parameter values
        if self._kernel == 'exponential':
            check_pars(kernel_pars, 'radius', 'pos_num')
            kernel_pars = {'radius': kernel_pars['radius']}
        elif self._kernel == 'tricubic':
            check_pars(kernel_pars, ['radius', 'exponent'], 'pos_num')
            kernel_pars = {key: kernel_pars[key]
                           for key in ['radius', 'exponent']}
        else:  # 'depth'
            check_pars(kernel_pars, 'radius', 'pos_frac')
            kernel_pars = {'radius': kernel_pars['radius']}

        # Create numba dictionary
        self._kernel_pars = TypedDict()
        for key in kernel_pars:
            self._kernel_pars[key] = float(kernel_pars[key])

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
    def distance(self, distance: Optional[str]) -> None:
        """Set distance function name.

        Parameters
        ----------
        distance : {'euclidean', 'hierarchical', None}
            Distance function name.

        Raises
        ------
        TypeError
            If `distance` is not a str or None.
        ValueError
            If `distance` is not a valid distance function.

        """
        # Set defaults
        if distance is None:
            if self._kernel == 'depth':
                distance = 'hierarchical'
            else:
                distance = 'euclidean'

        # Check type
        if not isinstance(distance, str):
            raise TypeError('`distance` is not a str.')

        # Check value
        if distance not in ('euclidean', 'hierarchical'):
            msg = '`distance` is not a valid distance function.'
            raise ValueError(msg)

        self._distance = distance
