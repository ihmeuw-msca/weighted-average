# pylint: disable=E0611, R0902
"""Smoothing dimension specifications.

Dimension class to specify smoothing dimension column name(s), distance
function, and kernel function.

TODO:
* Update tests
* Convert all ints to floats

"""
from typing import Dict, List, Union
import warnings

from numba.typed import Dict as NumbaDict
import numpy as np

from weave.utils import as_list


class Dimension:
    """Smoothing dimension specifications.

    Attributes
    ----------
    dimension : str or list of str
        Dimension column name(s).
    kernel : {'exponential', 'tricubic', 'depth'}
        Kernel function name.
    pars : numba dict of {str: float}
        Kernel function parameters.
    distance : {'continuous', 'euclidean', 'hierarchical'}
        Distance function name.

    """

    def __init__(self, dimension: Union[str, List[str]], kernel: str,
                 pars: Dict[str, Union[int, float]], distance: str = None) \
            -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        dimension : str or list of str
            Dimension column name(s).
        kernel : {'exponential', 'tricubic', 'depth'}
            Kernel function name.
        pars : dict of {str: int or float}
            Kernel function parameters.
        distance : {'continuous', 'euclidean', 'hierarchical'}, optional
            Distance function name.

        Distance function defaults
        --------------------------
        `kernel` : {'exponential', 'tricubic'}
            `dimension` : str
                `distance` : 'continuous'
            `dimension` : list of str
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
        self.dimension = dimension
        self.kernel = kernel
        self.pars = pars
        self.distance = distance

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
        AttributeError
            If `dimension` has already been set.
        TypeError
            If `dimension` not a str or list of str.
        ValueError
            If `dimension` contains duplicates.

        """
        # Once set, `dimension` cannot be changed
        if hasattr(self, 'dimension'):
            raise AttributeError('`dimension` cannot be changed.')

        # Check types
        dimension = as_list(dimension)
        if len(dimension) == 0:
            raise TypeError('`dimension` is an empty list.')
        if not all(isinstance(dim, str) for dim in dimension):
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
            Attribute `pars` is deleted when `kernel` is reset, so
            `pars` must also be reset.
        UserWarning
            If current `distance` attribute is not a valid distance
            function for `kernel`. Default `distance` used instead.

        """
        # Check type
        if not isinstance(kernel, str):
            raise TypeError('`kernel` is not a str.')

        # Check value
        if kernel not in ('exponential', 'tricubic', 'depth'):
            raise ValueError('`kernel` is not a valid kernel function.')

        # Delete kernel parameters
        if hasattr(self, 'pars'):
            warnings.warn('`kernel` has changed; must reset `pars`.')
            del self._pars

        self._kernel = kernel

        # Check distance
        if hasattr(self, 'distance'):
            self.distance = self.distance

    @property
    def pars(self) -> NumbaDict[str, float]:
        """Get kernel function parameters.

        Returns
        -------
        numba dict of {str: float}
            Kernel function parameters.

        """
        return self._pars

    @pars.setter
    def pars(self, pars: Dict[str, Union[int, float]]) -> None:
        """Set kernel function parameters.

        Parameters
        ----------
        pars : dict of {str: int or float}
            Kernel function parameters.

        """
        if self._kernel == 'exponential':
            self.check_pars(pars, 'radius', 'pos_num')
            pars = {'radius': pars['radius']}
        elif self._kernel == 'tricubic':
            self.check_pars(pars, ['radius', 'exponent'], 'pos_num')
            pars = {key: pars[key] for key in ['radius', 'exponent']}
        else:  # 'depth'
            self.check_pars(pars, 'radius', 'pos_frac')
            pars = {'radius': pars['radius']}
        self._pars = NumbaDict()
        for key in pars:
            self._pars[key] = float(pars[key])

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
    def distance(self, distance: Union[str, None]) -> None:
        """Set distance function name.

        Parameters
        ----------
        distance : {'continuous', 'euclidean', 'hierarchical', None}
            Distance function name.

        Raises
        ------
        TypeError
            If `distance` is not a str or None.
        ValueError
            If `distance` is not a valid distance function.

        Warns
        -----
        UserWarning
            If `distance` is not a valid distance function for current
            `dimension` and `kernel` attributes. Default used instead.

        """
        # Set defaults
        if distance is None:
            if self._kernel == 'depth':
                distance = 'hierarchical'
            else:
                if len(self._dimension) == 1:
                    distance = 'continuous'
                else:
                    distance = 'euclidean'

        # Check type
        if not isinstance(distance, str):
            raise TypeError('`distance` is not a str.')

        # Check value
        if distance not in ('continuous', 'euclidean', 'hierarchical'):
            msg = '`distance` is not a valid distance function.'
            raise ValueError(msg)

        # Check kernel and dimension
        if self._kernel == 'depth' and distance != 'hierarchical':
            msg = "`kernel` == 'depth' but `distance` != 'hierarchical'. "
            msg += "Using 'hierarchical' instead."
            warnings.warn(msg)
            distance = 'hierarchical'
        else:
            if len(self._dimension) > 1 and distance == 'continuous':
                msg = "`dimension` is a list of str but `distance` == "
                msg += "'continuous'. Using 'euclidean' instead."
                warnings.warn(msg)
                distance = 'euclidean'

        self._distance = distance

    @staticmethod
    def check_pars(pars: Dict[str, Union[int, float]],
                   names: Union[str, List[str]],
                   types: Union[str, List[str]]) -> None:
        """Check parameter types and values.

        Parameters
        ----------
        pars : dict of {str: int or float}
            Kernel parameters
        names : str or list of str
            Parameter names.
        types : str or list of str
            Parameter types. Valid types are 'pos_num' or 'pos_frac'.

        Raises
        ------
        KeyError
            If `pars` is missing a kernel parameter.
        TypeError
            If a kernel parameter is an invalid type.
        ValueError
            If a kernel parameter is an invalid value.

        """
        names = as_list(names)
        if isinstance(types, str):
            types = [types]*len(names)

        for idx_par, par_name in enumerate(names):
            # Check key
            if par_name not in pars:
                raise KeyError(f"`{par_name}` is not in `pars`.")
            par_val = pars[par_name]

            if types[idx_par] == 'pos_num':
                # Check type
                is_bool = isinstance(par_val, bool)
                is_int = isinstance(par_val, (int, np.integer))
                is_float = isinstance(par_val, (float, np.floating))
                if is_bool or not (is_int or is_float):
                    raise TypeError(f"`{par_name}` is not an int or float.")

                # Check value
                if par_val <= 0.0:
                    raise ValueError(f"`{par_name}` is not positive.")

            else:  # 'pos_frac'
                # Check type
                if not isinstance(par_val, (float, np.floating)):
                    raise TypeError(f"`{par_name}` is not a float.")

                # Check value
                if par_val <= 0.0 or par_val >= 1.0:
                    raise ValueError(f"`{par_name}` is not in (0, 1).")
