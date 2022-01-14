# pylint: disable=E0611, R0902, R0903, R0913
"""Smoothing dimension specifications.

Dimension class to specify smoothing dimension column name(s), distance
function, and kernel function.

TODO:
* Write tests for new changes and new class
* Fix mypy errors

"""
from typing import Dict, List, Optional, Tuple, Union

from numba.experimental import jitclass
from numba.typed import Dict as TypedDict, List as TypedList
from numba.types import DictType, float64, ListType, unicode_type, UniTuple

from weave.distance import check_dict
from weave.kernels import check_pars
from weave.utils import as_list

Numeric = Union[str, float]
DistanceDict = Dict[Tuple[Numeric, Numeric], Numeric]
TypedDistanceDict = TypedDict[UniTuple(float64, 2), float64]


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
    distance : {'dictionary', 'euclidean', 'hierarchical'}
        Distance function name.
    distance_dict : dict of {(float, float): float}
        Dictionary of distances between points if
        `distance` == 'dictionary'.

    """

    def __init__(self, name: str, columns: Union[str, List[str]], kernel: str,
                 kernel_pars: Dict[str, Numeric],
                 distance: Optional[str] = None,
                 distance_dict: Optional[DistanceDict] = None) -> None:
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
        distance : {'dictionary', 'euclidean', 'hierarchical'}, optional
            Distance function name.
        distance_dict : dict of {(numeric, numeric): numeric}, optional
            Dictionary of distance between points if
            `distance` == 'dictionary'.

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

        Dictionary `distance_dict` contains the distance between points
        `x` and `y`. Dictionary keys are tuples of point pairs
        `(x, y)`, where `x` and `y` are floats (e.g., location IDs),
        and dictionary values are the corresponding distances. Because
        distances are assumed to be symmetric, point pairs are listed
        from smallest to largest, e.g., `x` <= `y`.

        """
        self.name = name
        self.columns = columns
        self.kernel = kernel
        self.kernel_pars = kernel_pars
        self.distance = distance
        self.distance_dict = distance_dict

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
    def columns(self) -> TypedList[unicode_type]:
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
        if not all(isinstance(col, str) for col in columns):
            raise TypeError('`columns` contains invalid types.')

        # Check duplicates
        if len(columns) > len(set(columns)):
            raise ValueError('`columns` contains duplicates.')

        self._columns = TypedList(columns)

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
        AttributeError
            If `kernel` has already been set.
        TypeError
            If `kernel` not a str.
        ValueError
            If `kernel` is not a valid kernel function.

        """
        # Once set, `kernel` cannot be changed
        if hasattr(self, 'kernel'):
            raise AttributeError('`kernel` cannot be changed.')

        # Check type
        if not isinstance(kernel, str):
            raise TypeError('`kernel` is not a str.')

        # Check value
        if kernel not in ('exponential', 'tricubic', 'depth'):
            raise ValueError('`kernel` is not a valid kernel function.')

        self._kernel = kernel

    @property
    def kernel_pars(self) -> TypedDict[unicode_type, float64]:
        """Get kernel function parameters.

        Returns
        -------
        dict of {str: float}
            Kernel function parameters.

        """
        return self._kernel_pars

    @kernel_pars.setter
    def kernel_pars(self, kernel_pars: Dict[str, Numeric]) -> None:
        """Set kernel function parameters.

        Parameters
        ----------
        kernel_pars : dict of {str: numeric}
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
        distance : {'dictionary', 'euclidean', 'hierarchical', None}
            Distance function name.

        Raises
        ------
        AttributeError
            If `distance` has already been set.
        TypeError
            If `distance` is not a str or None.
        ValueError
            If `distance` is not a valid distance function.

        """
        # Once set, `distance` cannot be changed
        if hasattr(self, 'distance'):
            raise AttributeError('`distance` cannot be changed.')

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
        if distance not in ('dictionary', 'euclidean', 'hierarchical'):
            msg = '`distance` is not a valid distance function.'
            raise ValueError(msg)

        self._distance = distance

    @property
    def distance_dict(self) -> TypedDistanceDict:
        """Get dictionary of distances between points.

        Returns
        -------
        dict of {(float, float): float}
            Dictionary of distances between points.

        """
        return self._distance_dict

    @distance_dict.setter
    def distance_dict(self, distance_dict: Optional[DistanceDict]) -> None:
        """Set dictionary of distances between points.

        Parameters
        ----------
        distance_dict : dict of {(numeric, numeric): numeric}
            Dictionary of distances between points.

        Raises
        ------
        AttributeError
            If `distance_dict` has already been set.
        ValueError
            If `distance` == 'dictionary' but `distance_dict` is None.

        """
        # Once set, `distance_dict` cannot be changed
        if hasattr(self, 'distance_dict'):
            raise AttributeError('`distance_dict` cannot be changed.')

        # Create numba dictionary
        self._distance_dict = TypedDict.empty(
            key_type=UniTuple(float64, 2),
            value_type=float64
        )
        if self._distance == 'dictionary':
            if distance_dict is None:
                msg = "`distance` == 'dictionary', "
                msg += "but `distance_dict` is None."
                raise ValueError(msg)
            check_dict(distance_dict)
            for key in distance_dict:
                float_key = tuple(float(point) for point in key)
                self._distance_dict[float_key] = float(distance_dict[key])


@jitclass([('name', unicode_type),
           ('columns', ListType(unicode_type)),
           ('kernel', unicode_type),
           ('kernel_pars', DictType(unicode_type, float64)),
           ('distance', unicode_type),
           ('distance_dict', DictType(UniTuple(float64, 2), float64))])
class TypedDimension:
    """class docstring"""
    def __init__(self, name: unicode_type, columns: TypedList[unicode_type],
                 kernel: unicode_type,
                 kernel_pars: TypedDict[unicode_type, float64],
                 distance: unicode_type,
                 distance_dict: TypedDistanceDict) -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        name : str
            Dimension name.
        columns : list of str
            Dimension column name(s).
        kernel : {'exponential', 'tricubic', 'depth'}
            Kernel function name.
        kernel_pars : dict of {str: float}
            Kernel function parameters.
        distance : {'dictionary', 'euclidean', 'hierarchical'}
            Distance function name.
        distance_dict : dict of {(float, float): float}
            Dictionary of distance between points if
            `distance` == 'dictionary'.

        """
        self.name = name
        self.columns = columns
        self.kernel = kernel
        self.kernel_pars = kernel_pars
        self.distance = distance
        self.distance_dict = distance_dict
