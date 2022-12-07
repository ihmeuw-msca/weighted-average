# pylint: disable=C0103, E0611, R0902, R0903, R0912, R0913
"""Smoothing dimension specifications."""
from typing import Dict, List, Optional, Tuple, Union

from numba.experimental import jitclass  # type: ignore
from numba.typed import Dict as TypedDict  # type: ignore
from numba.types import DictType, UniTuple  # type: ignore
from numba.types import boolean, float32, unicode_type  # type: ignore
import numpy as np
from pandas import DataFrame

from weave.distance import euclidean, tree
from weave.kernels import exponential, depth, tricubic
from weave.utils import as_list, is_int, is_float, is_number

number = Union[int, float]
pars = Union[number, bool]
DistanceDict = Dict[Tuple[number, number], number]
WeightDict = Dict[Tuple[float, float], float]


@jitclass([('name', unicode_type),
           ('weight_dict', DictType(UniTuple(float32, 2), float32)),
           ('normalize', boolean)])
class TypedDimension:
    """Smoothing dimension specifications."""
    def __init__(self, name: str, weight_dict: WeightDict, normalize: bool) \
            -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        name : unicode_type
            Dimension name.
        weight_dict : numba.typed.Dict of {(float32, float32): float32}
            Dictionary of dimension smoothing weights.
        normalize : bool
            Normalize dimension weights in groups.

        """
        self.name = name
        self.weight_dict = weight_dict
        self.normalize = normalize


class Dimension:
    """Smoothing dimension specifications.

    Dimension class to specify smoothing dimension column names, kernel
    function, distance function, and relevant parameters.

    Attributes
    ----------
    name : str
        Dimension name.

        Column in data frame containing the ID of points in the given
        dimension. For example, 'age_id', 'year_id', or 'location_id'.

    coordinates : list of str
        Dimension coordinates.

        Column(s) in data frame containing the coordinates of points in
        the given dimension. For example, `['lat', 'lon']` or
        `['super_region', 'region', 'country']`. Can be same as `name`
        attribute if dimension is 1D.

    kernel : {'identity', 'exponential', 'tricubic', 'depth'}
        Kernel function name.

        Name of kernel function to compute smoothing weights.

        See Also
        --------
        weave.kernels

    kernel_pars : dict of {str: number or bool}
        Kernel function parameters.

        Dictionary of kernel function parameters corresponding to
        `kernel` attribute.

    distance : {'euclidean', 'tree', 'dictionary'}
        Distance function name.

        Name of distance function to compute distance between points.

        See Also
        --------
        weave.distance

    distance_dict : dict of {(number, number): number}
        Dictionary of distances between points.

        User-defined dictionary of distances between points if
        `distance` attribute is 'dictionary'. Dictionary keys are
        tuples of point ID pairs, and dictionary values are the
        corresponding distances.

    normalize : bool
        Normalize dimension weights in groups.

        Whether or not the preceding dimension weights should be
        normalized in groups based on the current dimension weight
        values. This corresponds to the CODEm [1]_ framework where the
        product of age and time weights are normalized in groups based
        on the location hierarchy before being multiplied by location
        weights. For example, for points :math:`i, j, k` from the same
        country:

        .. math:: w_{i, j} = w_{\\ell_{i, j}} \\cdot
                  \\frac{w_{a_{i, j}} w_{t_{i, j}}} {\\sum_{k}
                  w_{a_{i, k}} w_{t_{i, k}}}

        This option may be inefficient if there is a large number of
        possible weight values for the given dimension.

    """

    def __init__(self, name: str, coordinates: Union[str, List[str]] = None,
                 kernel: str = 'identity',
                 kernel_pars: Optional[Dict[str, pars]] = None,
                 distance: Optional[str] = None,
                 distance_dict: Optional[DistanceDict] = None,
                 normalize: Optional[bool] = None) -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        name : str
            Dimension name.
        coordinates : str or list of str, optional
            Dimension coordinates, if different from `name`.
        kernel : {'identity', 'exponential', 'tricubic', 'depth'}, optional
            Kernel function name. Default is 'identity'.
        kernel_pars : dict of {str: number or bool}
            Kernel function parameters. Optional if `kernel` is
            'identity'.
        distance : {'euclidean', 'tree', 'dictionary'}, optional
            Distance function name. If None, default distance function
            is used based on `kernel`.
        distance_dict : dict of {(number, number): number}, optional
            Dictionary of distances between points, if `distance` is
            'dictionary'.
        normalize : bool, optional
            Normalize dimension weights in groups. If None, default is
            used based on `kernel`.

        Notes
        -----

        Kernel-specific parameters and default attributes are given in
        the table below.

          .. list-table::
             :header-rows: 1

             * - Kernel
               - Parameters
               - Parameter types
               - Default `distance`
               - Default `normalize`
             * - ``identity``
               -
               -
               - ``euclidean``
               - ``False``
             * - ``exponential``
               - ``radius``
               - Positive number
               - ``euclidean``
               - ``False``
             * - ``tricubic``
               - ``exponent``
               - Positive number
               - ``euclidean``
               - ``False``
             * - ``depth``
               - ``radius``
               - Float in :math:`(0, 1)`
               - ``tree``
               - ``True``
             * -
               - ``version``
               - Integer in :math:`\\{1, 2\\}`, optional (default is 1)
               -
               -

        The identity kernel does not have any kernel parameters because
        the weight values are equal to the distance values.

        References
        ----------
        .. [1] `Cause of Death Ensemble model
               <https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-10-1>`_

        Examples
        --------
        Dimensions with exponential kernel and default Euclidean
        distance.

        >>> from weave.dimension import Dimension
        >>> age = Dimension(
                name='age_id',
                coordinates='age_mean',
                kernel='exponential',
                kernel_pars={'radius': 0.5}
            )
        >>> location = Dimension(
                name='location_id',
                coordinates=['lat', 'lon'],
                kernel='exponential',
                kernel_pars={'radius': 0.5}
            )

        Dimension with tricubic kernel and default Euclidean distance.

        >>> from weave.dimension import Dimension
        >>> year = Dimension(
                name='year_id',
                kernel='tricubic',
                kernel_pars={'exponent': 3}
            )

        Dimension with tricubic kernel and dictionary distance.

        >>> from weave.dimension import Dimension
        >>> location = Dimension(
                name='location_id',
                kernel='tricubic',
                kernel_pars={'exponent': 3},
                distance='dictionary',
                distance_dict={
                    (4, 4): 0.,
                    (4, 5): 1.,
                    (4, 6): 2.,
                    (5, 6): 2.
                }
            )

        Dimension with depth kernel and default tree distance.

        >>> from weave.dimension import Dimension
        >>> location = Dimension(
                name='location_id',
                coordinates=['super_region', 'region', 'country'],
                kernel='depth',
                kernel_pars={'radius': 0.9}
            )

        Dimension with identity kernel and default Euclidean distance.

        >>> from weave.dimension import Dimension
        >>> location = Dimension(
                name='location_id',
                coordinates=['lat', 'lon'],
                kernel='identity'
            )

        """
        self.name = name
        self.coordinates = coordinates  # type: ignore
        self.kernel = kernel
        self.kernel_pars = kernel_pars  # type: ignore
        self.distance = distance  # type: ignore
        self.distance_dict = distance_dict  # type: ignore
        self.normalize = normalize

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
    def coordinates(self) -> List[str]:
        """Get dimension coordinates.

        Returns
        -------
        list of str
            Dimension coordinates.

        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Optional[Union[str, List[str]]]) \
            -> None:
        """Set dimension coordinates.

        Parameters
        ----------
        coordinates : str or list of str, optional
            Dimension coordinates. If None, set equal to `name`.

        Raises
        ------
        AttributeError
            If `coordinates` has already been set.
        TypeError
            If `coordinates` not a str or list of str or None.
        ValueError
            If `coordinates` is an empty list or contains duplicates.

        """
        # Once set, `coordinates` cannot be changed
        if hasattr(self, 'coordinates'):
            raise AttributeError('`coordinates` cannot be changed.')

        # Set default
        if coordinates is None:
            coordinates = self._name

        # Check types
        coordinates = as_list(coordinates)
        if not all(isinstance(coord, str) for coord in coordinates):
            raise TypeError('`coordinates` contains invalid types.')

        # Check values
        if len(coordinates) == 0:
            raise ValueError('`coordinates` is an empty list.')
        if len(coordinates) > len(set(coordinates)):
            raise ValueError('`coordinates` contains duplicates.')

        self._coordinates = coordinates

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
        kernel : {'identity', 'exponential', 'tricubic', 'depth'}
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
        if kernel not in ('identity', 'exponential', 'tricubic', 'depth'):
            raise ValueError('`kernel` is not a valid kernel function.')

        self._kernel = kernel

    @property
    def kernel_pars(self) -> Dict[str, pars]:
        """Get kernel function parameters.

        Returns
        -------
        dict of {str: number or bool}
            Kernel function parameters.

        """
        return self._kernel_pars

    @kernel_pars.setter
    def kernel_pars(self, kernel_pars: Optional[Dict[str, pars]]) -> None:
        """Set kernel function parameters.

        Parameters
        ----------
        kernel_pars : dict of {str: number or bool} or None
            Kernel function parameters.

        """
        # Check values
        if self._kernel != 'identity':
            if self._kernel == 'exponential':
                check_pars(kernel_pars, 'radius', 'pos_num')
                kernel_pars = {'radius': kernel_pars['radius']}
            if self._kernel == 'tricubic':
                check_pars(kernel_pars, 'exponent', 'pos_num')
                kernel_pars = {'exponent': kernel_pars['exponent']}
            if self._kernel == 'depth':
                if 'version' not in kernel_pars:
                    kernel_pars['version'] = 1
                kpars = ['radius', 'version']
                check_pars(kernel_pars, kpars, ['pos_frac', 'pos_int'])
                kernel_pars = {key: kernel_pars[key] for key in kpars}
            self._kernel_pars = kernel_pars

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
        distance : {'dictionary', 'euclidean', 'tree', None}
            Distance function name.

        Raises
        ------
        AttributeError
            If `distance` has already been set.
        TypeError
            If `distance` is not a str or None.
        ValueError
            If `distance` is not a valid distance function.
            If `distance` is 'dictionary' but `coordinates` not 1D.

        """
        # Once set, `distance` cannot be changed
        if hasattr(self, 'distance'):
            raise AttributeError('`distance` cannot be changed.')

        # Set default
        if distance is None:
            if self._kernel == 'depth':
                distance = 'tree'
            else:
                distance = 'euclidean'

        # Check type
        if not isinstance(distance, str):
            raise TypeError('`distance` is not a str.')

        # Check value
        if distance not in ('euclidean', 'tree', 'dictionary'):
            msg = '`distance` is not a valid distance function.'
            raise ValueError(msg)
        if distance == 'dictionary' and len(self._coordinates) > 1:
            msg = "`distance` is 'dictionary' but `coordinates` not 1D."
            raise ValueError(msg)

        self._distance = distance

    @property
    def distance_dict(self) -> DistanceDict:
        """Get dictionary of distances between points.

        Returns
        -------
        dict of {(number, number): number}
            Dictionary of distances between points.

        """
        return self._distance_dict

    @distance_dict.setter
    def distance_dict(self, distance_dict: Optional[DistanceDict]) -> None:
        """Set dictionary of distances between points.

        Parameters
        ----------
        distance_dict : dict of {(number, number): number}
            Dictionary of distances between points.

        Raises
        ------
        AttributeError
            If `distance_dict` has already been set.
        ValueError
            If `distance` is 'dictionary' but `distance_dict` is None.

        """
        # Once set, `distance_dict` cannot be changed
        if hasattr(self, 'distance_dict'):
            raise AttributeError('`distance_dict` cannot be changed.')

        # Check values
        if self._distance == 'dictionary':
            if distance_dict is None:
                msg = "`distance` is 'dictionary', "
                msg += 'but `distance_dict` is None.'
                raise ValueError(msg)
            check_dict(distance_dict)
            self._distance_dict = distance_dict

    @property
    def normalize(self) -> bool:
        """Get `normalize` attribute.

        Returns
        -------
        bool
            Normalize dimension weights in groups.

        """
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: Optional[bool]) -> None:
        """Set `normalize` attribute.

        Parameters
        ----------
        normalize : bool, optional
            Normalize dimension weights in groups.

        Raises
        ------
        AttributeError
            If `normalize` has already been set.
        TypeError
            If `normalize` is not a bool or None.

        """
        # Once set, `normalize` cannot be changed
        if hasattr(self, 'normalize'):
            raise AttributeError('`normalize` cannot be changed.')

        # Set default
        if normalize is None:
            normalize = self._distance == 'depth'

        # Check type
        if not isinstance(normalize, bool):
            raise TypeError('`normalize` is not a bool.')

        self._normalize = normalize

    def get_typed_dimension(self, data: DataFrame) -> TypedDimension:
        """Get smoothing dimension cast as jitclass object.

        Parameters
        ----------
        data : DataFrame
            Input data structure.

        Returns
        -------
        TypedDimension
            Smoothing dimension cast as jitclass object.

        """
        weight_dict = self.get_weight_dict(data)
        return TypedDimension(self._name, weight_dict, self._normalize)

    def get_weight_dict(self, data: DataFrame) -> WeightDict:
        """Get dictionary of dimension smoothing weights.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data structure.

        Returns
        -------
        dict of {(float32, float32): float32}
            Dictionary of dimension smoothing weights.

        """
        # Get point names and coordinates
        points = data[[self._name] + [self._coordinates]]
        points = np.array(points.drop_duplicates(), dtype=np.float32)
        names = points[:, 0]
        coords = points[:, 1:]

        # Initialize weight dictionary
        weight_dict = TypedDict.empty(
            key_type=UniTuple(float32, 2),
            value_type=float32
        )

        # Compute weights
        for idx_x, x in enumerate(names):
            distances = {y: self.get_distance(coords[idx_x], coords[idx_y])
                         for idx_y, y in enumerate(names)}
            radius = max(distances.values()) + 1  # tricubic kernel
            levels = len(coords[idx_x])  # depth kernel
            weights = {(x, y): self.get_weight(distances[y], radius, levels)
                       for y in names}
            weight_dict.update(weights)

        return weight_dict

    def get_distance(self, x: np.ndarray, y: np.ndarray) -> np.float32:
        """Get distance between `x` and `y`.

        Parameters
        ----------
        x : 1D numpy.ndarray of float
            Current point.
        y : 1D numpy.ndarray of float
            Nearby point.

        Returns
        -------
        nonnegative float32
            Distance between `x` and `y`.

        """
        if self._distance == 'euclidean':
            return euclidean(x, y)
        if self._distance == 'tree':
            return tree(x, y)
        return np.float32(self._distance_dict[(x[0], y[0])])

    def get_weight(self, distance: number, radius: number, levels: int) \
            -> np.float32:
        """Get dimension smoothing weight.

        Parameters
        ----------
        distance : nonnegative int or float
            Distance between points.
        radius : positive int or float, optional
            Kernel radius for `kernels.tricubic`.
        levels : positive int, optional
            Number of levels for `kernels.depth`.

        Returns
        -------
        nonnegative float32
            Dimension smoothing weight.

        """
        if self._kernel == 'identity':
            return np.float32(distance)
        if self._kernel == 'exponential':
            return exponential(distance, **self._kernel_pars)
        if self._kernel == 'tricubic':
            return tricubic(distance, radius, **self._kernel_pars)
        return depth(distance, levels, **self._kernel_pars)


def check_pars(kernel_pars: Dict[str, number], names: Union[str, List[str]],
               types: Union[str, List[str]]) -> None:
    """Check kernel parameter types and values.

    Parameters
    ----------
    pars : dict of {str: number}
        Kernel parameters
    names : str or list of str
        Parameter names.
    types : str or list of str
        Parameter types. Valid types are 'pos_num', 'pos_int', and
        'pos_frac'.

    Raises
    ------
    KeyError
        If `pars` is missing a kernel parameter.
    TypeError
        If a kernel parameter is an invalid type.
    ValueError
        If a kernel parameter is an invalid value.

    """
    # Check type
    if not isinstance(kernel_pars, dict):
        raise TypeError('`kernel_pars` is not a dict.')

    # Get parameter names
    names = as_list(names)
    if isinstance(types, str):
        types = [types]*len(names)

    for idx_par, par_name in enumerate(names):
        msg = f"`{par_name}` is not "

        # Check key
        if par_name not in kernel_pars:
            raise KeyError(msg + 'in `pars`.')
        par_val = kernel_pars[par_name]

        # Check type and value
        if par_val <= 0.0:
            raise ValueError(msg + 'positive.')
        if types[idx_par] == 'pos_num':
            if not is_number(par_val):
                raise TypeError(msg + 'an int or float.')
        elif types[idx_par] == 'pos_int':
            if not is_int(par_val):
                raise TypeError(msg + 'an int.')
            if par_name == 'version' and par_val not in (1, 2):
                raise ValueError(msg + 'in {1, 2}.')
        else:  # 'pos_frac'
            if not is_float(par_val):
                raise TypeError(msg + 'a float.')
            if par_val >= 1.0:
                raise ValueError(msg + 'in (0, 1).')


def check_dict(distance_dict: Dict[Tuple[number, number], number]) -> None:
    """Check distance dictionary keys and values.

    Parameters
    ----------
    distance_dict : dict of {(number, number): number}
        Dictionary of distances between points.

    Raises
    ------
    TypeError
        If `distance_dict`, keys, or values are an invalid type.
    ValueError
        If `dictionary_dict` is empty, dictionary keys are not all
        length 2, or dictionary values are not all nonnegative.

    Notes
    -----
    Does not check that the values in `distance_dict` satisfy
    properties 2-4 in `weave.distance`.

    """
    # Check types
    if not isinstance(distance_dict, dict):
        raise TypeError('`distance_dict` is not a dict.')
    if not all(isinstance(key, tuple) for key in distance_dict):
        raise TypeError('`distance_dict` keys not all tuple.')
    if not all(is_number(point) for key in distance_dict for point in key):
        raise TypeError('`distance_dict` key entries not all int or float.')
    if not all(is_number(value) for value in distance_dict.values()):
        raise TypeError('`distance_dict` values not all int or float.')

    # Check values
    if len(distance_dict) == 0:
        raise ValueError('`distance_dict` is an empty dict.')
    if any(len(key) != 2 for key in distance_dict):
        raise ValueError('`distance_dict` keys are not all length 2.')
    if any(value < 0.0 for value in distance_dict.values()):
        raise ValueError('`distance_dict` contains negative values.')
