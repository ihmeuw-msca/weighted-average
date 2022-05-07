# pylint: disable=C0103, E0611, R0902, R0903, R0913
"""Smoothing dimension specifications."""
from typing import Dict, List, Optional, Tuple, Union

from numba.experimental import jitclass  # type: ignore
from numba.typed import List as TypedList  # type: ignore
from numba.types import DictType, ListType, UniTuple  # type: ignore
from numba.types import float32, unicode_type  # type: ignore

from weave.distance import get_typed_dict, _check_dict
from weave.kernels import get_typed_pars, _check_pars
from weave.utils import as_list

number = Union[int, float]
pars = Union[number, bool]
DistanceDict = Dict[Tuple[number, number], number]


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

    kernel : {'depth', 'exponential', 'identity', 'tricubic'}
        Kernel function name.

        Name of kernel function to compute smoothing weights.

        See Also
        --------
        weave.kernels

    kernel_pars : dict of {str: number or bool}
        Kernel function parameters.

        Dictionary of kernel function parameters corresponding to
        `kernel` attribute.

    distance : {'dictionary', 'euclidean', 'tree'}
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
        corresponding distances. Because distances are assumed to be
        symmetric, point IDs are listed from smallest to largest.

        See Also
        --------
        weave.distance.dictionary

    """

    def __init__(self, name: str, coordinates: Union[str, List[str]] = None,
                 kernel: str = 'identity',
                 kernel_pars: Optional[Dict[str, pars]] = None,
                 distance: Optional[str] = None,
                 distance_dict: Optional[DistanceDict] = None) -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        name : str
            Dimension name.
        coordinates : str or list of str, optional
            Dimension coordinates, if different from `name`.
        kernel : {'depth', 'exponential', 'identity', 'tricubic'}, optional
            Kernel function name. Default is 'identity'.
        kernel_pars : dict of {str: number or bool}
            Kernel function parameters. Optional if `kernel` is
            'identity'.
        distance : {'dictionary', 'euclidean', 'tree'}, optional
            Distance function name. If None, default distance function
            is used based on `kernel`.
        distance_dict : dict of {(number, number): number}, optional
            Dictionary of distances between points, if `distance` is
            'dictionary'.

        Notes
        -----

        * Kernel-specific parameters and default distance functions are
          given in the table below. The given dictionary `kernel_pars`
          is converted to an instance of `numba.typed.Dict
          <https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict>`_
          within :func:`weave.smoother.Smoother`.

          .. list-table::
             :header-rows: 1

             * - Kernel
               - Parameter
               - Parameter type
               - Default distance
             * - ``depth``
               - ``radius``
               - Float in :math:`(0, 1)`
               - ``tree``
             * -
               - ``levels``
               - Positive int
               -
             * -
               - ``normalize``
               - Boolean, optional (default is ``True``)
               -
             * - ``exponential``
               - ``radius``
               - Positive number
               - ``euclidean``
             * -
               - ``normalize``
               - Boolean, optional (default is ``False``)
               -
             * - ``identity``
               - ``normalize``
               - Boolean, optional (default is ``False``)
               -  ``euclidean``
             * - ``tricubic``
               - ``radius``
               - Positive number
               - ``euclidean``
             * -
               - ``exponent``
               - Positive float
               -
             * -
               - ``normalize``
               - Boolean, optional (default is ``False``)
               -

        * The optional kernel parameter `normalize` indicates whether
          or not the preceding dimension weights should be normalized
          in groups based on the current dimension weight values. This
          corresponds to the CODEm [1]_ framework where the product of
          age and time weights are normalized in groups based on the
          location hierarchy before being multiplied by location
          weights. For example, for points :math:`i, j, k` from the
          same country:

          .. math:: w_{i, j} = w_{\\ell_{i, j}} \\cdot
                    \\frac{w_{a_{i, j}} w_{t_{i, j}}} {\\sum_{k}
                    w_{a_{i, k}} w_{t_{i, k}}}

          This option may be inefficient if there is a large number of
          possible weight values for the given dimension.

        * The parameters for the identity kernel are optional because
          the weight values are equal to the distance values. For
          increased efficiency, you can precompute all dimension
          weights as a dictionary and then use the identity kernel with
          the dictionary distance. This is done automatically within
          :func:`weave.smoother.__call__` if `precompute` is True.

        * The parameter `distance_dict` contains the user-defined
          distances between points if the distance attribute is
          'dictionary'. Dictionary keys are tuples of point ID pairs,
          and dictionary values are the corresponding distances.
          Because distances are assumed to be symmetric, point IDs are
          listed from smallest to largest. The given `distance_dict` is
          converted to an instance of `numba.typed.Dict
          <https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict>`_
          within :func:`weave.smoother.Smoother`.

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
                kernel_pars={'radius': 2, 'exponent': 3}
            )

        Dimension with tricubic kernel and dictionary distance.

        >>> from weave.dimension import Dimension
        >>> location = Dimension(
                name='location_id',
                kernel='tricubic',
                kernel_pars={'radius': 2, 'exponent': 3},
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
                kernel_pars={'radius': 0.9, 'levels': 3}
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
        kernel : {'depth', 'exponential', 'identity', 'tricubic'}
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
        if kernel not in ('depth', 'exponential', 'identity', 'tricubic'):
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
        if kernel_pars is None:
            kernel_pars = {}

        # Check parameter values
        if self._kernel == 'depth':
            if 'normalize' not in kernel_pars:
                kernel_pars['normalize'] = True
            _check_pars(kernel_pars, ['radius', 'levels', 'normalize'],
                        ['pos_frac', 'pos_int', 'bool'])
            kernel_pars = {key: kernel_pars[key]
                           for key in ['radius', 'levels', 'normalize']}
        else:
            if 'normalize' not in kernel_pars:
                kernel_pars['normalize'] = False
            if self._kernel == 'exponential':
                _check_pars(kernel_pars, ['radius', 'normalize'],
                            ['pos_num', 'bool'])
                kernel_pars = {key: kernel_pars[key]
                               for key in ['radius', 'normalize']}
            elif self._kernel == 'identity':
                _check_pars(kernel_pars, 'normalize', 'bool')
                kernel_pars = {'normalize': kernel_pars['normalize']}
            elif self._kernel == 'tricubic':
                _check_pars(kernel_pars, ['radius', 'exponent', 'normalize'],
                            ['pos_num', 'pos_num', 'bool'])
                kernel_pars = {key: kernel_pars[key]
                               for key in ['radius', 'exponent', 'normalize']}
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

        # Set defaults
        if distance is None:
            if self._kernel == 'depth':
                distance = 'tree'
            else:
                distance = 'euclidean'

        # Check type
        if not isinstance(distance, str):
            raise TypeError('`distance` is not a str.')

        # Check value
        if distance not in ('dictionary', 'euclidean', 'tree'):
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
            _check_dict(distance_dict)
            self._distance_dict = distance_dict


@jitclass([('name', unicode_type),
           ('coordinates', ListType(unicode_type)),
           ('kernel', unicode_type),
           ('kernel_pars', DictType(unicode_type, float32)),
           ('distance', unicode_type),
           ('distance_dict', DictType(UniTuple(float32, 2), float32))])
class TypedDimension:
    """Smoothing dimension specifications."""
    def __init__(self, name: str, coordinates: List[str], kernel: str,
                 kernel_pars: Dict[str, float], distance: str,
                 distance_dict: Dict[Tuple[float, float], float]) -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        name : unicode_type
            Dimension name.
        coordinates : numba.typed.List of unicode_type
            Dimension coordinates.
        kernel : {'depth', 'exponential', 'identity', 'tricubic'}
            Kernel function name.
        kernel_pars : numba.typed.Dict of {unicode_type: float32}
            Kernel function parameters.
        distance : {'dictionary', 'euclidean', 'tree'}
            Distance function name.
        distance_dict : numba.typed.Dict of {(float32, float32): float32}
            Dictionary of distances between points if `distance` is
            'dictionary'.

        """
        self.name = name
        self.coordinates = coordinates
        self.kernel = kernel
        self.kernel_pars = kernel_pars
        self.distance = distance
        self.distance_dict = distance_dict


def get_typed_dimension(dim: Dimension) -> TypedDimension:
    """Get smoothing dimension cast as jitclass object.

    Returns
    -------
    TypedDimension
        Smoothing dimension cast as jitclass object.

    """
    # Get typed version of attributes
    coordinates = TypedList(dim.coordinates)
    kernel_pars = get_typed_pars(dim.kernel_pars)
    if hasattr(dim, 'distance_dict'):
        distance_dict = get_typed_dict(dim.distance_dict)
    else:
        distance_dict = get_typed_dict()

    # Create typed dimension
    typed_dim = TypedDimension(dim.name, coordinates, dim.kernel, kernel_pars,
                               dim.distance, distance_dict)
    return typed_dim
