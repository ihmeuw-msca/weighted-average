# pylint: disable=C0103, E0611, R0902, R0903, R0912, R0913
"""Smoothing dimension specifications."""
from typing import Dict, List, Optional, Tuple, Union

from numba.experimental import jitclass  # type: ignore
from numba.typed import Dict as TypedDict  # type: ignore
from numba.types import DictType, UniTuple  # type: ignore
from numba.types import float32, unicode_type  # type: ignore
import numpy as np
from pandas import DataFrame  # type: ignore

from weave.distance import euclidean, tree
from weave.kernels import exponential, tricubic, depth, inverse
from weave.utils import as_list, is_float, is_number

number = Union[int, float]
DistanceDict = Dict[Tuple[number, number], number]
WeightDict = Dict[Tuple[float, float], float]


@jitclass(
    [
        ("name", unicode_type),
        ("kernel", unicode_type),
        ("weight_dict", DictType(UniTuple(float32, 2), float32)),
    ]
)
class TypedDimension:
    """Smoothing dimension specifications."""

    def __init__(self, name: str, kernel: str, weight_dict: WeightDict) -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        name : unicode_type
            Dimension name.
        kernel : unicode_type
            Kernel function name.
        weight_dict : numba.typed.Dict of {(float32, float32): float32}
            Dictionary of dimension smoothing weights.

        """
        self.name = name
        self.kernel = kernel
        self.weight_dict = weight_dict


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
        the given dimension. For example, `'age_mid'`, `['lat', 'lon']`,
        or `['super_region', 'region', 'country']`. Can be same as
        `name` attribute if dimension is 1D.

    kernel : {'exponential', 'tricubic', 'depth', 'inverse', 'identity'}
        Kernel function name.

        Name of kernel function to compute smoothing weights.

        See Also
        --------
        weave.kernels

    distance : {'euclidean', 'tree', 'dictionary'}
        Distance function name.

        Name of distance function to compute distance between points.

        See Also
        --------
        weave.distance

    radius : positive number, optional
        Kernel radius.

        Kernel radius if `kernel` is 'exponential', 'depth', or 'inverse'.

    exponent : positive number, optional
        Kernel exponent.

        Kernel exponent if `kernel` is 'tricubic'.

    version : {'codem', 'stgpr'}, optional
        Kernel version.

        Kernel version if `kernel` is 'depth'.

    distance_dict : dict of {(number, number): number}, optional
        Dictionary of distances between points.

        User-defined dictionary of distances between points if
        `distance` attribute is 'dictionary'. Dictionary keys are
        tuples of point ID pairs, and dictionary values are the
        corresponding distances.

    """

    def __init__(
        self,
        name: str,
        coordinates: Optional[Union[str, List[str]]] = None,
        kernel: Optional[str] = "identity",
        distance: Optional[str] = None,
        radius: Optional[number] = None,
        exponent: Optional[number] = None,
        version: Optional[str] = None,
        distance_dict: Optional[DistanceDict] = None,
    ) -> None:
        """Create smoothing dimension.

        Parameters
        ----------
        name : str
            Dimension name.
        coordinates : str or list of str, optional
            Dimension coordinates, if different from `name`.
        kernel : {'exponential', 'tricubic', 'depth', 'inverse', 'identity'}, optional
            Kernel function name. Default is 'identity'.
        distance : {'euclidean', 'tree', 'dictionary'}, optional
            Distance function name. If None, default distance function
            is assigned based on `kernel`.

        Other Parameters
        ----------------
        radius : positive number, optional
            Kernel radius if `kernel` is 'exponential', 'depth', or
            'inverse'. For depth kernel, `radius` must be a float in
            (0.5, 1).
        exponent : positive number, optional
            Kernel exponent if `kernel` is 'tricubic'.
        version : {'codem', 'stgpr'}, optional
            Kernel version if `kernel` is 'depth'. Default is 'codem'.
        distance_dict : dict of {(number, number): number}, optional
            Dictionary of distances between points if `distance` is
            'dictionary'. Dictionary values must be nonnegative.

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
             * - ``exponential``
               - ``radius``
               - Positive number
               - ``euclidean``
             * - ``tricubic``
               - ``exponent``
               - Positive number
               - ``euclidean``
             * - ``depth``
               - ``radius``
               - float in :math:`(0.5, 1)`
               - ``tree``
             * -
               - ``version``
               - \\{'codem', 'stgpr'\\}, optional (default is 'codem')
               -
             * - ``inverse``
               - ``radius``
               - Positive number
               - ``euclidean``
             * - ``identity``
               -
               -
               - ``euclidean``

        The identity kernel does not have any kernel parameters because
        the weight values are equal to the distance values.

        Examples
        --------
        Dimensions with exponential kernel and default Euclidean
        distance.

        >>> from weave.dimension import Dimension
        >>> age = Dimension(
                name='age_id',
                coordinates='age_mean',
                kernel='exponential',
                radius=0.5
            )
        >>> location = Dimension(
                name='location_id',
                coordinates=['lat', 'lon'],
                kernel='exponential',
                radius=0.5
            )

        Dimension with tricubic kernel and default Euclidean distance.

        >>> from weave.dimension import Dimension
        >>> year = Dimension(
                name='year_id',
                kernel='tricubic',
                exponent=3
            )

        Dimension with tricubic kernel and dictionary distance.

        >>> from weave.dimension import Dimension
        >>> location = Dimension(
                name='location_id',
                kernel='tricubic',
                exponent=3,
                distance='dictionary',
                distance_dict={
                    (4, 4): 0,
                    (4, 5): 1,
                    (4, 6): 2,
                    (5, 4): 1,
                    (5, 5): 0,
                    (5, 6): 2,
                    (6, 4): 2,
                    (6, 5): 2,
                    (6, 6): 0
                }
            )

        Dimension with depth kernel and default tree distance.

        >>> from weave.dimension import Dimension
        >>> location = Dimension(
                name='location_id',
                coordinates=['super_region', 'region', 'country'],
                kernel='depth',
                radius=0.9
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
        self.coordinates = coordinates
        self.kernel = kernel
        self.distance = distance
        self.radius = radius
        self.exponent = exponent
        self.version = version
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
        if hasattr(self, "name"):
            raise AttributeError("`name` cannot be changed")

        # Check type
        if not isinstance(name, str):
            raise TypeError("`name` is not a str")

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
    def coordinates(self, coordinates: Optional[Union[str, List[str]]]) -> None:
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
        if hasattr(self, "coordinates"):
            raise AttributeError("`coordinates` cannot be changed")

        # Set default
        if coordinates is None:
            coordinates = self._name

        # Check types
        coordinates = as_list(coordinates)
        if not all(isinstance(coord, str) for coord in coordinates):
            raise TypeError("`coordinates` contains invalid types")

        # Check values
        if len(coordinates) == 0:
            raise ValueError("`coordinates` is an empty list")
        if len(coordinates) > len(set(coordinates)):
            raise ValueError("`coordinates` contains duplicates")

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
        kernel : {'exponential', 'tricubic', 'depth', 'inverse', 'identity'}
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
        if hasattr(self, "kernel"):
            raise AttributeError("`kernel` cannot be changed")

        # Check type
        if not isinstance(kernel, str):
            raise TypeError("`kernel` is not a str")

        # Check value
        if kernel not in ("exponential", "tricubic", "depth", "inverse", "identity"):
            raise ValueError("`kernel` is not a valid kernel function")

        self._kernel = kernel

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
        distance : {'euclidean', 'tree', 'dictionary'} or None
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
        if hasattr(self, "distance"):
            raise AttributeError("`distance` cannot be changed")

        # Set default
        if distance is None:
            if self._kernel == "depth":
                distance = "tree"
            else:
                distance = "euclidean"

        # Check type
        if not isinstance(distance, str):
            raise TypeError("`distance` is not a str")

        # Check value
        if distance not in ("euclidean", "tree", "dictionary"):
            msg = "`distance` is not a valid distance function"
            raise ValueError(msg)

        self._distance = distance

    @property
    def radius(self) -> number:
        """Get kernel radius if `kernel` is 'exponential', 'depth', or 'inverse'.

        Returns
        -------
        positive number
            Kernel radius.

        """
        return self._radius

    @radius.setter
    def radius(self, radius: Optional[number]) -> None:
        """Set kernel radius if `kernel` is 'exponential', 'depth', or 'inverse'.

        Parameters
        ----------
        radius : positive number or None
            Kernel radius.

        Raises
        ------
        AttributeError
            If `kernel` is 'exponential', 'depth', or 'inverse' but `radius` is None.
        TypeError
            If `kernel` is 'exponential' or 'inverse' but `radius` is not a number.
            If `kernel` is 'depth' but `radius` is not a float.
        ValueError
            If `kernel` is 'exponential' or 'inverse' but `radius` is not positive.
            If `kernel` is 'depth' but `radius` is not in (0.5, 1).

        """
        if self._kernel in ("exponential", "depth", "inverse"):
            if radius is None:
                msg = f"`radius` is required for '{self._kernel}' kernel"
                raise AttributeError(msg)
            if self._kernel in ("exponential", "inverse"):
                if not is_number(radius):
                    raise TypeError("`radius` is not an int or float")
                if radius <= 0:
                    raise ValueError("`radius` is not positive")
            elif self._kernel == "depth":
                if not is_float(radius):
                    raise TypeError("`radius` is not a float")
                if radius <= 0.5 or radius >= 1:
                    raise ValueError("`radius` is not in (0.5, 1)")
            self._radius = radius

    @property
    def exponent(self) -> number:
        """Get kernel exponent if `kernel` is 'tricubic'.

        Returns
        -------
        positive number
            Kernel exponent.

        """
        return self._exponent

    @exponent.setter
    def exponent(self, exponent: Optional[number]) -> None:
        """Set kernel exponent if `kernel` is 'tricubic'.

        Parameters
        ----------
        exponent : positive number or None
            Kernel exponent.

        Raises
        ------
        AttributeError
            If `kernel` is 'tricubic' but `exponent` is None.
        TypeError
            If `kernel` is 'tricubic' but `exponent` is not a number.
        ValueError
            If `kernel` is 'tricubic' but `exponent` is not positive.

        """
        if self._kernel == "tricubic":
            if exponent is None:
                msg = "`exponent` is required for 'tricubic' kernel"
                raise AttributeError(msg)
            if not is_number(exponent):
                raise TypeError("`exponent` is not an int or float")
            if exponent <= 0:
                raise ValueError("`exponent` is not positive")
            self._exponent = exponent

    @property
    def version(self) -> str:
        """Get kernel version if `kernel` is 'depth'.

        Returns
        -------
        str
            Kernel version.

        """
        return self._version

    @version.setter
    def version(self, version: Optional[str]) -> None:
        """Set kernel version if `kernel` is 'depth'.

        Parameters
        ----------
        version : str or None
            Kernel version.

        Raises
        ------
        TypeError
            If `kernel` is 'depth' but `version` is not a str or None.
        ValueError
            If `kernel` is 'depth' but `version` not in
            {'codem', 'stgpr'}.

        """
        if self._kernel == "depth":
            if version is None:
                self._version = "codem"
            else:
                if not isinstance(version, str):
                    raise TypeError("`version` is not a str")
                if version not in ("codem", "stgpr"):
                    raise ValueError("`version` not in {'codem', 'stgpr'}")
                self._version = version

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
        if hasattr(self, "distance_dict"):
            raise AttributeError("`distance_dict` cannot be changed")

        # Check values
        if self._distance == "dictionary":
            if distance_dict is None:
                msg = "`distance` is 'dictionary', "
                msg += "but `distance_dict` is None"
                raise ValueError(msg)
            check_dict(distance_dict)
            self._distance_dict = distance_dict

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
        return TypedDimension(self._name, self._kernel, weight_dict)

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
        points = data[[self._name] + self._coordinates]
        points = np.array(points.drop_duplicates(), dtype=np.float32)

        # Initialize weight dictionary
        weight_dict = TypedDict.empty(key_type=UniTuple(float32, 2), value_type=float32)

        # Compute weights
        for idx_x, x in enumerate(points[:, 0]):
            distances = {
                y: self.get_distance(points[idx_x], points[idx_y])
                for idx_y, y in enumerate(points[:, 0])
            }
            radius = max(distances.values()) + 1  # tricubic kernel
            levels = len(points[idx_x, 1:])  # depth kernel
            weights = {
                (x, y): self.get_weight(distances[y], radius, levels)
                for y in points[:, 0]
            }
            weight_dict.update(weights)

        return weight_dict

    def get_distance(self, x: np.ndarray, y: np.ndarray) -> np.float32:
        """Get distance between `x` and `y`.

        Parameters
        ----------
        x : float or 1D numpy.ndarray of float
            Current point.
        y : float or 1D numpy.ndarray of float
            Nearby point.

        Returns
        -------
        nonnegative float32
            Distance between `x` and `y`.

        """
        if self._distance == "euclidean":
            return euclidean(x[1:], y[1:])
        if self._distance == "tree":
            return tree(x[1:], y[1:])
        return np.float32(self._distance_dict[(x[0], y[0])])

    def get_weight(self, distance: number, radius: number, levels: int) -> np.float32:
        """Get dimension smoothing weight.

        Parameters
        ----------
        distance : nonnegative int or float
            Distance between points.
        radius : positive int or float
            Kernel radius for `kernels.tricubic`.
        levels : positive int
            Number of levels for `kernels.depth`.

        Returns
        -------
        nonnegative float32
            Dimension smoothing weight.

        """
        if self._kernel == "exponential":
            return exponential(distance, self._radius)
        if self._kernel == "tricubic":
            return tricubic(distance, radius, self._exponent)
        if self._kernel == "depth":
            return depth(distance, levels, self._radius, self._version)
        if self._kernel == "inverse":
            return inverse(distance, self._radius)
        return np.float32(distance)  # identity


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
        raise TypeError("`distance_dict` is not a dict")
    if not all(isinstance(key, tuple) for key in distance_dict):
        raise TypeError("`distance_dict` keys not all tuple")
    if not all(is_number(point) for key in distance_dict for point in key):
        raise TypeError("`distance_dict` key entries not all int or float")
    if not all(is_number(value) for value in distance_dict.values()):
        raise TypeError("`distance_dict` values not all int or float")

    # Check values
    if len(distance_dict) == 0:
        raise ValueError("`distance_dict` is an empty dict")
    if any(len(key) != 2 for key in distance_dict):
        raise ValueError("`distance_dict` keys are not all length 2")
    if any(value < 0.0 for value in distance_dict.values()):
        raise ValueError("`distance_dict` contains negative values")
