# pylint: disable=C0103
"""Tests for depth kernel values."""
import numpy as np
import pytest

from weave.kernels import depth

radius_vals = 0.1*np.arange(1, 10).astype(np.float32)


@pytest.mark.parametrize('radius', radius_vals)
def test_levels_1(radius):
    """Depth kernel with 1 level."""
    levels = np.int32(1)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [1, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


@pytest.mark.parametrize('radius', radius_vals)
def test_levels_2(radius):
    """Depth kernel with 2 levels."""
    levels = np.int32(2)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [radius, 1 - radius, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


@pytest.mark.parametrize('radius', radius_vals)
def test_levels_3(radius):
    """Depth kernel with 3 levels."""
    levels = np.int32(3)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [radius, radius*(1 - radius), (1 - radius)**2, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


@pytest.mark.parametrize('radius', radius_vals)
def test_levels_4(radius):
    """Depth kernel with 4 levels."""
    levels = np.int32(4)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [radius, radius*(1 - radius), radius*(1 - radius)**2,
               (1 - radius)**3, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])


@pytest.mark.parametrize('radius', radius_vals)
def test_levels_5(radius):
    """Depth kernel with 5 levels."""
    levels = np.int32(5)
    distances = np.arange(levels + 1).astype(np.float32)
    weights = [radius, radius*(1 - radius), radius*(1 - radius)**2,
               radius*(1 - radius)**3, (1 - radius)**4, 0]
    for ii, distance in enumerate(distances):
        weight = depth(distance, radius, levels)
        assert np.isclose(weight, weights[ii])
