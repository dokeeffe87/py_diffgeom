"""Shared fixtures for differential geometry tests."""

import pytest
from sympy import diag, sin, symbols

from diffgeom import MetricTensor


@pytest.fixture
def schwarzschild():
    """4D Schwarzschild metric in (t, r, θ, φ) coordinates.

    ds² = -(1 - r_s/r) dt² + (1 - r_s/r)⁻¹ dr² + r² dθ² + r² sin²θ dφ²
    """
    t, r, theta, phi = symbols("t r theta phi")
    r_s = symbols("r_s", positive=True)

    f = 1 - r_s / r
    g = diag(-f, 1 / f, r**2, r**2 * sin(theta) ** 2)
    return MetricTensor(g, (t, r, theta, phi)), {"r_s": r_s, "r": r, "theta": theta}


@pytest.fixture
def flat_5d():
    """5D Minkowski metric: ds² = -dt² + dx² + dy² + dz² + dw²."""
    t, x, y, z, w = symbols("t x y z w")
    g = diag(-1, 1, 1, 1, 1)
    return MetricTensor(g, (t, x, y, z, w))


@pytest.fixture
def sphere_2d():
    """2D sphere of radius R: ds² = R² dθ² + R² sin²θ dφ²."""
    R, theta, phi = symbols("R theta phi")
    g = diag(R**2, R**2 * sin(theta) ** 2)
    return MetricTensor(g, (theta, phi)), {"R": R, "theta": theta}
