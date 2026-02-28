"""Tests for MetricTensor using known GR solutions."""

import pytest
import sympy
from sympy import Matrix, Rational, diag, simplify, sin, symbols

from diffgeom import MetricTensor


# ---------------------------------------------------------------------------
# Fixtures: well-known metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Basic property tests
# ---------------------------------------------------------------------------


class TestBasicProperties:
    def test_dimension_4d(self, schwarzschild):
        metric, _ = schwarzschild
        assert metric.dim == 4

    def test_dimension_5d(self, flat_5d):
        assert flat_5d.dim == 5

    def test_dimension_2d(self, sphere_2d):
        metric, _ = sphere_2d
        assert metric.dim == 2

    def test_inverse_is_true_inverse(self, schwarzschild):
        metric, _ = schwarzschild
        product = simplify(metric.matrix * metric.inverse)
        assert product.equals(sympy.eye(4))

    def test_inverse_5d(self, flat_5d):
        product = simplify(flat_5d.matrix * flat_5d.inverse)
        assert product.equals(sympy.eye(5))

    def test_indexing(self, schwarzschild):
        metric, params = schwarzschild
        # g_{22} = r² for Schwarzschild
        r = params["r"]
        assert simplify(metric[2, 2] - r**2) == 0

    def test_symmetry_required(self):
        t, x = symbols("t x")
        asymmetric = Matrix([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="symmetric"):
            MetricTensor(asymmetric, (t, x))

    def test_shape_mismatch(self):
        t, x, y = symbols("t x y")
        g = Matrix([[1, 0], [0, 1]])
        with pytest.raises(ValueError, match="does not match"):
            MetricTensor(g, (t, x, y))


# ---------------------------------------------------------------------------
# Christoffel symbol tests
# ---------------------------------------------------------------------------


class TestChristoffelSymbols:
    def test_flat_space_christoffels_vanish(self, flat_5d):
        """All Christoffel symbols vanish in flat Minkowski space."""
        gamma = flat_5d.christoffel_second_kind
        for sigma in range(5):
            for mu in range(5):
                for nu in range(5):
                    assert gamma[sigma, mu, nu] == 0

    def test_schwarzschild_christoffel_gamma_r_tt(self, schwarzschild):
        """Γ^r_{tt} = (r_s / 2r²)(1 - r_s/r) for Schwarzschild.

        This is the component responsible for gravitational acceleration.
        """
        metric, params = schwarzschild
        r_s, r = params["r_s"], params["r"]

        gamma = metric.christoffel_second_kind
        # indices: t=0, r=1, theta=2, phi=3
        expected = r_s * (r - r_s) / (2 * r**3)
        assert simplify(gamma[1, 0, 0] - expected) == 0

    def test_schwarzschild_christoffel_gamma_t_tr(self, schwarzschild):
        """Γ^t_{tr} = r_s / (2r(r - r_s)) for Schwarzschild."""
        metric, params = schwarzschild
        r_s, r = params["r_s"], params["r"]

        gamma = metric.christoffel_second_kind
        expected = r_s / (2 * r * (r - r_s))
        assert simplify(gamma[0, 0, 1] - expected) == 0

    def test_schwarzschild_christoffel_gamma_theta_r_theta(self, schwarzschild):
        """Γ^θ_{rθ} = 1/r for Schwarzschild."""
        metric, params = schwarzschild
        r = params["r"]

        gamma = metric.christoffel_second_kind
        expected = 1 / r
        assert simplify(gamma[2, 1, 2] - expected) == 0

    def test_schwarzschild_christoffel_gamma_phi_r_phi(self, schwarzschild):
        """Γ^φ_{rφ} = 1/r for Schwarzschild."""
        metric, params = schwarzschild
        r = params["r"]

        gamma = metric.christoffel_second_kind
        expected = 1 / r
        assert simplify(gamma[3, 1, 3] - expected) == 0

    def test_schwarzschild_christoffel_gamma_theta_phi_phi(self, schwarzschild):
        """Γ^θ_{φφ} = -sin(θ)cos(θ) for Schwarzschild."""
        metric, params = schwarzschild
        theta = params["theta"]

        gamma = metric.christoffel_second_kind
        expected = -sin(theta) * sympy.cos(theta)
        assert simplify(gamma[2, 3, 3] - expected) == 0

    def test_sphere_christoffel(self, sphere_2d):
        """On a 2-sphere: Γ^θ_{φφ} = -sin(θ)cos(θ) and Γ^φ_{θφ} = cos(θ)/sin(θ)."""
        metric, params = sphere_2d
        theta = params["theta"]

        gamma = metric.christoffel_second_kind
        assert simplify(gamma[0, 1, 1] - (-sin(theta) * sympy.cos(theta))) == 0
        assert simplify(gamma[1, 0, 1] - sympy.cos(theta) / sin(theta)) == 0

    def test_christoffel_symmetry_in_lower_indices(self, schwarzschild):
        """Γ^σ_{μν} = Γ^σ_{νμ} (torsion-free connection)."""
        metric, _ = schwarzschild
        gamma = metric.christoffel_second_kind
        n = metric.dim
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    assert simplify(gamma[sigma, mu, nu] - gamma[sigma, nu, mu]) == 0


# ---------------------------------------------------------------------------
# Determinant tests
# ---------------------------------------------------------------------------


class TestDeterminant:
    def test_schwarzschild_determinant(self, schwarzschild):
        """det(g) = -r⁴ sin²θ for Schwarzschild (the f factors cancel)."""
        metric, params = schwarzschild
        r, theta = params["r"], params["theta"]
        expected = -(r**4) * sin(theta) ** 2
        assert simplify(metric.determinant - expected) == 0

    def test_flat_5d_determinant(self, flat_5d):
        assert flat_5d.determinant == -1
