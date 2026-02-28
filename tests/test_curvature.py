"""Tests for curvature tensors using known GR solutions."""

from sympy import simplify, sin

# ---------------------------------------------------------------------------
# Riemann tensor tests
# ---------------------------------------------------------------------------


class TestRiemannTensor:
    def test_flat_space_riemann_vanishes(self, flat_5d):
        """All Riemann tensor components vanish in flat Minkowski space."""
        Riem = flat_5d.riemann_tensor
        n = flat_5d.dim
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        assert Riem[rho, sigma, mu, nu] == 0

    def test_antisymmetry_in_last_two_indices(self, schwarzschild):
        """R^ρ_{σμν} = -R^ρ_{σνμ} for Schwarzschild."""
        metric, _ = schwarzschild
        Riem = metric.riemann_tensor
        n = metric.dim
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        assert simplify(
                            Riem[rho, sigma, mu, nu] + Riem[rho, sigma, nu, mu]
                        ) == 0

    def test_sphere_riemann_components(self, sphere_2d):
        """On a 2-sphere: R^θ_{φθφ} = sin²θ and R^φ_{θφθ} = 1."""
        metric, params = sphere_2d
        theta = params["theta"]
        Riem = metric.riemann_tensor
        # indices: theta=0, phi=1
        assert simplify(Riem[0, 1, 0, 1] - sin(theta) ** 2) == 0
        assert simplify(Riem[1, 0, 1, 0] - 1) == 0


# ---------------------------------------------------------------------------
# Ricci tensor tests
# ---------------------------------------------------------------------------


class TestRicciTensor:
    def test_schwarzschild_ricci_vanishes(self, schwarzschild):
        """Schwarzschild is a vacuum solution: R_{μν} = 0 everywhere."""
        metric, _ = schwarzschild
        Ric = metric.ricci_tensor
        n = metric.dim
        for mu in range(n):
            for nu in range(n):
                assert simplify(Ric[mu, nu]) == 0

    def test_flat_space_ricci_vanishes(self, flat_5d):
        """All Ricci tensor components vanish in flat Minkowski space."""
        Ric = flat_5d.ricci_tensor
        n = flat_5d.dim
        for mu in range(n):
            for nu in range(n):
                assert Ric[mu, nu] == 0

    def test_sphere_ricci_components(self, sphere_2d):
        """On a 2-sphere of radius R: R_{θθ} = 1, R_{φφ} = sin²θ."""
        metric, params = sphere_2d
        theta = params["theta"]
        Ric = metric.ricci_tensor
        assert simplify(Ric[0, 0] - 1) == 0
        assert simplify(Ric[1, 1] - sin(theta) ** 2) == 0
        assert simplify(Ric[0, 1]) == 0
        assert simplify(Ric[1, 0]) == 0

    def test_ricci_symmetry(self, schwarzschild):
        """R_{μν} = R_{νμ}."""
        metric, _ = schwarzschild
        Ric = metric.ricci_tensor
        n = metric.dim
        for mu in range(n):
            for nu in range(n):
                assert simplify(Ric[mu, nu] - Ric[nu, mu]) == 0


# ---------------------------------------------------------------------------
# Ricci scalar tests
# ---------------------------------------------------------------------------


class TestRicciScalar:
    def test_schwarzschild_ricci_scalar_vanishes(self, schwarzschild):
        """R = 0 for Schwarzschild (vacuum)."""
        metric, _ = schwarzschild
        assert simplify(metric.ricci_scalar) == 0

    def test_flat_space_ricci_scalar_vanishes(self, flat_5d):
        """R = 0 for flat Minkowski."""
        assert simplify(flat_5d.ricci_scalar) == 0

    def test_sphere_ricci_scalar(self, sphere_2d):
        """On a 2-sphere of radius R: R = 2/R²."""
        metric, params = sphere_2d
        R_sym = params["R"]
        expected = 2 / R_sym**2
        assert simplify(metric.ricci_scalar - expected) == 0


# ---------------------------------------------------------------------------
# Einstein tensor tests
# ---------------------------------------------------------------------------


class TestEinsteinTensor:
    def test_schwarzschild_einstein_vanishes(self, schwarzschild):
        """G_{μν} = 0 for Schwarzschild (vacuum)."""
        metric, _ = schwarzschild
        G = metric.einstein_tensor
        n = metric.dim
        for mu in range(n):
            for nu in range(n):
                assert simplify(G[mu, nu]) == 0

    def test_flat_space_einstein_vanishes(self, flat_5d):
        """G_{μν} = 0 for flat Minkowski."""
        G = flat_5d.einstein_tensor
        n = flat_5d.dim
        for mu in range(n):
            for nu in range(n):
                assert G[mu, nu] == 0

    def test_sphere_einstein_vanishes(self, sphere_2d):
        """Einstein tensor vanishes identically in 2D."""
        metric, _ = sphere_2d
        G = metric.einstein_tensor
        n = metric.dim
        for mu in range(n):
            for nu in range(n):
                assert simplify(G[mu, nu]) == 0

    def test_einstein_symmetry(self, schwarzschild):
        """G_{μν} = G_{νμ}."""
        metric, _ = schwarzschild
        G = metric.einstein_tensor
        n = metric.dim
        for mu in range(n):
            for nu in range(n):
                assert simplify(G[mu, nu] - G[nu, mu]) == 0
