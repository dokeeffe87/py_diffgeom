"""Tests for geodesic equation computation."""

from sympy import Derivative, Eq, Function, Symbol, cos, diag, simplify, sin, symbols

from diffgeom import MetricTensor
from diffgeom.formatting import format_geodesic_equations

lam = Symbol("lambda")


class TestGeodesicFlat:
    """In flat space all geodesic equations reduce to d²x/dλ² = 0."""

    def test_flat_2d(self):
        t, x = symbols("t x")
        g = diag(-1, 1)
        metric = MetricTensor(g, (t, x))
        eqs = metric.geodesic_equations

        t_f = Function("t")(lam)
        x_f = Function("x")(lam)

        assert len(eqs) == 2
        assert eqs[0] == Eq(Derivative(t_f, lam, 2), 0)
        assert eqs[1] == Eq(Derivative(x_f, lam, 2), 0)

    def test_flat_5d(self, flat_5d):
        eqs = flat_5d.geodesic_equations
        assert len(eqs) == 5
        for eq in eqs:
            # Each equation should be d²x/dλ² = 0
            assert eq.rhs == 0
            assert eq.lhs.is_Derivative


class TestGeodesicSphere:
    """Geodesic equations on a 2-sphere of radius R."""

    def test_sphere_geodesic_theta(self, sphere_2d):
        """d²θ/dλ² - sin(θ)cos(θ)(dφ/dλ)² = 0."""
        metric, params = sphere_2d
        eqs = metric.geodesic_equations

        theta_f = Function("theta")(lam)
        phi_f = Function("phi")(lam)

        # θ equation: d²θ/dλ² - sin(θ)cos(θ)(dφ/dλ)² = 0
        expected_lhs = (
            Derivative(theta_f, lam, 2)
            - sin(theta_f) * cos(theta_f) * Derivative(phi_f, lam) ** 2
        )
        assert simplify(eqs[0].lhs - expected_lhs) == 0

    def test_sphere_geodesic_phi(self, sphere_2d):
        """d²φ/dλ² + 2(cos(θ)/sin(θ))(dθ/dλ)(dφ/dλ) = 0."""
        metric, params = sphere_2d
        eqs = metric.geodesic_equations

        theta_f = Function("theta")(lam)
        phi_f = Function("phi")(lam)

        # φ equation: d²φ/dλ² + 2 cos(θ)/sin(θ) dθ/dλ dφ/dλ = 0
        expected_lhs = (
            Derivative(phi_f, lam, 2)
            + 2 * cos(theta_f) / sin(theta_f)
            * Derivative(theta_f, lam)
            * Derivative(phi_f, lam)
        )
        assert simplify(eqs[1].lhs - expected_lhs) == 0


class TestGeodesicArbitraryFunction:
    """Geodesic equations with an undetermined function contain its derivatives."""

    def test_contains_function_derivative(self):
        t, x = symbols("t x")
        f = Function("f")
        g = diag(1, f(x))
        metric = MetricTensor(g, (t, x))
        eqs = metric.geodesic_equations

        assert len(eqs) == 2

        # The t equation should be simple (d²t/dλ² = 0) since the metric
        # component for t is constant
        t_f = Function("t")(lam)
        assert eqs[0] == Eq(Derivative(t_f, lam, 2), 0)

        # The x equation should contain f and/or its derivative
        x_eq_str = str(eqs[1])
        assert "f" in x_eq_str


class TestGeodesicFormatting:
    """Test plain and LaTeX formatting of geodesic equations."""

    def test_plain_format(self):
        t, x = symbols("t x")
        g = diag(-1, 1)
        metric = MetricTensor(g, (t, x))
        eqs = metric.geodesic_equations

        output = format_geodesic_equations(eqs, ["t", "x"], latex=False)
        assert "Geodesic equations" in output
        assert "\u03bb" in output  # λ in header
        assert "t:" in output
        assert "x:" in output

    def test_latex_format(self):
        t, x = symbols("t x")
        g = diag(-1, 1)
        metric = MetricTensor(g, (t, x))
        eqs = metric.geodesic_equations

        output = format_geodesic_equations(eqs, ["t", "x"], latex=True)
        assert "Geodesic equations" in output
        # LaTeX output should contain $ delimiters
        assert "$" in output

    def test_equation_count_matches_coordinates(self, sphere_2d):
        metric, _ = sphere_2d
        eqs = metric.geodesic_equations
        coord_names = [str(c) for c in metric.coordinates]
        output = format_geodesic_equations(eqs, coord_names, latex=False)
        # Should have header + one line per coordinate
        lines = output.strip().splitlines()
        assert len(lines) == 1 + len(coord_names)
