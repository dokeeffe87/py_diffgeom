"""Tests for Tensor wrapper, index operations, and contractions."""

import pytest
import sympy
from sympy import Array, simplify, sin

from diffgeom import Tensor, contract, trace

# ---------------------------------------------------------------------------
# Tensor class basics
# ---------------------------------------------------------------------------


class TestTensorBasics:
    def test_creation_and_rank(self):
        arr = Array([[1, 0], [0, 1]])
        t = Tensor(arr, ("down", "down"))
        assert t.rank == 2
        assert t.shape == (2, 2)
        assert t.index_pos == ("down", "down")

    def test_components_property(self):
        arr = Array([1, 2, 3])
        t = Tensor(arr, ("up",))
        assert t.components is arr

    def test_getitem_delegates_to_array(self):
        arr = Array([[1, 2], [3, 4]])
        t = Tensor(arr, ("up", "down"))
        assert t[0, 0] == 1
        assert t[0, 1] == 2
        assert t[1, 0] == 3
        assert t[1, 1] == 4

    def test_repr(self):
        arr = Array([[[0] * 2] * 2] * 2)
        t = Tensor(arr, ("up", "down", "down"))
        r = repr(t)
        assert "rank=3" in r
        assert "up" in r
        assert "down" in r

    def test_invalid_index_pos_length(self):
        arr = Array([[1, 0], [0, 1]])
        with pytest.raises(ValueError, match="does not match"):
            Tensor(arr, ("up",))

    def test_invalid_index_pos_value(self):
        arr = Array([1, 2])
        with pytest.raises(ValueError, match="'up' or 'down'"):
            Tensor(arr, ("left",))

    def test_invalid_components_type(self):
        with pytest.raises(TypeError, match="sympy.Array"):
            Tensor([[1, 2], [3, 4]], ("up", "down"))


# ---------------------------------------------------------------------------
# MetricTensor returns Tensor objects
# ---------------------------------------------------------------------------


class TestMetricReturnsTensor:
    def test_christoffel_first_returns_tensor(self, sphere_2d):
        metric, _ = sphere_2d
        result = metric.christoffel_first_kind
        assert isinstance(result, Tensor)
        assert result.index_pos == ("down", "down", "down")
        assert result.rank == 3

    def test_christoffel_second_returns_tensor(self, sphere_2d):
        metric, _ = sphere_2d
        result = metric.christoffel_second_kind
        assert isinstance(result, Tensor)
        assert result.index_pos == ("up", "down", "down")
        assert result.rank == 3

    def test_riemann_returns_tensor(self, sphere_2d):
        metric, _ = sphere_2d
        result = metric.riemann_tensor
        assert isinstance(result, Tensor)
        assert result.index_pos == ("up", "down", "down", "down")
        assert result.rank == 4

    def test_ricci_returns_tensor(self, sphere_2d):
        metric, _ = sphere_2d
        result = metric.ricci_tensor
        assert isinstance(result, Tensor)
        assert result.index_pos == ("down", "down")
        assert result.rank == 2

    def test_einstein_returns_tensor(self, sphere_2d):
        metric, _ = sphere_2d
        result = metric.einstein_tensor
        assert isinstance(result, Tensor)
        assert result.index_pos == ("down", "down")
        assert result.rank == 2

    def test_ricci_scalar_still_expr(self, sphere_2d):
        metric, _ = sphere_2d
        result = metric.ricci_scalar
        assert isinstance(result, sympy.Expr)
        assert not isinstance(result, Tensor)


# ---------------------------------------------------------------------------
# raise_index and lower_index
# ---------------------------------------------------------------------------


class TestRaiseAndLowerIndex:
    def test_raise_ricci_index_sphere(self, sphere_2d):
        """Raise index 0 of R_{μν} on 2-sphere to get R^μ_ν.

        R^0_0 = g^{00} R_{00} = (1/R²)(1) = 1/R²
        R^1_1 = g^{11} R_{11} = (1/(R² sin²θ))(sin²θ) = 1/R²
        Off-diagonals are zero.
        """
        metric, params = sphere_2d
        R_sym = params["R"]
        Ric_up = metric.raise_index(metric.ricci_tensor, 0)

        assert Ric_up.index_pos == ("up", "down")
        assert simplify(Ric_up[0, 0] - 1 / R_sym**2) == 0
        assert simplify(Ric_up[1, 1] - 1 / R_sym**2) == 0
        assert simplify(Ric_up[0, 1]) == 0
        assert simplify(Ric_up[1, 0]) == 0

    def test_raise_then_lower_is_identity(self, sphere_2d):
        """Raising then lowering the same index recovers the original tensor."""
        metric, _ = sphere_2d
        Ric = metric.ricci_tensor
        Ric_up = metric.raise_index(Ric, 0)
        Ric_back = metric.lower_index(Ric_up, 0)

        n = metric.dim
        for mu in range(n):
            for nu in range(n):
                assert simplify(Ric_back[mu, nu] - Ric[mu, nu]) == 0

    def test_lower_riemann_first_index_antisymmetry(self, sphere_2d):
        """Lowering R^ρ_{σμν} → R_{ρσμν} should satisfy R_{abcd} = -R_{bacd}."""
        metric, _ = sphere_2d
        Riem_lower = metric.lower_index(metric.riemann_tensor, 0)

        assert Riem_lower.index_pos == ("down", "down", "down", "down")
        n = metric.dim
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        assert simplify(
                            Riem_lower[a, b, c, d] + Riem_lower[b, a, c, d]
                        ) == 0

    def test_raise_already_up_raises_error(self, sphere_2d):
        metric, _ = sphere_2d
        with pytest.raises(ValueError, match="already 'up'"):
            metric.raise_index(metric.riemann_tensor, 0)

    def test_lower_already_down_raises_error(self, sphere_2d):
        metric, _ = sphere_2d
        with pytest.raises(ValueError, match="already 'down'"):
            metric.lower_index(metric.ricci_tensor, 0)

    def test_raise_index_out_of_range(self, sphere_2d):
        metric, _ = sphere_2d
        with pytest.raises(ValueError, match="out of range"):
            metric.raise_index(metric.ricci_tensor, 5)


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


class TestTrace:
    def test_trace_ricci_gives_scalar(self, sphere_2d):
        """trace(R^μ_ν, 0, 1) = Ricci scalar = 2/R² on 2-sphere."""
        metric, params = sphere_2d
        R_sym = params["R"]
        Ric_up = metric.raise_index(metric.ricci_tensor, 0)

        result = trace(Ric_up, 0, 1)
        assert isinstance(result, sympy.Expr)
        assert simplify(result - 2 / R_sym**2) == 0

    def test_trace_riemann_gives_ricci(self, sphere_2d):
        """trace(R^λ_{μλν}, 0, 2) reproduces the Ricci tensor."""
        metric, params = sphere_2d
        theta = params["theta"]
        Riem = metric.riemann_tensor

        Ric_from_trace = trace(Riem, 0, 2)
        assert isinstance(Ric_from_trace, Tensor)
        assert Ric_from_trace.index_pos == ("down", "down")

        assert simplify(Ric_from_trace[0, 0] - 1) == 0
        assert simplify(Ric_from_trace[1, 1] - sin(theta) ** 2) == 0
        assert simplify(Ric_from_trace[0, 1]) == 0

    def test_trace_same_position_raises_error(self, sphere_2d):
        metric, _ = sphere_2d
        with pytest.raises(ValueError, match="Cannot trace"):
            trace(metric.ricci_tensor, 0, 1)

    def test_trace_same_index_raises_error(self, sphere_2d):
        metric, _ = sphere_2d
        with pytest.raises(ValueError, match="must be different"):
            trace(metric.riemann_tensor, 0, 0)


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_kretschner_scalar_sphere(self, sphere_2d):
        """Kretschner scalar K = R_{abcd} R^{abcd} = 4/R⁴ for a 2-sphere."""
        metric, params = sphere_2d
        R_sym = params["R"]

        Riem = metric.riemann_tensor
        R_lower = metric.lower_index(Riem, 0)

        R_upper = R_lower
        for i in range(4):
            R_upper = metric.raise_index(R_upper, i)
        assert R_upper.index_pos == ("up", "up", "up", "up")

        K = contract(R_lower, R_upper, [(0, 0), (1, 1), (2, 2), (3, 3)])
        assert isinstance(K, sympy.Expr)
        assert simplify(K - 4 / R_sym**4) == 0

    def test_contract_metric_with_inverse_gives_identity(self, sphere_2d):
        """g_{μα} g^{αν} = δ^ν_μ."""
        metric, _ = sphere_2d
        n = metric.dim
        g_down = Tensor(Array(metric.matrix.tolist()), ("down", "down"))
        g_up = Tensor(Array(metric.inverse.tolist()), ("up", "up"))

        result = contract(g_down, g_up, [(1, 0)])
        assert isinstance(result, Tensor)
        assert result.index_pos == ("down", "up")
        assert result.rank == 2

        for mu in range(n):
            for nu in range(n):
                expected = 1 if mu == nu else 0
                assert simplify(result[mu, nu] - expected) == 0

    def test_contract_incompatible_positions_raises_error(self, sphere_2d):
        metric, _ = sphere_2d
        Ric = metric.ricci_tensor
        with pytest.raises(ValueError, match="same position"):
            contract(Ric, Ric, [(0, 0)])

    def test_contract_duplicate_index_raises_error(self, sphere_2d):
        metric, _ = sphere_2d
        Ric_up = metric.raise_index(metric.ricci_tensor, 0)
        with pytest.raises(ValueError, match="multiple pairs"):
            contract(metric.riemann_tensor, Ric_up, [(1, 0), (2, 0)])
