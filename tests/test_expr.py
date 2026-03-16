"""Tests for the tensor expression engine (parser + evaluator)."""

import subprocess
import sys
from pathlib import Path

import pytest
from sympy import Rational, Symbol, simplify

from diffgeom.config import build_metric, load_config
from diffgeom.expr import (
    BinOp,
    ExpressionError,
    NumericLiteral,
    ScalarRef,
    TensorRef,
    UnaryMinus,
    evaluate_expression,
    parse_expression,
)
from diffgeom.tensor import Tensor

METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"


# ---------------------------------------------------------------------------
# Helpers — build metrics used across multiple tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sphere_metric():
    config = load_config(METRICS_DIR / "sphere_2d.yaml")
    metric, _ = build_metric(config)
    return metric


@pytest.fixture(scope="module")
def schwarzschild_metric():
    config = load_config(METRICS_DIR / "schwarzschild.yaml")
    metric, _ = build_metric(config)
    return metric


@pytest.fixture(scope="module")
def flat_metric():
    config = load_config(METRICS_DIR / "minkowski_flat.yaml")
    metric, _ = build_metric(config)
    return metric


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_parse_tensor_two_down(self):
        ast = parse_expression("Ric{_a_b}")
        assert isinstance(ast, TensorRef)
        assert ast.name == "Ric"
        assert len(ast.indices) == 2
        assert ast.indices[0].label == "a"
        assert ast.indices[0].position == "down"
        assert ast.indices[1].label == "b"
        assert ast.indices[1].position == "down"

    def test_parse_tensor_mixed_indices(self):
        ast = parse_expression("Riem{^a_b_c_d}")
        assert isinstance(ast, TensorRef)
        assert ast.name == "Riem"
        assert len(ast.indices) == 4
        assert ast.indices[0].position == "up"
        assert ast.indices[1].position == "down"

    def test_parse_scalar_ref(self):
        ast = parse_expression("R")
        assert isinstance(ast, ScalarRef)
        assert ast.name == "R"
        assert ast.power == 1

    def test_parse_scalar_power(self):
        ast = parse_expression("R^2")
        assert isinstance(ast, ScalarRef)
        assert ast.name == "R"
        assert ast.power == 2

    def test_parse_product(self):
        ast = parse_expression("Ric{_a_b}*Ric{^a^b}")
        assert isinstance(ast, BinOp)
        assert ast.op == "*"
        assert isinstance(ast.left, TensorRef)
        assert isinstance(ast.right, TensorRef)

    def test_parse_gauss_bonnet(self):
        ast = parse_expression(
            "R^2 - 4*Ric{_a_b}*Ric{^a^b} + Riem{_a_b_c_d}*Riem{^a^b^c^d}"
        )
        # Should be BinOp('+', BinOp('-', ...), ...)
        assert isinstance(ast, BinOp)
        assert ast.op == "+"

    def test_parse_numeric_coefficient(self):
        ast = parse_expression("4*R")
        assert isinstance(ast, BinOp)
        assert ast.op == "*"
        assert isinstance(ast.left, NumericLiteral)
        assert ast.left.value == 4

    def test_parse_parenthesized(self):
        ast = parse_expression("(R + K)")
        assert isinstance(ast, BinOp)
        assert ast.op == "+"

    def test_parse_unary_minus(self):
        ast = parse_expression("-R")
        assert isinstance(ast, UnaryMinus)
        assert isinstance(ast.operand, ScalarRef)

    def test_parse_error_unknown_char(self):
        with pytest.raises(ExpressionError, match="Unexpected character"):
            parse_expression("R & K")

    def test_parse_error_empty_braces(self):
        with pytest.raises(ExpressionError, match="Empty index list"):
            parse_expression("Ric{}")

    def test_parse_error_multi_char_index(self):
        with pytest.raises(ExpressionError, match="single letters"):
            parse_expression("Ric{_ab_c}")


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_scalar_R_identity(self, sphere_metric):
        """R evaluates to the Ricci scalar."""
        result, free = evaluate_expression("R", sphere_metric)
        expected = sphere_metric.ricci_scalar
        assert simplify(result - expected) == 0
        assert free == []

    def test_scalar_R_squared(self, sphere_metric):
        """R^2 evaluates to the square of the Ricci scalar."""
        result, free = evaluate_expression("R^2", sphere_metric)
        expected = sphere_metric.ricci_scalar ** 2
        assert simplify(result - expected) == 0

    def test_kretschmann_scalar(self, sphere_metric):
        """K evaluates to the Kretschmann scalar."""
        result, free = evaluate_expression("K", sphere_metric)
        expected = sphere_metric.kretschmann_scalar
        assert simplify(result - expected) == 0

    def test_metric_tensor_identity(self, sphere_metric):
        """g{_a_b} matches the metric matrix."""
        result, free = evaluate_expression("g{_a_b}", sphere_metric)
        assert isinstance(result, Tensor)
        assert result.index_pos == ("down", "down")
        for i in range(sphere_metric.dim):
            for j in range(sphere_metric.dim):
                assert simplify(result[i, j] - sphere_metric[i, j]) == 0

    def test_ricci_raised_index(self, sphere_metric):
        """Ric{^a_b} produces rank-2 (up, down)."""
        result, free = evaluate_expression("Ric{^a_b}", sphere_metric)
        assert isinstance(result, Tensor)
        assert result.index_pos == ("up", "down")
        assert len(free) == 2
        assert free[0].position == "up"
        assert free[1].position == "down"

    def test_ricci_squared_sphere(self, sphere_metric):
        """Ric{_a_b}*Ric{^a^b} on 2-sphere matches manual computation."""
        result, free = evaluate_expression("Ric{_a_b}*Ric{^a^b}", sphere_metric)
        assert free == []
        # On a 2-sphere of radius R: R_ab = (1/R^2) g_ab
        # R_{ab} R^{ab} = g^{ac} g^{bd} R_{ab} R_{cd} = (1/R^4) * g^{ac}g^{bd} g_{ab} g_{cd}
        # = (1/R^4) * delta^c_b delta^d_d ... = (1/R^4) * n = 2/R^4
        # Actually: Ric_{ab} = g_{ab}/R^2 on unit sphere, so
        # Ric_{ab}Ric^{ab} = (1/R^4) g_{ab} g^{ab} = (1/R^4)*n = 2/R^4
        R = Symbol("R", positive=True)
        expected = Rational(2, 1) / R**4
        assert simplify(result - expected) == 0

    def test_kretschmann_via_expression_schwarzschild(self, schwarzschild_metric):
        """Riem{_a_b_c_d}*Riem{^a^b^c^d} matches kretschmann_scalar."""
        result, free = evaluate_expression(
            "Riem{_a_b_c_d}*Riem{^a^b^c^d}", schwarzschild_metric
        )
        assert free == []
        expected = schwarzschild_metric.kretschmann_scalar
        assert simplify(result - expected) == 0

    def test_gauss_bonnet_sphere(self, sphere_metric):
        """Gauss-Bonnet invariant on 2-sphere.

        G_GB = R^2 - 4*R_{ab}*R^{ab} + R_{abcd}*R^{abcd}
        On a 2-sphere: R = 2/R^2, Ric_{ab}Ric^{ab} = 2/R^4,
        Kretschmann = 4/R^4.
        G_GB = 4/R^4 - 8/R^4 + 4/R^4 = 0
        """
        result, free = evaluate_expression(
            "R^2 - 4*Ric{_a_b}*Ric{^a^b} + Riem{_a_b_c_d}*Riem{^a^b^c^d}",
            sphere_metric,
        )
        assert free == []
        assert simplify(result) == 0

    def test_numeric_times_tensor(self, sphere_metric):
        """Numeric coefficient multiplied with a tensor."""
        result, free = evaluate_expression("2*Ric{_a_b}", sphere_metric)
        assert isinstance(result, Tensor)
        ric = sphere_metric.ricci_tensor
        for i in range(sphere_metric.dim):
            for j in range(sphere_metric.dim):
                assert simplify(result[i, j] - 2 * ric[i, j]) == 0

    def test_flat_ricci_vanishes(self, flat_metric):
        """Ric{_a_b}*Ric{^a^b} = 0 on flat space."""
        result, free = evaluate_expression("Ric{_a_b}*Ric{^a^b}", flat_metric)
        assert simplify(result) == 0

    def test_negation(self, sphere_metric):
        """-R gives the negation of the Ricci scalar."""
        result, free = evaluate_expression("-R", sphere_metric)
        expected = -sphere_metric.ricci_scalar
        assert simplify(result - expected) == 0


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestEvaluatorErrors:
    def test_unknown_tensor(self, sphere_metric):
        with pytest.raises(ExpressionError, match="Unknown"):
            evaluate_expression("Foo{_a_b}", sphere_metric)

    def test_wrong_index_count(self, sphere_metric):
        with pytest.raises(ExpressionError, match="rank"):
            evaluate_expression("Ric{_a_b_c}", sphere_metric)

    def test_scalar_with_indices(self, sphere_metric):
        with pytest.raises(ExpressionError, match="scalar"):
            evaluate_expression("R{_a}", sphere_metric)

    def test_tensor_without_indices(self, sphere_metric):
        with pytest.raises(ExpressionError, match="rank"):
            evaluate_expression("Ric", sphere_metric)

    def test_mismatched_free_indices_add(self, sphere_metric):
        with pytest.raises(ExpressionError, match="free indices"):
            evaluate_expression("Ric{_a_b} + Ric{_a_c}", sphere_metric)

    def test_repeated_same_position(self, sphere_metric):
        with pytest.raises(ExpressionError, match="same position"):
            evaluate_expression("Ric{_a_b}*Ric{_a_b}", sphere_metric)

    def test_triple_repeated_index(self, sphere_metric):
        """Index appearing 3 times should error."""
        with pytest.raises(ExpressionError):
            evaluate_expression("Gam{^a_a_b}*g{_a_b}", sphere_metric)

    def test_add_scalar_to_tensor(self, sphere_metric):
        with pytest.raises(ExpressionError, match="scalar.*tensor"):
            evaluate_expression("R + Ric{_a_b}", sphere_metric)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIExpr:
    def test_expr_flag(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "sphere_2d.yaml"),
                "--quantities", "ricci_scalar",
                "--expr", "Ric{_a_b}*Ric{^a^b}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "Ric{_a_b}*Ric{^a^b}" in result.stdout

    def test_expr_flag_latex(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "sphere_2d.yaml"),
                "--quantities", "ricci_scalar",
                "--expr", "R^2",
                "--latex",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "$" in result.stdout

    def test_expr_invalid_expression(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "sphere_2d.yaml"),
                "--quantities", "ricci_scalar",
                "--expr", "Foo{_a_b}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode != 0
        assert "Error" in result.stderr
