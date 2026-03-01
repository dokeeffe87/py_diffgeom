"""Tests for config loading, formatting, and CLI."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import sympy
from sympy import Symbol, simplify, sin

from diffgeom.config import VALID_QUANTITIES, build_metric, load_config, parse_quantities_flag
from diffgeom.formatting import format_metric_summary, format_scalar, format_tensor
from diffgeom.tensor import Tensor

METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_schwarzschild(self):
        config = load_config(METRICS_DIR / "schwarzschild.yaml")
        assert config["name"] == "schwarzschild"
        assert config["coordinates"] == ["t", "r", "theta", "phi"]
        assert len(config["metric"]) == 4
        names = [name for name, _ in config["compute"]]
        assert "christoffel" in names

    def test_load_sphere_2d(self):
        config = load_config(METRICS_DIR / "sphere_2d.yaml")
        assert len(config["coordinates"]) == 2
        assert config["assumptions"]["R"] == {"positive": True}

    def test_load_minkowski(self):
        config = load_config(METRICS_DIR / "minkowski_flat.yaml")
        assert len(config["coordinates"]) == 5
        assert config["assumptions"] == {}

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_missing_coordinates(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("metric:\n  - [1, 0]\n  - [0, 1]\n")
        with pytest.raises(ValueError, match="coordinates"):
            load_config(p)

    def test_missing_metric(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("coordinates: [x, y]\n")
        with pytest.raises(ValueError, match="metric"):
            load_config(p)

    def test_dimension_mismatch(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(textwrap.dedent("""\
            coordinates: [x, y, z]
            metric:
              - [1, 0]
              - [0, 1]
        """))
        with pytest.raises(ValueError, match="3x3"):
            load_config(p)

    def test_row_length_mismatch(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(textwrap.dedent("""\
            coordinates: [x, y]
            metric:
              - [1, 0, 0]
              - [0, 1]
        """))
        with pytest.raises(ValueError, match="row 0"):
            load_config(p)

    def test_invalid_quantity(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(textwrap.dedent("""\
            coordinates: [x, y]
            metric:
              - [1, 0]
              - [0, 1]
            compute:
              - not_a_quantity
        """))
        with pytest.raises(ValueError, match="Unknown quantity"):
            load_config(p)

    def test_defaults_when_omitted(self, tmp_path):
        p = tmp_path / "minimal.yaml"
        p.write_text(textwrap.dedent("""\
            coordinates: [x, y]
            metric:
              - [1, 0]
              - [0, 1]
        """))
        config = load_config(p)
        assert config["name"] is None
        assert config["assumptions"] == {}
        names = {name for name, _ in config["compute"]}
        assert names == VALID_QUANTITIES


# ---------------------------------------------------------------------------
# build_metric
# ---------------------------------------------------------------------------


class TestBuildMetric:
    def test_schwarzschild_metric(self):
        config = load_config(METRICS_DIR / "schwarzschild.yaml")
        metric, symbols_dict = build_metric(config)
        assert metric.dim == 4
        assert "r_s" in symbols_dict
        assert symbols_dict["r_s"].is_positive

    def test_sphere_2d_metric(self):
        config = load_config(METRICS_DIR / "sphere_2d.yaml")
        metric, symbols_dict = build_metric(config)
        assert metric.dim == 2
        R = symbols_dict["R"]
        theta = symbols_dict["theta"]
        # g_{00} = R^2
        assert simplify(metric[0, 0] - R**2) == 0
        # g_{11} = R^2 sin^2(theta)
        assert simplify(metric[1, 1] - R**2 * sin(theta)**2) == 0

    def test_flat_metric(self):
        config = load_config(METRICS_DIR / "minkowski_flat.yaml")
        metric, _ = build_metric(config)
        assert metric.dim == 5
        assert metric[0, 0] == -1
        assert metric[1, 1] == 1
        assert metric[0, 1] == 0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_metric_summary(self):
        config = load_config(METRICS_DIR / "schwarzschild.yaml")
        metric, _ = build_metric(config)
        output = format_metric_summary(config, metric)
        assert "schwarzschild" in output
        assert "4D" in output
        assert "t, r, theta, phi" in output

    def test_format_metric_summary_latex(self):
        config = load_config(METRICS_DIR / "schwarzschild.yaml")
        metric, _ = build_metric(config)
        output = format_metric_summary(config, metric, latex=True)
        assert "$" in output

    def test_format_scalar_zero(self):
        output = format_scalar(sympy.Integer(0), "Ricci scalar", "R")
        assert "R = 0" in output

    def test_format_scalar_nonzero(self):
        R = Symbol("R")
        output = format_scalar(2 / R**2, "Ricci scalar", "R")
        assert "Ricci scalar" in output

    def test_format_scalar_latex(self):
        output = format_scalar(sympy.Integer(0), "Ricci scalar", "R", latex=True)
        assert "$" in output

    def test_format_tensor_all_zero(self):
        arr = sympy.Array([[0, 0], [0, 0]])
        t = Tensor(arr, ("down", "down"))
        output = format_tensor(t, "Test tensor", "T", ["x", "y"])
        assert "All components vanish" in output

    def test_format_tensor_nonzero(self):
        arr = sympy.Array([[1, 0], [0, 2]])
        t = Tensor(arr, ("down", "down"))
        output = format_tensor(t, "Test tensor", "T", ["x", "y"])
        assert "= 1" in output
        assert "= 2" in output
        assert "non-zero" in output

    def test_format_tensor_latex(self):
        arr = sympy.Array([[1, 0], [0, 0]])
        t = Tensor(arr, ("up", "down"))
        output = format_tensor(t, "Test", "T", ["x", "y"], latex=True)
        assert "$" in output


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


class TestCLI:
    def test_compute_sphere_ricci_scalar(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "sphere_2d.yaml"),
                "--quantities", "ricci_scalar",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "Ricci scalar" in result.stdout

    def test_compute_minkowski_christoffel(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "minkowski_flat.yaml"),
                "--quantities", "christoffel",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "All components vanish" in result.stdout

    def test_compute_latex_flag(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "minkowski_flat.yaml"),
                "--quantities", "ricci_scalar",
                "--latex",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "$" in result.stdout

    def test_compute_geodesic_equations(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "sphere_2d.yaml"),
                "--quantities", "geodesic",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "Geodesic equations" in result.stdout
        # Should have one equation per coordinate (theta, phi)
        assert "theta:" in result.stdout
        assert "phi:" in result.stdout

    def test_no_command_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "diffgeom.cli"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "diffgeom" in result.stdout.lower()

    def test_invalid_quantity_flag(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "minkowski_flat.yaml"),
                "--quantities", "bogus",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_compute_with_index_spec_cli(self):
        """--quantities riemann:dddd produces all-down index labels."""
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "minkowski_flat.yaml"),
                "--quantities", "riemann:dddd",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        output = result.stdout
        assert "Riemann tensor" in output
        # All-down: header should NOT contain '^' (no up indices)
        for line in output.splitlines():
            if "Riemann tensor" in line:
                assert "^" not in line
                break

    def test_invalid_index_spec_wrong_length(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "minkowski_flat.yaml"),
                "--quantities", "riemann:dd",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "length 2" in result.stderr

    def test_invalid_index_spec_scalar(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "diffgeom.cli",
                "compute", str(METRICS_DIR / "minkowski_flat.yaml"),
                "--quantities", "ricci_scalar:d",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "scalar" in result.stderr


# ---------------------------------------------------------------------------
# Index spec parsing
# ---------------------------------------------------------------------------


class TestIndexSpecParsing:
    def test_parse_quantities_flag_plain(self):
        result = parse_quantities_flag("christoffel,riemann")
        assert result == [("christoffel", None), ("riemann", None)]

    def test_parse_quantities_flag_with_indices(self):
        result = parse_quantities_flag("riemann:dddd,ricci_tensor:ud")
        assert result == [("riemann", "dddd"), ("ricci_tensor", "ud")]

    def test_parse_quantities_flag_mixed(self):
        result = parse_quantities_flag("christoffel,riemann:dddd")
        assert result == [("christoffel", None), ("riemann", "dddd")]

    def test_parse_quantities_flag_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown quantity"):
            parse_quantities_flag("bogus:dd")

    def test_parse_quantities_flag_wrong_length(self):
        with pytest.raises(ValueError, match="length 2"):
            parse_quantities_flag("riemann:dd")

    def test_parse_quantities_flag_invalid_chars(self):
        with pytest.raises(ValueError, match="only 'u'"):
            parse_quantities_flag("riemann:abcd")

    def test_parse_quantities_flag_scalar_indices(self):
        with pytest.raises(ValueError, match="scalar"):
            parse_quantities_flag("ricci_scalar:d")

    def test_config_dict_entry_with_indices(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text(textwrap.dedent("""\
            coordinates: [x, y]
            metric:
              - [1, 0]
              - [0, 1]
            compute:
              - ricci_tensor: {indices: uu}
        """))
        config = load_config(p)
        assert config["compute"] == [("ricci_tensor", "uu")]

    def test_config_mixed_string_and_dict(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text(textwrap.dedent("""\
            coordinates: [x, y]
            metric:
              - [1, 0]
              - [0, 1]
            compute:
              - christoffel
              - ricci_tensor: {indices: ud}
              - ricci_scalar
        """))
        config = load_config(p)
        assert config["compute"] == [
            ("christoffel", None),
            ("ricci_tensor", "ud"),
            ("ricci_scalar", None),
        ]

    def test_config_invalid_indices_length(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text(textwrap.dedent("""\
            coordinates: [x, y]
            metric:
              - [1, 0]
              - [0, 1]
            compute:
              - riemann: {indices: dd}
        """))
        with pytest.raises(ValueError, match="length 2"):
            load_config(p)

    def test_config_indices_on_scalar(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text(textwrap.dedent("""\
            coordinates: [x, y]
            metric:
              - [1, 0]
              - [0, 1]
            compute:
              - ricci_scalar: {indices: d}
        """))
        with pytest.raises(ValueError, match="scalar"):
            load_config(p)
