# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

py_diffgeom is a Python library for symbolic differential geometry computations, with applications to General Relativity. It uses SymPy for all symbolic math. All geometric objects support arbitrary spacetime dimensions.

## Commands

```bash
# Setup (one-time)
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_metric.py::TestChristoffelSymbols::test_schwarzschild_christoffel_gamma_r_tt -v

# Lint
ruff check src/ tests/

# CLI — compute geometric quantities from a YAML config
diffgeom compute metrics/schwarzschild.yaml
diffgeom compute metrics/sphere_2d.yaml --latex
diffgeom compute metrics/minkowski_flat.yaml --quantities christoffel,ricci_scalar
```

## CLI & Config

The `diffgeom compute` command reads a YAML config defining a metric and prints computed geometric quantities. Use `--latex` for LaTeX output. Use `--quantities` (comma-separated) to override the config's compute list.

YAML config format:
- `name` (optional): label for the metric
- `coordinates`: list of coordinate symbol names
- `assumptions` (optional): dict mapping symbol names to SymPy assumption kwargs (e.g. `positive: true`)
- `metric`: n×n list-of-lists of symbolic expressions
- `compute` (optional): list from `christoffel`, `riemann`, `ricci_tensor`, `ricci_scalar`, `einstein`. Defaults to all.

Example configs live in `metrics/`.

## Architecture

- **`src/diffgeom/`** — main package (installed as `diffgeom`)
- **`src/diffgeom/tensor.py`** — `Tensor` class: lightweight wrapper around `sympy.Array` that tracks index positions (`'up'`/`'down'`). Also provides standalone `trace()` and `contract()` functions for index contraction operations.
- **`src/diffgeom/metric.py`** — `MetricTensor` class: core object that holds a symbolic metric matrix and coordinate symbols. Computes inverse metric, determinant, Christoffel symbols (both kinds), Riemann tensor, Ricci tensor, Ricci scalar, and Einstein tensor — all returned as `Tensor` objects with index metadata. Provides `raise_index()` and `lower_index()` methods for index manipulation.
- **`src/diffgeom/config.py`** — `load_config()` and `build_metric()`: YAML config loading, validation, and MetricTensor construction. Reusable by future interfaces (GUI, notebook).
- **`src/diffgeom/formatting.py`** — `format_tensor()`, `format_scalar()`, `format_metric_summary()`: output formatting for pretty-print and LaTeX. Shows only non-zero components with coordinate-name indices.
- **`src/diffgeom/cli.py`** — argparse CLI entry point. Thin orchestration: loads config, computes quantities, prints formatted output.
- **`metrics/`** — example YAML metric configs (Schwarzschild, 2-sphere, 5D Minkowski).
- **`tests/`** — pytest tests using known exact solutions (Schwarzschild, 2-sphere, flat 5D Minkowski) to verify computations against textbook results.

## Design Conventions

- All computations are symbolic via SymPy (no numerics). Results are simplified with `sympy.simplify`.
- Expensive derived quantities (inverse, Christoffel symbols) use `@cached_property` so they are computed once on first access.
- Tensor components are wrapped in `Tensor` objects that pair a `sympy.Array` with index position metadata (`'up'`/`'down'`). Index ordering is documented in docstrings.
- Tests verify against known analytic results from GR textbooks rather than using snapshot/regression testing.
