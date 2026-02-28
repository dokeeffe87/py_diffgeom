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
```

## Architecture

- **`src/diffgeom/`** — main package (installed as `diffgeom`)
- **`src/diffgeom/metric.py`** — `MetricTensor` class: core object that holds a symbolic metric matrix and coordinate symbols. Computes inverse metric, determinant, and Christoffel symbols (both kinds). Dimension is inferred from the coordinates tuple, not hardcoded.
- **`tests/`** — pytest tests using known exact solutions (Schwarzschild, 2-sphere, flat 5D Minkowski) to verify computations against textbook results.

## Design Conventions

- All computations are symbolic via SymPy (no numerics). Results are simplified with `sympy.simplify`.
- Expensive derived quantities (inverse, Christoffel symbols) use `@cached_property` so they are computed once on first access.
- Tensor components are stored as `sympy.Array` (rank-3 for Christoffel symbols) with explicit index ordering documented in docstrings.
- Tests verify against known analytic results from GR textbooks rather than using snapshot/regression testing.
