# py_diffgeom

A Python library for symbolic differential geometry computations, with applications to General Relativity. Built on [SymPy](https://www.sympy.org/) for fully symbolic math in arbitrary spacetime dimensions.

Given a metric tensor defined in a simple YAML file, py_diffgeom computes:

- **Christoffel symbols** (first and second kind)
- **Riemann curvature tensor**
- **Ricci tensor**
- **Ricci scalar**
- **Einstein tensor**
- **Geodesic equations**

All quantities support index raising/lowering, custom index position specs, and output in both pretty-print and LaTeX formats.

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/dokeeffe87/py_diffgeom.git
cd py_diffgeom
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Optional: install with GUI support
pip install -e ".[gui]"
```

Requires Python 3.10+.

## Quick start

Define a metric in a YAML config file:

```yaml
# metrics/schwarzschild.yaml
name: schwarzschild
coordinates: [t, r, theta, phi]
assumptions:
  r_s: {positive: true}
metric:
  - [-(1 - r_s/r), 0, 0, 0]
  - [0, 1/(1 - r_s/r), 0, 0]
  - [0, 0, r**2, 0]
  - [0, 0, 0, r**2 * sin(theta)**2]
compute:
  - christoffel
  - riemann: {indices: dddd}
  - ricci_tensor
  - ricci_scalar
  - einstein
  - geodesic
```

Then compute:

```bash
diffgeom compute metrics/schwarzschild.yaml
```

## CLI usage

```bash
# Compute all quantities defined in the config
diffgeom compute metrics/sphere_2d.yaml

# Output in LaTeX format
diffgeom compute metrics/sphere_2d.yaml --latex

# Override which quantities to compute
diffgeom compute metrics/schwarzschild.yaml --quantities christoffel,ricci_scalar

# Specify index positions (u=up, d=down)
diffgeom compute metrics/schwarzschild.yaml --quantities riemann:dddd,ricci_tensor
```

Example output for the 2-sphere:

```
2-sphere (2D)
Coordinates: (theta, phi)

Christoffel symbols (non-zero components):
  Gamma^theta_phi,phi = -sin(2*theta)/2
  Gamma^phi_theta,phi = 1/tan(theta)
  Gamma^phi_phi,theta = 1/tan(theta)

Ricci scalar:
  R = 2/R**2

Geodesic equations (affine parameter: lambda):
  theta: d^2(theta)/dlambda^2 - sin(2*theta)*d(phi)/dlambda^2/2 = 0
  phi: d^2(phi)/dlambda^2 + 2*d(phi)/dlambda*d(theta)/dlambda/tan(theta) = 0
```

## Web GUI

A Streamlit-based GUI provides an interactive interface with LaTeX-rendered output:

```bash
diffgeom-gui
```

The GUI supports three input modes:
- **Example config** -- load any YAML file from the `metrics/` directory
- **Upload YAML** -- upload your own config file
- **Manual entry** -- type coordinates and metric components directly in the browser

## YAML config format

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Label for the metric |
| `coordinates` | Yes | List of coordinate symbol names |
| `assumptions` | No | SymPy assumptions for symbols (e.g. `r_s: {positive: true}`) |
| `functions` | No | Names to treat as arbitrary/undefined functions (e.g. `[f, g]`) |
| `metric` | Yes | n x n list-of-lists of symbolic expressions |
| `compute` | No | List of quantities to compute (defaults to all). Supports index specs like `riemann: {indices: dddd}` |

Available quantities for `compute`: `christoffel`, `riemann`, `ricci_tensor`, `ricci_scalar`, `einstein`, `geodesic`.

### Arbitrary functions

You can derive curvature in terms of unspecified functions by declaring them in `functions` and using them in metric components:

```yaml
name: 2D metric with arbitrary function
coordinates: [t, x]
functions: [f]
metric:
  - [1, 0]
  - [0, f(x)]
compute:
  - christoffel
  - ricci_scalar
```

This is useful for exploring how curvature depends on a general metric function without specifying its form.

## Example metrics

The `metrics/` directory includes several example configs:

| File | Description |
|------|-------------|
| `schwarzschild.yaml` | Schwarzschild black hole (4D) |
| `sphere_2d.yaml` | 2-sphere of radius R |
| `minkowski_flat.yaml` | Flat Minkowski spacetime (5D) |
| `kasner.yaml` | Kasner cosmological solution (4D) |
| `morris_thorne_wormhole.yaml` | Morris-Thorne traversable wormhole (4D) |
| `arbitrary_function_2d.yaml` | 2D metric with an unspecified function f(x) |

## Python API

You can also use py_diffgeom as a library:

```python
from diffgeom.config import load_config, build_metric

config = load_config("metrics/schwarzschild.yaml")
metric, coord_names = build_metric(config)

# Access computed quantities (cached on first access)
christoffel = metric.christoffel_second_kind  # Tensor with index_pos ('up', 'down', 'down')
riemann = metric.riemann_tensor               # Tensor with index_pos ('up', 'down', 'down', 'down')
ricci = metric.ricci_tensor                   # Tensor with index_pos ('down', 'down')
scalar = metric.ricci_scalar                  # SymPy expression
einstein = metric.einstein_tensor             # Tensor with index_pos ('down', 'down')
geodesics = metric.geodesic_equations         # List of SymPy equations

# Raise/lower indices
riemann_all_down = metric.lower_index(riemann, 0)

# Inverse metric and determinant
g_inv = metric.inverse
det = metric.determinant
```

## Running tests

```bash
python -m pytest tests/ -v
```

Tests verify computations against known analytic results from GR textbooks (Schwarzschild, 2-sphere, flat Minkowski).

## License

MIT
