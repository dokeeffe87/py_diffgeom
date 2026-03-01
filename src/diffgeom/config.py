"""YAML config loading and MetricTensor construction."""

from pathlib import Path

import yaml
from sympy import Function, Matrix, Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from diffgeom.metric import MetricTensor

VALID_QUANTITIES = frozenset({
    "christoffel",
    "riemann",
    "ricci_tensor",
    "ricci_scalar",
    "einstein",
    "geodesic",
})

# Default index positions for each quantity (None = not applicable, e.g. scalar or geodesic).
DEFAULT_INDEX_POS = {
    "christoffel": "udd",
    "riemann": "uddd",
    "ricci_tensor": "dd",
    "einstein": "dd",
    "ricci_scalar": None,
    "geodesic": None,
}

# Transformations for parse_expr: standard + implicit multiplication + ^ as **
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def validate_config(config: dict) -> dict:
    """Validate and normalize a metric config dictionary.

    Parameters
    ----------
    config : dict
        A raw config dictionary (e.g. from ``yaml.safe_load``).

    Returns
    -------
    dict
        The validated config dictionary with keys: ``name``, ``coordinates``,
        ``assumptions``, ``metric``, ``compute``.

    Raises
    ------
    ValueError
        If required fields are missing or the metric dimensions are wrong.
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")

    # Required fields
    if "coordinates" not in config:
        raise ValueError("Config must include 'coordinates'.")
    if "metric" not in config:
        raise ValueError("Config must include 'metric'.")

    coords = config["coordinates"]
    if not isinstance(coords, list) or len(coords) == 0:
        raise ValueError("'coordinates' must be a non-empty list.")

    n = len(coords)
    metric_rows = config["metric"]
    if not isinstance(metric_rows, list) or len(metric_rows) != n:
        raise ValueError(
            f"'metric' must be a {n}x{n} list-of-lists matching "
            f"the number of coordinates ({n})."
        )
    for i, row in enumerate(metric_rows):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(
                f"Metric row {i} has length {len(row) if isinstance(row, list) else 'N/A'}, "
                f"expected {n}."
            )

    # Validate functions field
    functions = config.get("functions", [])
    if not isinstance(functions, list):
        raise ValueError("'functions' must be a list of function names.")
    for fn in functions:
        if not isinstance(fn, str):
            raise ValueError(
                f"Each entry in 'functions' must be a string, got: {type(fn).__name__}"
            )

    # Defaults
    config.setdefault("name", None)
    config.setdefault("assumptions", {})
    config.setdefault("functions", [])
    config.setdefault("compute", list(VALID_QUANTITIES))

    # Normalize compute list into (name, indices_or_None) tuples
    config["compute"] = _parse_compute_list(config["compute"])

    return config


def load_config(path: str | Path) -> dict:
    """Load and validate a metric config from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
        The validated config dictionary with keys: ``name``, ``coordinates``,
        ``assumptions``, ``metric``, ``compute``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required fields are missing or the metric dimensions are wrong.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return validate_config(config)


def _parse_compute_list(raw: list) -> list[tuple[str, str | None]]:
    """Normalize a compute list into ``(quantity_name, indices_or_None)`` tuples.

    Accepts both plain strings (``"christoffel"``) and dicts
    (``{"riemann": {"indices": "dddd"}}``).
    """
    result = []
    for entry in raw:
        if isinstance(entry, str):
            name, indices = entry, None
        elif isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(
                    f"Dict entries in 'compute' must have exactly one key, got: {entry}"
                )
            name = next(iter(entry))
            opts = entry[name]
            indices = opts.get("indices") if isinstance(opts, dict) else None
        else:
            raise ValueError(
                f"Each 'compute' entry must be a string or dict, got: {type(entry).__name__}"
            )

        if name not in VALID_QUANTITIES:
            raise ValueError(
                f"Unknown quantity '{name}'. "
                f"Valid values: {', '.join(sorted(VALID_QUANTITIES))}"
            )

        if indices is not None:
            _validate_indices(name, indices)

        result.append((name, indices))
    return result


def _validate_indices(name: str, indices: str) -> None:
    """Validate an index spec string against the quantity's rank."""
    default = DEFAULT_INDEX_POS[name]
    if default is None:
        raise ValueError(
            f"'{name}' is a scalar quantity and does not accept an index spec."
        )
    if not all(c in "ud" for c in indices):
        raise ValueError(
            f"Invalid index spec '{indices}' for '{name}': "
            f"only 'u' (up) and 'd' (down) characters are allowed."
        )
    expected_len = len(default)
    if len(indices) != expected_len:
        raise ValueError(
            f"Index spec '{indices}' for '{name}' has length {len(indices)}, "
            f"expected {expected_len} (default: {default})."
        )


def parse_quantities_flag(flag: str) -> list[tuple[str, str | None]]:
    """Parse a ``--quantities`` CLI flag value into ``(name, indices)`` tuples.

    Supports ``name`` and ``name:indices`` syntax, comma-separated.
    """
    result = []
    for token in flag.split(","):
        token = token.strip()
        if ":" in token:
            name, indices = token.split(":", 1)
        else:
            name, indices = token, None

        if name not in VALID_QUANTITIES:
            raise ValueError(
                f"Unknown quantity '{name}'. "
                f"Valid values: {', '.join(sorted(VALID_QUANTITIES))}"
            )
        if indices is not None:
            _validate_indices(name, indices)

        result.append((name, indices))
    return result


def build_metric(config: dict) -> tuple[MetricTensor, dict[str, Symbol]]:
    """Build a MetricTensor from a validated config dictionary.

    Parameters
    ----------
    config : dict
        A config dictionary as returned by :func:`load_config`.

    Returns
    -------
    tuple of (MetricTensor, dict)
        The metric tensor and a dict mapping all symbol names (coordinates
        and assumption parameters) to their SymPy Symbol objects.
    """
    # Build symbols with assumptions
    symbols_dict: dict[str, Symbol] = {}

    # Assumption parameters (e.g. r_s: {positive: true})
    for name, assumptions in config.get("assumptions", {}).items():
        if isinstance(assumptions, dict):
            symbols_dict[name] = Symbol(name, **assumptions)
        else:
            symbols_dict[name] = Symbol(name)

    # Coordinate symbols
    coord_symbols = []
    for name in config["coordinates"]:
        if name in symbols_dict:
            coord_symbols.append(symbols_dict[name])
        else:
            sym = Symbol(name)
            symbols_dict[name] = sym
            coord_symbols.append(sym)

    # Build local_dict for parse_expr with standard SymPy names
    import sympy
    local_dict = dict(symbols_dict)
    for fn_name in ("sin", "cos", "tan", "exp", "sqrt", "log", "asin", "acos", "atan",
                     "sinh", "cosh", "tanh", "pi", "E", "oo"):
        local_dict.setdefault(fn_name, getattr(sympy, fn_name))

    # Register user-declared arbitrary functions (e.g. f, g, h)
    for fn_name in config.get("functions", []):
        local_dict.setdefault(fn_name, Function(fn_name))

    # Parse metric entries
    rows = []
    for i, row in enumerate(config["metric"]):
        parsed_row = []
        for j, entry in enumerate(row):
            expr = parse_expr(
                str(entry),
                local_dict=local_dict,
                transformations=_TRANSFORMATIONS,
            )
            parsed_row.append(expr)
        rows.append(parsed_row)

    matrix = Matrix(rows)
    metric = MetricTensor(matrix, tuple(coord_symbols))

    return metric, symbols_dict
