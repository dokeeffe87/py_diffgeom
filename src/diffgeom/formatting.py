"""Output formatting for tensors and scalars (pretty-print and LaTeX)."""

from itertools import product as iproduct

import sympy

from diffgeom.tensor import Tensor


def format_metric_summary(config: dict, metric, latex: bool = False) -> str:
    """Format a header with metric name, dimension, and coordinates.

    Parameters
    ----------
    config : dict
        The config dictionary from :func:`diffgeom.config.load_config`.
    metric : MetricTensor
        The metric tensor.
    latex : bool
        If True, use LaTeX formatting.

    Returns
    -------
    str
        The formatted header string.
    """
    name = config.get("name") or "Unnamed metric"
    dim = metric.dim
    coord_names = [str(c) for c in metric.coordinates]
    coord_str = ", ".join(coord_names)

    lines = []
    if latex:
        lines.append(f"{name} ({dim}D)")
        lines.append(f"Coordinates: $({coord_str})$")
    else:
        lines.append(f"{name} ({dim}D)")
        lines.append(f"Coordinates: ({coord_str})")
    return "\n".join(lines)


def format_tensor(
    tensor: Tensor,
    name: str,
    symbol: str,
    coord_names: list[str],
    latex: bool = False,
) -> str:
    """Format a Tensor showing only non-zero components.

    Parameters
    ----------
    tensor : Tensor
        The tensor to format.
    name : str
        Display name (e.g. "Christoffel symbols").
    symbol : str
        The symbol used for components (e.g. "Gamma", "R").
    coord_names : list of str
        Coordinate names for index labels.
    latex : bool
        If True, use LaTeX formatting.

    Returns
    -------
    str
        The formatted tensor string.
    """
    rank = tensor.rank
    dim = tensor.shape[0]
    index_pos = tensor.index_pos

    # Build index label string for the header
    header_indices = _index_label(symbol, index_pos, latex=latex)

    lines = []
    if latex:
        lines.append(f"{name} ${header_indices}$ (non-zero components):")
    else:
        lines.append(f"{name} {header_indices} (non-zero components):")

    # Collect non-zero components
    nonzero = []
    for indices in iproduct(range(dim), repeat=rank):
        value = tensor[indices]
        if sympy.simplify(value) != 0:
            comp_str = _component_str(symbol, indices, index_pos, coord_names, latex=latex)
            if latex:
                expr_str = sympy.latex(value)
                nonzero.append(f"  ${comp_str} = {expr_str}$")
            else:
                expr_str = str(value)
                nonzero.append(f"  {comp_str} = {expr_str}")

    if nonzero:
        lines.extend(nonzero)
    else:
        lines.append("  All components vanish.")

    return "\n".join(lines)


def format_scalar(expr: sympy.Expr, name: str, symbol: str, latex: bool = False) -> str:
    """Format a scalar quantity.

    Parameters
    ----------
    expr : sympy.Expr
        The scalar expression.
    name : str
        Display name (e.g. "Ricci scalar").
    symbol : str
        The symbol (e.g. "R").
    latex : bool
        If True, use LaTeX formatting.

    Returns
    -------
    str
        The formatted scalar string.
    """
    if latex:
        expr_str = sympy.latex(expr)
        return f"{name}:\n  ${symbol} = {expr_str}$"
    else:
        return f"{name}:\n  {symbol} = {expr}"


# Greek letter names that need a backslash prefix in LaTeX
_GREEK_NAMES = frozenset({
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma",
    "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega",
})


def _latex_coord(name: str) -> str:
    """Prefix a coordinate name with backslash if it's a Greek letter."""
    if name in _GREEK_NAMES:
        return f"\\{name}"
    # Handle multi-char names with underscore subscripts (e.g. r_s -> r_{s})
    if "_" in name:
        base, sub = name.split("_", 1)
        prefix = f"\\{base}" if base in _GREEK_NAMES else base
        return f"{prefix}_{{{sub}}}"
    return name


def _index_label(symbol: str, index_pos: tuple[str, ...], latex: bool = False) -> str:
    """Build an abstract index label like Gamma^a_bc or \\Gamma^{a}_{bc}."""
    # Use generic Greek letters for abstract indices
    greek = ["sigma", "mu", "nu", "rho", "alpha", "beta", "gamma", "delta"]
    up_indices = []
    down_indices = []
    for i, pos in enumerate(index_pos):
        letter = greek[i] if i < len(greek) else f"i{i}"
        if pos == "up":
            up_indices.append(letter)
        else:
            down_indices.append(letter)

    if latex:
        up_str = "".join(f"\\{letter}" for letter in up_indices)
        down_str = "".join(f"\\{letter}" for letter in down_indices)
        result = f"\\{symbol}" if symbol in ("Gamma",) else symbol
        if up_str:
            result += f"^{{{up_str}}}"
        if down_str:
            result += f"_{{{down_str}}}"
        return result
    else:
        # Unicode-style: Gamma^sigma_mu_nu
        _greek_map = {
            "sigma": "\u03c3", "mu": "\u03bc", "nu": "\u03bd", "rho": "\u03c1",
            "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "delta": "\u03b4",
        }
        up_str = "".join(_greek_map.get(letter, letter) for letter in up_indices)
        down_str = "".join(_greek_map.get(letter, letter) for letter in down_indices)
        _symbol_map = {"Gamma": "\u0393", "R": "R", "G": "G"}
        sym = _symbol_map.get(symbol, symbol)
        result = sym
        if up_str:
            result += f"^{up_str}"
        if down_str:
            result += f"_{down_str}"
        return result


def _component_str(
    symbol: str,
    indices: tuple[int, ...],
    index_pos: tuple[str, ...],
    coord_names: list[str],
    latex: bool = False,
) -> str:
    """Build a component string like Gamma^r_tt or \\Gamma^{r}_{t t}."""
    up_parts = []
    down_parts = []
    for idx, pos in zip(indices, index_pos):
        name = coord_names[idx]
        if pos == "up":
            up_parts.append(name)
        else:
            down_parts.append(name)

    if latex:
        base = f"\\{symbol}" if symbol in ("Gamma",) else symbol
        up_str = " ".join(_latex_coord(p) for p in up_parts)
        down_str = " ".join(_latex_coord(p) for p in down_parts)
        result = base
        if up_str:
            result += f"^{{{up_str}}}"
        if down_str:
            result += f"_{{{down_str}}}"
        return result
    else:
        _symbol_map = {"Gamma": "\u0393", "R": "R", "G": "G"}
        sym = _symbol_map.get(symbol, symbol)
        up_str = ",".join(up_parts)
        down_str = ",".join(down_parts)
        result = sym
        if up_str:
            result += f"^{up_str}"
        if down_str:
            result += f"_{down_str}"
        return result
