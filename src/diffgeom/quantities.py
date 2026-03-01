"""Shared quantity definitions used by CLI and GUI."""

# Maps quantity name -> (MetricTensor attribute, display name, symbol, is_scalar)
QUANTITY_MAP = {
    "christoffel": ("christoffel_second_kind", "Christoffel symbols", "Gamma", False),
    "riemann": ("riemann_tensor", "Riemann tensor", "R", False),
    "ricci_tensor": ("ricci_tensor", "Ricci tensor", "R", False),
    "ricci_scalar": ("ricci_scalar", "Ricci scalar", "R", True),
    "einstein": ("einstein_tensor", "Einstein tensor", "G", False),
}


def apply_index_spec(metric, tensor, indices: str):
    """Raise/lower indices on *tensor* to match the requested *indices* spec.

    Parameters
    ----------
    metric : MetricTensor
        The metric tensor (provides raise_index/lower_index).
    tensor : Tensor
        The tensor whose indices will be adjusted.
    indices : str
        A string of 'u' and 'd' characters specifying the desired index positions.

    Returns
    -------
    Tensor
        A new tensor with indices matching *indices*.
    """
    target = tuple("up" if c == "u" else "down" for c in indices)
    for i, (current, desired) in enumerate(zip(tensor.index_pos, target)):
        if current != desired:
            if desired == "up":
                tensor = metric.raise_index(tensor, i)
            else:
                tensor = metric.lower_index(tensor, i)
    return tensor
