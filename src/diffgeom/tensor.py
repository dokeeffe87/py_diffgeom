"""Tensor wrapper and contraction operations."""

from __future__ import annotations

import sympy
from sympy import Array, simplify, tensorcontraction, tensorproduct


class Tensor:
    """A lightweight wrapper around a sympy.Array that tracks index positions.

    Parameters
    ----------
    components : sympy.Array
        The tensor component data.
    index_pos : tuple of str
        One entry per index, each either ``'up'`` or ``'down'``.
        Length must match the rank of *components*.

    Examples
    --------
    >>> from sympy import Array
    >>> T = Tensor(Array([[1, 0], [0, 1]]), ('down', 'down'))
    >>> T.rank
    2
    >>> T[0, 1]
    0
    """

    def __init__(self, components: Array, index_pos: tuple[str, ...]):
        if not isinstance(components, Array):
            raise TypeError(
                f"components must be a sympy.Array, got {type(components).__name__}."
            )
        rank = components.rank()
        if len(index_pos) != rank:
            raise ValueError(
                f"index_pos length ({len(index_pos)}) does not match "
                f"tensor rank ({rank})."
            )
        for pos in index_pos:
            if pos not in ("up", "down"):
                raise ValueError(
                    f"Each index position must be 'up' or 'down', got '{pos}'."
                )
        self._components = components
        self._index_pos = tuple(index_pos)

    @property
    def components(self) -> Array:
        """Raw sympy.Array of tensor components."""
        return self._components

    @property
    def index_pos(self) -> tuple[str, ...]:
        """Tuple of index positions ('up' or 'down')."""
        return self._index_pos

    @property
    def rank(self) -> int:
        """Tensor rank (number of indices)."""
        return self._components.rank()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying component array."""
        return self._components.shape

    def __getitem__(self, key):
        """Access components by index, delegating to the underlying Array."""
        return self._components[key]

    def __repr__(self) -> str:
        pos_str = ", ".join(self._index_pos)
        return f"Tensor(rank={self.rank}, index_pos=({pos_str}))"


def trace(tensor: Tensor, pos1: int, pos2: int) -> Tensor | sympy.Expr:
    """Contract two indices of a tensor (trace operation).

    One index must be ``'up'`` and the other ``'down'``.

    Parameters
    ----------
    tensor : Tensor
        The tensor to trace over.
    pos1, pos2 : int
        Positions of the two indices to contract.

    Returns
    -------
    Tensor or sympy.Expr
        The traced result. A scalar ``sympy.Expr`` if the result has rank 0.
    """
    if pos1 == pos2:
        raise ValueError("pos1 and pos2 must be different indices.")
    r = tensor.rank
    for p in (pos1, pos2):
        if not (0 <= p < r):
            raise ValueError(f"Index position {p} out of range for rank-{r} tensor.")
    if tensor.index_pos[pos1] == tensor.index_pos[pos2]:
        raise ValueError(
            f"Cannot trace two '{tensor.index_pos[pos1]}' indices. "
            f"One must be 'up' and the other 'down'."
        )

    result = tensorcontraction(tensor.components, (pos1, pos2))

    remaining = tuple(
        tensor.index_pos[i] for i in range(r) if i != pos1 and i != pos2
    )

    if isinstance(result, sympy.Expr) and not isinstance(result, Array):
        return simplify(result)

    result = Array(result).applyfunc(simplify)
    return Tensor(result, remaining)


def contract(
    tensor_a: Tensor,
    tensor_b: Tensor,
    index_pairs: list[tuple[int, int]],
) -> Tensor | sympy.Expr:
    """Contract indices between two tensors.

    Forms the tensor product and contracts specified index pairs.
    For each pair ``(i, j)``, index *i* of *tensor_a* is contracted with
    index *j* of *tensor_b*. One must be ``'up'`` and the other ``'down'``.

    Parameters
    ----------
    tensor_a, tensor_b : Tensor
        The two tensors to contract.
    index_pairs : list of (int, int)
        Each tuple ``(i, j)`` specifies that index *i* of *tensor_a*
        contracts with index *j* of *tensor_b*.

    Returns
    -------
    Tensor or sympy.Expr
        The result of the contraction. A scalar ``sympy.Expr`` if rank
        becomes 0.
    """
    ra = tensor_a.rank
    rb = tensor_b.rank

    a_used: set[int] = set()
    b_used: set[int] = set()
    contraction_axes = []

    for i, j in index_pairs:
        if not (0 <= i < ra):
            raise ValueError(f"Index {i} out of range for tensor_a (rank {ra}).")
        if not (0 <= j < rb):
            raise ValueError(f"Index {j} out of range for tensor_b (rank {rb}).")
        if i in a_used:
            raise ValueError(f"Index {i} of tensor_a appears in multiple pairs.")
        if j in b_used:
            raise ValueError(f"Index {j} of tensor_b appears in multiple pairs.")
        if tensor_a.index_pos[i] == tensor_b.index_pos[j]:
            raise ValueError(
                f"Cannot contract indices with same position "
                f"('{tensor_a.index_pos[i]}'). One must be 'up' and the other 'down'."
            )
        a_used.add(i)
        b_used.add(j)
        contraction_axes.append((i, ra + j))

    product = tensorproduct(tensor_a.components, tensor_b.components)
    result = tensorcontraction(product, *contraction_axes)

    remaining = tuple(
        [tensor_a.index_pos[i] for i in range(ra) if i not in a_used]
        + [tensor_b.index_pos[j] for j in range(rb) if j not in b_used]
    )

    if isinstance(result, sympy.Expr) and not isinstance(result, Array):
        return simplify(result)

    result = Array(result).applyfunc(simplify)
    return Tensor(result, remaining)
