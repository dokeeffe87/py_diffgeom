"""Metric tensor and derived geometric quantities."""

from functools import cached_property

import sympy
from sympy import Array, MutableDenseNDimArray, Matrix, Symbol, simplify


class MetricTensor:
    """A metric tensor on an n-dimensional manifold.

    Parameters
    ----------
    matrix : sympy.Matrix
        An n x n symmetric matrix representing the metric components g_{μν}.
    coordinates : tuple of sympy.Symbol
        The coordinate symbols (e.g., (t, r, theta, phi) for Schwarzschild).
        The length of this tuple determines the manifold dimension.

    Examples
    --------
    Define the 2D metric for a sphere of radius R:

    >>> import sympy
    >>> R, theta, phi = sympy.symbols("R theta phi")
    >>> g = sympy.Matrix([
    ...     [R**2, 0],
    ...     [0, R**2 * sympy.sin(theta)**2],
    ... ])
    >>> sphere = MetricTensor(g, (theta, phi))
    >>> sphere.dim
    2
    """

    def __init__(self, matrix: Matrix, coordinates: tuple[Symbol, ...]):
        n = len(coordinates)
        if matrix.shape != (n, n):
            raise ValueError(
                f"Metric matrix shape {matrix.shape} does not match "
                f"the number of coordinates ({n})."
            )
        if not matrix.equals(matrix.T):
            raise ValueError("Metric tensor must be symmetric: g_{μν} = g_{νμ}.")

        self._matrix = matrix.applyfunc(simplify)
        self._coordinates = tuple(coordinates)

    @property
    def dim(self) -> int:
        """Manifold dimension."""
        return len(self._coordinates)

    @property
    def coordinates(self) -> tuple[Symbol, ...]:
        """Coordinate symbols."""
        return self._coordinates

    @property
    def matrix(self) -> Matrix:
        """Covariant metric components g_{μν} as a sympy Matrix."""
        return self._matrix

    def __getitem__(self, key: tuple[int, int]) -> sympy.Expr:
        """Get component g_{μν} by index: metric[mu, nu]."""
        mu, nu = key
        return self._matrix[mu, nu]

    @cached_property
    def inverse(self) -> Matrix:
        """Contravariant metric components g^{μν}."""
        return simplify(self._matrix.inv())

    @cached_property
    def determinant(self) -> sympy.Expr:
        """Determinant of the metric, det(g_{μν})."""
        return simplify(self._matrix.det())

    @cached_property
    def christoffel_first_kind(self) -> Array:
        """Christoffel symbols of the first kind, Γ_{λμν}.

        Defined as:
            Γ_{λμν} = (1/2)(∂_μ g_{νλ} + ∂_ν g_{μλ} - ∂_λ g_{μν})

        Returns
        -------
        sympy.Array
            A rank-3 array with shape (n, n, n) where n is the manifold dimension.
            Index order is [lambda, mu, nu].
        """
        n = self.dim
        coords = self._coordinates
        g = self._matrix

        gamma = MutableDenseNDimArray.zeros(n, n, n)
        for lam in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    value = sympy.Rational(1, 2) * (
                        sympy.diff(g[nu, lam], coords[mu])
                        + sympy.diff(g[mu, lam], coords[nu])
                        - sympy.diff(g[mu, nu], coords[lam])
                    )
                    value = simplify(value)
                    gamma[lam, mu, nu] = value
                    gamma[lam, nu, mu] = value  # symmetric in mu, nu

        return Array(gamma)

    @cached_property
    def christoffel_second_kind(self) -> Array:
        """Christoffel symbols of the second kind, Γ^σ_{μν}.

        Defined as:
            Γ^σ_{μν} = g^{σλ} Γ_{λμν}

        Returns
        -------
        sympy.Array
            A rank-3 array with shape (n, n, n) where n is the manifold dimension.
            Index order is [sigma, mu, nu].
        """
        n = self.dim
        g_inv = self.inverse
        gamma_first = self.christoffel_first_kind

        gamma = MutableDenseNDimArray.zeros(n, n, n)
        for sigma in range(n):
            for mu in range(n):
                for nu in range(mu, n):
                    value = sum(
                        g_inv[sigma, lam] * gamma_first[lam, mu, nu] for lam in range(n)
                    )
                    value = simplify(value)
                    gamma[sigma, mu, nu] = value
                    gamma[sigma, nu, mu] = value  # symmetric in mu, nu

        return Array(gamma)
