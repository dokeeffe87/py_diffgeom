"""Metric tensor and derived geometric quantities."""

from functools import cached_property
from itertools import product as iproduct

import sympy
from sympy import (
    Array,
    Eq,
    Function,
    Matrix,
    MutableDenseNDimArray,
    Rational,
    Symbol,
    simplify,
    trigsimp,
)

from diffgeom.tensor import Tensor


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

        return Tensor(Array(gamma), ("down", "down", "down"))

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

        return Tensor(Array(gamma), ("up", "down", "down"))

    @cached_property
    def riemann_tensor(self) -> Array:
        """Riemann curvature tensor, R^ρ_{σμν}.

        Defined as:
            R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ}
                        + Γ^ρ_{μλ} Γ^λ_{νσ} - Γ^ρ_{νλ} Γ^λ_{μσ}

        Returns
        -------
        sympy.Array
            A rank-4 array with shape (n, n, n, n).
            Index order is [rho, sigma, mu, nu].
        """
        n = self.dim
        coords = self._coordinates
        Gamma = self.christoffel_second_kind

        R = MutableDenseNDimArray.zeros(n, n, n, n)
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(mu + 1, n):
                        term = (
                            sympy.diff(Gamma[rho, nu, sigma], coords[mu])
                            - sympy.diff(Gamma[rho, mu, sigma], coords[nu])
                        )
                        for lam in range(n):
                            term += (
                                Gamma[rho, mu, lam] * Gamma[lam, nu, sigma]
                                - Gamma[rho, nu, lam] * Gamma[lam, mu, sigma]
                            )
                        term = simplify(term)
                        R[rho, sigma, mu, nu] = term
                        R[rho, sigma, nu, mu] = -term

        return Tensor(Array(R), ("up", "down", "down", "down"))

    @cached_property
    def ricci_tensor(self) -> Array:
        """Ricci curvature tensor, R_{μν}.

        Defined as the contraction:
            R_{μν} = R^λ_{μλν}

        Returns
        -------
        sympy.Array
            A rank-2 array with shape (n, n).
            Index order is [mu, nu].
        """
        n = self.dim
        Riem = self.riemann_tensor

        Ric = MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(mu, n):
                value = sum(Riem[lam, mu, lam, nu] for lam in range(n))
                value = simplify(value)
                Ric[mu, nu] = value
                Ric[nu, mu] = value

        return Tensor(Array(Ric), ("down", "down"))

    @cached_property
    def ricci_scalar(self) -> sympy.Expr:
        """Ricci scalar curvature, R = g^{μν} R_{μν}."""
        n = self.dim
        g_inv = self.inverse
        Ric = self.ricci_tensor

        R = sum(g_inv[mu, nu] * Ric[mu, nu] for mu in range(n) for nu in range(n))
        return simplify(R)

    @cached_property
    def einstein_tensor(self) -> Array:
        """Einstein tensor, G_{μν} = R_{μν} - ½ g_{μν} R.

        Returns
        -------
        sympy.Array
            A rank-2 array with shape (n, n).
            Index order is [mu, nu].
        """
        n = self.dim
        g = self._matrix
        Ric = self.ricci_tensor
        R = self.ricci_scalar

        G = MutableDenseNDimArray.zeros(n, n)
        for mu in range(n):
            for nu in range(mu, n):
                value = Ric[mu, nu] - sympy.Rational(1, 2) * g[mu, nu] * R
                value = simplify(value)
                G[mu, nu] = value
                G[nu, mu] = value

        return Tensor(Array(G), ("down", "down"))

    @cached_property
    def kretschmann_scalar(self) -> sympy.Expr:
        """Kretschmann scalar K = R_{abcd} R^{abcd}."""
        n = self.dim
        Riem = self.riemann_tensor
        # All-down Riemann: lower the first (up) index
        R_lower = self.lower_index(Riem, 0)
        # All-up Riemann: raise all four indices of the all-down form
        R_upper = R_lower
        for idx in range(4):
            R_upper = self.raise_index(R_upper, idx)
        # Contract all four index pairs by direct summation
        K = sum(
            R_lower[a, b, c, d] * R_upper[a, b, c, d]
            for a in range(n)
            for b in range(n)
            for c in range(n)
            for d in range(n)
        )
        return simplify(K)

    @cached_property
    def weyl_tensor(self) -> Tensor:
        """Weyl conformal tensor C^a_{bcd}. Vanishes identically for n < 3."""
        n = self.dim
        if n < 3:
            return Tensor(
                Array(MutableDenseNDimArray.zeros(n, n, n, n)),
                ("up", "down", "down", "down"),
            )

        Riem = self.riemann_tensor
        Ric = self.ricci_tensor
        R = self.ricci_scalar
        g = self._matrix
        g_inv = self.inverse

        # Mixed Ricci: R^a_m = g^{ar} R_{rm}
        Ric_mixed = MutableDenseNDimArray.zeros(n, n)
        for a in range(n):
            for m in range(n):
                Ric_mixed[a, m] = sum(
                    g_inv[a, r] * Ric[r, m] for r in range(n)
                )

        delta = sympy.eye(n)
        coeff1 = Rational(1, n - 2)
        coeff2 = R * Rational(1, (n - 1) * (n - 2))

        C = MutableDenseNDimArray.zeros(n, n, n, n)
        for a in range(n):
            for s in range(n):
                for mu in range(n):
                    for nu in range(mu + 1, n):
                        term = Riem[a, s, mu, nu]
                        term -= coeff1 * (
                            delta[a, mu] * Ric[nu, s]
                            - delta[a, nu] * Ric[mu, s]
                            - g[s, mu] * Ric_mixed[a, nu]
                            + g[s, nu] * Ric_mixed[a, mu]
                        )
                        term += coeff2 * (
                            delta[a, mu] * g[nu, s] - delta[a, nu] * g[mu, s]
                        )
                        term = simplify(term)
                        C[a, s, mu, nu] = term
                        C[a, s, nu, mu] = -term

        return Tensor(Array(C), ("up", "down", "down", "down"))

    @cached_property
    def geodesic_equations(self) -> list[Eq]:
        """Geodesic equations for this metric.

        Builds the system of second-order ODEs:
            d²x^μ/dλ² + Γ^μ_{αβ} (dx^α/dλ)(dx^β/dλ) = 0

        Returns
        -------
        list of sympy.Eq
            One equation per coordinate, each of the form
            ``Eq(d²x^μ/dλ² + ..., 0)``.
        """
        n = self.dim
        lam = Symbol("lambda")
        Gamma = self.christoffel_second_kind

        # Create coordinate functions of the affine parameter
        coord_funcs = [Function(str(c))(lam) for c in self._coordinates]

        # Substitution map: bare coordinate symbol -> function of λ
        subs = dict(zip(self._coordinates, coord_funcs))

        equations = []
        for mu in range(n):
            # Acceleration term: d²x^μ/dλ²
            accel = sympy.diff(coord_funcs[mu], lam, 2)

            # Connection term: Γ^μ_{αβ} (dx^α/dλ)(dx^β/dλ)
            connection_sum = sympy.S.Zero
            for alpha in range(n):
                for beta in range(n):
                    gamma_val = Gamma[mu, alpha, beta]
                    if gamma_val == 0:
                        continue
                    gamma_sub = gamma_val.subs(subs)
                    connection_sum += (
                        gamma_sub
                        * sympy.diff(coord_funcs[alpha], lam)
                        * sympy.diff(coord_funcs[beta], lam)
                    )

            lhs = simplify(accel + connection_sum)
            equations.append(Eq(lhs, 0))

        return equations

    def raise_index(self, tensor: Tensor, pos: int) -> Tensor:
        """Raise an index by contracting with the inverse metric g^{μα}.

        Parameters
        ----------
        tensor : Tensor
            The tensor whose index to raise.
        pos : int
            Which index to raise (0-based). Must currently be ``'down'``.

        Returns
        -------
        Tensor
            A new tensor with the specified index raised to ``'up'``.
        """
        r = tensor.rank
        if not (0 <= pos < r):
            raise ValueError(f"Index position {pos} out of range for rank-{r} tensor.")
        if tensor.index_pos[pos] == "up":
            raise ValueError(f"Index at position {pos} is already 'up'. Cannot raise.")

        n = self.dim
        g_inv = self.inverse
        old = tensor.components

        result = MutableDenseNDimArray.zeros(*old.shape)
        for indices in iproduct(range(n), repeat=r):
            value = sum(
                g_inv[indices[pos], alpha]
                * old[indices[:pos] + (alpha,) + indices[pos + 1 :]]
                for alpha in range(n)
            )
            result[indices] = simplify(value)

        new_pos = list(tensor.index_pos)
        new_pos[pos] = "up"
        return Tensor(Array(result), tuple(new_pos))

    def covariant_derivative(self, tensor: Tensor) -> Tensor:
        """Compute the covariant derivative ∇_ρ T^{...}_{...}.

        For a tensor of rank r, returns a tensor of rank r+1 with one
        additional trailing 'down' index (the derivative index ρ).

        For each 'up' index: adds +Γ^a_{ρc} T^{...c...}_{...}
        For each 'down' index: adds -Γ^c_{ρa} T^{...}_{...c...}

        Parameters
        ----------
        tensor : Tensor
            The tensor to differentiate.

        Returns
        -------
        Tensor
            A new tensor with rank r+1. The last index is the derivative
            index (down).
        """
        n = self.dim
        r = tensor.rank
        coords = self._coordinates
        Gamma = self.christoffel_second_kind
        old = tensor.components
        index_pos = tensor.index_pos

        new_shape = tuple(n for _ in range(r + 1))
        result = MutableDenseNDimArray.zeros(*new_shape)

        for indices in iproduct(range(n), repeat=r + 1):
            # indices = (i_0, i_1, ..., i_{r-1}, rho)
            tensor_indices = indices[:-1]
            rho = indices[-1]

            # Partial derivative term
            value = sympy.diff(old[tensor_indices], coords[rho])

            # Connection terms for each existing index
            for k in range(r):
                if index_pos[k] == "up":
                    # +Γ^{i_k}_{ρ c} T^{...c...}
                    for c in range(n):
                        replaced = tensor_indices[:k] + (c,) + tensor_indices[k + 1 :]
                        value += Gamma[tensor_indices[k], rho, c] * old[replaced]
                else:
                    # -Γ^c_{ρ i_k} T_{...c...}
                    for c in range(n):
                        replaced = tensor_indices[:k] + (c,) + tensor_indices[k + 1 :]
                        value -= Gamma[c, rho, tensor_indices[k]] * old[replaced]

            result[indices] = value

        new_pos = tensor.index_pos + ("down",)
        return Tensor(Array(result), new_pos)

    @cached_property
    def killing_vectors(self) -> list[tuple[str, Tensor]]:
        """Auto-identify coordinate Killing vectors.

        If ∂_i g_{μν} = 0 for all μ,ν, then ∂/∂x^i is a Killing vector.

        Returns
        -------
        list of (str, Tensor)
            Each entry is (label, vector) where label is e.g. "∂/∂t" and
            vector is a Tensor with index_pos ('up',) and components
            [0, ..., 1, ..., 0].
        """
        n = self.dim
        g = self._matrix
        coords = self._coordinates
        result = []

        for i in range(n):
            is_killing = True
            for mu in range(n):
                for nu in range(mu, n):
                    if simplify(sympy.diff(g[mu, nu], coords[i])) != 0:
                        is_killing = False
                        break
                if not is_killing:
                    break

            if is_killing:
                components = MutableDenseNDimArray.zeros(n)
                components[i] = 1
                label = f"∂/∂{coords[i]}"
                result.append((label, Tensor(Array(components), ("up",))))

        return result

    def is_killing_vector(self, vector) -> bool:
        """Check if a vector satisfies the Killing equation.

        The Killing equation is:
            ξ_{(μ;ν)} = ∂_ν ξ_μ + ∂_μ ξ_ν − 2 Γ^σ_{μν} ξ_σ = 0

        Parameters
        ----------
        vector : Tensor, list, Array, or Matrix
            A contravariant vector (components with 'up' index). If not
            a Tensor, it is treated as contravariant components.

        Returns
        -------
        bool
            True if the vector satisfies the Killing equation.
        """
        n = self.dim
        g = self._matrix
        coords = self._coordinates
        Gamma = self.christoffel_second_kind

        # Normalize input to a list of expressions
        if isinstance(vector, Tensor):
            if vector.index_pos != ("up",):
                raise ValueError("Killing vector must have index_pos ('up',).")
            xi_up = [vector[i] for i in range(n)]
        elif isinstance(vector, (list, tuple)):
            xi_up = list(vector)
        elif isinstance(vector, Array):
            xi_up = [vector[i] for i in range(n)]
        elif isinstance(vector, Matrix):
            xi_up = [vector[i] for i in range(n)]
        else:
            raise TypeError(f"Unsupported vector type: {type(vector).__name__}")

        # Lower the index: ξ_μ = g_{μα} ξ^α
        xi_down = []
        for mu in range(n):
            xi_down.append(sum(g[mu, alpha] * xi_up[alpha] for alpha in range(n)))

        # Check Killing equation for all independent (mu, nu) with mu <= nu
        for mu in range(n):
            for nu in range(mu, n):
                # ∂_ν ξ_μ + ∂_μ ξ_ν - 2 Γ^σ_{μν} ξ_σ
                val = (
                    sympy.diff(xi_down[mu], coords[nu])
                    + sympy.diff(xi_down[nu], coords[mu])
                    - 2 * sum(Gamma[sigma, mu, nu] * xi_down[sigma] for sigma in range(n))
                )
                # Try trigsimp first (handles trig identities better before
                # simplify rewrites them into harder-to-reduce forms)
                val = trigsimp(val)
                if val != 0:
                    val = simplify(val)
                    if val != 0:
                        return False
        return True

    @cached_property
    def killing_tensors(self) -> list[tuple[str, Tensor]]:
        """Auto-identify Killing tensors.

        The metric tensor g_{μν} is always a rank-2 Killing tensor.

        Returns
        -------
        list of (str, Tensor)
            Each entry is (label, tensor).
        """
        g_tensor = Tensor(Array(self._matrix), ("down", "down"))
        return [("g (metric)", g_tensor)]

    def is_killing_tensor(self, tensor) -> bool:
        """Check if a rank-2 symmetric tensor satisfies the Killing tensor equation.

        The Killing tensor equation is:
            ∇_{(α} K_{βγ)} = 0

        (fully symmetrized covariant derivative vanishes).

        Parameters
        ----------
        tensor : Tensor, Matrix, or Array
            A rank-2 covariant (down, down) tensor.

        Returns
        -------
        bool
            True if the tensor satisfies the Killing tensor equation.
        """
        n = self.dim
        coords = self._coordinates
        Gamma = self.christoffel_second_kind

        # Normalize to components
        if isinstance(tensor, Tensor):
            if tensor.rank != 2:
                raise ValueError("Killing tensor check requires a rank-2 tensor.")
            K = tensor.components
        elif isinstance(tensor, Matrix):
            K = Array(tensor)
        elif isinstance(tensor, Array):
            K = tensor
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor).__name__}")

        # Check ∇_{(α} K_{βγ)} = 0 for all independent (α,β,γ) with α≤β≤γ
        # ∇_α K_{βγ} = ∂_α K_{βγ} - Γ^σ_{αβ} K_{σγ} - Γ^σ_{αγ} K_{βσ}
        for alpha in range(n):
            for beta in range(alpha, n):
                for gamma in range(beta, n):
                    # Sum of cyclic permutations: (α,β,γ) + (β,γ,α) + (γ,α,β)
                    val = sympy.S.Zero
                    for a, b, c in [
                        (alpha, beta, gamma),
                        (beta, gamma, alpha),
                        (gamma, alpha, beta),
                    ]:
                        term = sympy.diff(K[b, c], coords[a])
                        for sigma in range(n):
                            term -= Gamma[sigma, a, b] * K[sigma, c]
                            term -= Gamma[sigma, a, c] * K[b, sigma]
                        val += term

                    val = trigsimp(val)
                    if val != 0:
                        val = simplify(val)
                        if val != 0:
                            return False
        return True

    def lower_index(self, tensor: Tensor, pos: int) -> Tensor:
        """Lower an index by contracting with the metric g_{μα}.

        Parameters
        ----------
        tensor : Tensor
            The tensor whose index to lower.
        pos : int
            Which index to lower (0-based). Must currently be ``'up'``.

        Returns
        -------
        Tensor
            A new tensor with the specified index lowered to ``'down'``.
        """
        r = tensor.rank
        if not (0 <= pos < r):
            raise ValueError(f"Index position {pos} out of range for rank-{r} tensor.")
        if tensor.index_pos[pos] == "down":
            raise ValueError(
                f"Index at position {pos} is already 'down'. Cannot lower."
            )

        n = self.dim
        g = self._matrix
        old = tensor.components

        result = MutableDenseNDimArray.zeros(*old.shape)
        for indices in iproduct(range(n), repeat=r):
            value = sum(
                g[indices[pos], alpha]
                * old[indices[:pos] + (alpha,) + indices[pos + 1 :]]
                for alpha in range(n)
            )
            result[indices] = simplify(value)

        new_pos = list(tensor.index_pos)
        new_pos[pos] = "down"
        return Tensor(Array(result), tuple(new_pos))
