"""Tests for Killing vectors, Killing tensors, and covariant derivative."""

from sympy import Array, Matrix, MutableDenseNDimArray, cos, cot, simplify, sin

from diffgeom.tensor import Tensor

# ---------------------------------------------------------------------------
# Coordinate Killing vector identification
# ---------------------------------------------------------------------------


class TestKillingVectors:
    def test_schwarzschild_has_two_killing_vectors(self, schwarzschild):
        """Schwarzschild has ∂/∂t and ∂/∂φ as coordinate Killing vectors."""
        metric, _ = schwarzschild
        kvs = metric.killing_vectors
        labels = [label for label, _ in kvs]
        assert len(kvs) == 2
        assert "∂/∂t" in labels
        assert "∂/∂phi" in labels

    def test_sphere_2d_has_one_killing_vector(self, sphere_2d):
        """2-sphere has ∂/∂φ as the only coordinate Killing vector."""
        metric, _ = sphere_2d
        kvs = metric.killing_vectors
        labels = [label for label, _ in kvs]
        assert len(kvs) == 1
        assert "∂/∂phi" in labels

    def test_flat_5d_has_five_killing_vectors(self, flat_5d):
        """Flat 5D Minkowski has all 5 translation Killing vectors."""
        kvs = flat_5d.killing_vectors
        assert len(kvs) == 5

    def test_killing_vector_components(self, schwarzschild):
        """Check that ∂/∂t has components (1, 0, 0, 0)."""
        metric, _ = schwarzschild
        kvs = metric.killing_vectors
        t_vec = None
        for label, vec in kvs:
            if label == "∂/∂t":
                t_vec = vec
                break
        assert t_vec is not None
        assert t_vec[0] == 1
        assert t_vec[1] == 0
        assert t_vec[2] == 0
        assert t_vec[3] == 0


# ---------------------------------------------------------------------------
# Killing vector verification
# ---------------------------------------------------------------------------


class TestIsKillingVector:
    def test_coordinate_killing_vectors_pass(self, schwarzschild):
        """Auto-identified coordinate Killing vectors pass verification."""
        metric, _ = schwarzschild
        for label, vec in metric.killing_vectors:
            assert metric.is_killing_vector(vec), f"{label} failed verification"

    def test_sphere_coordinate_killing_passes(self, sphere_2d):
        """∂/∂φ on the 2-sphere passes verification."""
        metric, _ = sphere_2d
        for label, vec in metric.killing_vectors:
            assert metric.is_killing_vector(vec), f"{label} failed verification"

    def test_flat_5d_killing_vectors_pass(self, flat_5d):
        """All 5D Minkowski coordinate Killing vectors pass."""
        for label, vec in flat_5d.killing_vectors:
            assert flat_5d.is_killing_vector(vec), f"{label} failed verification"

    def test_non_killing_vector_rejected(self, schwarzschild):
        """A random vector should not satisfy the Killing equation."""
        metric, params = schwarzschild
        r = params["r"]
        # ξ = (r, 1, 0, 0) is not a Killing vector
        vec = [r, 1, 0, 0]
        assert not metric.is_killing_vector(vec)

    def test_sphere_rotation_generator(self, sphere_2d):
        """The rotation generator ξ = (sin φ, cot θ cos φ) is a Killing
        vector on the 2-sphere (one of the three SO(3) generators)."""
        metric, params = sphere_2d
        theta = params["theta"]
        phi = metric.coordinates[1]
        # ξ^θ = sin(φ), ξ^φ = cot(θ) cos(φ)
        xi = [sin(phi), cot(theta) * cos(phi)]
        assert metric.is_killing_vector(xi)

    def test_accepts_list_input(self, flat_5d):
        """is_killing_vector accepts a plain list."""
        assert flat_5d.is_killing_vector([1, 0, 0, 0, 0])

    def test_accepts_array_input(self, flat_5d):
        """is_killing_vector accepts a sympy Array."""
        arr = Array([0, 1, 0, 0, 0])
        assert flat_5d.is_killing_vector(arr)

    def test_accepts_matrix_input(self, flat_5d):
        """is_killing_vector accepts a sympy Matrix."""
        mat = Matrix([0, 0, 1, 0, 0])
        assert flat_5d.is_killing_vector(mat)


# ---------------------------------------------------------------------------
# Killing tensor identification and verification
# ---------------------------------------------------------------------------


class TestKillingTensors:
    def test_metric_always_killing_tensor(self, schwarzschild):
        """The metric g_{μν} is always a rank-2 Killing tensor."""
        metric, _ = schwarzschild
        kts = metric.killing_tensors
        assert len(kts) >= 1
        label, _ = kts[0]
        assert "metric" in label.lower() or "g" in label.lower()

    def test_metric_passes_is_killing_tensor(self, schwarzschild):
        """is_killing_tensor(g) returns True for Schwarzschild."""
        metric, _ = schwarzschild
        g_tensor = Tensor(Array(metric.matrix), ("down", "down"))
        assert metric.is_killing_tensor(g_tensor)

    def test_sphere_metric_is_killing_tensor(self, sphere_2d):
        """is_killing_tensor(g) returns True for the 2-sphere."""
        metric, _ = sphere_2d
        g_tensor = Tensor(Array(metric.matrix), ("down", "down"))
        assert metric.is_killing_tensor(g_tensor)

    def test_flat_metric_is_killing_tensor(self, flat_5d):
        """is_killing_tensor(g) returns True for flat 5D Minkowski."""
        g_tensor = Tensor(Array(flat_5d.matrix), ("down", "down"))
        assert flat_5d.is_killing_tensor(g_tensor)

    def test_non_killing_tensor_rejected(self, sphere_2d):
        """A random symmetric tensor should not be a Killing tensor."""
        metric, params = sphere_2d
        theta = params["theta"]
        # A non-trivial symmetric tensor that is not a Killing tensor
        K = Matrix([[theta, 1], [1, theta]])
        assert not metric.is_killing_tensor(K)

    def test_accepts_matrix_input(self, flat_5d):
        """is_killing_tensor accepts a sympy Matrix."""
        assert flat_5d.is_killing_tensor(flat_5d.matrix)


# ---------------------------------------------------------------------------
# Covariant derivative sanity checks
# ---------------------------------------------------------------------------


class TestCovariantDerivative:
    def test_covariant_derivative_increases_rank(self, sphere_2d):
        """Covariant derivative of a rank-1 tensor yields rank-2."""
        metric, _ = sphere_2d
        n = metric.dim
        components = MutableDenseNDimArray.zeros(n)
        components[0] = 1
        vec = Tensor(Array(components), ("up",))
        result = metric.covariant_derivative(vec)
        assert result.rank == 2
        assert result.index_pos == ("up", "down")

    def test_flat_space_covariant_equals_partial(self, flat_5d):
        """In flat space, covariant derivative equals partial derivative."""
        n = flat_5d.dim
        coords = flat_5d.coordinates
        # Vector with x^1 component = coords[1] (the x coordinate)
        components = MutableDenseNDimArray.zeros(n)
        components[1] = coords[1]  # x
        vec = Tensor(Array(components), ("up",))
        nabla_vec = flat_5d.covariant_derivative(vec)
        # ∇_ν ξ^μ should be ∂_ν ξ^μ in flat space
        # ∂/∂x (x) = 1, everything else 0
        assert simplify(nabla_vec[1, 1] - 1) == 0
        assert simplify(nabla_vec[1, 0]) == 0
        assert simplify(nabla_vec[0, 1]) == 0
