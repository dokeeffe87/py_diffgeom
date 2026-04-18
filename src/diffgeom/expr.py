"""Tensor algebra expression engine: registry, parser, and evaluator.

Supports Einstein summation notation for contracting curvature tensors.
Example expressions:
    "Ric{_a_b}*Ric{^a^b}"
    "R^2 - 4*Ric{_a_b}*Ric{^a^b} + Riem{_a_b_c_d}*Riem{^a^b^c^d}"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as iproduct

import sympy
from sympy import Array, MutableDenseNDimArray, simplify

from diffgeom.tensor import Tensor

# ---------------------------------------------------------------------------
# Tensor registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorDef:
    """Definition of a named tensor in the registry."""
    attr: str                     # MetricTensor attribute name
    rank: int
    default_pos: tuple[str, ...]  # default index positions


TENSOR_REGISTRY: dict[str, TensorDef] = {
    "g": TensorDef("metric_tensor", 2, ("down", "down")),
    "ginv": TensorDef("inverse_metric_tensor", 2, ("up", "up")),
    "Gam": TensorDef("christoffel_second_kind", 3, ("up", "down", "down")),
    "Riem": TensorDef("riemann_tensor", 4, ("up", "down", "down", "down")),
    "Ric": TensorDef("ricci_tensor", 2, ("down", "down")),
    "R": TensorDef("ricci_scalar", 0, ()),
    "G": TensorDef("einstein_tensor", 2, ("down", "down")),
    "C": TensorDef("weyl_tensor", 4, ("up", "down", "down", "down")),
    "K": TensorDef("kretschmann_scalar", 0, ()),
}


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class IndexSpec:
    """A single index: label (e.g. 'a') and position ('up' or 'down')."""
    label: str
    position: str  # 'up' or 'down'


@dataclass
class TensorRef:
    """Reference to a tensor with explicit indices, e.g. Ric{_a_b}."""
    name: str
    indices: list[IndexSpec]


@dataclass
class ScalarRef:
    """Reference to a scalar quantity, optionally raised to a power."""
    name: str
    power: int = 1


@dataclass
class NumericLiteral:
    """A numeric constant."""
    value: int | float


@dataclass
class BinOp:
    """Binary operation: +, -, *."""
    op: str
    left: object
    right: object


@dataclass
class UnaryMinus:
    """Unary negation."""
    operand: object


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

@dataclass
class Token:
    kind: str   # 'NUMBER', 'NAME', 'LBRACE', 'RBRACE', 'CARET', 'UNDERSCORE',
                # 'PLUS', 'MINUS', 'STAR', 'LPAREN', 'RPAREN', 'EOF'
    value: str


def _tokenize(expr: str) -> list[Token]:
    """Tokenize a tensor expression string."""
    tokens: list[Token] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '{':
            tokens.append(Token('LBRACE', ch))
            i += 1
        elif ch == '}':
            tokens.append(Token('RBRACE', ch))
            i += 1
        elif ch == '^':
            tokens.append(Token('CARET', ch))
            i += 1
        elif ch == '_':
            tokens.append(Token('UNDERSCORE', ch))
            i += 1
        elif ch == '+':
            tokens.append(Token('PLUS', ch))
            i += 1
        elif ch == '-':
            tokens.append(Token('MINUS', ch))
            i += 1
        elif ch == '*':
            tokens.append(Token('STAR', ch))
            i += 1
        elif ch == '(':
            tokens.append(Token('LPAREN', ch))
            i += 1
        elif ch == ')':
            tokens.append(Token('RPAREN', ch))
            i += 1
        elif ch.isdigit() or ch == '.':
            j = i
            has_dot = (ch == '.')
            j += 1
            while j < len(expr) and (expr[j].isdigit() or (expr[j] == '.' and not has_dot)):
                if expr[j] == '.':
                    has_dot = True
                j += 1
            tokens.append(Token('NUMBER', expr[i:j]))
            i = j
        elif ch.isalpha():
            j = i
            while j < len(expr) and expr[j].isalnum():
                j += 1
            tokens.append(Token('NAME', expr[i:j]))
            i = j
        else:
            raise ExpressionError(f"Unexpected character '{ch}' at position {i}")
    tokens.append(Token('EOF', ''))
    return tokens


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------

class ExpressionError(Exception):
    """Error in parsing or evaluating a tensor expression."""


class _Parser:
    """Recursive descent parser for tensor expressions.

    Grammar:
        expr   -> term (('+' | '-') term)*
        term   -> unary ('*' unary)*
        unary  -> '-' unary | atom
        atom   -> NUMBER
                | NAME '{' index_list '}'     # tensor with indices
                | NAME '^' NUMBER             # scalar power
                | NAME                        # plain scalar
                | '(' expr ')'
        index_list -> ('^' LETTER | '_' LETTER)+
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str) -> Token:
        tok = self.advance()
        if tok.kind != kind:
            raise ExpressionError(
                f"Expected {kind}, got {tok.kind} ('{tok.value}')"
            )
        return tok

    def parse(self):
        node = self.parse_expr()
        if self.peek().kind != 'EOF':
            raise ExpressionError(
                f"Unexpected token '{self.peek().value}' after expression"
            )
        return node

    def parse_expr(self):
        left = self.parse_term()
        while self.peek().kind in ('PLUS', 'MINUS'):
            op = self.advance().kind
            right = self.parse_term()
            left = BinOp('+' if op == 'PLUS' else '-', left, right)
        return left

    def parse_term(self):
        left = self.parse_unary()
        while self.peek().kind == 'STAR':
            self.advance()
            right = self.parse_unary()
            left = BinOp('*', left, right)
        return left

    def parse_unary(self):
        if self.peek().kind == 'MINUS':
            self.advance()
            operand = self.parse_unary()
            return UnaryMinus(operand)
        return self.parse_atom()

    def parse_atom(self):
        tok = self.peek()

        if tok.kind == 'NUMBER':
            self.advance()
            if '.' in tok.value:
                return NumericLiteral(float(tok.value))
            return NumericLiteral(int(tok.value))

        if tok.kind == 'NAME':
            self.advance()
            name = tok.value

            # Check for tensor with indices: NAME '{' ...
            if self.peek().kind == 'LBRACE':
                self.advance()  # consume '{'
                indices = self._parse_index_list()
                self.expect('RBRACE')
                return TensorRef(name, indices)

            # Check for scalar power: NAME '^' NUMBER
            if self.peek().kind == 'CARET':
                self.advance()  # consume '^'
                num_tok = self.expect('NUMBER')
                power = int(num_tok.value)
                return ScalarRef(name, power)

            # Plain scalar reference
            return ScalarRef(name)

        if tok.kind == 'LPAREN':
            self.advance()
            node = self.parse_expr()
            self.expect('RPAREN')
            return node

        raise ExpressionError(
            f"Unexpected token '{tok.value}' (kind={tok.kind})"
        )

    def _parse_index_list(self) -> list[IndexSpec]:
        """Parse indices inside braces: ^a_b^c_d etc."""
        indices = []
        while self.peek().kind in ('CARET', 'UNDERSCORE'):
            pos_tok = self.advance()
            position = 'up' if pos_tok.kind == 'CARET' else 'down'
            label_tok = self.expect('NAME')
            if len(label_tok.value) != 1:
                raise ExpressionError(
                    f"Index labels must be single letters, got '{label_tok.value}'"
                )
            indices.append(IndexSpec(label_tok.value, position))
        if not indices:
            raise ExpressionError("Empty index list in braces")
        return indices


def parse_expression(expr_str: str):
    """Parse a tensor expression string into an AST.

    Parameters
    ----------
    expr_str : str
        The expression to parse.

    Returns
    -------
    AST node (TensorRef, ScalarRef, NumericLiteral, BinOp, or UnaryMinus)
    """
    tokens = _tokenize(expr_str)
    parser = _Parser(tokens)
    return parser.parse()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class IndexedResult:
    """Result of evaluating a sub-expression, tracking free indices."""
    value: Tensor | sympy.Expr  # Tensor for indexed results, Expr for scalars
    free_indices: list[IndexSpec] = field(default_factory=list)


def _resolve_tensor(name: str, indices: list[IndexSpec], metric) -> IndexedResult:
    """Resolve a TensorRef to an IndexedResult, raising/lowering as needed."""
    if name not in TENSOR_REGISTRY:
        raise ExpressionError(f"Unknown tensor '{name}'")

    tdef = TENSOR_REGISTRY[name]

    if tdef.rank == 0:
        raise ExpressionError(
            f"'{name}' is a scalar (rank 0) and cannot have indices. "
            f"Use it without braces."
        )

    if len(indices) != tdef.rank:
        raise ExpressionError(
            f"'{name}' has rank {tdef.rank} but got {len(indices)} indices"
        )

    tensor = getattr(metric, tdef.attr)
    requested_pos = tuple(idx.position for idx in indices)

    # Raise/lower to match requested positions
    for i, (current, desired) in enumerate(zip(tdef.default_pos, requested_pos)):
        if current != desired:
            if desired == "up":
                tensor = metric.raise_index(tensor, i)
            else:
                tensor = metric.lower_index(tensor, i)

    return IndexedResult(tensor, list(indices))


def _resolve_scalar(name: str, power: int, metric) -> IndexedResult:
    """Resolve a ScalarRef to an IndexedResult."""
    if name not in TENSOR_REGISTRY:
        raise ExpressionError(f"Unknown quantity '{name}'")

    tdef = TENSOR_REGISTRY[name]
    if tdef.rank != 0:
        raise ExpressionError(
            f"'{name}' is a rank-{tdef.rank} tensor, not a scalar. "
            f"Use index notation like {name}{{_a_b}}."
        )

    value = getattr(metric, tdef.attr)
    if power != 1:
        value = value ** power
    return IndexedResult(simplify(value), [])


def _contract_product(left: IndexedResult, right: IndexedResult, dim: int) -> IndexedResult:
    """Multiply two IndexedResults, contracting repeated indices (Einstein sum)."""
    # If both are scalars
    if not left.free_indices and not right.free_indices:
        return IndexedResult(simplify(left.value * right.value), [])

    # If one side is a pure scalar
    if not left.free_indices:
        # scalar * tensor
        scaled = left.value * right.value.components
        scaled = Array(scaled).applyfunc(simplify)
        return IndexedResult(
            Tensor(scaled, right.value.index_pos),
            list(right.free_indices),
        )
    if not right.free_indices:
        # tensor * scalar
        scaled = right.value * left.value.components
        scaled = Array(scaled).applyfunc(simplify)
        return IndexedResult(
            Tensor(scaled, left.value.index_pos),
            list(left.free_indices),
        )

    # Both sides are tensors — find contraction pairs
    left_labels = {idx.label: (i, idx) for i, idx in enumerate(left.free_indices)}
    right_labels = {idx.label: (i, idx) for i, idx in enumerate(right.free_indices)}

    contraction_pairs = []  # (left_slot, right_slot) in tensor indices
    contracted_labels = set()

    for label in left_labels:
        if label in right_labels:
            li, lidx = left_labels[label]
            ri, ridx = right_labels[label]
            if lidx.position == ridx.position:
                raise ExpressionError(
                    f"Repeated index '{label}' has the same position "
                    f"('{lidx.position}') on both sides. "
                    f"One must be up and the other down."
                )
            contraction_pairs.append((li, ri))
            contracted_labels.add(label)

    # Check for triple-repeated indices
    all_labels = [idx.label for idx in left.free_indices] + [
        idx.label for idx in right.free_indices
    ]
    for label in set(all_labels):
        count = all_labels.count(label)
        if count > 2:
            raise ExpressionError(
                f"Index '{label}' appears {count} times. "
                f"An index may appear at most twice (once up, once down)."
            )

    # Build free indices for the result
    result_free = [
        idx for idx in left.free_indices if idx.label not in contracted_labels
    ] + [
        idx for idx in right.free_indices if idx.label not in contracted_labels
    ]

    if not contraction_pairs:
        # Outer product — no contractions, just tensor product
        from sympy import tensorproduct
        product = tensorproduct(left.value.components, right.value.components)
        result_pos = left.value.index_pos + right.value.index_pos
        return IndexedResult(
            Tensor(Array(product).applyfunc(simplify), result_pos),
            result_free,
        )

    # Use contract() from tensor.py
    from diffgeom.tensor import contract
    result = contract(left.value, right.value, contraction_pairs)

    if isinstance(result, Tensor):
        return IndexedResult(result, result_free)
    else:
        # Full contraction produced a scalar
        return IndexedResult(simplify(result), [])


def _add_results(left: IndexedResult, right: IndexedResult, op: str, dim: int) -> IndexedResult:
    """Add or subtract two IndexedResults."""
    # Both scalars
    if not left.free_indices and not right.free_indices:
        if op == '+':
            return IndexedResult(simplify(left.value + right.value), [])
        else:
            return IndexedResult(simplify(left.value - right.value), [])

    # Both must be tensors with matching free indices
    if not left.free_indices or not right.free_indices:
        raise ExpressionError(
            "Cannot add/subtract a scalar and a tensor with free indices"
        )

    left_labels = [(idx.label, idx.position) for idx in left.free_indices]
    right_labels = [(idx.label, idx.position) for idx in right.free_indices]

    if sorted(left_labels) != sorted(right_labels):
        raise ExpressionError(
            f"Cannot add tensors with different free indices: "
            f"{left_labels} vs {right_labels}"
        )

    # If indices are in the same order, add directly
    if left_labels == right_labels:
        if op == '+':
            result_arr = left.value.components + right.value.components
        else:
            result_arr = left.value.components - right.value.components
        result_arr = Array(result_arr).applyfunc(simplify)
        return IndexedResult(
            Tensor(result_arr, left.value.index_pos),
            list(left.free_indices),
        )

    # Indices are in different order — need to rearrange right to match left
    label_to_right_slot = {idx.label: i for i, idx in enumerate(right.free_indices)}
    perm = [label_to_right_slot[idx.label] for idx in left.free_indices]

    # Permute right's components
    rank = right.value.rank
    shape = right.value.shape
    new_arr = MutableDenseNDimArray.zeros(*shape)
    for indices in iproduct(range(dim), repeat=rank):
        permuted = tuple(indices[perm[k]] for k in range(rank))
        new_arr[indices] = right.value.components[permuted]

    if op == '+':
        result_arr = left.value.components + Array(new_arr)
    else:
        result_arr = left.value.components - Array(new_arr)
    result_arr = Array(result_arr).applyfunc(simplify)
    return IndexedResult(
        Tensor(result_arr, left.value.index_pos),
        list(left.free_indices),
    )


def _eval_node(node, metric) -> IndexedResult:
    """Recursively evaluate an AST node."""
    dim = metric.dim

    if isinstance(node, NumericLiteral):
        return IndexedResult(sympy.Integer(node.value) if isinstance(node.value, int)
                             else sympy.Float(node.value), [])

    if isinstance(node, ScalarRef):
        return _resolve_scalar(node.name, node.power, metric)

    if isinstance(node, TensorRef):
        return _resolve_tensor(node.name, node.indices, metric)

    if isinstance(node, UnaryMinus):
        inner = _eval_node(node.operand, metric)
        if not inner.free_indices:
            return IndexedResult(-inner.value, [])
        negated = (-1) * inner.value.components
        negated = Array(negated).applyfunc(simplify)
        return IndexedResult(
            Tensor(negated, inner.value.index_pos),
            inner.free_indices,
        )

    if isinstance(node, BinOp):
        left = _eval_node(node.left, metric)
        right = _eval_node(node.right, metric)

        if node.op == '*':
            return _contract_product(left, right, dim)
        else:
            return _add_results(left, right, node.op, dim)

    raise ExpressionError(f"Unknown AST node type: {type(node).__name__}")


def evaluate_expression(
    expr_str: str, metric
) -> tuple[Tensor | sympy.Expr, list[IndexSpec]]:
    """Parse and evaluate a tensor expression against a metric.

    Parameters
    ----------
    expr_str : str
        The tensor expression (e.g. "Ric{_a_b}*Ric{^a^b}").
    metric : MetricTensor
        The metric to evaluate against.

    Returns
    -------
    tuple of (result, free_indices)
        result is a Tensor (if free indices remain) or sympy.Expr (if scalar).
        free_indices is the list of remaining IndexSpec objects.
    """
    ast = parse_expression(expr_str)
    result = _eval_node(ast, metric)
    return result.value, result.free_indices
