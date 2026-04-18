"""Command-line interface for diffgeom."""

import argparse
import sys

from diffgeom.config import VALID_QUANTITIES, build_metric, load_config, parse_quantities_flag
from diffgeom.expr import ExpressionError, evaluate_expression
from diffgeom.formatting import (
    format_expression_result,
    format_geodesic_equations,
    format_killing_tensors,
    format_killing_vectors,
    format_metric_summary,
    format_scalar,
    format_tensor,
)
from diffgeom.quantities import QUANTITY_MAP, apply_index_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffgeom",
        description="Symbolic differential geometry computations.",
    )
    subparsers = parser.add_subparsers(dest="command")

    compute_parser = subparsers.add_parser(
        "compute",
        help="Compute geometric quantities from a YAML metric config.",
    )
    compute_parser.add_argument(
        "config",
        help="Path to a YAML metric config file.",
    )
    compute_parser.add_argument(
        "--latex",
        action="store_true",
        default=False,
        help="Output in LaTeX format instead of pretty-print.",
    )
    compute_parser.add_argument(
        "--quantities",
        type=str,
        default=None,
        help=(
            "Comma-separated list of quantities to compute "
            "(overrides config). Use name:indices to specify index "
            "positions (e.g. riemann:dddd). Valid: "
            + ", ".join(sorted(VALID_QUANTITIES))
        ),
    )
    compute_parser.add_argument(
        "--expr",
        action="append",
        default=None,
        dest="expressions",
        metavar="EXPR",
        help=(
            "Tensor expression to evaluate using Einstein summation. "
            "Can be specified multiple times. "
            'Example: --expr "Ric{_a_b}*Ric{^a^b}"'
        ),
    )

    return parser


def _run_compute(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    metric, symbols_dict = build_metric(config)
    coord_names = [str(c) for c in metric.coordinates]
    latex = args.latex

    # Determine what to compute — always a list of (name, indices_or_None)
    if args.quantities:
        try:
            quantities = parse_quantities_flag(args.quantities)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        quantities = config["compute"]

    # Header
    print(format_metric_summary(config, metric, latex=latex))
    print()

    # Compute and format each quantity
    for qty_name, indices in quantities:
        attr_name, display_name, symbol, is_scalar = QUANTITY_MAP[qty_name]
        value = getattr(metric, attr_name)

        if qty_name == "geodesic":
            print(format_geodesic_equations(value, coord_names, latex=latex))
        elif qty_name == "killing_vectors":
            if indices is not None:
                value = [
                    (label, apply_index_spec(metric, vec, indices))
                    for label, vec in value
                ]
            print(format_killing_vectors(value, coord_names, latex=latex))
        elif qty_name == "killing_tensors":
            if indices is not None:
                value = [
                    (label, apply_index_spec(metric, t, indices))
                    for label, t in value
                ]
            print(format_killing_tensors(value, coord_names, latex=latex))
        elif is_scalar:
            print(format_scalar(value, display_name, symbol, latex=latex))
        else:
            if indices is not None:
                value = apply_index_spec(metric, value, indices)
            print(format_tensor(value, display_name, symbol, coord_names, latex=latex))
        print()

    # Evaluate tensor expressions from --expr flags and config
    all_expressions = []
    if args.expressions:
        for expr_str in args.expressions:
            all_expressions.append({"name": expr_str, "expr": expr_str})
    for entry in config.get("expressions", []):
        all_expressions.append(entry)

    for entry in all_expressions:
        expr_str = entry["expr"]
        expr_name = entry.get("name", expr_str)
        try:
            result, free_indices = evaluate_expression(expr_str, metric)
            print(format_expression_result(result, expr_name, coord_names, latex=latex))
            print()
        except ExpressionError as exc:
            print(f"Error evaluating '{expr_str}': {exc}", file=sys.stderr)
            sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the diffgeom CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "compute":
        _run_compute(args)


if __name__ == "__main__":
    main()
