"""Command-line interface for diffgeom."""

import argparse
import sys

from diffgeom.config import VALID_QUANTITIES, build_metric, load_config, parse_quantities_flag
from diffgeom.formatting import (
    format_geodesic_equations,
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
        elif is_scalar:
            print(format_scalar(value, display_name, symbol, latex=latex))
        else:
            if indices is not None:
                value = apply_index_spec(metric, value, indices)
            print(format_tensor(value, display_name, symbol, coord_names, latex=latex))
        print()


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
