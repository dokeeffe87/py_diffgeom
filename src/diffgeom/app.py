"""Streamlit web GUI for diffgeom."""

from pathlib import Path

import streamlit as st
import yaml

from diffgeom.config import DEFAULT_INDEX_POS, build_metric, load_config, validate_config
from diffgeom.formatting import format_metric_summary, format_scalar, format_tensor
from diffgeom.quantities import QUANTITY_MAP, apply_index_spec

METRICS_DIR = Path(__file__).resolve().parent.parent.parent / "metrics"


def _config_key(config: dict) -> tuple:
    """Build a hashable key from a config dict for caching."""
    coords = tuple(config["coordinates"])
    assumptions = tuple(sorted(
        (k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
        for k, v in config.get("assumptions", {}).items()
    ))
    functions = tuple(sorted(config.get("functions", [])))
    metric_rows = tuple(
        tuple(str(entry) for entry in row)
        for row in config["metric"]
    )
    return (coords, assumptions, functions, metric_rows)


@st.cache_resource
def _build_metric_cached(_key: tuple, config: dict):
    """Cache MetricTensor construction keyed by config content."""
    return build_metric(config)


def _discover_example_configs() -> dict[str, Path]:
    """Find YAML files in the metrics directory."""
    if not METRICS_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(METRICS_DIR.glob("*.yaml"))}


def _parse_assumptions(text: str) -> dict:
    """Parse a YAML-formatted assumptions string."""
    if not text.strip():
        return {}
    parsed = yaml.safe_load(text)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("Assumptions must be a YAML mapping (e.g. r_s: {positive: true})")
    return parsed


def _parse_functions(text: str) -> list[str]:
    """Parse a comma-separated list of function names."""
    if not text.strip():
        return []
    return [f.strip() for f in text.split(",") if f.strip()]


def _build_config_from_form(
    coords_str: str,
    assumptions_str: str,
    functions_str: str,
    metric_entries: list[list[str]],
    name: str | None = None,
) -> dict:
    """Assemble a config dict from form inputs."""
    coords = [c.strip() for c in coords_str.split(",") if c.strip()]
    if not coords:
        raise ValueError("Enter at least one coordinate.")

    n = len(coords)
    if len(metric_entries) != n or any(len(row) != n for row in metric_entries):
        raise ValueError(f"Metric must be {n}x{n} to match {n} coordinates.")

    assumptions = _parse_assumptions(assumptions_str)
    functions = _parse_functions(functions_str)

    metric_rows = []
    for row in metric_entries:
        metric_rows.append([entry.strip() for entry in row])

    raw = {
        "coordinates": coords,
        "assumptions": assumptions,
        "functions": functions,
        "metric": metric_rows,
    }
    if name:
        raw["name"] = name
    return validate_config(raw)


def _format_assumptions(assumptions: dict) -> str:
    """Format assumptions dict back to YAML text for the text area."""
    if not assumptions:
        return ""
    return yaml.dump(assumptions, default_flow_style=True).strip()


def _run_app():
    st.set_page_config(page_title="diffgeom", page_icon="∇", layout="wide")
    st.title("∇ diffgeom")
    st.caption("Symbolic differential geometry computations")

    # ── Sidebar: input mode ──────────────────────────────────────────────
    examples = _discover_example_configs()

    input_modes = ["Manual entry"]
    if examples:
        input_modes.insert(0, "Example config")
    input_modes.append("Upload YAML")

    mode = st.sidebar.radio("Input mode", input_modes)

    # Determine initial values based on input mode
    init_name = ""
    init_coords = ""
    init_assumptions = ""
    init_functions = ""
    init_metric: list[list[str]] | None = None

    if mode == "Example config":
        selected = st.sidebar.selectbox("Choose an example", list(examples.keys()))
        if selected:
            config = load_config(examples[selected])
            init_name = config.get("name") or ""
            init_coords = ", ".join(config["coordinates"])
            init_assumptions = _format_assumptions(config.get("assumptions", {}))
            init_functions = ", ".join(config.get("functions", []))
            init_metric = [[str(entry) for entry in row] for row in config["metric"]]

    elif mode == "Upload YAML":
        uploaded = st.sidebar.file_uploader("Upload a YAML config", type=["yaml", "yml"])
        if uploaded is not None:
            try:
                raw = yaml.safe_load(uploaded.read())
                config = validate_config(raw)
                init_name = config.get("name") or ""
                init_coords = ", ".join(config["coordinates"])
                init_assumptions = _format_assumptions(config.get("assumptions", {}))
                init_functions = ", ".join(config.get("functions", []))
                init_metric = [[str(entry) for entry in row] for row in config["metric"]]
            except Exception as e:
                st.sidebar.error(f"Invalid config: {e}")

    # ── Main area: metric form ───────────────────────────────────────────
    st.subheader("Metric definition")

    col_name, col_coords = st.columns([1, 2])
    with col_name:
        name = st.text_input("Name (optional)", value=init_name)
    with col_coords:
        coords_str = st.text_input(
            "Coordinates (comma-separated)",
            value=init_coords,
            placeholder="t, r, theta, phi",
        )

    assumptions_str = st.text_area(
        "Assumptions (YAML format)",
        value=init_assumptions,
        placeholder="r_s: {positive: true}",
        height=68,
    )

    functions_str = st.text_input(
        "Arbitrary functions (comma-separated)",
        value=init_functions,
        placeholder="f, g, h",
        help="Declare function names used in metric components, e.g. f(x), A(r).",
    )

    # Parse coordinates to size the metric grid
    coords = [c.strip() for c in coords_str.split(",") if c.strip()]
    n = len(coords)

    st.markdown("**Metric matrix** $g_{\\mu\\nu}$")

    if n == 0:
        st.info("Enter coordinates above to define the metric matrix.")
    else:
        # Build n x n grid of text inputs
        metric_entries: list[list[str]] = []
        # Column header
        header_cols = st.columns(n)
        for j, col in enumerate(header_cols):
            col.markdown(f"**{coords[j]}**")

        for i in range(n):
            row_cols = st.columns(n)
            row_entries = []
            for j, col in enumerate(row_cols):
                default = ""
                if init_metric and i < len(init_metric) and j < len(init_metric[i]):
                    default = init_metric[i][j]
                elif i == j:
                    default = "1"
                else:
                    default = "0"
                val = col.text_input(
                    f"g_{{{coords[i]},{coords[j]}}}",
                    value=default,
                    key=f"metric_{i}_{j}",
                    label_visibility="collapsed",
                )
                row_entries.append(val)
            metric_entries.append(row_entries)

    # ── Quantity selection ────────────────────────────────────────────────
    st.subheader("Quantities to compute")

    qty_cols = st.columns(len(QUANTITY_MAP))
    selected_quantities: list[tuple[str, str | None]] = []

    for col, (qty_name, (_, display_name, _, is_scalar)) in zip(qty_cols, QUANTITY_MAP.items()):
        with col:
            checked = st.checkbox(display_name, value=True, key=f"qty_{qty_name}")
            if checked:
                indices = None
                if not is_scalar:
                    default_idx = DEFAULT_INDEX_POS.get(qty_name, "")
                    idx_input = st.text_input(
                        "Indices",
                        value=default_idx or "",
                        key=f"idx_{qty_name}",
                        help=f"Index spec (u=up, d=down). Default: {default_idx}",
                    )
                    idx_input = idx_input.strip()
                    if idx_input and idx_input != default_idx:
                        indices = idx_input
                selected_quantities.append((qty_name, indices))

    # ── Compute ──────────────────────────────────────────────────────────
    st.divider()

    if st.button("Compute", type="primary", disabled=(n == 0)):
        try:
            config = _build_config_from_form(
                coords_str, assumptions_str, functions_str, metric_entries, name
            )
        except Exception as e:
            st.error(f"Config error: {e}")
            return

        try:
            key = _config_key(config)
            with st.spinner("Building metric tensor..."):
                metric, _ = _build_metric_cached(key, config)
        except Exception as e:
            st.error(f"Error building metric: {e}")
            return

        coord_names = [str(c) for c in metric.coordinates]

        # Show metric summary
        summary = format_metric_summary(config, metric, latex=True)
        for line in summary.splitlines():
            if "$" in line:
                # Split text and latex parts
                st.markdown(line)
            else:
                st.markdown(f"**{line}**")

        # Compute each selected quantity
        for qty_name, indices in selected_quantities:
            attr_name, display_name, symbol, is_scalar = QUANTITY_MAP[qty_name]

            try:
                with st.spinner(f"Computing {display_name}..."):
                    value = getattr(metric, attr_name)

                if is_scalar:
                    output = format_scalar(value, display_name, symbol, latex=True)
                else:
                    if indices is not None:
                        value = apply_index_spec(metric, value, indices)
                    output = format_tensor(
                        value, display_name, symbol, coord_names, latex=True
                    )

                # Render: header line as markdown, math lines with st.latex
                lines = output.splitlines()
                if lines:
                    st.markdown(f"#### {lines[0]}")
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("$") and line.endswith("$"):
                        st.latex(line[1:-1])
                    elif line.startswith("All components"):
                        st.info(line)
                    else:
                        st.markdown(line)

            except Exception as e:
                st.error(f"Error computing {display_name}: {e}")


# Streamlit runs the file as a script — this block handles both direct
# ``streamlit run`` invocation and the ``diffgeom-gui`` entry point.
if __name__ == "__main__" or __name__ == "diffgeom.app":
    _run_app()


def main():
    """Entry point for the ``diffgeom-gui`` console script."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--server.headless=true"]
    st_main()
