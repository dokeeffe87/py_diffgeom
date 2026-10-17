"""Tests for the Streamlit GUI."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from sympy import Matrix, latex, symbols

from diffgeom.config import build_metric, load_config
from diffgeom.metric import MetricTensor

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "diffgeom" / "app.py")
METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"


@pytest.fixture
def app():
    """Create a fresh AppTest instance."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


class TestAppStarts:
    def test_no_exception_on_startup(self, app):
        assert not app.exception, f"App raised an exception: {app.exception}"

    def test_title_present(self, app):
        titles = [t.value for t in app.title]
        assert any("diffgeom" in t for t in titles)

    def test_compute_button_present(self, app):
        assert len(app.button) > 0
        labels = [b.label for b in app.button]
        assert any("Compute" in label for label in labels)


class TestExampleConfig:
    def test_example_config_populates_coords(self, app):
        """Selecting an example config should populate the coordinates field."""
        # The sidebar radio should default to "Example config" if examples exist
        if METRICS_DIR.is_dir() and list(METRICS_DIR.glob("*.yaml")):
            # Coordinates field should be populated from the first example
            text_inputs = app.text_input
            coord_inputs = [ti for ti in text_inputs if "oordinate" in (ti.label or "")]
            if coord_inputs:
                assert coord_inputs[0].value != ""


class TestMetricPreview:
    @pytest.mark.parametrize("example", ["sphere_2d", "schwarzschild", "arbitrary_function_2d"])
    def test_preview_example_metric(self, app, example):
        app.sidebar.selectbox[0].set_value(example).run()
        app.button(key="preview_metric").click().run()

        metric, _ = build_metric(load_config(METRICS_DIR / f"{example}.yaml"))
        expected = r"g_{\mu\nu} = " + latex(metric.matrix)
        assert not app.exception
        assert not app.error
        assert [block.value.strip("$\n ") for block in app.latex] == [expected]
        order = ", ".join(str(c) for c in metric.coordinates)
        assert any(f"Row/column order: ({order})" == caption.value for caption in app.caption)

    def test_preview_manual_metric_with_off_diagonal_components(self, app):
        app.sidebar.radio[0].set_value("Manual entry").run()
        app.text_input(key="_coords").set_value("x, y").run()
        app.text_input(key="metric_0_0").set_value("1 + x**2")
        app.text_input(key="metric_0_1").set_value("x")
        app.text_input(key="metric_1_0").set_value("x")
        app.button(key="preview_metric").click().run()

        x = symbols("x")
        expected = r"g_{\mu\nu} = " + latex(Matrix([[1 + x**2, x], [x, 1]]))
        assert not app.exception
        assert not app.error
        assert [block.value.strip("$\n ") for block in app.latex] == [expected]

        app.text_input(key="metric_0_0").set_value("2 + x**2").run()
        assert not app.latex, "Editing must clear the outdated preview"
        app.button(key="preview_metric").click().run()
        updated = r"g_{\mu\nu} = " + latex(Matrix([[2 + x**2, x], [x, 1]]))
        assert not app.error
        assert [block.value.strip("$\n ") for block in app.latex] == [updated]

    def test_preview_does_not_compute_curvature(self, app, monkeypatch):
        def unexpected_curvature(_metric):
            raise AssertionError("Preview must not compute curvature")

        monkeypatch.setattr(MetricTensor, "riemann_tensor", property(unexpected_curvature))
        app.sidebar.selectbox[0].set_value("sphere_2d").run()
        app.button(key="preview_metric").click().run()

        assert not app.exception
        assert not app.error
        assert len(app.latex) == 1

    def test_invalid_metric_shows_error_and_can_be_corrected(self, app):
        app.sidebar.radio[0].set_value("Manual entry").run()
        app.text_input(key="_coords").set_value("x, y").run()
        app.text_input(key="metric_0_0").set_value("(")
        app.button(key="preview_metric").click().run()

        assert not app.exception
        assert any("Error building metric:" in error.value for error in app.error)
        assert not app.latex

        app.text_input(key="metric_0_0").set_value("1")
        app.button(key="preview_metric").click().run()
        assert not app.exception
        assert not app.error
        assert len(app.latex) == 1

    def test_preview_disabled_without_coordinates(self, app):
        app.sidebar.radio[0].set_value("Manual entry").run()
        app.text_input(key="_coords").set_value("").run()

        assert not app.exception
        assert app.button(key="preview_metric").disabled
        assert not app.latex


class TestCompute:
    def test_compute_flat_metric_produces_output(self):
        """Computing on a 2D flat metric should produce LaTeX output."""
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

        # Switch to manual mode
        at.sidebar.radio[0].set_value("Manual entry")
        at.run()

        # Set coordinates
        coord_inputs = [ti for ti in at.text_input if "oordinate" in (ti.label or "")]
        if coord_inputs:
            coord_inputs[0].set_value("x, y").run()

        # Set identity explicitly; switching modes may retain example cell values.
        for i in range(2):
            for j in range(2):
                at.text_input(key=f"metric_{i}_{j}").set_value("1" if i == j else "0")

        # Click compute
        compute_btns = [b for b in at.button if "Compute" in b.label]
        if compute_btns:
            compute_btns[0].click().run()

        # Check that no exceptions occurred
        assert not at.exception, f"App raised an exception: {at.exception}"

        # There should be some latex output or info messages for vanishing components
        has_output = len(at.latex) > 0 or any(
            "vanish" in str(m.value).lower() for m in at.info
        )
        assert has_output, "Expected LaTeX output or 'vanish' info after compute"
        expected_metric = r"g_{\mu\nu} = " + latex(Matrix.eye(2))
        assert any(block.value.strip("$\n ") == expected_metric for block in at.latex)
