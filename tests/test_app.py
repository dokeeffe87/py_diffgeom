"""Tests for the Streamlit GUI."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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

        # The metric grid should now have 4 cells defaulting to identity
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
