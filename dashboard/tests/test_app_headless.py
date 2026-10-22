"""Boot the Streamlit app without a browser and without downloading weights."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_boots_headless():
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    blob = " ".join(getattr(el, "value", "") for el in list(at.title) + list(at.warning) + list(at.info))
    assert "Not financial advice" in blob or "not financial advice" in blob.lower()
    assert "Kronos" in blob or "random forest" in blob.lower() or "Random forest" in blob
