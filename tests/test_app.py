"""Open every page through the app's navigation and check it renders without exceptions."""
import os
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MLPREP_PROGRESS", str(Path(tempfile.mkdtemp()) / "progress.json"))


def _pages():
    from mlprep.plan import LABS
    from mlprep.ui import PAGES
    paths = [p for p in PAGES.values() if (ROOT / p).exists()]
    paths += [v[1] for v in LABS.values() if (ROOT / v[1]).exists()]
    return paths


@pytest.mark.parametrize("path", _pages())
def test_page_renders(path):
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    at.switch_page(path)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_theory_every_week_and_section():
    from mlprep import content
    for wk, w in content.weeks().items():
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
        at.run()
        at.session_state["theory_week"] = wk
        at.switch_page("app_pages/theory.py")
        at.run()
        assert not at.exception, (wk, [e.message for e in at.exception])


def test_theory_electives_and_reading():
    from mlprep import content
    for key in list(content.electives()):
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
        at.run()
        at.session_state["theory_week"] = key
        at.switch_page("app_pages/theory.py")
        at.run()
        assert not at.exception, [e.message for e in at.exception]
    assert len(content.reading()) == 27
