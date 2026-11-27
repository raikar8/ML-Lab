"""Week 0 content, the diagnostic quiz, and their pages."""
import os
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MLPREP_PROGRESS", str(Path(tempfile.mkdtemp()) / "progress.json"))

from mlprep import content, diagnostic  # noqa: E402


def test_week0_content():
    f0 = content.foundations()["F0"]
    assert [d.day for d in f0.days] == [1, 2, 3, 4, 5] and f0.project
    assert len(f0.questions) >= 15 and all(q.qid.startswith("f0q") for q in f0.questions)
    assert {q.day for q in f0.questions} == {1, 2, 3, 4, 5}
    assert sorted(e.day for e in content.exercises_for("F0")) == [1, 2, 3, 4, 5]
    assert "F0" in content.elective_reading()
    # optional weeks must not leak into each other's exercises
    assert all(e.key.startswith("e1_") for e in content.exercises_for("E1"))
    assert all(e.key.startswith("w") for wk in range(1, 28) for e in content.exercises_for(wk))


def test_diagnostic_bank_is_consistent():
    qs = diagnostic.QUESTIONS
    assert len(qs) == 24
    for area in diagnostic.AREAS:
        assert sum(q.area == area for q in qs) == 4
    for q in qs:
        assert 0 <= q.answer < len(q.options) and len(set(q.options)) == len(q.options) and q.why


def test_scoring_and_recommendation():
    perfect = {i: q.answer for i, q in enumerate(diagnostic.QUESTIONS)}
    rec = diagnostic.recommend(diagnostic.score(perfect))
    assert rec["skip_week0"] and all(p["level"] == "skip" for p in rec["areas"].values())
    blank = diagnostic.recommend(diagnostic.score({}))
    assert blank["study_days"] == [1, 2, 3, 4, 5] and not blank["skip_week0"]
    # miss exactly one linear-algebra question -> skim D1 only
    one_off = dict(perfect)
    first_la = next(i for i, q in enumerate(diagnostic.QUESTIONS) if q.area == "Linear algebra")
    one_off[first_la] = (perfect[first_la] + 1) % 4
    rec = diagnostic.recommend(diagnostic.score(one_off))
    assert rec["skim_days"] == [1] and rec["study_days"] == [] and rec["skip_week0"]


def app():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
    at.run()
    return at


def test_diagnostic_page_submit():
    at = app()
    at.switch_page("app_pages/diagnostic.py")
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    for i, q in enumerate(diagnostic.QUESTIONS):
        at.radio(key=f"dq-{i}").set_value(q.options[q.answer if i % 3 else (q.answer + 1) % len(q.options)])
    at.button[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    last = at.session_state["progress"]["diagnostic"][-1]
    assert last["total"] == sum(1 for i in range(24) if i % 3)


def test_theory_week0_every_section():
    for day in range(1, 6):
        at = app()
        at.session_state["theory_week"] = "F0"
        at.session_state["theory_day"] = day
        at.switch_page("app_pages/theory.py")
        at.run()
        assert not at.exception, (day, [e.message for e in at.exception])
