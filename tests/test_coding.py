"""Coding track: metadata, re-solve ladder, and the Coding interview page."""
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MLPREP_PROGRESS", str(Path(tempfile.mkdtemp()) / "progress.json"))

from mlprep import coding, content, plan  # noqa: E402


def test_metadata_complete():
    ps = coding.problems()
    assert len(ps) >= 55
    for p in ps.values():
        assert p.pattern in coding.PATTERNS and p.level in coding.LEVELS and p.frequency in coding.FREQ, p.key
        assert 1 <= p.week <= plan.TOTAL_WEEKS and 1 <= p.day <= 5, p.key
        assert len(p.hints) >= 2 and p.followups and p.complexity and len(p.spec) > 200, p.key
        assert p.stub_path.exists() and p.test_path.exists(), p.key
    assert set(coding.PATTERNS) <= set(coding.guide())
    assert all(any(p.pattern == pat for p in ps.values()) for pat in coding.PATTERNS)


def test_every_week_has_coding_and_tasks_exist():
    weeks = {p.week for p in coding.problems().values()}
    assert weeks == set(range(1, plan.TOTAL_WEEKS + 1))
    tasks = [t for t in plan.all_tasks() if t.kind == "coding"]
    assert len(tasks) == len(coding.problems())


def test_ml_most_asked_refers_to_real_exercises():
    exs, ps = content.exercises(), coding.problems()
    for key, freq, check, mins, role in coding.ML_MOST_ASKED:
        assert key in exs or key in ps, key
        assert freq in coding.FREQ and mins > 0 and check


def test_ladder():
    d = {}
    day0 = dt.date(2026, 1, 1)
    key = "c01_two_sum"
    target = coding.problems()[key].minutes
    e = coding.record(d, key, False, 5, 0, day0)
    assert e["next"] == "2026-01-02" and not coding.solved(d, key)
    e = coding.record(d, key, True, target + 5, 0, day0)            # slow
    assert e["next"] == "2026-01-03" and e["rung"] == 0
    e = coding.record(d, key, True, None, 0, day0)                  # untimed
    assert e["next"] == "2026-01-03" and e["rung"] == 0
    e = coding.record(d, key, True, 3, 1, day0)                     # hinted
    assert e["rung"] == 0
    for gap, rung in zip(coding.LADDER, range(1, 5)):
        e = coding.record(d, key, True, 3, 0, day0)
        assert e["next"] == (day0 + dt.timedelta(days=gap)).isoformat() and e["rung"] == rung
    assert coding.mastered(d, key) and coding.best_minutes(d, key) == 3
    assert coding.due(d, day0 + dt.timedelta(days=200))[0].key == key
    assert coding.due(d, day0) == []


def app():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180)
    at.run()
    return at


def test_page_renders_all_tabs():
    at = app()
    at.switch_page("app_pages/coding.py")
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Coding interview" in t.value for t in at.title)


def test_practice_solve_with_timer_hint_and_reset(tmp_path):
    key = "c01_two_sum"
    p = coding.problems()[key]
    backup = tmp_path / "stub.py"
    shutil.copy(p.stub_path, backup)
    try:
        at = app()
        at.session_state["coding_key"] = key
        at.switch_page("app_pages/coding.py")
        at.run()
        at.button(key="cp-start").click().run()
        assert at.session_state["coding_timer"]["key"] == key
        at.button(key=f"cp-hint-{key}-0").click().run()
        assert at.session_state["coding_timer"]["hints"] == 1
        at.text_area(key=f"cp-code-{key}").input(p.solution_path.read_text())
        at.button(key=f"cp-run-{key}").click().run()
        assert not at.exception, [e.message for e in at.exception]
        e = at.session_state["progress"]["coding"][key]
        assert e["attempts"][-1]["passed"] and e["attempts"][-1]["hints"] == 1 and e["rung"] == 0
        at.button(key="cp-reset").click().run()
        assert "NotImplementedError" in p.stub_path.read_text()
    finally:
        shutil.copy(backup, p.stub_path)


def test_mock_start_and_finish():
    at = app()
    at.switch_page("app_pages/coding.py")
    at.run()
    at.button(key="cm-start").click().run()
    mock = at.session_state["coding_mock"]
    assert len(mock["keys"]) == 2
    at.checkbox(key="cm-c1").check()
    at.button(key="cm-finish").click().run()
    assert not at.exception, [e.message for e in at.exception]
    log = at.session_state["progress"]["coding_mocks"][-1]
    assert log["clarified"] and log["passed"] == {}
    for k in mock["keys"]:
        assert not at.session_state["progress"]["coding"][k]["attempts"][-1]["passed"]


def test_sprint_has_daily_coding_drills():
    at = app()
    at.switch_page("app_pages/sprint.py")
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    drills = [m for m in at.markdown if "Coding-round drill" in m.value]
    solve = [b for b in at.button if b.key and b.key.startswith("sp-code-")]
    assert len(drills) >= 5 and len(solve) >= 10
    assert len({b.key.rsplit("-", 1)[-1] for b in solve}) >= 8     # different problems on different days


def test_today_links_to_resolves():
    at = app()
    at.session_state["progress"]["coding"] = {"c01_two_sum": {"attempts": [], "rung": 0, "next": "2000-01-01"}}
    at.run()
    assert not at.exception


def test_stubs_do_not_reveal_hints_or_complexity():
    for p in coding.problems().values():
        head = p.stub_path.read_text().split('"""')[1]
        assert "hint:" not in head and "follow-up:" not in head and "complexity:" not in head, p.key
        assert p.title in head and p.spec.splitlines()[0] in head
