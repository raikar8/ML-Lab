"""Reading list parser, link sanity, read/notes state, and the Papers page."""
import os
import re
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MLPREP_PROGRESS", str(Path(tempfile.mkdtemp()) / "progress.json"))

from mlprep import content, papers  # noqa: E402

ARXIV = re.compile(r"^https://arxiv\.org/abs/\d{4}\.\d{4,5}$")


def test_every_reading_line_parses():
    text = (ROOT / "content" / "reading.md").read_text(encoding="utf-8")
    bullets = [ln for ln in text.splitlines() if ln.startswith("- [")]
    assert len(papers.all_papers()) == len(bullets) >= 130


def test_sections_and_weeks():
    ps = papers.all_papers()
    assert {p.week for p in ps if p.week} == set(range(1, 28))
    assert {p.section for p in ps if p.week is None} == set(content.optional_weeks())
    # the elective block must not leak into week 27's reading
    assert "Recommender" not in content.reading()[27]
    assert "E1" in content.elective_reading()


def test_links_are_well_formed_and_unique():
    ps = papers.all_papers()
    assert len({p.pid for p in ps}) == len(ps)
    for p in ps:
        assert p.url.startswith("https://"), p.title
        assert "scholar.google" not in p.url, f"unresolved search link: {p.title}"
        if "arxiv.org" in p.url:
            assert ARXIV.match(p.url), p.url
        if "doi.org" in p.url:
            assert re.match(r"^https://doi\.org/10\.\d{4,5}/\S+$", p.url), p.url
    arxiv_ids = [p.url for p in ps if "arxiv.org" in p.url]
    assert len(arxiv_ids) == len(set(arxiv_ids)), "same arXiv paper listed twice"


def test_meta_and_topics():
    for p in papers.all_papers():
        assert p.meta and len(p.meta) < 100, (p.title, p.meta)
        assert p.topics and p.topics[0] == papers.WEEK_TOPIC[p.section]
        assert set(p.topics) <= set(papers.TOPICS)
    by = papers.by_title()
    assert "Chinese labs" in by["DeepSeekMoE"].topics
    assert "Chinese labs" not in by["Gated Delta Networks"].topics  # NVIDIA/MIT, only *used* by Qwen
    assert "Speech & audio" in by["Hello Edge: Keyword Spotting on Microcontrollers"].topics


def test_parse_fallbacks():
    ps = papers.parse("## W3\n- [A](https://x.org) — Someone, Lab. No year here.\n"
                      "- [B](https://arxiv.org/abs/1234.5678) — X et al., 2024 (blog). Why.\n"
                      "not a bullet\n## E1\n- [A](https://y.org) — Z, 2020. Dup title.\n")
    assert [(p.section, p.meta, p.why) for p in ps] == [
        ("W3", "Someone, Lab", "No year here."), ("W3", "X et al., 2024 (blog)", "Why."), ("E1", "Z, 2020", "Dup title.")]
    assert ps[0].pid != ps[2].pid
    assert ps[1].source == "arXiv" and ps[0].source == "Blog / docs"


def test_read_and_note_roundtrip():
    d = {}
    pid = papers.all_papers()[0].pid
    papers.set_read(d, pid, True)
    assert papers.is_read(d, pid)
    papers.set_note(d, pid, {"changed": " x ", "why": "", "evidence": "y"})
    assert papers.note(d, pid) == {"changed": "x", "evidence": "y"}
    md = papers.notes_markdown(d)
    assert papers.all_papers()[0].title in md and "**What changed:** x" in md
    papers.set_read(d, pid, False)
    papers.set_note(d, pid, {})
    assert d["papers"] == {}


def test_refs_markdown_uses_reading_list():
    out = papers.refs_markdown(["Attention Is All You Need", "DeepSeek-V2"])
    assert "https://arxiv.org/abs/1706.03762" in out and "Week 18" in out


def _app():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
    at.run()
    return at


def test_papers_page_filters_and_note():
    at = _app()
    at.session_state["papers_section"] = "W12"
    at.switch_page("app_pages/papers.py")
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    shown = [m.value for m in at.markdown if m.value.startswith("**[")]
    w12 = [p for p in papers.all_papers() if p.section == "W12"]
    assert len(shown) == len(w12)
    # tick the first one as read
    at.checkbox(key=f"read-{w12[0].pid}").check().run()
    assert not at.exception
    assert w12[0].pid in at.session_state["progress"]["papers"]
    # write a note on the second
    pid = w12[1].pid
    # form widgets are committed on submit, so input + click happen in the same run
    at.text_input(key=f"n-{pid}-changed").input("Routing with a bias instead of an aux loss")
    [b for b in at.button if b.key and f"note-{pid}-" in b.key][0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    entry = at.session_state["progress"]["papers"][pid]
    assert entry["note"]["changed"].startswith("Routing") and entry.get("read")


def test_papers_page_search_and_status():
    at = _app()
    at.switch_page("app_pages/papers.py")
    at.run()
    at.text_input[0].input("Moonshot").run()
    shown = [m.value for m in at.markdown if m.value.startswith("**[")]
    assert shown and all("moonshot" in s.lower() or "kimi" in s.lower() for s in shown), shown


def test_theory_and_today_link_to_papers():
    at = _app()
    at.session_state["theory_week"] = 12
    at.switch_page("app_pages/theory.py")
    at.run()
    assert not at.exception
