"""ML Interview Lab — a 27-week daily plan for ML / Deep Learning / Embedded AI engineer interviews.

Run from the repository root:
    streamlit run streamlit_app.py
"""
import streamlit as st

from mlprep.content import ROOT
from mlprep.plan import LABS
from mlprep.ui import PAGES

st.set_page_config(page_title="ML Interview Lab", page_icon=":material/neurology:", layout="wide")


def page(key: str, title: str, icon: str, **kw):
    return st.Page(PAGES[key], title=title, icon=icon, **kw) if (ROOT / PAGES[key]).exists() else None


sections = {
    "Daily": [
        page("today", "Today", ":material/today:", default=True),
        page("roadmap", "6-month roadmap", ":material/map:"),
        page("review", "Review (spaced repetition)", ":material/replay:"),
        page("mock", "Mock interview", ":material/timer:"),
        page("grader", "Answer grader", ":material/grading:"),
        page("errorlog", "Error log", ":material/bug_report:"),
    ],
    "Learn": [
        page("theory", "Theory by week", ":material/menu_book:"),
        page("exercises", "Coding exercises", ":material/code:"),
        page("cpp", "C/C++ track", ":material/memory:"),
        page("notebooks", "GPU notebooks", ":material/developer_board:"),
        page("designs", "System design bank", ":material/architecture:"),
        page("papers", "Papers", ":material/article:"),
        page("capstone", "Hardware capstone", ":material/precision_manufacturing:"),
    ],
    "Deep dives": [
        page("transformers", "Transformer hub", ":material/account_tree:"),
        page("atlas", "Chinese model atlas", ":material/public:"),
        page("frontier", "Frontier watch", ":material/radar:"),
    ],
    "Interview soon?": [
        page("sprint", "Interview in a week", ":material/rocket_launch:"),
    ],
    "Labs": [st.Page(path, title=title, icon=":material/science:", url_path=f"lab_{key}")
             for key, (title, path, _) in LABS.items() if (ROOT / path).exists()],
    "Setup": [page("settings", "Settings & data", ":material/settings:")],
}
pages = {name: [p for p in ps if p is not None] for name, ps in sections.items()}
st.navigation({k: v for k, v in pages.items() if v}).run()
