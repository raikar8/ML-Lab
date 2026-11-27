"""Load the curriculum from plain files so it is easy to read and edit.

content/weeks/wNN.md   one file per week (theory for days 1-5, a project, questions)
exercises/wNN_*.py     coding exercises; the module docstring holds metadata + the spec
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEKS_DIR = ROOT / "content" / "weeks"
EX_DIR = ROOT / "exercises"
SOL_DIR = ROOT / "solutions"
TEST_DIR = ROOT / "tests" / "exercises"

PHASES = {
    1: ("Foundations", "Python fluency and classical ML, the part every interview loop still tests."),
    2: ("Deep learning & PyTorch", "Backprop, training dynamics, CNNs, sequence/audio models and PyTorch engineering."),
    3: ("Transformers & LLM architecture", "Attention, inference mechanics, scaling, efficient attention and generative models."),
    4: ("Post-training, Chinese playbook & LLM behaviour", "Fine-tuning, RL for LLMs, how DeepSeek, Qwen, Kimi, GLM and MiniMax got ahead, and how LLMs behave in products."),
    5: ("Embedded AI & model optimisation", "Hardware, quantization, pruning, distillation and on-device deployment."),
    6: ("System design & interview sprint", "Embedded ML system design, timed coding and full mock loops."),
}


@dataclass
class Question:
    qid: str
    week: int
    day: int
    text: str
    answer: str
    tags: list[str] = field(default_factory=list)
    core: bool = False  # marked "*" in the source: high-yield, used by the sprint mode


@dataclass
class DaySection:
    day: int
    title: str
    body: str


@dataclass
class Week:
    week: int
    phase: int
    title: str
    goal: str
    lab: str | None
    days: list[DaySection]
    project: str
    questions: list[Question]
    tracks: list[str]


@dataclass
class Exercise:
    key: str  # file stem, e.g. w02_softmax
    title: str
    week: int
    day: int
    level: str
    minutes: int
    tags: list[str]
    spec: str
    tracks: list[str]
    core: bool

    @property
    def stub_path(self) -> Path:
        return EX_DIR / f"{self.key}.py"

    @property
    def solution_path(self) -> Path:
        return SOL_DIR / f"{self.key}.py"

    @property
    def test_path(self) -> Path:
        return TEST_DIR / f"test_{self.key}.py"


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if text.startswith("---"):
        _, head, text = text.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta, text


_DAY_RE = re.compile(r"^## D(\d) · (.+)$", re.M)
_Q_RE = re.compile(r"^\?\?(\*?) d(\d)(?: \[([^\]]*)\])? \| (.+)$", re.M)


def parse_week(path: Path) -> Week:
    meta, body = _frontmatter(path.read_text(encoding="utf-8"))
    wk = int(meta["week"])
    # split off Project and Questions
    q_part = ""
    if "\n## Questions" in body:
        body, q_part = body.split("\n## Questions", 1)
    project = ""
    if "\n## Project" in body:
        body, project = body.split("\n## Project", 1)
    days: list[DaySection] = []
    marks = list(_DAY_RE.finditer(body))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        days.append(DaySection(int(m.group(1)), m.group(2).strip(), body[m.end():end].strip()))
    questions: list[Question] = []
    qm = list(_Q_RE.finditer(q_part))
    for i, m in enumerate(qm):
        end = qm[i + 1].start() if i + 1 < len(qm) else len(q_part)
        ans = q_part[m.end():end].strip()
        tags = [t.strip() for t in (m.group(3) or "").split(",") if t.strip()]
        questions.append(Question(f"w{wk:02d}q{i + 1:02d}", wk, int(m.group(2)), m.group(4).strip(), ans,
                                  tags, core=bool(m.group(1))))
    return Week(wk, int(meta.get("phase", 1)), meta.get("title", f"Week {wk}"), meta.get("goal", ""),
                meta.get("lab") or None, days, project.strip(), questions,
                [t.strip() for t in meta.get("tracks", "all").split(",")])


def parse_exercise(path: Path) -> Exercise:
    src = path.read_text(encoding="utf-8")
    m = re.match(r'\s*"""(.*?)"""', src, re.S)
    doc = m.group(1).strip("\n") if m else ""
    head, _, spec = doc.partition("\n\n")
    meta = {}
    for line in head.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return Exercise(path.stem, meta.get("title", path.stem), int(meta.get("week", 0)), int(meta.get("day", 1)),
                    meta.get("level", "medium"), int(meta.get("time", 30)),
                    [t.strip() for t in meta.get("tags", "").split(",") if t.strip()], spec.strip(),
                    [t.strip() for t in meta.get("tracks", "all").split(",")], meta.get("core", "no") == "yes")


@lru_cache(maxsize=1)
def weeks() -> dict[int, Week]:
    out = {}
    for p in sorted(WEEKS_DIR.glob("w*.md")):
        w = parse_week(p)
        out[w.week] = w
    return out


@lru_cache(maxsize=1)
def exercises() -> dict[str, Exercise]:
    return {p.stem: parse_exercise(p)
            for p in sorted(EX_DIR.glob("f*.py")) + sorted(EX_DIR.glob("w*.py")) + sorted(EX_DIR.glob("e*.py"))}


@lru_cache(maxsize=1)
def electives() -> dict[str, Week]:
    """Optional weeks outside the schedule (content/electives/e<n>.md), keyed 'E<n>'."""
    out = {}
    for p in sorted((ROOT / "content" / "electives").glob("e*.md")):
        w = parse_week(p)
        for q in w.questions:
            q.qid = p.stem + q.qid[3:]  # e1q01 ...
        out[p.stem.upper()] = w
    return out


@lru_cache(maxsize=1)
def foundations() -> dict[str, Week]:
    """The optional Week 0 (content/foundations/f0.md), keyed 'F0'. Question ids are f0qNN."""
    out = {}
    for p in sorted((ROOT / "content" / "foundations").glob("f*.md")):
        w = parse_week(p)
        for q in w.questions:
            q.qid = p.stem + q.qid[3:]
        out[p.stem.upper()] = w
    return out


def optional_weeks() -> dict[str, Week]:
    """Week 0 first, then the electives — everything outside the 27-week schedule."""
    return {**foundations(), **electives()}


def all_questions() -> list[Question]:
    return [q for w in weeks().values() for q in w.questions]


def question_index() -> dict[str, Question]:
    return {q.qid: q for q in all_questions()}


def exercises_for(week: int | str, day: int | None = None) -> list[Exercise]:
    """Exercises of a scheduled week (int) or of an optional week by key ("E1", "F0" → files e1_*, f0_*)."""
    if isinstance(week, str):
        prefix = week.lower() + "_"
        return [e for e in exercises().values() if e.key.startswith(prefix) and (day is None or e.day == day)]
    return [e for e in exercises().values() if e.key.startswith("w") and e.week == week
            and (day is None or e.day == day)]


@lru_cache(maxsize=1)
def _reading_sections() -> dict[str, str]:
    """content/reading.md split on '## W<n>' / '## E<n>' headings, keyed 'W<n>' / 'E<n>'."""
    p = ROOT / "content" / "reading.md"
    if not p.exists():
        return {}
    parts = re.split(r"^## ([WEF]\d+)\s*$", p.read_text(encoding="utf-8"), flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def reading() -> dict[int, str]:
    """Reading list per scheduled week (markdown links to primary sources)."""
    return {int(k[1:]): v for k, v in _reading_sections().items() if k.startswith("W")}


def elective_reading() -> dict[str, str]:
    """Reading list per optional week (Week 0 'F0', electives 'E<n>')."""
    return {k: v for k, v in _reading_sections().items() if k[0] in "EF"}
