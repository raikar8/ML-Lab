"""The reading list as structured data: one Paper per line of content/reading.md.

Each line looks like ``- [Title](url) — Authors, Venue Year. Why it matters.``
Read state and five-line notes live in the progress file under ``papers``.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import content

_LINE = re.compile(r"^- \[(?P<title>[^\]]+)\]\((?P<url>[^)\s]+)\)\s*[—-]\s*(?P<rest>.*)$")
# meta = everything up to the first year (optionally a range and a "(blog)"-style tag), then ". why"
_META = re.compile(r"^(?P<meta>.*?\b(?:19|20)\d\d(?:[–-]\d\d)?(?:\s*\([^)]*\))?)\.\s*(?P<why>.*)$")

NOTE_FIELDS = [
    ("changed", "What changed", "One sentence: what does the paper do differently from the baseline?"),
    ("why", "Why it works", "The mechanism or intuition, in your words."),
    ("evidence", "Evidence", "The one result that convinced you (number + setting)."),
    ("cost", "Cost / limits", "What it costs (compute, memory, complexity) or where it breaks."),
    ("question", "One question", "A question you'd ask the authors, or an interviewer would ask you."),
]

# Primary topic = the week the paper is assigned to (accurate by construction).
WEEK_TOPIC: dict[str, str] = {
    **{f"W{n}": "Classical ML & evaluation" for n in (1, 2, 3, 4)},
    "W5": "Training & optimisation", "W6": "Training & optimisation",
    "W7": "CNNs & efficient architectures", "W8": "Speech & audio", "W9": "PyTorch & ML engineering",
    "W10": "Transformers & attention", "W11": "Transformers & attention", "W12": "Scaling & Mixture-of-Experts",
    "W13": "Long context & linear attention", "W14": "Inference & serving", "W15": "Distributed training & GPU",
    "W16": "Generative models", "W17": "Post-training & RL", "W18": "Chinese labs", "W19": "Chinese labs",
    "W20": "LLM behaviour & evaluation", "W21": "Embedded hardware & deployment",
    "W22": "Quantization & number formats", "W23": "Quantization & number formats",
    "W24": "Pruning & distillation", "W25": "Embedded hardware & deployment",
    "W26": "ML system design", "W27": "ML system design", "E1": "Recommender systems", "F0": "Foundations",
}
# Cross-tags: strict patterns (word boundaries) matched against title + summary.
CROSS_TAGS: dict[str, str] = {
    "Chinese labs": r"deepseek|qwen|\bkimi\b|moonshot|minimax|\bglm-|zhipu|bytedance|meituan|longcat|stepfun|\bstep-3\b",
    "Scaling & Mixture-of-Experts": r"\bmoe\b|mixture-of-experts|\bexperts\b",
    "Speech & audio": r"\bspeech\b|\baudio\b|keyword spotting",
    "Quantization & number formats": r"quantiz|\bint[48]\b|\bfp[48]\b|mxfp|microscaling|1-bit|1\.58",
    "Embedded hardware & deployment": r"microcontroller|tinyml|on-device|\biot\b",
    "Long context & linear attention": r"long-context|long context|linear attention|sparse attention|delta rule|deltanet|lightning attention|1m ctx",
    "Inference & serving": r"speculative|kv ?cache|\bserving\b|decoding cost|cost-effective decoding",
    "Post-training & RL": r"\bgrpo\b|\bcispo\b|reinforcement learning|\brlhf\b",
}
TOPICS = list(dict.fromkeys(list(WEEK_TOPIC.values()) + list(CROSS_TAGS)))


@dataclass
class Paper:
    pid: str
    section: str          # "W12" or "E1"
    week: int | None      # None for electives
    title: str
    url: str
    meta: str             # "Authors, Venue Year"
    why: str
    source: str           # arXiv | Proceedings / DOI | Blog / docs | Book
    topics: list[str] = field(default_factory=list)

    @property
    def section_label(self) -> str:
        if self.week is not None:
            return f"Week {self.week}"
        return "Week 0 (foundations)" if self.section.startswith("F") else f"Elective {self.section}"


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:60].rstrip("-")


def _source(url: str, meta: str) -> str:
    if "arxiv.org" in url:
        return "arXiv"
    if "(book)" in meta.lower():
        return "Book"
    if any(h in url for h in ("doi.org", "dl.acm.org", "proceedings.", "aclanthology", "openreview", "proceedings.mlr")):
        return "Proceedings / DOI"
    return "Blog / docs"


def _topics(section: str, title: str, meta: str, why: str) -> list[str]:
    tags = [WEEK_TOPIC.get(section, "Other")]
    for name, pat in CROSS_TAGS.items():
        # "Chinese labs" is about who wrote it, so only title + authors count, not the summary.
        text = f"{title} {meta}" if name == "Chinese labs" else f"{title} {meta} {why}"
        if name not in tags and re.search(pat, text.lower()):
            tags.append(name)
    return tags


def parse(text: str) -> list[Paper]:
    out: list[Paper] = []
    section = None
    seen: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        h = re.match(r"^## ([WEF]\d+)\s*$", line)
        if h:
            section = h.group(1)
            continue
        m = _LINE.match(line)
        if not (m and section):
            continue
        rest = m.group("rest").strip()
        mm = _META.match(rest)
        if mm:
            meta, why = mm.group("meta").strip(), mm.group("why").strip()
        else:
            meta, _, why = rest.partition(". ")
            meta, why = meta.strip().rstrip("."), why.strip()
        title = m.group("title").strip()
        pid = slug(title)
        if pid in seen:  # same title listed twice: keep ids unique and stable
            seen[pid] += 1
            pid = f"{pid}-{seen[pid]}"
        else:
            seen[pid] = 1
        week = int(section[1:]) if section.startswith("W") else None
        tags = _topics(section, title, meta, why)
        out.append(Paper(pid, section, week, title, m.group("url"), meta, why, _source(m.group("url"), meta), tags))
    return out


@lru_cache(maxsize=1)
def all_papers() -> list[Paper]:
    p = content.ROOT / "content" / "reading.md"
    if not p.exists():
        return []
    return parse(p.read_text(encoding="utf-8"))


def by_title() -> dict[str, Paper]:
    return {p.title: p for p in all_papers()}


def find(title: str) -> Paper:
    """Look a paper up by its exact title in the reading list (KeyError if missing)."""
    return by_title()[title]


def refs_markdown(titles: list[str]) -> str:
    """A compact markdown list of links for the given reading-list titles."""
    lines = []
    for t in titles:
        p = find(t)
        lines.append(f"- [{p.title}]({p.url}) — {p.meta} · *{p.section_label}*")
    return "\n".join(lines)


# ---- progress -------------------------------------------------------------

def state(data: dict) -> dict:
    return data.setdefault("papers", {})


def is_read(data: dict, pid: str) -> bool:
    return bool(state(data).get(pid, {}).get("read"))


def set_read(data: dict, pid: str, read: bool) -> None:
    entry = state(data).setdefault(pid, {})
    if read:
        entry["read"] = dt.date.today().isoformat()
    else:
        entry.pop("read", None)
    if not entry:
        state(data).pop(pid, None)


def note(data: dict, pid: str) -> dict:
    return dict(state(data).get(pid, {}).get("note", {}))


def set_note(data: dict, pid: str, fields: dict[str, str]) -> None:
    clean = {k: v.strip() for k, v in fields.items() if v and v.strip()}
    entry = state(data).setdefault(pid, {})
    if clean:
        entry["note"] = clean
    else:
        entry.pop("note", None)
    if not entry:
        state(data).pop(pid, None)


def notes_markdown(data: dict) -> str:
    """All written notes as one Markdown document, in reading-list order."""
    labels = {k: lab for k, lab, _ in NOTE_FIELDS}
    out = ["# Paper notes", ""]
    for p in all_papers():
        n = note(data, p.pid)
        if not n:
            continue
        out += [f"## {p.title}", f"{p.meta} · {p.section_label} · <{p.url}>", ""]
        out += [f"- **{labels[k]}:** {n[k]}" for k, _, _ in NOTE_FIELDS if k in n]
        out.append("")
    return "\n".join(out).rstrip() + "\n"
