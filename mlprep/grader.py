"""Grade a spoken/typed interview answer against the reference answer.

Backends (all optional, chosen on the Grader page):
  * "claude"  — Anthropic Messages API; key read from the ANTHROPIC_API_KEY environment variable (never stored).
  * "local"   — any OpenAI-compatible server, e.g. llama.cpp `llama-server` (http://localhost:8080/v1) or
                Ollama (http://localhost:11434/v1). Fully offline.
  * "offline" — a built-in key-concept coverage check (no model). Crude but useful as a fallback.
Only the standard library is used for HTTP.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import Counter

RUBRIC = """You are a senior interviewer for ML / deep-learning / embedded-AI engineer roles.
Grade the CANDIDATE ANSWER to the QUESTION using the REFERENCE ANSWER as the key facts (the candidate may phrase things
differently or add correct extra detail — that's fine). Be strict about technical correctness and concrete.
Return ONLY a JSON object with these keys:
  "score": integer 0-3 (0 = wrong or blank, 1 = partial / major gaps, 2 = solid, 3 = strong, precise, with an example or numbers),
  "covered": list of key points the candidate got right (short phrases),
  "missed": list of important key points from the reference that are missing,
  "errors": list of statements that are technically wrong (quote briefly and correct them),
  "vague": list of places where the answer was hand-wavy and what precise statement was expected,
  "follow_up": one probing follow-up question a real interviewer would ask next.
"""

STOP = set("""a an the and or of to in on for with by is are was were be been being it its this that these those as at from
into than then so such can could would should will may might do does did not no yes if when which who whom what where why how
also more most less much many very just only each other both all any some there their they them we you your our i he she his her
one two three use used uses using make makes made get gets give gives e g i e eg ie vs via per about over under between because""".split())


def _terms(text: str) -> Counter:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+|\d+(?:\.\d+)?", text.lower())
    words = [w for w in words if w not in STOP and len(w) > 2]
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return Counter(words + bigrams)


def offline_grade(question: str, reference: str, answer: str) -> dict:
    """Key-concept coverage: which distinctive reference terms (incl. bigrams) appear in the answer."""
    ref, ans = _terms(reference), _terms(answer)
    qterms = _terms(question)
    keys = [t for t, _ in ref.most_common() if t not in qterms][:25]
    if not keys:
        return {"score": 0, "covered": [], "missed": [], "errors": [], "vague": [], "follow_up": "", "backend": "offline"}
    hit = [k for k in keys if k in ans]
    cov = len(hit) / len(keys)
    words = len(answer.split())
    score = 0 if words < 5 else 3 if cov >= 0.55 else 2 if cov >= 0.35 else 1 if cov >= 0.15 else 0
    return {"score": score, "covered": hit[:12], "missed": [k for k in keys if k not in ans][:12], "errors": [],
            "vague": ["Offline mode can't judge correctness — only coverage of key terms from the reference."],
            "follow_up": "", "coverage": round(cov, 2), "backend": "offline"}


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model did not return JSON")
    out = json.loads(m.group(0))
    out["score"] = int(max(0, min(3, int(out.get("score", 0)))))
    for k in ("covered", "missed", "errors", "vague"):
        out[k] = [str(x) for x in out.get(k, [])] if isinstance(out.get(k, []), list) else [str(out[k])]
    out["follow_up"] = str(out.get("follow_up", ""))
    return out


def _post(url: str, payload: dict, headers: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def prompt(question: str, reference: str, answer: str) -> str:
    return f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nCANDIDATE ANSWER:\n{answer}\n"


def grade(question: str, reference: str, answer: str, backend: str = "offline", model: str = "",
          base_url: str = "", transport=_post) -> dict:
    if backend == "offline" or not answer.strip():
        return offline_grade(question, reference, answer)
    if backend == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable before starting the app.")
        r = transport("https://api.anthropic.com/v1/messages",
                      {"model": model or "claude-sonnet-5-5", "max_tokens": 1200, "system": RUBRIC,
                       "messages": [{"role": "user", "content": prompt(question, reference, answer)}]},
                      {"x-api-key": key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in r.get("content", []))
    elif backend == "local":
        r = transport(base_url.rstrip("/") + "/chat/completions",
                      {"model": model or "local", "temperature": 0,
                       "messages": [{"role": "system", "content": RUBRIC},
                                    {"role": "user", "content": prompt(question, reference, answer)}]}, {})
        text = r["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"unknown backend {backend}")
    out = parse_json(text)
    out["backend"] = backend
    return out


RATING_FROM_SCORE = {0: 0, 1: 1, 2: 2, 3: 3}  # grader score -> spaced-repetition rating
