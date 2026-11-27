"""Turn the curriculum into a dated, day-by-day task list.

Every week has the same rhythm:
  Days 1-5  Theory section + coding exercise(s) + quiz on that day's questions
  Day 6     Lab day: the week's interactive lab and/or a mini-project
  Day 7     Review day: spaced-repetition review + a timed weekly mock + teach-back note
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from . import content

TOTAL_WEEKS = 27
DAY_NAMES = {1: "Theory + code", 2: "Theory + code", 3: "Theory + code", 4: "Theory + code",
             5: "Theory + code", 6: "Lab & project", 7: "Review & mock"}

# lab key -> (title, page file, short description)
LABS = {
    "bias_variance": ("Bias–variance & regularisation", "app_pages/labs/bias_variance.py",
                      "Fit polynomials, watch train/test error and the effect of L2."),
    "metrics": ("Classification metrics & thresholds", "app_pages/labs/metrics.py",
                "ROC, PR, calibration and choosing an operating point under class imbalance."),
    "clustering": ("k-means vs GMM (EM)", "app_pages/labs/clustering.py",
                   "Run EM step by step on 2-D data and compare with k-means."),
    "optimizers": ("Optimizers & LR schedules", "app_pages/labs/optimizers.py",
                   "SGD, momentum, Adam, AdamW and Muon-style updates on hard loss surfaces."),
    "cnn_cost": ("CNN cost calculator", "app_pages/labs/cnn_cost.py",
                 "Params, MACs, activation memory and receptive field for standard vs depthwise convs."),
    "attention": ("Attention & RoPE", "app_pages/labs/attention.py",
                  "Masks, temperature, and why RoPE gives relative-position scores."),
    "kv_cache": ("KV cache, GQA/MLA & decoding", "app_pages/labs/kv_cache.py",
                 "KV-cache memory for MHA/GQA/MQA/MLA/hybrids, and sampling with temperature/top-k/top-p."),
    "moe": ("Mixture-of-Experts routing", "app_pages/labs/moe.py",
            "Top-k routing, aux-loss vs DeepSeek's aux-loss-free bias balancing."),
    "efficient_attention": ("Efficient attention", "app_pages/labs/efficient_attention.py",
                            "Cost of softmax vs linear vs sparse attention; delta-rule memory recall."),
    "diffusion": ("Diffusion on a toy distribution", "app_pages/labs/diffusion.py",
                  "Forward noising, the score, and DDPM/DDIM sampling with an exact score."),
    "grpo": ("RL post-training: PPO vs GRPO vs CISPO", "app_pages/labs/grpo.py",
             "Group advantages, clipping, and what each objective does to the gradient."),
    "roofline": ("Roofline & fixed point", "app_pages/labs/roofline.py",
                 "Is your layer compute- or memory-bound? Q-format arithmetic and requantisation."),
    "quantization": ("Quantization lab", "app_pages/labs/quantization.py",
                     "Per-tensor vs per-channel vs group-wise, outliers, SmoothQuant and Hadamard rotation."),
    "pruning": ("Pruning & low-rank", "app_pages/labs/pruning.py",
                "Unstructured vs 2:4 vs structured pruning and SVD truncation on real weight-like matrices."),
}


@dataclass
class Task:
    tid: str
    kind: str  # theory | exercise | quiz | lab | project | review | mock | note
    title: str
    minutes: int
    week: int
    day: int
    ref: str = ""  # exercise key / lab key / etc.


def week_day_for(start: dt.date, date: dt.date) -> tuple[int, int] | None:
    idx = (date - start).days
    if idx < 0 or idx >= TOTAL_WEEKS * 7:
        return None
    return idx // 7 + 1, idx % 7 + 1


def date_for(start: dt.date, week: int, day: int) -> dt.date:
    return start + dt.timedelta(days=(week - 1) * 7 + day - 1)


def tasks_for(week: int, day: int) -> list[Task]:
    w = content.weeks().get(week)
    if w is None:
        return []
    out: list[Task] = []
    base = f"w{week:02d}d{day}"
    if day <= 5:
        sec = next((s for s in w.days if s.day == day), None)
        if sec:
            words = len(sec.body.split())
            out.append(Task(f"{base}-theory", "theory", f"Read: {sec.title}", max(20, words // 12), week, day))
        for ex in content.exercises_for(week, day):
            out.append(Task(f"{base}-ex-{ex.key}", "exercise", f"Code: {ex.title}", ex.minutes, week, day, ex.key))
        qs = [q for q in w.questions if q.day == day]
        if qs:
            out.append(Task(f"{base}-quiz", "quiz", f"Answer {len(qs)} interview questions out loud, then check",
                            5 * len(qs), week, day))
    elif day == 6:
        if w.lab and w.lab in LABS:
            out.append(Task(f"{base}-lab", "lab", f"Lab: {LABS[w.lab][0]}", 45, week, day, w.lab))
        if w.project:
            out.append(Task(f"{base}-project", "project", "Mini-project (see the week page)", 90, week, day))
    else:
        out.append(Task(f"{base}-review", "review", "Spaced-repetition review of all due cards", 30, week, day))
        out.append(Task(f"{base}-mock", "mock", f"Timed weekly mock: week {week} ({w.title})", 30, week, day))
        out.append(Task(f"{base}-note", "note", "Teach-back: write the 5 things you'd say about this week", 15,
                        week, day))
    return out


def all_tasks() -> list[Task]:
    return [t for wk in range(1, TOTAL_WEEKS + 1) for d in range(1, 8) for t in tasks_for(wk, d)]


def unlocked_questions(start: dt.date, today: dt.date):
    """Questions whose scheduled day has arrived (they join the review deck)."""
    out = []
    for q in content.all_questions():
        if date_for(start, q.week, q.day) <= today:
            out.append(q)
    return out
