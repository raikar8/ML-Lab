"""A 24-question diagnostic that decides which parts of Week 0 (and how much coding warm-up) you need."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCQ:
    area: str
    text: str
    options: list[str]
    answer: int
    why: str


# area -> (Week 0 day, what to do if weak)
AREAS = {
    "Linear algebra": (1, "Week 0 · D1 (and 3Blue1Brown chapters 1–9)"),
    "Probability & statistics": (2, "Week 0 · D2 (and Seeing Theory chapters 1–4)"),
    "Calculus & optimisation": (3, "Week 0 · D3 (and Karpathy's micrograd lecture)"),
    "Python": (4, "Week 0 · D4, then the first five Coding-interview problems"),
    "NumPy & tensors": (5, "Week 0 · D5 and the NumPy drills exercise"),
    "Algorithms & complexity": (5, "Week 0 · D5 complexity table, then the Patterns tab and easy coding problems first"),
}

QUESTIONS: list[MCQ] = [
    # Linear algebra
    MCQ("Linear algebra", "A is 3×4 and B is 4×2. What is the shape of (AB)ᵀ?",
        ["3×2", "2×3", "4×4", "AB is undefined"], 1, "AB is 3×2, so its transpose is 2×3."),
    MCQ("Linear algebra", "What is the rank of the outer product u vᵀ for non-zero vectors u ∈ ℝᵐ, v ∈ ℝⁿ?",
        ["1", "min(m, n)", "m", "It depends on u and v"], 0,
        "Every column is a multiple of u, so the column space is one-dimensional."),
    MCQ("Linear algebra", "Which gives the best rank-k approximation of a matrix (in Frobenius norm)?",
        ["Keeping the k rows with the largest norms", "Truncating the SVD to the top k singular values",
         "A QR factorisation with k columns", "Gaussian elimination stopped after k steps"], 1,
        "Eckart–Young: the truncated SVD is optimal in both Frobenius and spectral norm."),
    MCQ("Linear algebra", "In code, how should you solve Ax = b for a square, well-conditioned A?",
        ["x = np.linalg.inv(A) @ b", "x = np.linalg.solve(A, b)", "Invert AᵀA and multiply by Aᵀb",
         "Iterate x ← Ax until it converges"], 1,
        "solve() uses an LU factorisation: faster and more accurate than forming the inverse; the normal equations "
        "square the condition number."),
    # Probability & statistics
    MCQ("Probability & statistics", "A test has 99% sensitivity and a 1% false-positive rate; prevalence is 0.1%. "
        "Roughly what is P(disease | positive)?", ["99%", "50%", "9%", "1%"], 2,
        "0.99·0.001 / (0.99·0.001 + 0.01·0.999) ≈ 0.09 — the base rate dominates."),
    MCQ("Probability & statistics", "X and Y are independent with Var(X) = 4 and Var(Y) = 9. What is Var(X − Y)?",
        ["5", "−5", "13", "36"], 2, "Variances of independent variables add, even for a difference: 4 + 9."),
    MCQ("Probability & statistics", "Expected number of rolls of a fair die until the first 6?",
        ["3.5", "5", "6", "36"], 2, "Geometric distribution with p = 1/6: mean 1/p = 6."),
    MCQ("Probability & statistics", "A p-value of 0.03 means…",
        ["There is a 3% chance the null hypothesis is true",
         "If the null were true, data at least this extreme would appear 3% of the time",
         "There is a 97% chance the effect is real", "The effect size is large"], 1,
        "A p-value is computed assuming the null; it is not the probability of the null, nor an effect size."),
    # Calculus & optimisation
    MCQ("Calculus & optimisation", "What is ∇ₓ of f(x) = ‖Ax − b‖²?",
        ["2(Ax − b)", "2Aᵀ(Ax − b)", "Aᵀ(Ax − b) / 2", "2A(Ax − b)"], 1,
        "Chain rule: the gradient of ‖r‖² is 2r, then multiply by the Jacobian transpose Aᵀ."),
    MCQ("Calculus & optimisation", "What is the derivative of the sigmoid σ(z) = 1 / (1 + e⁻ᶻ)?",
        ["σ(z)²", "1 − σ(z)", "σ(z)(1 − σ(z))", "e⁻ᶻ"], 2,
        "σ' = e⁻ᶻ / (1 + e⁻ᶻ)² = σ(1 − σ). It peaks at 0.25, which is one reason deep sigmoid nets vanish."),
    MCQ("Calculus & optimisation", "Gradient descent on a quadratic whose Hessian's largest eigenvalue is 10 converges "
        "only if the learning rate is…", ["< 0.1", "< 0.2", "< 1", "< 10"], 1,
        "Along that eigen-direction the error is multiplied by (1 − ηλ); |1 − 10η| < 1 needs η < 2/10."),
    MCQ("Calculus & optimisation", "Why prefer (f(x+h) − f(x−h)) / 2h over (f(x+h) − f(x)) / h for gradient checks?",
        ["It's cheaper", "Its truncation error is O(h²) instead of O(h)", "It works without float64",
         "It avoids evaluating f"], 1, "The first-order error terms cancel in the central difference."),
    # Python
    MCQ("Python", "Average time complexity of `x in s` when s is a Python set?",
        ["O(1)", "O(log n)", "O(n)", "O(n log n)"], 0, "Sets are hash tables (O(n) only with pathological collisions)."),
    MCQ("Python", "After `grid = [[0] * 3] * 3` and `grid[0][0] = 1`, what is grid?",
        ["[[1, 0, 0], [0, 0, 0], [0, 0, 0]]", "[[1, 0, 0], [1, 0, 0], [1, 0, 0]]", "[[1, 1, 1], [0, 0, 0], [0, 0, 0]]",
         "An error"], 1, "The outer * repeats a reference to the same inner list three times."),
    MCQ("Python", "With `def f(x, acc=[]): acc.append(x); return acc`, what does the second call f(2) return "
        "after f(1)?", ["[2]", "[1, 2]", "[1]", "None"], 1,
        "Default values are evaluated once at definition time, so both calls share one list."),
    MCQ("Python", "Which is the right tool for a FIFO queue with many pushes and pops?",
        ["list.append + list.pop(0)", "collections.deque with append + popleft", "list.insert(0, x) + list.pop()",
         "heapq"], 1, "pop(0) and insert(0, x) shift every element (O(n)); deque is O(1) at both ends."),
    # NumPy & tensors
    MCQ("NumPy & tensors", "What is the shape of a (5, 1) array plus a (1, 3) array?",
        ["(5, 1)", "(1, 3)", "(5, 3)", "Error: shapes don't match"], 2,
        "Broadcasting stretches size-1 dimensions: (5, 1) and (1, 3) give (5, 3)."),
    MCQ("NumPy & tensors", "X has shape (100, 8). What is the shape of X.mean(axis=0)?",
        ["(100,)", "(8,)", "(1, 8)", "()"], 1, "axis=0 reduces over rows: one mean per column."),
    MCQ("NumPy & tensors", "After `a = np.arange(5); b = a[1:3]; b[:] = 0`, what is a?",
        ["[0, 1, 2, 3, 4]", "[0, 0, 0, 3, 4]", "[0, 0, 2, 3, 4]", "[0, 1, 0, 0, 4]"], 1,
        "Basic slicing returns a view, so writing through b changes a[1:3]."),
    MCQ("NumPy & tensors", "An int8 NumPy array holds 127. What does adding 1 (in int8) give?",
        ["128", "127 (saturates)", "−128", "OverflowError"], 2,
        "Integer array arithmetic wraps around in two's complement — why int8 kernels accumulate in int32 and clip."),
    # Algorithms & complexity
    MCQ("Algorithms & complexity", "n = 10⁵. Which time complexity will run in about a second in Python?",
        ["O(n²)", "O(n log n)", "O(n² log n)", "O(2ⁿ)"], 1,
        "About 10⁷ simple operations per second in Python: n log n ≈ 1.7·10⁶ is fine, n² = 10¹⁰ is not."),
    MCQ("Algorithms & complexity", "About how many comparisons does binary search need on a sorted array of 10⁶ items?",
        ["~10", "~20", "~1 000", "~500 000"], 1, "log₂(10⁶) ≈ 20."),
    MCQ("Algorithms & complexity", "Which structure gives the smallest element of a changing collection in O(log n) "
        "per insert/remove?", ["A sorted Python list", "A hash set", "A min-heap", "A stack"], 2,
        "heapq: push/pop O(log n), peek O(1). Insertion into a sorted list is O(n)."),
    MCQ("Algorithms & complexity", "Shortest path (fewest edges) between two nodes of an unweighted graph?",
        ["DFS", "BFS", "Topological sort", "Dijkstra is the only correct choice"], 1,
        "BFS visits nodes in order of distance; Dijkstra also works but is unnecessary without weights."),
]


def score(answers: dict[int, int | None]) -> dict[str, tuple[int, int]]:
    """answers: question index -> chosen option (None = skipped). Returns area -> (correct, total)."""
    out = {a: [0, 0] for a in AREAS}
    for i, q in enumerate(QUESTIONS):
        out[q.area][1] += 1
        if answers.get(i) == q.answer:
            out[q.area][0] += 1
    return {a: (c, t) for a, (c, t) in out.items()}


def recommend(scores: dict[str, tuple[int, int]]) -> dict:
    """Which Week 0 days to do, and how to pace the coding track."""
    plan = {}
    for area, (c, t) in scores.items():
        frac = c / t if t else 0
        day, what = AREAS[area]
        level = "skip" if frac == 1 else "skim" if frac >= 0.75 else "study"
        plan[area] = {"level": level, "day": day, "what": what, "frac": frac}
    days = sorted({p["day"] for p in plan.values() if p["level"] == "study"})
    skim = sorted({p["day"] for p in plan.values() if p["level"] == "skim"} - set(days))
    algo = plan["Algorithms & complexity"]["level"]
    coding = {"study": "Start the coding track with the easy problems and read the Patterns tab first; "
                       "add one extra easy problem a week for the first month.",
              "skim": "Follow the scheduled two problems a week; time every solve.",
              "skip": "Follow the schedule and try the timed mocks early (week 4+)."}[algo]
    return {"areas": plan, "study_days": days, "skim_days": skim, "coding": coding,
            "skip_week0": not days}
