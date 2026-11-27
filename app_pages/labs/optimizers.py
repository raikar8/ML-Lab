import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 6 · Lab")
st.title("Optimisers & LR schedules")
st.markdown("Race SGD, momentum, Adam and AdamW on an ill-conditioned quadratic or the Rosenbrock valley, and preview "
            "learning-rate schedules.")
c1, c2, c3 = st.columns(3)
surf = c1.radio("Surface", ["Ill-conditioned quadratic", "Rosenbrock"])
lr = c2.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], 1e-2)
steps = c3.slider("Steps", 50, 2000, 500)
if surf.startswith("Ill"):
    kappa = st.slider("Condition number", 1, 200, 50)
    f = lambda p: 0.5 * (kappa * p[0] ** 2 + p[1] ** 2)
    g = lambda p: np.array([kappa * p[0], p[1]])
    start, rng_ = np.array([1.0, 3.0]), (-3.5, 3.5)
else:
    f = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
    g = lambda p: np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])
    start, rng_ = np.array([-1.5, 2.0]), (-2, 2.5)


def run(kind):
    p = start.copy(); m = np.zeros(2); v = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        gr = np.clip(g(p), -1e3, 1e3)
        if kind == "SGD":
            p -= lr * gr
        elif kind == "Momentum":
            m = 0.9 * m + gr; p -= lr * m
        else:
            m = 0.9 * m + 0.1 * gr; v = 0.999 * v + 0.001 * gr ** 2
            p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            if kind == "AdamW":
                p -= lr * 0.01 * p
        path.append(p.copy())
        if not np.isfinite(p).all() or np.abs(p).max() > 1e3:
            break
    return np.array(path)


xs = np.linspace(*rng_, 120); ys = np.linspace(*rng_, 120)
Z = np.array([[f(np.array([a, b])) for a in xs] for b in ys])
fig = go.Figure(go.Contour(x=xs, y=ys, z=np.log1p(Z), showscale=False, colorscale="Greys", opacity=0.5))
res = {}
for kind in ["SGD", "Momentum", "Adam", "AdamW"]:
    P = run(kind); res[kind] = f(P[-1]) if np.isfinite(P[-1]).all() else np.inf
    fig.add_scatter(x=P[:, 0], y=P[:, 1], mode="lines", name=kind)
fig.update_layout(height=420, xaxis_range=rng_, yaxis_range=rng_, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
st.write({k: f"{v:.2e}" for k, v in res.items()})
st.subheader("Schedules")
T, W = 1000, 50
t = np.arange(T)
cos = np.where(t < W, (t + 1) / W, 0.1 + 0.45 * (1 + np.cos(np.pi * (t - W) / (T - W))))
wsd = np.where(t < W, (t + 1) / W, np.where(t < 0.8 * T, 1.0, 1 - (t - 0.8 * T) / (0.2 * T)))
st.line_chart({"warmup + cosine": cos, "WSD (warmup-stable-decay)": wsd}, height=220)
