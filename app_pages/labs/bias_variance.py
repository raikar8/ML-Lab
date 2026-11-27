import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 1 · Lab")
st.title("Bias–variance & regularisation")
st.markdown("Fit polynomials to noisy samples of a sine wave. Watch train vs test error as degree and L2 strength change, "
            "and how much the fit varies across resampled training sets (variance).")
c1, c2, c3 = st.columns(3)
deg = c1.slider("Polynomial degree", 0, 15, 3)
lam = c2.select_slider("L2 (ridge) λ", [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0], 0.0)
n = c3.slider("Training points", 8, 200, 20)
noise = st.slider("Noise σ", 0.0, 1.0, 0.3)
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
xt = np.linspace(0, 1, 300)


def feats(x, d):
    return np.vander(x, d + 1, increasing=True)


def fit(x, y, d, lam):
    A = feats(x, d)
    return np.linalg.solve(A.T @ A + lam * np.eye(d + 1), A.T @ y)


fig = go.Figure()
preds = []
for r in range(20):
    x = rng.uniform(0, 1, n); y = f(x) + noise * rng.normal(size=n)
    w = fit(x, y, deg, lam)
    p = feats(xt, deg) @ w
    preds.append(p)
    if r < 8:
        fig.add_scatter(x=xt, y=p, mode="lines", line=dict(width=1), opacity=0.5, showlegend=False)
fig.add_scatter(x=xt, y=f(xt), name="true", line=dict(width=3, color="black"))
fig.update_layout(height=340, yaxis_range=[-2, 2], margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
P = np.array(preds)
bias2 = ((P.mean(0) - f(xt)) ** 2).mean(); var = P.var(0).mean()
degs = list(range(0, 16))
tr, te = [], []
x = rng.uniform(0, 1, n); y = f(x) + noise * rng.normal(size=n)
xv = rng.uniform(0, 1, 500); yv = f(xv) + noise * rng.normal(size=500)
for d in degs:
    w = fit(x, y, d, lam)
    tr.append(((feats(x, d) @ w - y) ** 2).mean()); te.append(((feats(xv, d) @ w - yv) ** 2).mean())
c1, c2 = st.columns([2, 1])
f2 = go.Figure()
f2.add_scatter(x=degs, y=tr, name="train MSE"); f2.add_scatter(x=degs, y=te, name="test MSE")
f2.update_layout(height=300, yaxis_type="log", xaxis_title="degree", margin=dict(l=10, r=10, t=10, b=10))
c1.plotly_chart(f2, width="stretch")
c2.metric("Bias²", f"{bias2:.3f}"); c2.metric("Variance", f"{var:.3f}"); c2.metric("Noise σ²", f"{noise ** 2:.3f}")
st.markdown("**Try:** degree 15 with 20 points (variance explodes) → add λ = 1e-4 → add more points.")
