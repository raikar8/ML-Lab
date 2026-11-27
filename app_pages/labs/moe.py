import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Weeks 12 & 18 · Lab")
st.title("Mixture-of-Experts routing")
with st.container(border=True):
    st.markdown("**What this lab is for.** Simulate a router with a built-in preference for some experts and compare "
                "three balancing strategies: none, an auxiliary loss (as a gradient on router logits), and DeepSeek-V3's "
                "aux-loss-free bias updates.")
    st.markdown("**Start here.** Increase the skew, then compare the load histograms and the max/mean load ratio over "
                "steps. Notice that the bias method balances *selection* without changing the gate weights.")

c1, c2, c3, c4 = st.columns(4)
E = c1.select_slider("Experts", [8, 16, 32, 64, 128, 256], 64)
k = c2.select_slider("Top-k", [1, 2, 4, 6, 8], 8)
skew = c3.slider("Router skew", 0.0, 4.0, 2.0)
steps = c4.slider("Steps", 20, 400, 200)
c1, c2 = st.columns(2)
gamma = c1.slider("Bias update speed γ", 0.001, 0.1, 0.01, format="%.3f")
alpha = c2.slider("Aux-loss strength", 0.0, 1.0, 0.1)
rng = np.random.default_rng(0)
pref = np.linspace(skew, -skew, E)
N = 2048


def run(mode):
    bias, delta = np.zeros(E), np.zeros(E)
    ratios, loads = [], None
    for _ in range(steps):
        logits = rng.normal(0, 1, (N, E)) + pref + delta
        s = 1 / (1 + np.exp(-logits))
        sel = s + (bias if mode == "bias" else 0)
        idx = np.argpartition(-sel, k - 1, axis=1)[:, :k]
        loads = np.bincount(idx.ravel(), minlength=E)
        ratios.append(loads.max() / loads.mean())
        if mode == "bias":
            bias -= gamma * np.sign(loads - loads.mean())
        elif mode == "aux":
            f = loads / idx.size
            P = (np.exp(logits) / np.exp(logits).sum(1, keepdims=True)).mean(0)
            delta -= alpha * E * (f - 1 / E) * 5  # gradient-like push on preferred experts
    return np.array(ratios), loads


res = {m: run(m) for m in ("none", "aux", "bias")}
fig = go.Figure()
for m, (r, _) in res.items():
    fig.add_scatter(y=r, name={"none": "no balancing", "aux": "aux loss", "bias": "aux-loss-free bias"}[m])
fig.update_layout(height=300, yaxis_title="max / mean expert load", xaxis_title="step", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
fig2 = go.Figure()
for m, (_, ld) in res.items():
    fig2.add_bar(y=ld, name=m, opacity=0.6)
fig2.update_layout(barmode="overlay", height=280, xaxis_title="expert", yaxis_title="tokens (last step)",
                   margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, width="stretch")
st.markdown("""
**What to notice.** Without balancing a few experts take most tokens (with expert parallelism, they set the step time).
The aux loss balances by changing the router's logits, which competes with the language-modelling objective in real
training; the bias method reaches balance with a tiny per-expert offset used only for *selection*. Too large γ makes the
bias oscillate.
""")
