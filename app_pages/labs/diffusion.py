import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 16 · Lab")
st.title("Diffusion on a toy distribution")
st.markdown("Data is a 1-D mixture of Gaussians, so the noisy marginal and its **exact score** are known in closed form "
            "— no network needed. See the forward process blur the data, then sample back with DDPM (stochastic) or "
            "DDIM (deterministic) using the exact ε.")
c1, c2, c3 = st.columns(3)
steps = c1.select_slider("Sampling steps", [5, 10, 20, 50, 100, 250, 1000], 50)
sampler = c2.radio("Sampler", ["DDPM", "DDIM"]); t_show = c3.slider("Show forward marginal at t/T", 0.0, 1.0, 0.3)
mus, sig, w = np.array([-2.0, 1.0, 3.0]), 0.3, np.array([0.3, 0.5, 0.2])
T = 1000
betas = np.linspace(1e-4, 0.02, T); ab = np.cumprod(1 - betas)


def eps_star(x, t):
    a = ab[t]; m = np.sqrt(a) * mus; v = a * sig ** 2 + (1 - a)
    logp = np.log(w) - 0.5 * (x[:, None] - m) ** 2 / v
    r = np.exp(logp - logp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    score = (r * (m - x[:, None]) / v).sum(1)
    return -np.sqrt(1 - a) * score


rng = np.random.default_rng(0)
x0 = np.concatenate([rng.normal(m, sig, int(5000 * ww)) for m, ww in zip(mus, w)])
tt = int(t_show * (T - 1))
xt = np.sqrt(ab[tt]) * x0 + np.sqrt(1 - ab[tt]) * rng.normal(size=x0.size)
ts = np.linspace(T - 1, 0, steps).round().astype(int)
x = rng.normal(size=5000)
for i, t in enumerate(ts):
    e = eps_star(x, t)
    a_t = ab[t]; a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else 1.0
    x0_hat = (x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t)
    if sampler == "DDIM":
        x = np.sqrt(a_prev) * x0_hat + np.sqrt(1 - a_prev) * e
    else:
        beta_eff = 1 - a_t / a_prev
        var = beta_eff * (1 - a_prev) / (1 - a_t)
        mean = np.sqrt(a_prev) * beta_eff / (1 - a_t) * x0_hat + np.sqrt(1 - beta_eff) * (1 - a_prev) / (1 - a_t) * x
        x = mean + (np.sqrt(var) * rng.normal(size=x.size) if i + 1 < len(ts) else 0)
fig = go.Figure()
fig.add_histogram(x=x0, name="data x₀", opacity=0.5, nbinsx=120, histnorm="probability density")
fig.add_histogram(x=xt, name=f"x_t at t={tt}", opacity=0.5, nbinsx=120, histnorm="probability density")
fig.add_histogram(x=x, name=f"{sampler} samples ({steps} steps)", opacity=0.5, nbinsx=120, histnorm="probability density")
fig.update_layout(barmode="overlay", height=380, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
st.caption("With the exact score, DDIM stays accurate at few steps; DDPM needs more. A learned ε-network adds its own error.")
