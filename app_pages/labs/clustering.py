import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 4 · Lab")
st.title("k-means vs GMM (EM)")
st.markdown("Step through k-means and EM on elongated clusters. k-means assumes round, equal clusters; a full-covariance "
            "GMM adapts its ellipses and gives soft assignments.")
c1, c2, c3 = st.columns(3)
K = c1.slider("K", 2, 5, 3); steps = c2.slider("Iterations", 0, 40, 10); stretch = c3.slider("Cluster elongation", 1.0, 6.0, 4.0)
rng = np.random.default_rng(1)
C = np.array([[0, 0], [6, 1], [2, 6]])
X = np.vstack([rng.normal(size=(150, 2)) @ np.array([[stretch, 0], [0, 1]]) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * 0.6 + c
               for a, c in zip([0.3, 1.2, -0.8], C)])
init = X[rng.choice(len(X), K, replace=False)]
Ck = init.copy()
for _ in range(steps):
    lab = ((X[:, None] - Ck[None]) ** 2).sum(-1).argmin(1)
    Ck = np.array([X[lab == k].mean(0) if (lab == k).any() else Ck[k] for k in range(K)])
lab_k = ((X[:, None] - Ck[None]) ** 2).sum(-1).argmin(1)
mu, S, pi = init.copy(), np.stack([np.cov(X.T)] * K), np.full(K, 1 / K)
ll = []
for _ in range(max(1, steps)):
    lp = np.stack([np.log(pi[k]) - 0.5 * np.einsum("ni,ij,nj->n", X - mu[k], np.linalg.inv(S[k]), X - mu[k])
                   - 0.5 * np.log(np.linalg.det(2 * np.pi * S[k])) for k in range(K)], 1)
    mx = lp.max(1, keepdims=True); lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(1)); ll.append(lse.sum())
    R = np.exp(lp - lse[:, None]); Nk = R.sum(0); pi = Nk / len(X); mu = R.T @ X / Nk[:, None]
    S = np.stack([(R[:, k, None] * (X - mu[k])).T @ (X - mu[k]) / Nk[k] + 1e-6 * np.eye(2) for k in range(K)])
c1, c2 = st.columns(2)
f1 = go.Figure(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=lab_k, size=5)))
f1.add_scatter(x=Ck[:, 0], y=Ck[:, 1], mode="markers", marker=dict(symbol="x", size=14, color="black"))
f1.update_layout(height=360, showlegend=False, title="k-means", margin=dict(l=10, r=10, t=30, b=10))
c1.plotly_chart(f1, width="stretch")
f2 = go.Figure(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=R.argmax(1), size=5, opacity=np.clip(R.max(1), 0.2, 1))))
t = np.linspace(0, 2 * np.pi, 80)
for k in range(K):
    w, V = np.linalg.eigh(S[k]); e = (V * 2 * np.sqrt(w)) @ np.vstack([np.cos(t), np.sin(t)])
    f2.add_scatter(x=mu[k, 0] + e[0], y=mu[k, 1] + e[1], mode="lines", line=dict(color="black"))
f2.update_layout(height=360, showlegend=False, title="GMM (2σ ellipses; opacity = confidence)", margin=dict(l=10, r=10, t=30, b=10))
c2.plotly_chart(f2, width="stretch")
st.line_chart({"GMM log-likelihood": ll}, height=180)
st.caption("The EM log-likelihood never decreases — a property you can prove (week 4, day 2).")
