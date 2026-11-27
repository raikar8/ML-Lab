import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 23 · Lab")
st.title("Pruning & low-rank")
st.markdown("A weight matrix with low-rank structure, applied to inputs with a few outlier features. Compare output "
            "error of unstructured magnitude pruning, Wanda (|W|·‖x‖), 2:4, structured (row) pruning and SVD truncation "
            "at the same parameter budget.")
keep = st.slider("Fraction of parameters kept", 0.1, 1.0, 0.5, 0.05)
rng = np.random.default_rng(0)
out, inp = 128, 256
W = rng.normal(size=(out, 16)) @ rng.normal(size=(16, inp)) / 4 + 0.3 * rng.normal(size=(out, inp))
X = rng.normal(size=(512, inp)); X[:, :6] *= 15
Y = X @ W.T
err = lambda Wp: np.linalg.norm(Y - X @ Wp.T) / np.linalg.norm(Y)


def row_prune(score, frac):
    k = int(round((1 - frac) * score.shape[1])); M = np.ones_like(score, bool)
    if k:
        idx = np.argpartition(score, k - 1, axis=1)[:, :k]; np.put_along_axis(M, idx, False, 1)
    return M


res = {"Magnitude (unstructured)": err(W * row_prune(np.abs(W), keep)),
       "Wanda |W|·‖x‖": err(W * row_prune(np.abs(W) * np.linalg.norm(X, axis=0), keep))}
g = (np.abs(W) * np.linalg.norm(X, axis=0)).reshape(out, -1, 4)
m24 = np.zeros_like(g, bool); np.put_along_axis(m24, np.argsort(-g, -1)[..., :2], True, -1)
res["2:4 (Wanda scores, fixed 50%)"] = err(W * m24.reshape(out, inp))
rows = np.argsort(-np.linalg.norm(W, axis=1))[: int(keep * out)]
Ws = np.zeros_like(W); Ws[rows] = W[rows]
res["Structured: keep rows"] = err(Ws)
r = max(1, int(keep * out * inp / (out + inp)))
U, S, Vt = np.linalg.svd(W, full_matrices=False)
res[f"SVD rank {r}"] = err((U[:, :r] * S[:r]) @ Vt[:r])
fig = go.Figure(go.Bar(x=list(res.values()), y=list(res.keys()), orientation="h", text=[f"{v:.3f}" for v in res.values()]))
fig.update_layout(height=320, xaxis_title="relative output error", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
st.line_chart({"singular values": S}, height=180)
st.caption("Unstructured/Wanda give the lowest error but need sparse kernels to be faster; structured and SVD give dense, "
           "smaller layers that are faster everywhere. Activation-aware scores (Wanda) matter when inputs have outliers.")
