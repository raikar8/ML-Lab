import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Weeks 21–22 · Lab")
st.title("Quantization lab")
with st.container(border=True):
    st.markdown("**What this lab is for.** See how bit width, granularity, outliers and the classic fixes "
                "(SmoothQuant scaling, Hadamard rotation) change quantization error on a weight-like matrix and an "
                "LLM-like activation matrix with outlier channels.")
    st.markdown("**Start here.** Set 4 bits and per-tensor, then switch to per-channel and group-wise. Then add "
                "outliers and compare the activation error with and without SmoothQuant / rotation.")

c1, c2, c3, c4 = st.columns(4)
bits = c1.select_slider("Bits", [2, 3, 4, 6, 8], 4, help="Integer bit width (symmetric, signed).")
gran = c2.radio("Weight granularity", ["per-tensor", "per-channel", "group-wise"], index=1)
group = c3.select_slider("Group size", [16, 32, 64, 128, 256], 64, disabled=gran != "group-wise")
outlier = c4.slider("Activation outlier scale", 1.0, 100.0, 30.0, help="Magnitude of a few outlier channels.")
seed = st.number_input("Seed", 0, 1000, 0)

rng = np.random.default_rng(int(seed))
d_in, d_out, n_tok = 512, 256, 128
W = rng.normal(0, 1, (d_out, d_in)) * rng.lognormal(0, 0.5, (d_out, 1))
X = rng.normal(0, 1, (n_tok, d_in))
out_ch = rng.choice(d_in, 6, replace=False)
X[:, out_ch] *= outlier
qmax = 2 ** (bits - 1) - 1


def q_sym(x, axis=None, group=None):
    if group:
        xs = x.reshape(x.shape[0], -1, group)
        s = np.abs(xs).max(-1, keepdims=True) / qmax
        return (np.clip(np.round(xs / s), -qmax, qmax) * s).reshape(x.shape)
    s = np.abs(x).max(axis=axis, keepdims=True) / qmax if axis is not None else np.abs(x).max() / qmax
    return np.clip(np.round(x / s), -qmax, qmax) * s


def sqnr(ref, est):
    return 10 * np.log10((ref ** 2).sum() / ((ref - est) ** 2).sum())


Wq = q_sym(W, None) if gran == "per-tensor" else q_sym(W, 1) if gran == "per-channel" else q_sym(W, group=group)
Y = X @ W.T
res = {"Weights only (fp activations)": sqnr(Y, X @ Wq.T)}
Xq = q_sym(X, 1)  # per-token activations
res["W + A, per-token activations"] = sqnr(Y, Xq @ Wq.T)
s = np.abs(X).max(0) ** 0.5 / np.abs(W).max(0) ** 0.5
Xs, Ws = X / s, W * s
Wsq = q_sym(Ws, 1) if gran != "group-wise" else q_sym(Ws, group=group)
res["W + A with SmoothQuant (α=0.5)"] = sqnr(Y, q_sym(Xs, 1) @ Wsq.T)
H = np.array([[1.0]])
while H.shape[0] < d_in:
    H = np.block([[H, H], [H, -H]])
Q = H / np.sqrt(d_in) * rng.choice([-1, 1], d_in)
Xr, Wr = X @ Q, W @ Q
Wrq = q_sym(Wr, 1) if gran != "group-wise" else q_sym(Wr, group=group)
res["W + A with Hadamard rotation"] = sqnr(Y, q_sym(Xr, 1) @ Wrq.T)

st.subheader("Output SQNR (higher is better)")
fig = go.Figure(go.Bar(x=list(res.values()), y=list(res.keys()), orientation="h",
                       text=[f"{v:.1f} dB" for v in res.values()], textposition="auto"))
fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="dB")
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Per-channel max |activation| before/after**")
    f2 = go.Figure()
    f2.add_scatter(y=np.abs(X).max(0), name="original", mode="lines")
    f2.add_scatter(y=np.abs(Xs).max(0), name="SmoothQuant", mode="lines")
    f2.add_scatter(y=np.abs(Xr).max(0), name="rotated", mode="lines")
    f2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_type="log", xaxis_title="channel")
    st.plotly_chart(f2, width="stretch")
with c2:
    st.markdown("**Weight value histogram and quantization levels (first row)**")
    f3 = go.Figure()
    f3.add_histogram(x=W[0], nbinsx=60, name="weights")
    for lv in np.unique(Wq[0])[:: max(1, len(np.unique(Wq[0])) // 40)]:
        f3.add_vline(x=lv, line_width=0.5, line_color="orange")
    f3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(f3, width="stretch")

st.markdown("""
**What to notice**
* Each bit adds ~6 dB; per-channel and smaller groups recover several dB at 4 bits because rows have different ranges.
* With outliers, per-token activation quantization collapses: the outlier channels set the scale for every token.
* **SmoothQuant** divides outlier channels down and multiplies the matching weight columns up — the product is
  unchanged but both sides quantize better. **Rotation** spreads outlier energy over all channels.
""")
