import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 2 · Lab")
st.title("Classification metrics & thresholds")
st.markdown("Two score distributions (negatives and positives). Change the class imbalance and separation and see why "
            "ROC-AUC can look great while precision collapses, and how the threshold trades precision for recall.")
c1, c2, c3 = st.columns(3)
pos_rate = c1.select_slider("Positive rate", [0.5, 0.2, 0.05, 0.01, 0.001], 0.05)
sep = c2.slider("Separation (d′)", 0.0, 4.0, 2.0)
thr = c3.slider("Threshold", -3.0, 6.0, 1.5)
rng = np.random.default_rng(0)
N = 50000
npos = max(10, int(N * pos_rate))
s = np.r_[rng.normal(0, 1, N - npos), rng.normal(sep, 1, npos)]
y = np.r_[np.zeros(N - npos), np.ones(npos)]
order = np.argsort(-s); ys = y[order]
tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
tpr, fpr, prec = tp / npos, fp / (N - npos), tp / (tp + fp)
auc = np.trapezoid(tpr, fpr); ap = (prec * ys).sum() / npos
pred = s >= thr
TP = int((pred & (y == 1)).sum()); FP = int((pred & (y == 0)).sum()); FN = npos - TP
P = TP / max(1, TP + FP); R = TP / npos
c1, c2 = st.columns(2)
f1 = go.Figure(); f1.add_scatter(x=fpr, y=tpr, name=f"ROC (AUC {auc:.3f})")
f1.add_scatter(x=[FP / (N - npos)], y=[R], mode="markers", marker=dict(size=12), name="threshold")
f1.update_layout(height=320, xaxis_title="FPR", yaxis_title="TPR", margin=dict(l=10, r=10, t=30, b=10), title="ROC")
c1.plotly_chart(f1, width="stretch")
f2 = go.Figure(); f2.add_scatter(x=tpr, y=prec, name=f"PR (AP {ap:.3f})")
f2.add_scatter(x=[R], y=[P], mode="markers", marker=dict(size=12), name="threshold")
f2.add_hline(y=pos_rate, line_dash="dot", annotation_text="baseline = positive rate")
f2.update_layout(height=320, xaxis_title="recall", yaxis_title="precision", margin=dict(l=10, r=10, t=30, b=10), title="PR")
c2.plotly_chart(f2, width="stretch")
a, b, c, d = st.columns(4)
a.metric("Precision", f"{P:.3f}"); b.metric("Recall", f"{R:.3f}")
c.metric("F1", f"{2 * P * R / max(1e-9, P + R):.3f}"); d.metric("False alarms", f"{FP:,}")
st.markdown("**Try:** keep d′ = 2 and lower the positive rate to 0.001 — AUC barely moves, average precision collapses.")
