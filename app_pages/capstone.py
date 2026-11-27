"""Hardware capstone: one model taken from training to measured integer inference on real hardware."""
import json

import streamlit as st

from mlprep import progress
from mlprep.content import ROOT
from mlprep.ui import data

d = data()
cap = d.setdefault("capstone", {"done": {}, "report": {}})
OUT = ROOT / "capstone" / "out"

st.title("Hardware capstone")
st.markdown(
    "One project that ties the embedded half of the plan together and gives you **measured numbers on a device** — the "
    "strongest thing you can bring to an embedded-AI interview. The pipeline in `capstone/` is written from scratch: a "
    "DS-CNN keyword spotter is trained as a teacher, **distilled** into a student, **channel-pruned** by BatchNorm |γ|, "
    "**BN-folded** and trained with **int8 QAT**, then executed by a NumPy **integer-only engine** (int8 activations with "
    "zero-points, per-channel weights, int32 accumulators, Q31 multiplier + shift requantisation — what CMSIS-NN does). "
    "It exports the int8 weights and requantisation constants as a **C header** for an MCU build."
)
st.code("python -m capstone.pipeline --data synthetic --fast    # ~2-minute smoke test (accuracies not meaningful)\n"
        "python -m capstone.pipeline --data speechcommands      # the real run (downloads Speech Commands once)\n"
        "python -m capstone.pipeline --bench-only               # on the target device, after copying capstone/out/", language="bash")

MILESTONES = [
    ("run", "Run the pipeline on Speech Commands (10 keywords) and read the report", "Weeks 7–9, 24"),
    ("teacher", "Teacher ≥ 92% test accuracy; explain its params/MACs", "Week 7"),
    ("distil", "Distilled student within ~2 points of the teacher at ≤ ¼ of the MACs", "Week 24"),
    ("prune", "Prune 25–50% of channels and recover accuracy with fine-tuning; plot accuracy vs MACs", "Week 24"),
    ("qat", "QAT int8: integer-engine accuracy within 1 point of float; agreement ≥ 99%", "Weeks 22–23"),
    ("stream", "Make it streaming (causal convs with state, week-7 exercise) and verify streaming = offline", "Weeks 7, 25"),
    ("device", "Measure on a device: Raspberry Pi 5 (p50/p99 latency, RSS memory, power with a USB meter)", "Weeks 21, 25"),
    ("mcu", "Stretch: run kws_model.h on an MCU (ESP32-S3 / Cortex-M) with your C++ track kernels (int8 GEMM, im2col, "
            "requantisation); report cycles per inference and SRAM", "Week 21 + C++ track"),
    ("writeup", "Write the one-page report below and add it to your GitHub README with a diagram", "Week 27"),
]
st.subheader("Milestones")
for key, text, wk in MILESTONES:
    v = st.checkbox(f"{text}  \n:gray[{wk}]", value=cap["done"].get(key, False), key=f"cap-{key}")
    if v != cap["done"].get(key, False):
        cap["done"][key] = v
        progress.save(d)

st.subheader("Target-specific notes")
t1, t2, t3 = st.tabs(["Raspberry Pi 5 / Linux SBC", "MCU (ESP32-S3, Cortex-M)", "Jetson / NPU"])
with t1:
    st.markdown("""
* `pip install torch numpy` on the Pi (64-bit OS), copy `capstone/out/model_fp32.pt`, run `--bench-only`.
* Report p50/p99 latency per 1 s window and per 10 ms hop if you made it streaming; RSS memory (`/usr/bin/time -v`);
  energy with a USB-C power meter: (average W − idle W) × time per inference.
* Optional: export with `torch.export` and run via ExecuTorch (XNNPACK) or LiteRT and compare.""")
with t2:
    st.markdown("""
* `kws_model.h` contains per-layer int8 weights, int32 biases, Q31 multipliers and shifts, and zero-points. The C++ track
  exercises 08 (quantisation), 09 (int8 GEMM + requantisation) and 12 (im2col) are exactly the kernels you need; depthwise layers
  are the groups = C case.
* Compute features on the device with the same front-end (40 log-mels, 25 ms / 10 ms); check bit-exactness against
  `capstone.quant.Int8Engine` on a few recorded inputs before measuring.
* Report cycles (DWT counter on Cortex-M, `esp_cpu_get_cycle_count` on ESP32), peak SRAM (the planner from the week-21 exercise),
  and flash size.""")
with t3:
    st.markdown("""
* Export with `torch.export`/ONNX, build a TensorRT INT8 engine (or the vendor NPU SDK), and compare latency and accuracy with your
  own integer engine — differences point to calibration or rounding mismatches.""")

st.subheader("Your report")
fields = ["Device & runtime", "Accuracy (float → int8)", "Params / MACs", "Model size (KB)", "Peak RAM (KB)",
          "Latency p50 / p99", "Energy per inference", "What limited performance", "What you'd do next"]
for f in fields:
    v = st.text_input(f, cap["report"].get(f, ""), key=f"rep-{f}")
    if v != cap["report"].get(f, ""):
        cap["report"][f] = v
        progress.save(d)
if (OUT / "report.md").exists():
    with st.expander("Latest pipeline report (capstone/out/report.md)", expanded=True):
        st.markdown((OUT / "report.md").read_text())
    if (OUT / "metrics.json").exists():
        st.json(json.loads((OUT / "metrics.json").read_text()), expanded=False)
else:
    st.caption("Run the pipeline to see its report here.")
