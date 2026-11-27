"""Capstone pipeline: teacher → distilled student → channel pruning → QAT → integer-only engine → benchmark → C export.

    python -m capstone.pipeline --data synthetic --fast        # 1-minute smoke run
    python -m capstone.pipeline --data speechcommands          # real data (downloads ~2.3 GB once)
Outputs go to capstone/out/: report.md, metrics.json, kws_model.h (int8 weights for an MCU build), model_fp32.pt.
On a Raspberry Pi 5 / Jetson run with --bench-only to measure on that device.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .model import DSCNN, macs, prune_channels
from .quant import FoldedNet, Int8Engine, calibrate, export_c_header

OUT = Path(__file__).resolve().parent / "out"


def accuracy(model, x, y, batch=512):
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[i:i + batch]).argmax(-1) for i in range(0, len(x), batch)])
    return float((pred == y).float().mean())


def train(model, x, y, epochs, lr, teacher=None, T=4.0, alpha=0.3, batch=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(x) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], y[idx]
            xb = torch.roll(xb, int(torch.randint(-5, 6, (1,), generator=g)), dims=2)     # time-shift augmentation
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if teacher is not None:   # knowledge distillation (week 24): T² keeps gradient scale comparable
                with torch.no_grad():
                    tl = teacher(xb)
                kd = F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(tl / T, -1), log_target=True, reduction="batchmean")
                loss = alpha * loss + (1 - alpha) * T * T * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model


def latency_ms(model, runs=200):
    model.eval()
    x = torch.zeros(1, data.N_MELS, data.N_FRAMES)
    with torch.inference_mode():
        for _ in range(20):
            model(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 99))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "speechcommands"], default="synthetic")
    p.add_argument("--fast", action="store_true", help="tiny epochs for a smoke test")
    p.add_argument("--keep", type=float, default=0.75, help="fraction of channels kept by pruning")
    p.add_argument("--bench-only", action="store_true", help="only benchmark a saved model on this device")
    a = p.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    if a.bench_only:
        m = torch.load(OUT / "model_fp32.pt", weights_only=False)
        print("p50/p99 ms per 1 s window:", latency_ms(m))
        return
    if a.data == "synthetic":
        x, y = data.synthetic(3000 if a.fast else 6000)
        xt, yt = data.synthetic(600 if a.fast else 2000, seed=1)
    else:
        x, y = data.speechcommands("data", "training")
        xt, yt = data.speechcommands("data", "testing")
    ep = 6 if a.fast else 25
    tw, sw = ((64,) * 5, (32,) * 4) if a.fast else ((128,) * 6, (64,) * 5)
    res = {}
    teacher = train(DSCNN(tw), x, y, ep, 3e-3)
    res["teacher"] = dict(acc=accuracy(teacher, xt, yt), params=sum(q.numel() for q in teacher.parameters()), macs=macs(teacher))
    student = train(DSCNN(sw), x, y, ep, 3e-3, teacher=teacher)
    res["student (distilled)"] = dict(acc=accuracy(student, xt, yt), params=sum(q.numel() for q in student.parameters()), macs=macs(student))
    pruned = prune_channels(student, a.keep)
    res["pruned (before fine-tune)"] = dict(acc=accuracy(pruned, xt, yt), params=sum(q.numel() for q in pruned.parameters()), macs=macs(pruned))
    pruned = train(pruned, x, y, max(2, ep // 2), 1e-3, teacher=teacher)
    res["pruned + fine-tuned"] = dict(acc=accuracy(pruned, xt, yt), params=sum(q.numel() for q in pruned.parameters()), macs=macs(pruned))
    net = FoldedNet(pruned)
    calibrate(net, x[:1000])
    net.quant = True
    res["int8 fake-quant (PTQ)"] = dict(acc=accuracy(net, xt, yt))
    net.observing = False
    net.train()
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    for _ in range(max(1, ep // 5)):                      # QAT fine-tune (week 22): STE through rounding
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(net(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    res["int8 fake-quant (QAT)"] = dict(acc=accuracy(net, xt, yt))
    eng = Int8Engine(net)
    n_eval = 300 if a.fast else len(xt)
    logits = eng.run(xt[:n_eval].numpy())
    int_pred = torch.tensor(logits.argmax(1))
    with torch.no_grad():
        fq_pred = net(xt[:n_eval]).argmax(-1)
    res["int8 integer-only engine"] = dict(acc=float((int_pred == yt[:n_eval]).float().mean()),
                                           agreement_with_fake_quant=float((int_pred == fq_pred).float().mean()),
                                           size_bytes=eng.size_bytes(), peak_activation_bytes=eng.peak_activation_bytes())
    res["latency_fp32_cpu_ms_p50_p99"] = latency_ms(pruned)
    export_c_header(eng, OUT / "kws_model.h", n_classes=10)
    torch.save(pruned, OUT / "model_fp32.pt")
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    lines = ["# Capstone report", "", f"Data: {a.data}. Test samples: {len(xt)}." + (" **--fast smoke test: accuracies are not meaningful.**" if a.fast else ""), "",
             "| Stage | Accuracy | Params | MACs |", "|---|---|---|---|"]
    for k, v in res.items():
        if isinstance(v, dict):
            lines.append(f"| {k} | {v['acc']:.3f} | {v.get('params', '')} | {v.get('macs', '')} |")
    e = res["int8 integer-only engine"]
    lines += ["", f"Integer engine: {e['size_bytes'] / 1024:.1f} KB weights+constants, peak activations "
              f"{e['peak_activation_bytes'] / 1024:.1f} KB (int8), agreement with fake-quant {e['agreement_with_fake_quant']:.3f}.",
              f"FP32 CPU latency per 1 s window (this machine): p50 {res['latency_fp32_cpu_ms_p50_p99'][0]:.2f} ms, "
              f"p99 {res['latency_fp32_cpu_ms_p50_p99'][1]:.2f} ms.", "",
              "Next: measure on your target (see the Capstone page) and fill in device latency, energy and memory."]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main()
