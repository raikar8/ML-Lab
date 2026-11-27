"""Capstone building blocks: pruning is exact at keep=1, the integer engine matches fake-quant, the C header compiles."""
import shutil
import subprocess

import numpy as np
import pytest
import torch

from capstone.model import DSCNN, prune_channels
from capstone.quant import FoldedNet, Int8Engine, calibrate, export_c_header, quantize_multiplier, requantize


def random_model():
    torch.manual_seed(0)
    m = DSCNN((16, 16, 16)).eval()
    for b in m.modules():
        if isinstance(b, torch.nn.BatchNorm2d):
            b.running_mean.uniform_(-0.5, 0.5); b.running_var.uniform_(0.5, 2); b.weight.data.uniform_(0.2, 2)
    return m


def test_prune_identity_and_shrink():
    m, x = random_model(), torch.randn(3, 40, 98)
    torch.testing.assert_close(prune_channels(m, 1.0).eval()(x), m(x))
    assert prune_channels(m, 0.5).widths == [8, 8, 8]


def test_requantize_and_engine_matches_fake_quant(tmp_path):
    M0, sh = quantize_multiplier(0.25)
    assert requantize(np.array([1000]), M0, np.array(sh))[0] == 250
    M0, sh = quantize_multiplier(1.5)  # multiplier ≥ 1 -> left shift
    assert requantize(np.array([100]), M0, np.array(sh))[0] == 150
    m = random_model()
    net = FoldedNet(m)
    x = torch.randn(64, 40, 98)
    with torch.no_grad():
        torch.testing.assert_close(net(x), m(x), atol=1e-4, rtol=1e-4)   # BN folding is exact
    calibrate(net, x)
    net.quant = True
    net.eval()
    eng = Int8Engine(net)
    with torch.no_grad():
        fq = net(x[:16]).numpy()
    ie = eng.run(x[:16].numpy())
    assert (fq.argmax(1) == ie.argmax(1)).mean() >= 0.9
    assert np.abs(fq - ie).max() < 0.1 * np.abs(fq).max() + 1e-3
    hdr = tmp_path / "kws_model.h"
    export_c_header(eng, hdr, 10)
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    src = tmp_path / "t.cpp"
    src.write_text('#include "kws_model.h"\nint main() { return L0_W[0] == 127 ? 0 : 0; }\n')
    assert subprocess.run([cxx, "-std=c++17", f"-I{tmp_path}", str(src), "-o", str(tmp_path / "t")]).returncode == 0
