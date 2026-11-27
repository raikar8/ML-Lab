"""CPU-checkable parts of the GPU notebooks: static-KV decode equals the full forward; DDP script runs with Gloo."""
import subprocess
import sys
from pathlib import Path

import torch

from gpu.decode import TinyDecoder

ROOT = Path(__file__).resolve().parent.parent


def test_static_cache_decode_matches_full():
    torch.manual_seed(0)
    m = TinyDecoder(vocab=50, d=32, n_layers=2, n_heads=4, max_len=16).eval()
    toks = torch.randint(0, 50, (3, 10))
    with torch.inference_mode():
        full = m.forward_full(toks)
        cache = m.new_cache(3)
        steps = [m.step(toks[:, t], torch.tensor(t), cache) for t in range(10)]
    torch.testing.assert_close(torch.stack(steps, 1), full, atol=1e-5, rtol=1e-5)


def test_ddp_script_runs_on_cpu():
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29531",
                        str(ROOT / "gpu" / "dist_train.py"), "--mode", "ddp", "--d", "64", "--layers", "2", "--steps", "4",
                        "--batch", "2", "--seq", "32"], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "mode=ddp world=2" in r.stdout
