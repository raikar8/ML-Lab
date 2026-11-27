"""Correctness of the Triton kernels via Triton's CPU interpreter (no GPU needed). Performance is measured in the notebooks."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")
import pytest  # noqa: E402

triton = pytest.importorskip("triton")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpu import flash_attention as fa  # noqa: E402
from gpu import triton_kernels as tk  # noqa: E402


def test_softmax_and_rmsnorm():
    x = torch.randn(7, 100) * 5
    torch.testing.assert_close(tk.softmax(x), torch.softmax(x, -1), atol=1e-5, rtol=1e-5)
    w, r = torch.rand(100), torch.randn(7, 100)
    y, res = tk.add_rmsnorm(x, w, r)
    torch.testing.assert_close(res, x + r)
    torch.testing.assert_close(y, F.rms_norm(x + r, (100,), w, eps=1e-6), atol=1e-5, rtol=1e-5)


def test_matmuls():
    a, b = torch.randn(70, 50), torch.randn(50, 90)
    torch.testing.assert_close(tk.matmul(a, b, 32, 32, 16), a @ b, atol=1e-3, rtol=1e-3)
    wq, s = tk.quantize_w8(b)
    torch.testing.assert_close(tk.w8_matmul(a, wq, s, 32, 32, 16), a @ (wq.float() * s), atol=1e-3, rtol=1e-3)


def test_flash_attention():
    q, k, v = (torch.randn(2, 2, 50, 16) for _ in range(3))
    for causal in (False, True):
        torch.testing.assert_close(fa.flash_attention(q, k, v, causal, 16, 16),
                                   F.scaled_dot_product_attention(q, k, v, is_causal=causal), atol=1e-4, rtol=1e-4)
