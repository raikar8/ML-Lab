import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_sizes_and_cost():
    assert m.conv_out_size(224, 7, 2, 3) == 112 and m.conv_out_size(10, 3, dilation=2) == 6
    assert m.conv_cost(3, 64, 7, 112, 112) == (9408, 9408 * 112 * 112)
    assert m.conv_cost(32, 32, 3, 10, 10, groups=32) == (288, 28800)


def test_im2col_and_conv():
    x = torch.randn(2, 3, 9, 8, dtype=torch.double)
    torch.testing.assert_close(m.im2col(x, 3, 2, 1), F.unfold(x, 3, padding=1, stride=2))
    w, b = torch.randn(5, 3, 3, 3, dtype=torch.double), torch.randn(5, dtype=torch.double)
    torch.testing.assert_close(m.conv2d(x, w, b, stride=2, pad=1), F.conv2d(x, w, b, stride=2, padding=1))


def test_no_library_conv(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("don't call F.conv2d / F.unfold")
    monkeypatch.setattr(F, "conv2d", boom)
    monkeypatch.setattr(F, "unfold", boom)
    m.conv2d(torch.randn(1, 2, 5, 5), torch.randn(3, 2, 3, 3))
