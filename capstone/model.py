"""DS-CNN keyword spotter (Hello Edge style) and structured channel pruning by BatchNorm |γ|."""
import torch
import torch.nn as nn


def conv_bn(c_in, c_out, k, stride, groups=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(c_out), nn.ReLU())


class DSCNN(nn.Module):
    """stem conv → [depthwise 3×3 → pointwise 1×1] × n_blocks → global average pool → linear."""

    def __init__(self, widths=(64, 64, 64, 64, 64), n_classes=10):
        super().__init__()
        self.widths = list(widths)
        self.stem = conv_bn(1, widths[0], 3, 2)
        self.blocks = nn.ModuleList()
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            self.blocks.append(nn.Sequential(conv_bn(c_in, c_in, 3, 1, groups=c_in), conv_bn(c_in, c_out, 1, 1)))
        self.fc = nn.Linear(widths[-1], n_classes)

    def forward(self, x):  # x (B, 40, 98)
        h = self.stem(x[:, None])
        for b in self.blocks:
            h = b(h)
        return self.fc(h.mean((2, 3)))


def macs(model, shape=(1, 40, 98)):
    total = 0

    def hook(m, i, o):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape))
    model.train(was)
    for h in hs:
        h.remove()
    return total


@torch.no_grad()
def prune_channels(model: DSCNN, keep_ratio: float) -> DSCNN:
    """Structured pruning: for each stage output, keep the channels with the largest BN |γ| and copy their weights into a
    narrower DSCNN. A pruned output channel also removes the matching depthwise filter and pointwise input in the next block."""
    stage_bns = [model.stem[1]] + [b[1][1] for b in model.blocks]
    keeps = [torch.argsort(bn.weight.abs(), descending=True)[: max(4, int(round(len(bn.weight) * keep_ratio)))].sort().values
             for bn in stage_bns]
    new = DSCNN([len(k) for k in keeps], model.fc.out_features)

    def copy_cbr(dst, src, out_idx, in_idx=None, depthwise=False):
        w = src[0].weight[out_idx]
        if in_idx is not None and not depthwise:
            w = w[:, in_idx]
        dst[0].weight.copy_(w)
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(dst[1], name).copy_(getattr(src[1], name)[out_idx])

    copy_cbr(new.stem, model.stem, keeps[0])
    for i, (nb, ob) in enumerate(zip(new.blocks, model.blocks)):
        copy_cbr(nb[0], ob[0], keeps[i], depthwise=True)
        copy_cbr(nb[1], ob[1], keeps[i + 1], in_idx=keeps[i])
    new.fc.weight.copy_(model.fc.weight[:, keeps[-1]])
    new.fc.bias.copy_(model.fc.bias)
    return new
