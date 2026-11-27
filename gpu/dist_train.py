"""Train a small transformer LM with DDP or FSDP2 and report memory and step time per rank.

    torchrun --nproc_per_node=2 gpu/dist_train.py --mode ddp
    torchrun --nproc_per_node=2 gpu/dist_train.py --mode fsdp
Uses NCCL on GPUs and Gloo on CPU (DDP only on CPU). Random tokens — this measures systems behaviour, not accuracy.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.n2(x))


class LM(nn.Module):
    def __init__(self, vocab, d, h, layers, T):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ddp", "fsdp"], default="ddp")
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=256)
    a = p.parse_args()
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(dev)
    torch.manual_seed(0)
    model = LM(32000, a.d, max(1, a.d // 64), a.layers, a.seq).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if a.mode == "ddp":
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if cuda else None)
    else:
        from torch.distributed.fsdp import fully_shard  # FSDP2 (PyTorch ≥ 2.6)
        for b in model.blocks:
            fully_shard(b)          # each block's params are all-gathered just before use, then freed
        fully_shard(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    g = torch.Generator(device="cpu").manual_seed(rank)
    times = []
    for step in range(a.steps):
        x = torch.randint(0, 32000, (a.batch, a.seq), generator=g).to(dev)
        t0 = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
            loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1).float(), x[:, 1:].flatten())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if cuda else float("nan")
    stats = torch.tensor([sum(times[3:]) / max(1, len(times) - 3), mem], device=dev)
    allstats = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(allstats, stats)
    if rank == 0:
        print(f"mode={a.mode} world={world} params={n_params / 1e6:.0f}M loss={loss.item():.3f}")
        for r, s in enumerate(allstats):
            print(f"  rank {r}: step {s[0].item() * 1e3:.0f} ms, peak memory {s[1].item():.2f} GiB")
        print(f"  expected optimiser+weights+grads ≈ {16 * n_params / 2 ** 30 / (world if a.mode == 'fsdp' else 1):.2f} GiB "
              f"per rank ({'sharded' if a.mode == 'fsdp' else 'replicated'}), plus activations")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
