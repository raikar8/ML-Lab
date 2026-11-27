import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_tp_mlp():
    torch.manual_seed(0)
    x, A, B = torch.randn(5, 16, dtype=torch.double), torch.randn(16, 64, dtype=torch.double), torch.randn(64, 16, dtype=torch.double)
    ref = F.gelu(x @ A) @ B
    shards = m.shard_mlp(A, B, 4)
    assert shards[0][0].shape == (16, 16) and shards[0][1].shape == (16, 16)
    y, n = m.tp_forward(x, shards)
    torch.testing.assert_close(y, ref)
    assert n == 1
    y2, n2 = m.bad_tp_forward(x, shards)
    torch.testing.assert_close(y2, ref)
    assert n2 == 2
