import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_ternary_and_training():
    torch.manual_seed(0)
    Wt, s = m.ternarize(torch.randn(16, 16))
    assert set(Wt.unique().tolist()) <= {-1.0, 0.0, 1.0} and s > 0
    X = torch.randn(512, 16)
    teacher = torch.sign(torch.randn(4, 16)) * (torch.rand(4, 16) > 0.3)
    y = X @ teacher.T
    layer = m.BitLinear(16, 4)
    opt = torch.optim.Adam(layer.parameters(), 0.01)
    for _ in range(400):
        opt.zero_grad(); loss = ((layer(X) - y) ** 2).mean(); loss.backward(); opt.step()
    assert loss.item() < 0.15 * (y ** 2).mean().item()
