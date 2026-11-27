import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_advantages():
    a = m.group_advantages(torch.tensor([1.0, 0, 0, 1]))
    torch.testing.assert_close(a, torch.tensor([1.0, -1, -1, 1]), atol=1e-5, rtol=0)
    assert m.group_advantages(torch.ones(4)).abs().max() < 1e-6
    torch.testing.assert_close(m.group_advantages(torch.tensor([2.0, 0]), scale=False), torch.tensor([1.0, -1]))


def test_loss_properties():
    G, T = 4, 5
    lp = torch.randn(G, T)
    mask = torch.ones(G, T); mask[0, 3:] = 0
    adv = torch.tensor([1.0, -1, 1, -1])
    new = lp.clone().requires_grad_()
    loss = m.grpo_loss(new, lp, lp, adv, mask)
    assert abs(loss.item()) < 1e-6  # ratio 1, zero-mean adv, kl 0
    loss.backward()
    assert (new.grad[0, :3] < 0).all() and (new.grad[0, 3:] == 0).all() and (new.grad[1] > 0).all()
    big = (lp + 1.0).requires_grad_()  # ratio e > 1.2 -> clipped for positive adv
    m.grpo_loss(big, lp, big.detach(), torch.tensor([1.0, 1, 1, 1]), torch.ones(G, T), beta=0.0).backward()
    assert big.grad.abs().sum() == 0
    big.grad = None
    m.grpo_loss(big, lp, big.detach(), torch.tensor([1.0, 1, 1, 1]), torch.ones(G, T), beta=0.0, clip_high=3.0).backward()
    assert big.grad.abs().sum() > 0
