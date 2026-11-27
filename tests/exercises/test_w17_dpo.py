import math

import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_seq_logprob():
    logits = torch.log(torch.tensor([[[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]]]))
    lp = m.sequence_logprob(logits, torch.tensor([[0, 0, 1]]), torch.tensor([[0.0, 1, 1]]))
    assert abs(lp.item() - (math.log(0.9) + math.log(0.8))) < 1e-6


def test_dpo():
    z = torch.zeros(3)
    loss, rw, rl = m.dpo_loss(z, z, z, z)
    assert abs(loss.item() - math.log(2)) < 1e-6
    pw = torch.tensor([0.0, 0, 0], requires_grad=True)
    loss, rw, rl = m.dpo_loss(pw, torch.tensor([-5.0] * 3), z, z, beta=0.5)
    assert loss.item() < math.log(2) and (rl < rw).all()
    loss.backward()
    assert (pw.grad < 0).all()  # increasing chosen log-prob lowers the loss
