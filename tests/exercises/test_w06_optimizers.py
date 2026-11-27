import torch
from helpers import key_of, load

m = load(key_of(__file__))


def run(opt_cls, ref_cls, kw, ref_kw, steps=15):
    torch.manual_seed(0)
    w0 = torch.randn(4, 3, dtype=torch.double)
    x = torch.randn(10, 4, dtype=torch.double)
    outs = []
    for cls, k in ((opt_cls, kw), (ref_cls, ref_kw)):
        w = w0.clone().requires_grad_()
        o = cls([w], **k)
        for _ in range(steps):
            o.zero_grad()
            ((x @ w).sin() ** 2).sum().backward()
            o.step()
        outs.append(w.detach())
    torch.testing.assert_close(outs[0], outs[1])


def test_sgd_momentum():
    run(m.SGDMomentum, torch.optim.SGD, dict(lr=0.05, momentum=0.9, weight_decay=0.01),
        dict(lr=0.05, momentum=0.9, weight_decay=0.01))


def test_adamw():
    run(m.AdamW, torch.optim.AdamW, dict(lr=0.01, weight_decay=0.1), dict(lr=0.01, weight_decay=0.1))


def test_newton_schulz_orthogonalises():
    torch.manual_seed(1)
    G = torch.randn(32, 64) @ torch.diag(torch.logspace(-2, 1, 64))
    X = m.newton_schulz(G, steps=5)
    s = torch.linalg.svdvals(X)
    assert X.shape == G.shape and s.min() > 0.5 and s.max() < 1.5
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    assert torch.nn.functional.cosine_similarity(X.flatten(), (U @ Vh).flatten(), dim=0) > 0.9
    assert m.newton_schulz(G.T).shape == (64, 32)


def test_muon_reduces_loss():
    torch.manual_seed(2)
    W = torch.zeros(8, 8, requires_grad=True)
    T = torch.randn(8, 8)
    o = m.Muon([W], lr=0.05)
    first = None
    for _ in range(50):
        o.zero_grad()
        loss = ((W - T) ** 2).sum()
        first = first or loss.item()
        loss.backward()
        o.step()
    assert loss.item() < 0.2 * first
