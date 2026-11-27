import torch
from helpers import key_of, load

m = load(key_of(__file__))


def data():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(600, 4, generator=g) * torch.tensor([1.0, 10.0, 0.1, 5.0]) + 3
    y = ((X[:, 0] - 3) * (X[:, 1] - 3) > 0).long() + 2 * ((X[:, 3] - 3) > 0).long()
    return X, y


def test_learns():
    X, y = data()
    model = m.train_classifier(X[:500].clone(), y[:500], 4, epochs=40)
    assert m.evaluate(model, X[500:], y[500:]) > 0.85


def test_inputs_not_mutated_and_eval_is_deterministic():
    X, y = data()
    X0 = X.clone()
    model = m.train_classifier(X, y, 4, epochs=2)
    assert torch.equal(X, X0)
    model.train()
    a1, a2 = m.evaluate(model, X, y), m.evaluate(model, X, y)
    assert a1 == a2 and model.training
