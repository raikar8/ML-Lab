import numpy as np
import torch
import torch.nn.functional as F
from helpers import key_of, load

m = load(key_of(__file__))


def test_matches_torch_ctc():
    rng = np.random.default_rng(0)
    for target in ([1, 2, 2, 3], [4], [1, 1, 1]):
        T, V = 12, 5
        lp = torch.log_softmax(torch.tensor(rng.normal(size=(T, V))), -1)
        ref = F.ctc_loss(lp[:, None], torch.tensor([target]), torch.tensor([T]), torch.tensor([len(target)]),
                         blank=0, reduction="sum")
        assert abs(m.ctc_loss(lp.numpy(), target) - ref.item()) < 1e-6


def test_impossible_and_greedy():
    lp = np.log(np.full((2, 3), 1 / 3))
    assert m.ctc_loss(lp, [1, 1]) == np.inf  # needs at least 3 frames
    probs = np.array([[.1, .8, .1], [.1, .8, .1], [.9, .05, .05], [.1, .8, .1], [.1, .1, .8]])
    assert m.greedy_decode(np.log(probs)) == [1, 1, 2]
