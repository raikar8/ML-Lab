import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_schedule_and_q_sample_statistics():
    s = m.linear_schedule(1000)
    assert s["alpha_bars"][-1] < 1e-4 and abs(float(s["alpha_bars"][0]) - (1 - 1e-4)) < 1e-12
    x0 = torch.full((20000, 1), 2.0, dtype=torch.double)
    t = torch.full((20000,), 300, dtype=torch.long)
    xt = m.q_sample(x0, t, torch.randn(20000, 1, dtype=torch.double), s)
    ab = float(s["alpha_bars"][300])
    assert abs(xt.mean().item() - 2 * ab ** 0.5) < 0.03 and abs(xt.var().item() - (1 - ab)) < 0.03


def test_loss_zero_for_oracle_and_reverse_step():
    s = m.linear_schedule(100)
    x0 = torch.randn(8, 3, dtype=torch.double)
    g = torch.Generator().manual_seed(0)
    t = torch.randint(0, 100, (8,), generator=torch.Generator().manual_seed(0))
    noise = torch.randn(x0.shape, generator=g, dtype=x0.dtype)
    class Oracle:
        def __call__(self, xt, tt):
            ab = s["alpha_bars"][tt].reshape(-1, 1)
            return (xt - ab.sqrt() * x0) / (1 - ab).sqrt()
    assert m.ddpm_loss(Oracle(), x0, s, torch.Generator().manual_seed(0)).item() < 1e-20
    xt = m.q_sample(x0, torch.full((8,), 1, dtype=torch.long), noise, s)
    x_prev = m.p_sample_step(Oracle(), xt, 1, s, torch.zeros_like(xt))
    assert (x_prev - x0).norm() < (xt - x0).norm()
