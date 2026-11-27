import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_matches_nn_lstm():
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(5, 7, batch_first=True).double()
    x = torch.randn(3, 11, 5, dtype=torch.double)
    ref, (h, c) = lstm(x)
    out, (h2, c2) = m.lstm_forward(x, lstm)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(h2, h[0]); torch.testing.assert_close(c2, c[0])
