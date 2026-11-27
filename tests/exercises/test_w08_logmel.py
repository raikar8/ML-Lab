import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_spectrogram_matches_stft():
    x = torch.randn(4000, dtype=torch.double)
    ref = torch.stft(x, 512, 160, window=torch.hann_window(512, dtype=torch.double), center=False,
                     return_complex=True).abs().T ** 2
    torch.testing.assert_close(m.power_spectrogram(x, 512, 160), ref)


def test_filterbank():
    fb = m.mel_filterbank(40, 512, 16000)
    assert fb.shape == (40, 257) and (fb >= 0).all()
    mx = fb.max(1).values
    assert (mx > 0.4).all() and (mx <= 1).all() and mx[20:].min() > 0.85
    peaks = fb.argmax(1)
    assert (peaks[1:] >= peaks[:-1]).all()
    assert abs(float(m.mel_to_hz(m.hz_to_mel(1000.0))) - 1000) < 1e-6


def test_log_mel_tone():
    sr = 16000
    t = torch.arange(sr) / sr
    feats = m.log_mel(torch.sin(2 * torch.pi * 1000 * t), sr)
    assert feats.shape == (97, 40)
    band = feats.mean(0).argmax()
    centre = m.mel_to_hz(torch.linspace(0, float(m.hz_to_mel(8000)), 42)[1:-1])[band]
    assert abs(float(centre) - 1000) < 150
