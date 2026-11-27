"""Data for the capstone.

* `speech_commands(root)` — Google Speech Commands v2 (10 keywords) via torchaudio's downloader, featurised with the
  log-mel front-end below (the same maths as the week-8 exercise).
* `synthetic(n, n_classes)` — a fast stand-in: each class is a smooth time–frequency template, randomly shifted and noised.
  Use it to check the pipeline end to end in seconds before downloading 2 GB of audio.
"""
import math

import torch

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
N_MELS, N_FRAMES = 40, 98


def mel_filterbank(n_mels=N_MELS, n_fft=512, sr=16000):
    mel = lambda f: 2595 * math.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = torch.tensor([inv(m) for m in torch.linspace(mel(20), mel(sr / 2), n_mels + 2).tolist()])
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    lo, ce, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    return torch.clamp(torch.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)), min=0)


_FB = None


def log_mel(wave, n_fft=512, win=400, hop=160):
    """wave (16000,) float in [-1, 1] -> (40, 98) log-mel; 25 ms Hann window, 10 ms hop, no centring."""
    global _FB
    _FB = mel_filterbank() if _FB is None else _FB
    wave = torch.nn.functional.pad(wave, (0, max(0, 16000 - wave.numel())))[:16000]
    frames = wave.unfold(0, win, hop)[:N_FRAMES] * torch.hann_window(win)
    spec = torch.fft.rfft(frames, n=n_fft).abs() ** 2
    return torch.log(spec @ _FB.T + 1e-6).T


def speech_commands(root="data", subset="training", limit=None):
    import torchaudio  # only needed for the real dataset
    ds = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset=subset)
    xs, ys = [], []
    for i in range(len(ds)):
        path = ds._walker[i]
        label = path.split("/")[-2]
        if label not in KEYWORDS:
            continue
        wave, sr, *_ = ds[i]
        xs.append(log_mel(wave[0]))
        ys.append(KEYWORDS.index(label))
        if limit and len(xs) >= limit:
            break
    x = torch.stack(xs)
    return (x - x.mean()) / x.std(), torch.tensor(ys)


def synthetic(n=2000, n_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, N_FRAMES)
    f = torch.linspace(0, 1, N_MELS)[:, None]
    templates = []
    for c in range(n_classes):
        f0, slope, width = 0.15 + 0.07 * c, (c % 3 - 1) * 0.3, 0.05 + 0.01 * (c % 4)
        templates.append(torch.exp(-((f - (f0 + slope * (t - 0.5))) ** 2) / (2 * width ** 2)) * torch.exp(-((t - 0.5) ** 2) / 0.08))
    y = torch.randint(0, n_classes, (n,), generator=g)
    x = torch.stack([templates[i] for i in y.tolist()])
    shifts = torch.randint(-15, 16, (n,), generator=g)
    x = torch.stack([torch.roll(xi, int(s), dims=1) for xi, s in zip(x, shifts)])
    x = x + 0.35 * torch.randn(x.shape, generator=g)
    return (x - x.mean()) / x.std(), y
