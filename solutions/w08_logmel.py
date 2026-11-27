"""
title: STFT framing, mel filterbank and log-mel features in PyTorch
week: 8
day: 4
level: medium
time: 40
tags: pytorch, audio, front-end
core: no

* `power_spectrogram(x, n_fft, hop)` — `x (T,)`; frames of length `n_fft` every `hop` samples (no centring/padding),
  periodic Hann window (`torch.hann_window(n_fft)`), `|rfft|²` → `(n_frames, n_fft//2 + 1)`.
* `hz_to_mel(f)` / `mel_to_hz(m)` — HTK formula `2595·log10(1 + f/700)`.
* `mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None)` → `(n_mels, n_fft//2+1)` triangular filters with peaks of 1 at
  `n_mels` centre frequencies equally spaced on the mel scale between fmin and fmax (edges included as the extra 2 points).
* `log_mel(x, sr, n_fft=512, hop=160, n_mels=40, eps=1e-6)` → `(n_frames, n_mels)` = `log(power @ fbank.T + eps)`.
"""
import torch


def power_spectrogram(x, n_fft, hop):
    frames = x.unfold(-1, n_fft, hop) * torch.hann_window(n_fft, dtype=x.dtype)
    return torch.fft.rfft(frames).abs() ** 2


def hz_to_mel(f):
    return 2595.0 * torch.log10(1.0 + torch.as_tensor(f, dtype=torch.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10 ** (torch.as_tensor(mel, dtype=torch.float64) / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, sr, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    pts = mel_to_hz(torch.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(fmax)), n_mels + 2))
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, dtype=torch.float64)
    lo, ce, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    up = (freqs - lo) / (ce - lo)
    down = (hi - freqs) / (hi - ce)
    return torch.clamp(torch.minimum(up, down), min=0).float()


def log_mel(x, sr, n_fft=512, hop=160, n_mels=40, eps=1e-6):
    p = power_spectrogram(x, n_fft, hop)
    fb = mel_filterbank(n_mels, n_fft, sr).to(p.dtype)
    return torch.log(p @ fb.T + eps)
