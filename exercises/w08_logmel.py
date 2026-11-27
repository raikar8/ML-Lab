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
    raise NotImplementedError  # TODO: implement


def hz_to_mel(f):
    raise NotImplementedError  # TODO: implement


def mel_to_hz(mel):
    raise NotImplementedError  # TODO: implement


def mel_filterbank(n_mels, n_fft, sr, fmin=0.0, fmax=None):
    raise NotImplementedError  # TODO: implement


def log_mel(x, sr, n_fft=512, hop=160, n_mels=40, eps=1e-6):
    raise NotImplementedError  # TODO: implement
