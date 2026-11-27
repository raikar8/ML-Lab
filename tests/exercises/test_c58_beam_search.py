import itertools
import math

import numpy as np

from helpers import key_of, load

m = load(key_of(__file__))

V, EOS = 4, 0


def make_lm(seed):
    rng = np.random.default_rng(seed)
    table = {}

    def step(prefix):
        key = prefix[-2:]
        if key not in table:
            logits = rng.normal(size=V) * 2
            table[key] = (logits - np.log(np.exp(logits).sum())).tolist()
        return table[key]

    return step


def test_beam1_is_greedy():
    lm = make_lm(0)
    toks = ()
    for _ in range(6):
        lp = lm(toks)
        toks += (int(np.argmax(lp)),)
        if toks[-1] == EOS:
            break
    out = m.beam_search(lm, beam=1, max_len=6, eos=EOS)
    assert out[0][0] == list(toks)


def test_exhaustive_when_beam_is_large():
    lm = make_lm(1)
    L = 3
    best = None
    for n in range(1, L + 1):
        for seq in itertools.product(range(V), repeat=n):
            if EOS in seq[:-1] or (n < L and seq[-1] != EOS):
                continue
            lp = sum(lm(seq[:i])[seq[i]] for i in range(n))
            if best is None or lp > best[1] + 1e-12:
                best = (list(seq), lp)
    out = m.beam_search(lm, beam=V ** L, max_len=L, eos=EOS)
    assert out[0][0] == best[0] and math.isclose(out[0][1], best[1])


def test_eos_finishes_and_length_norm():
    def lm(prefix):
        # EOS is likely immediately; token 1 is a bit less likely but then very likely to continue cheaply
        if not prefix:
            return [math.log(0.6), math.log(0.4), -math.inf, -math.inf]
        if prefix[-1] == 1 and len(prefix) < 4:
            return [math.log(0.05), math.log(0.95), -math.inf, -math.inf]
        return [0.0, -math.inf, -math.inf, -math.inf]

    plain = m.beam_search(lm, beam=2, max_len=5, eos=EOS, alpha=0.0)
    assert plain[0][0] == [0]                       # shortest wins without normalisation
    norm = m.beam_search(lm, beam=2, max_len=5, eos=EOS, alpha=1.0)
    assert norm[0][0][0] == 1 and norm[0][0][-1] == EOS
    for toks, _ in plain + norm:
        assert EOS not in toks[:-1]
    assert len(plain) <= 2
