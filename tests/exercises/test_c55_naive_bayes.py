import math

import numpy as np

from helpers import key_of, load

m = load(key_of(__file__))

DOCS = [["good", "great", "fun"], ["great", "acting"], ["bad", "boring"], ["boring", "plot", "bad"], ["fun", "fun"]]
Y = ["pos", "pos", "neg", "neg", "pos"]


def test_hand_computed():
    nb = m.MultinomialNB(alpha=1.0).fit(DOCS, Y)
    assert nb.classes_ == ["neg", "pos"]
    V = 7  # acting bad boring fun good great plot
    # P(bad|neg) = (2+1)/(5+7); P(bad|pos) = (0+1)/(7+7)
    lp = nb.predict_log_proba([["bad"]])[0]
    neg = math.log(2 / 5) + math.log(3 / 12)
    pos = math.log(3 / 5) + math.log(1 / 14)
    z = math.log(math.exp(neg) + math.exp(pos))
    assert np.allclose(lp, [neg - z, pos - z])
    assert nb.predict([["fun", "great"], ["boring"], ["unknown"]]) == ["pos", "neg", "pos"]


def test_normalised_and_stable():
    nb = m.MultinomialNB(alpha=0.5).fit(DOCS, Y)
    lp = nb.predict_log_proba([["bad"] * 5000 + ["fun"] * 10, []])
    assert np.all(np.isfinite(lp))
    assert np.allclose(np.log(np.exp(lp).sum(axis=1)), 0)
    assert nb.predict([["bad"] * 5000])[0] == "neg"
    # empty / OOV-only doc -> posterior equals prior
    assert np.allclose(np.exp(lp[1]), [2 / 5, 3 / 5])


def test_learns_synthetic():
    rng = np.random.default_rng(0)
    va, vb = [f"a{i}" for i in range(20)], [f"b{i}" for i in range(20)]
    docs, ys = [], []
    for _ in range(200):
        c = rng.integers(2)
        pool = (va if c else vb) * 3 + va + vb
        docs.append(list(rng.choice(pool, size=15)))
        ys.append(int(c))
    nb = m.MultinomialNB().fit(docs[:150], ys[:150])
    acc = np.mean(np.array(nb.predict(docs[150:])) == np.array(ys[150:]))
    assert acc > 0.9
