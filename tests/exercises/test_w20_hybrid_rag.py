import numpy as np
from helpers import key_of, load

m = load(key_of(__file__))

DOCS = [
    "To pair your phone, open Settings > Bluetooth on the head unit and select Add device.",
    "Warning code P0420 indicates catalyst efficiency below threshold. Visit a service centre.",
    "Tyre pressure should be 2.4 bar front and 2.2 bar rear when cold.",
    "The wiper fluid reservoir is under the bonnet on the left side.",
    "Adaptive cruise control keeps a set distance to the vehicle ahead.",
    "To reset the tyre pressure monitoring system, hold the TPMS button for 5 seconds.",
]


def test_parts():
    v = m.hash_embed("tyre pressure")
    assert abs(np.linalg.norm(v) - 1) < 1e-9 and not m.hash_embed("!!!").any()
    s = m.BM25(DOCS).scores("P0420")
    assert s.argmax() == 1 and s[0] == 0
    assert m.rrf([[0, 1, 2], [2, 0, 1]])[0] == 0


def test_retrieval_recall():
    qs = ["how do I connect my phone with bluetooth", "what does P0420 mean", "correct tyre pressure",
          "where is washer fluid reservoir", "reset TPMS"]
    rel = [0, 1, 2, 3, 5]
    res = [m.retrieve(q, DOCS, 2) for q in qs]
    assert m.recall_at_k(res, rel) >= 0.8
