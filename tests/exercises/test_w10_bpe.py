from helpers import key_of, load

m = load(key_of(__file__))


def test_bpe():
    text = "low lower lowest low low"
    merges = m.train_bpe(text, 10)
    assert merges[0] == ("l", "o") and merges[1] == ("lo", "w")
    enc = m.encode(text, merges)
    assert m.decode(enc) == text and len(enc) < len(text) / 2
    assert m.encode("lowly", merges)[0] == "low"
    assert m.train_bpe("abc", 5) == []
