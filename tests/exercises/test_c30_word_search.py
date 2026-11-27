import random

from helpers import key_of, load

m = load(key_of(__file__))

B = ["ABCE", "SFCS", "ADEE"]


def valid(board, word, path):
    if path is None:
        return False
    if len(path) != len(word) or len(set(path)) != len(path):
        return False
    for i, (r, c) in enumerate(path):
        if board[r][c] != word[i]:
            return False
        if i and abs(r - path[i - 1][0]) + abs(c - path[i - 1][1]) != 1:
            return False
    return True


def brute(board, word):
    R, C = len(board), len(board[0])

    def go(r, c, i, seen):
        if board[r][c] != word[i]:
            return False
        if i == len(word) - 1:
            return True
        seen = seen | {(r, c)}
        return any(0 <= nr < R and 0 <= nc < C and (nr, nc) not in seen and go(nr, nc, i + 1, seen)
                   for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))

    return not word or any(go(r, c, 0, frozenset()) for r in range(R) for c in range(C))


def test_examples():
    assert m.exists(B, "ABCCED") and m.exists(B, "SEE") and not m.exists(B, "ABCB")
    assert valid(B, "ABCCED", m.find_path(B, "ABCCED"))
    assert m.find_path(B, "") == [] and m.find_path(B, "Z") is None


def test_random_against_brute_force():
    rng = random.Random(30)
    for _ in range(400):
        R, C = rng.randint(1, 4), rng.randint(1, 4)
        board = ["".join(rng.choice("ab") for _ in range(C)) for _ in range(R)]
        word = "".join(rng.choice("ab") for _ in range(rng.randint(1, 6)))
        copy = list(board)
        p = m.find_path(board, word)
        assert (p is not None) == brute(board, word), (board, word)
        if p is not None:
            assert valid(board, word, p)
        assert board == copy


def test_adversarial_prunes():
    board = ["A" * 6] * 6
    assert not m.exists(board, "A" * 20 + "B")      # letter-count pruning
    board = ["AAAAAA"] * 5 + ["AAAAAB"]
    assert m.exists(board, "B" + "A" * 12)
