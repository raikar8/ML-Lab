import random

from helpers import key_of, load

m = load(key_of(__file__))


def ref(grid):
    R = len(grid); C = len(grid[0]) if R else 0
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in range(R):
        for c in range(C):
            if grid[r][c] == "1":
                parent[(r, c)] = (r, c)
    for (r, c) in list(parent):
        for nb in ((r + 1, c), (r, c + 1)):
            if nb in parent:
                parent[find(nb)] = find((r, c))
    sizes = {}
    for cell in parent:
        root = find(cell)
        sizes[root] = sizes.get(root, 0) + 1
    return len(sizes), max(sizes.values(), default=0)


def test_examples():
    g = ["11000", "11000", "00100", "00011"]
    assert m.num_islands(g) == 3 and m.largest_island(g) == 4
    assert m.num_islands([]) == 0 and m.largest_island(["000"]) == 0
    assert m.num_islands(["101", "010", "101"]) == 5      # diagonals don't connect


def test_random_against_union_find():
    rng = random.Random(14)
    for _ in range(300):
        R, C = rng.randint(1, 9), rng.randint(1, 9)
        g = ["".join(rng.choice("01") for _ in range(C)) for _ in range(R)]
        copy = list(g)
        assert (m.num_islands(g), m.largest_island(g)) == ref(g)
        assert g == copy


def test_large_grid():
    g = ["1" * 400] * 400
    assert m.num_islands(g) == 1 and m.largest_island(g) == 160_000
