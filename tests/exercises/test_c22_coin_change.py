import functools
import random

from helpers import key_of, load

m = load(key_of(__file__))


def brute(coins, amount):
    coins = sorted(coins)

    @functools.lru_cache(None)
    def go(i, left):
        if left == 0:
            return (0, 1)
        if i == len(coins) or left < 0:
            return (float("inf"), 0)
        best, cnt = go(i + 1, left)
        b2, c2 = go(i, left - coins[i])
        return (min(best, b2 + 1), cnt + c2)

    b, c = go(0, amount)
    return (b if b != float("inf") else -1), c


def test_examples():
    assert m.fewest_coins([1, 2, 5], 11) == 3 and m.fewest_coins([2], 3) == -1
    assert m.fewest_coins([1], 0) == 0 and m.ways([1, 2, 5], 5) == 4
    assert m.ways([2], 3) == 0 and m.ways([7], 0) == 1
    assert m.fewest_coins([1, 3, 4], 6) == 2     # greedy would say 3


def test_random_against_brute_force():
    rng = random.Random(22)
    for _ in range(300):
        coins = rng.sample(range(1, 12), rng.randint(1, 4))
        amount = rng.randint(0, 40)
        assert (m.fewest_coins(coins, amount), m.ways(coins, amount)) == brute(coins, amount)


def bfs_fewest(coins, amount):
    # independent check: shortest path from 0 to amount, edges = coins
    from collections import deque
    dist = {0: 0}
    q = deque([0])
    while q:
        x = q.popleft()
        if x == amount:
            return dist[x]
        for c in coins:
            y = x + c
            if y <= amount and y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    return -1


def test_big_amount():
    assert m.fewest_coins([1, 7, 13, 29], 10_000) == bfs_fewest([1, 7, 13, 29], 10_000)
    assert m.fewest_coins([6, 10, 15], 10_007) == bfs_fewest([6, 10, 15], 10_007)
