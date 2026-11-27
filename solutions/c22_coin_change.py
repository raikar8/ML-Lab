"""
title: Coin change — fewest coins, and number of ways
pattern: Dynamic programming
level: medium
time: 25
frequency: very high
week: 11
day: 4
tags: dp, unbounded-knapsack
complexity: O(amount · len(coins)) time, O(amount) space
hint: fewest[x] = 1 + min over coins c ≤ x of fewest[x − c]; fewest[0] = 0; unreachable = ∞.
hint: Counting combinations (order doesn't matter): loop over coins in the OUTER loop so each multiset is counted once.
follow-up: Swap the loops in ways() — what do you count now? (ordered sequences / compositions)
follow-up: Why does greedy (largest coin first) fail for coins [1, 3, 4] and amount 6?

* `fewest_coins(coins, amount)` — minimum number of coins summing to `amount` (unlimited supply of each
  denomination), or `-1` if impossible; `amount == 0` → 0.
* `ways(coins, amount)` — number of **combinations** (multisets) of coins that sum to `amount`.

Coins are positive and distinct.

```
fewest_coins([1, 2, 5], 11) -> 3      # 5 + 5 + 1
fewest_coins([2], 3)        -> -1
ways([1, 2, 5], 5)          -> 4      # 5, 2+2+1, 2+1+1+1, 1×5
```
"""


def fewest_coins(coins: list[int], amount: int) -> int:
    INF = amount + 1
    best = [0] + [INF] * amount
    for x in range(1, amount + 1):
        for c in coins:
            if c <= x and best[x - c] + 1 < best[x]:
                best[x] = best[x - c] + 1
    return best[amount] if best[amount] <= amount else -1


def ways(coins: list[int], amount: int) -> int:
    count = [1] + [0] * amount
    for c in coins:
        for x in range(c, amount + 1):
            count[x] += count[x - c]
    return count[amount]
