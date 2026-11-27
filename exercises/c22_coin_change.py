"""
Coin change — fewest coins, and number of ways

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
    raise NotImplementedError  # TODO: implement


def ways(coins: list[int], amount: int) -> int:
    raise NotImplementedError  # TODO: implement
