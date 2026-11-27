"""
title: Word search in a letter grid (backtracking with pruning)
pattern: Backtracking
level: medium
time: 30
frequency: very high
week: 15
day: 4
tags: backtracking, dfs, grid
complexity: O(R·C·4·3^(L−1)) worst case for a word of length L; O(L) recursion depth
hint: DFS from every cell matching word[0]; mark a cell as used on the way down and unmark it on the way back.
hint: Prune early: if the grid lacks enough copies of some letter, return False before searching.
follow-up: Many words at once (Word Search II): build a trie and walk it during the DFS; delete found leaves.
follow-up: Count all paths that spell the word (memoise on (cell, index) only if cells may repeat).

`board` is a list of equal-length strings. A word is found if it can be traced through **4-directionally**
adjacent cells, using each cell at most once. Implement `exists(board, word) -> bool` and
`find_path(board, word)` → list of `(row, col)` cells for one valid tracing, or `None`.
The empty word exists (path `[]`). Don't modify `board`.

```
board = ["ABCE",
         "SFCS",
         "ADEE"]
exists(board, "ABCCED") -> True;  exists(board, "ABCB") -> False
```
"""

from collections import Counter

def find_path(board: list[str], word: str):
    raise NotImplementedError  # TODO: implement


def exists(board: list[str], word: str) -> bool:
    raise NotImplementedError  # TODO: implement
