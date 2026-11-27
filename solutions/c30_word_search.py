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
    if not word:
        return []
    R = len(board)
    C = len(board[0]) if R else 0
    have = Counter(ch for row in board for ch in row)
    if any(have[ch] < n for ch, n in Counter(word).items()):
        return None
    # search from the rarer end first (big win on adversarial inputs)
    rev = have[word[0]] > have[word[-1]]
    w = word[::-1] if rev else word
    used = [[False] * C for _ in range(R)]
    path: list[tuple[int, int]] = []

    def dfs(r, c, i):
        if board[r][c] != w[i]:
            return False
        used[r][c] = True
        path.append((r, c))
        if i == len(w) - 1:
            return True
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < R and 0 <= nc < C and not used[nr][nc] and dfs(nr, nc, i + 1):
                return True
        used[r][c] = False
        path.pop()
        return False

    for r in range(R):
        for c in range(C):
            if dfs(r, c, 0):
                return path[::-1] if rev else path
    return None


def exists(board: list[str], word: str) -> bool:
    return find_path(board, word) is not None
