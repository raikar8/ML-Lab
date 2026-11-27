"""
Word search in a letter grid (backtracking with pruning)

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
