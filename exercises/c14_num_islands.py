"""
Number of islands — and the size of the largest one

`grid` is a list of strings of `'1'` (land) and `'0'` (water). Cells connect **4-directionally**.

* `num_islands(grid)` — number of islands;
* `largest_island(grid)` — number of cells in the largest island (0 if none);
* `island_sizes(grid)` — the size of every island, in row-major order of each island's first cell (both functions above can use it).

Don't modify the input. Must handle a 400×400 grid of all land.

```
grid = ["11000",
        "11000",
        "00100",
        "00011"]
num_islands(grid) -> 3;  largest_island(grid) -> 4
```
"""

def island_sizes(grid: list[str]) -> list[int]:
    """Sizes of all islands, in the order their top-left-most cell is found (row-major)."""
    raise NotImplementedError  # TODO: implement


def num_islands(grid: list[str]) -> int:
    raise NotImplementedError  # TODO: implement


def largest_island(grid: list[str]) -> int:
    raise NotImplementedError  # TODO: implement
