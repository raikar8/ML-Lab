"""
Union–find (disjoint set union) — components, redundant edge, and accounts merge

* `DSU(n)` with `find(x)`, `union(a, b) -> bool` (False if already connected), `connected(a, b)`,
  `size(x)` (size of x's component) and `components` (number of components, an attribute or property).
  Use path compression **and** union by size (it must handle 200 000 elements in a chain quickly).
* `redundant_edge(n, edges)` — in an undirected graph that is a tree plus one extra edge, return the **last**
  edge in the input whose removal leaves a tree.
* `merge_accounts(accounts)` — each account is `[name, email1, email2, ...]`; accounts sharing any email belong to
  the same person. Return merged accounts `[name, *sorted unique emails]`, sorted by `(name, first email)`.
"""

class DSU:
    def __init__(self, n: int):
        raise NotImplementedError  # TODO: implement

    def find(self, x: int) -> int:
        raise NotImplementedError  # TODO: implement

    def union(self, a: int, b: int) -> bool:
        raise NotImplementedError  # TODO: implement

    def connected(self, a: int, b: int) -> bool:
        raise NotImplementedError  # TODO: implement

    def size(self, x: int) -> int:
        raise NotImplementedError  # TODO: implement


def redundant_edge(n: int, edges: list[tuple[int, int]]) -> tuple[int, int]:
    raise NotImplementedError  # TODO: implement


def merge_accounts(accounts: list[list[str]]) -> list[list[str]]:
    raise NotImplementedError  # TODO: implement
