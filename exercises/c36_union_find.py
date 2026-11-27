"""
title: Union–find (disjoint set union) — components, redundant edge, and accounts merge
pattern: Graphs
level: medium
time: 30
frequency: high
week: 18
day: 4
tags: union-find, graph, path-compression
complexity: O(α(n)) amortised per operation with path compression + union by size
hint: find(x) follows parents to the root and compresses the path; union(a, b) attaches the smaller tree under the larger.
hint: An edge whose endpoints already share a root closes a cycle — that's the redundant edge.
follow-up: Kruskal's minimum spanning tree = sort edges + union–find.
follow-up: Number of islands with land added online — each addition is a union with up to 4 neighbours.

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
