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
        self.parent = list(range(n))
        self.sz = [1] * n
        self.components = n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.sz[ra] += self.sz[rb]
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def size(self, x: int) -> int:
        return self.sz[self.find(x)]


def redundant_edge(n: int, edges: list[tuple[int, int]]) -> tuple[int, int]:
    d = DSU(n)
    last = None
    for u, v in edges:
        if not d.union(u, v):
            last = (u, v)
    return last


def merge_accounts(accounts: list[list[str]]) -> list[list[str]]:
    d = DSU(len(accounts))
    owner: dict[str, int] = {}
    for i, acc in enumerate(accounts):
        for e in acc[1:]:
            if e in owner:
                d.union(i, owner[e])
            else:
                owner[e] = i
    groups: dict[int, set[str]] = {}
    for e, i in owner.items():
        groups.setdefault(d.find(i), set()).add(e)
    out = [[accounts[root][0], *sorted(es)] for root, es in groups.items()]
    for i, acc in enumerate(accounts):            # accounts with no email still exist
        if len(acc) == 1:
            out.append([acc[0]])
    return sorted(out, key=lambda a: (a[0], a[1] if len(a) > 1 else ""))
