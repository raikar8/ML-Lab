## The method

Interviewers grade the process as much as the code. Say each step out loud.

1. **Clarify (2 min).** Restate the problem. Ask about input size, value ranges, duplicates, empty input, sortedness,
   and what to return on "no answer". Write two small examples, including one edge case.
2. **Brute force first (1–2 min).** Say the obvious solution and its complexity. It proves you understand the
   problem and gives you a fallback.
3. **Find the bottleneck and the pattern (3–5 min).** What work is repeated? Which pattern below removes it?
   State the target complexity *before* coding.
4. **Code it (10–20 min).** Use clear names and small helpers. Handle edge cases at the top. Talk while you type,
   but don't narrate syntax.
5. **Test by hand (3–5 min).** Run your example through the code line by line, then an edge case (empty, one
   element, all equal, negative). Fix bugs calmly; finding your own bug is a good signal.
6. **Complexity and follow-ups (2 min).** Give time and space, then say what you'd change if *n* were 100× larger,
   the data were a stream, or memory were tight.

Time budget for a 45-minute round: about 5 min understanding, 25 min coding, 10 min testing and follow-ups.
If you're stuck for more than about 3 minutes, say what you're considering and ask whether the direction is
reasonable. That's collaboration, not failure.

## Arrays & hashing

**Signals:** "find a pair/group with property X", "have we seen this before?", "count occurrences", "group by".

**Idea:** trade memory for time with a dict or set. Turn O(n²) pair checks into one pass that asks "have I already
seen the complement?". Use **prefix sums** for "sum of a range / subarray".

```python
seen = {}
for i, x in enumerate(nums):
    if target - x in seen:          # complement lookup is O(1)
        return seen[target - x], i
    seen.setdefault(x, i)

# subarrays with sum k: P[j] - P[i] == k
count, prefix, freq = 0, 0, {0: 1}
for x in nums:
    prefix += x
    count += freq.get(prefix - k, 0)
    freq[prefix] = freq.get(prefix, 0) + 1
```

**Pitfalls:** pairing an element with itself; insertion order vs first index; lists can't be dict keys (use tuples
or a canonical string); a sliding window doesn't work for subarray sums once negatives appear, so use prefix sums.

## Two pointers

**Signals:** sorted array (or you can sort), "pair/triplet with sum", "in place", "palindrome", "partition".

**Idea:** two indices that move monotonically, so the total work is O(n). When the array is sorted, moving the left
pointer increases the sum and moving the right pointer decreases it.

```python
lo, hi = 0, len(a) - 1
while lo < hi:
    s = a[lo] + a[hi]
    if s == target: return lo, hi
    if s < target: lo += 1
    else: hi -= 1
```

**Pitfalls:** skipping duplicates (3Sum); `<` vs `<=` in the loop condition; forgetting that sorting costs
O(n log n) and destroys the original indices.

## Sliding window

**Signals:** "longest/shortest **contiguous** substring/subarray such that…", "at most k distinct", "window of size k".

**Idea:** grow the right edge; while the window is invalid, shrink the left edge. Each index enters and leaves once,
so the whole scan is O(n).

```python
left = 0
for right, ch in enumerate(s):
    add(ch)
    while not valid():
        remove(s[left]); left += 1
    best = max(best, right - left + 1)
```

For *fixed* windows that need a max or min, use a **monotonic deque** to get O(1) per step.

**Pitfalls:** moving `left` backwards (use `max(left, last[ch] + 1)`); recording the answer at the wrong moment
(for a minimum window, record while the window is valid, before shrinking further).

## Stack

**Signals:** matching brackets, "next greater/smaller element", "previous …", histogram areas, evaluating
expressions, undo.

**Idea:** a stack holds the things still waiting for an answer. In a **monotonic stack** (increasing or
decreasing), every index is pushed and popped once, so the total is O(n).

```python
stack = []                      # indices with decreasing values
for i, x in enumerate(a):
    while stack and a[stack[-1]] < x:
        j = stack.pop()
        answer[j] = i           # i is j's next greater element
    stack.append(i)
```

**Pitfalls:** strict vs non-strict comparison when values are equal; forgetting the end-of-input sentinel (append a
0 for histogram problems).

## Binary search

**Signals:** sorted data, "minimum X such that…", "maximum feasible…", O(log n) required, monotone yes/no
questions.

**Idea:** keep an invariant over a half-open window `[lo, hi)`. For "search on the answer", binary-search the value
and check feasibility greedily.

```python
lo, hi = 0, len(a)            # first index with a[i] >= x
while lo < hi:
    mid = (lo + hi) // 2
    if a[mid] < x: lo = mid + 1
    else: hi = mid
return lo
```

**Pitfalls:** infinite loops (`lo = mid` with `mid = (lo + hi) // 2`); mixing closed and half-open windows;
overflow in C/C++ (use `lo + (hi - lo) / 2`); not proving the predicate is monotone.

## Linked list

**Signals:** pointer surgery, "in place", cycles, "k-th from the end", merging.

**Idea:** a **dummy head** removes special cases at the front. **Fast/slow pointers** find middles and cycles.
Reverse with `prev, cur`.

```python
prev, cur = None, head
while cur:
    cur.next, prev, cur = prev, cur, cur.next
return prev
```

**Pitfalls:** losing the rest of the list (save `next` first); off-by-one in "k-th from the end"; Python's
multiple assignment evaluates the right-hand side first, which is what makes the one-liner above correct.

## Trees

**Signals:** hierarchies, BSTs, "depth/height/diameter", "level by level", "lowest common ancestor".

**Idea:** most tree problems are a DFS that returns something from each subtree (height, a (take, skip) pair,
validity plus range). Level-by-level means BFS with a queue, processing `len(queue)` nodes per round.

```python
def dfs(node):
    if node is None: return BASE
    left, right = dfs(node.left), dfs(node.right)
    update_global(left, right, node)
    return combine(left, right, node)
```

**Pitfalls:** checking a BST only against the parent (pass `(lo, hi)` bounds down instead); recursion depth on
degenerate trees (Python's default limit is about 1000, so use an explicit stack); confusing height with depth.

## Trie

**Signals:** many strings sharing prefixes, autocomplete, "starts with", word search with a dictionary.

**Idea:** one node per character, with child links, plus counters (words passing through, words ending here).
Insert and lookup are O(word length).

**Pitfalls:** memory blow-up with large alphabets (use dict children); forgetting the "end of word" marker; not
pruning branches after deletion.

## Heap / priority queue

**Signals:** "k largest/smallest/closest", "merge k sorted", "running median", scheduling by the earliest end time.

**Idea:** `heapq` is a min-heap. Keep the k best in a heap of size k, giving O(n log k). For a max-heap, push
negated keys. Two heaps (a max-heap for the low half, a min-heap for the high half) give a running median.

```python
heap = []
for x in stream:
    if len(heap) < k: heapq.heappush(heap, x)
    elif x > heap[0]: heapq.heapreplace(heap, x)
# heap[0] is the k-th largest
```

**Pitfalls:** comparing payloads on ties (push `(key, tie_breaker, payload)`); `heapq` has no decrease-key, so push
duplicates and skip stale entries (as in Dijkstra).

## Intervals

**Signals:** meetings, bookings, ranges, "overlap", "merge", "minimum rooms", concurrent requests.

**Idea:** sort by start and merge greedily. Or use a **sweep line**: turn intervals into +1/−1 events and sort them,
with ends before starts when intervals are half-open.

**Pitfalls:** closed vs half-open intervals (does `[1, 3]` overlap `[3, 5]`?); sorting by end when you meant start.

## Graphs

**Signals:** grids, dependencies, networks, "shortest", "connected", "order", "reachable".

**Idea:**

* **BFS** for unweighted shortest paths. Multi-source BFS puts all sources in the queue at distance 0.
* **DFS or iterative stack** for reachability and flood fill.
* **Topological sort (Kahn)** for dependency order and cycle detection in directed graphs.
* **Dijkstra** for non-negative weights; **Bellman–Ford** if weights can be negative.
* **Union–find** for dynamic connectivity, Kruskal, and "redundant edge".

```python
dist = {src: 0}; q = deque([src])
while q:
    u = q.popleft()
    for v in adj[u]:
        if v not in dist:
            dist[v] = dist[u] + 1; q.append(v)
```

**Pitfalls:** marking visited when *popping* instead of when *pushing* (duplicates explode the queue); recursion
depth in DFS; directed vs undirected cycle detection.

## Backtracking

**Signals:** "all subsets/permutations/combinations", "place N things", puzzles, "generate every valid …".

**Idea:** choose, recurse, un-choose. Prune as early as possible (sorted input lets you `break`). Dedupe by skipping
equal values at the same depth.

```python
def go(start):
    out.append(cur[:])
    for i in range(start, len(a)):
        if i > start and a[i] == a[i - 1]: continue   # skip duplicate branch
        cur.append(a[i]); go(i + 1); cur.pop()
```

**Pitfalls:** appending `cur` instead of a copy; exponential blow-up without pruning; forgetting to undo state.

## Dynamic programming

**Signals:** "number of ways", "minimum/maximum cost", "can you…", optimal substructure, overlapping subproblems
(the brute-force recursion repeats calls).

**Recipe:** (1) define the state in words ("best[i] = best total for the first i houses"); (2) write the
transition; (3) set the base cases; (4) choose the iteration order so dependencies are ready; (5) reduce memory if
only the last row or two are needed.

| Shape | Example | Transition |
|---|---|---|
| 1-D | house robber, climbing stairs | `f[i] = max(f[i−1], f[i−2] + v[i])` |
| Knapsack | coin change, 0/1 knapsack | outer loop over items; capacity up (unbounded) or down (0/1) |
| Two sequences | LCS, edit distance | `D[i][j]` from `D[i−1][j−1], D[i−1][j], D[i][j−1]` |
| Grid | unique paths, min path sum | `f[r][c]` from up and left |
| LIS | longest increasing subsequence | patience sorting + binary search, O(n log n) |

**Pitfalls:** loop order in knapsack (combinations vs permutations); off-by-one in table sizes (use `n+1` with an
empty-prefix row); memoising on mutable arguments.

## Greedy

**Signals:** "minimum number of …", scheduling, reachability, and you can argue that a locally best choice never
hurts.

**Idea:** sort or scan, and make the choice that keeps the most options open (earliest end time, farthest reach).
**Always justify it**, usually with an exchange argument; otherwise say "greedy might fail, so here's the DP".

**Pitfalls:** greedy coin change with arbitrary denominations (it fails); not proving the choice.

## Bit manipulation & math

**Signals:** "without extra space", XOR pairs, powers of two, subsets as masks, embedded and DSP code.

**Toolbox:** `x & (x − 1)` clears the lowest set bit; `x & −x` isolates it; `a ^ a = 0`; masks for subsets
(`for mask in range(1 << n)`); shifts for ×/÷ by powers of two; `(a + b − 1) // b` for ceiling division.

**Pitfalls:** Python ints are unbounded (mask with `& 0xFFFFFFFF` to emulate 32-bit); operator precedence (`&`
binds looser than `==` in C, so add parentheses).

## Matrix

**Signals:** images, grids, rotate, spiral, "in place".

**Idea:** rotate 90° = transpose + reverse rows. For spiral traversal, shrink four boundaries. For in-place marking,
reuse the first row and column as flags.

**Pitfalls:** non-square matrices; boundaries crossing in the middle of a spiral; aliasing (`[[0] * C] * R` shares
rows!).

## Design

**Signals:** "implement a class that supports these operations in O(1)/O(log n)": LRU/LFU cache, rate limiter,
min-stack, time-versioned store, iterator.

**Idea:** combine structures so that each operation is cheap: a dict plus a doubly-linked list (LRU), dicts of
ordered buckets (LFU), a deque log (rate limiter), parallel sorted lists plus bisect (versioned store).

**Interview tip:** list the operations and the complexity each needs *first*, then choose structures. Mention
thread safety, eviction policy, and memory bounds as follow-ups.

## Randomised

**Signals:** sampling, shuffling, reservoir, "pick with probability proportional to…", quickselect pivots, hashing.

**Toolbox:** Fisher–Yates shuffle; reservoir sampling (keep item i with probability k/i); prefix sums plus binary
search for weighted picks; Efraimidis–Spirakis keys `u^(1/w)` for weighted sampling without replacement; random
pivots to avoid adversarial O(n²).

**Pitfalls:** off-by-one in `randrange` (a biased shuffle); calling `random()` more often than needed; not seeding
when tests need determinism.

## ML from scratch

**Signals:** "implement X in NumPy/PyTorch without the library function", e.g. attention, softmax, k-means, logistic
regression, metrics, NMS, sampling, beam search, a training loop.

**What interviewers check:** shapes stated before coding; numerical stability (subtract the max before `exp`,
work in log space); vectorisation (no Python loops over samples); train vs eval behaviour (dropout, BatchNorm);
correct gradients (do a numerical check); and complexity in terms of n, d, k and sequence length.

**Pitfalls:** broadcasting silently doing the wrong thing (assert shapes); in-place ops breaking autograd; forgetting
`model.eval()` / `torch.no_grad()`; division by zero in metrics (empty classes); ties in ranking metrics.
