"""
title: Hybrid retrieval for on-device RAG — hashing embedder, BM25, rank fusion, recall@k
week: 20
day: 5
level: medium
time: 45
tags: python, rag, retrieval, edge
core: yes

All NumPy / standard library, no models.

* `tokenize(text)` → lowercase alphanumeric tokens (`re.findall(r"[a-z0-9]+", text.lower())`).
* `hash_embed(text, dim=256)` — bag of words + character 3-grams of each token, hashed into `dim` buckets with
  `zlib.crc32(gram.encode()) % dim`, L2-normalised (zero vector stays zero).
* `BM25(docs, k1=1.5, b=0.75)` with `scores(query)` → array of BM25 scores, IDF = `log(1 + (N − df + 0.5)/(df + 0.5))`.
* `rrf(rankings, k=60)` — reciprocal rank fusion of several ranked lists of doc ids → fused ranked list.
* `retrieve(query, docs, top_k, emb=None, bm25=None)` → top_k doc ids from RRF of dense (cosine) and BM25 rankings.
* `recall_at_k(results, relevant)` — fraction of queries whose relevant id is in their result list.
"""
import math
import re
import zlib
from collections import Counter

import numpy as np


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def hash_embed(text, dim=256):
    v = np.zeros(dim)
    for tok in tokenize(text):
        grams = [tok] + [tok[i:i + 3] for i in range(max(1, len(tok) - 2))]
        for g in grams:
            v[zlib.crc32(g.encode()) % dim] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avg = sum(map(len, self.toks)) / max(1, self.N)
        self.df = Counter(t for ts in self.toks for t in set(ts))
        self.tf = [Counter(ts) for ts in self.toks]

    def scores(self, query):
        out = np.zeros(self.N)
        for t in set(tokenize(query)):
            if t not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.toks[i]) / self.avg))
        return out


def rrf(rankings, k=60):
    score = Counter()
    for r in rankings:
        for rank, d in enumerate(r):
            score[d] += 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def retrieve(query, docs, top_k, emb=None, bm25=None):
    emb = emb if emb is not None else np.stack([hash_embed(d) for d in docs])
    bm25 = bm25 or BM25(docs)
    dense = list(np.argsort(-(emb @ hash_embed(query)), kind="stable"))
    sparse = list(np.argsort(-bm25.scores(query), kind="stable"))
    return [int(i) for i in rrf([dense, sparse])[:top_k]]


def recall_at_k(results, relevant):
    return sum(r in res for res, r in zip(results, relevant)) / len(relevant)
