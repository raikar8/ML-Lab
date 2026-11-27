"""
title: Multinomial naive Bayes for text, in log space
pattern: ML from scratch
level: medium
time: 35
frequency: high
week: 3
day: 5
tags: numpy, probability, classic, nlp
complexity: O(nnz) training, O(V·C) per document scoring
hint: Score(c | doc) = log P(c) + Σ_w count(w) · log P(w | c). Never multiply raw probabilities.
hint: Laplace smoothing: P(w | c) = (count(w, c) + α) / (total(c) + α·V). Ignore words not in the vocabulary.
follow-up: Why does naive Bayes give badly calibrated probabilities even when its accuracy is good?
follow-up: Bernoulli vs multinomial NB — when would you pick each? What changes in the formula?

Implement `MultinomialNB(alpha=1.0)` working on **lists of token lists**:

* `fit(docs, labels)` — build the vocabulary from training docs, learn log priors and smoothed log likelihoods;
  returns `self`.
* `predict_log_proba(docs)` — `(n_docs, n_classes)` NumPy array of **normalised** log posteriors
  (rows log-sum-exp to 0); classes in sorted order, exposed as `self.classes_`.
* `predict(docs)` — list of predicted labels (ties → the first class in `classes_`).

Out-of-vocabulary tokens are ignored. Must stay finite for long documents (thousands of tokens).
"""

import numpy as np

class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        raise NotImplementedError  # TODO: implement

    def fit(self, docs, labels):
        raise NotImplementedError  # TODO: implement

    def _counts(self, docs):
        X = np.zeros((len(docs), len(self.vocab_)))
        for i, d in enumerate(docs):
            for w in d:
                j = self.vocab_.get(w)
                if j is not None:
                    X[i, j] += 1
        return X

    def predict_log_proba(self, docs):
        raise NotImplementedError  # TODO: implement

    def predict(self, docs):
        raise NotImplementedError  # TODO: implement
