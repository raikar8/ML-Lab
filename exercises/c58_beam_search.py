"""
title: Beam search decoding with length normalisation and EOS handling
pattern: ML from scratch
level: medium
time: 35
frequency: high
week: 14
day: 5
tags: decoding, search, asr, seq2seq
complexity: O(T · B · V log(B·V)) for T steps, beam B, vocab V (O(T·B·V) with a partial top-k)
hint: Keep B hypotheses (tokens, total log-prob). Each step, expand every live hypothesis by every token, keep the best B.
hint: A hypothesis that emits EOS is finished — move it to a finished list and stop expanding it; rank by log-prob / len^α.
follow-up: Why does pure log-prob ranking prefer short outputs, and what does α = 1 vs 0 do?
follow-up: CTC prefix beam search: what's different when blanks and repeats merge paths?

`step_fn(prefix) -> list[float]` returns the **log-probabilities** of the next token given a tuple `prefix` of
token ids (length `V`, token `eos` included). Implement
`beam_search(step_fn, beam, max_len, eos, alpha=0.0)` returning a list of `(tokens, score)` for the finished
hypotheses, best first, at most `beam` of them:

* start from the empty prefix; at each step expand every live hypothesis by every token and keep the top `beam`
  candidates by total log-prob (ties → lexicographically smaller token tuple);
* candidates ending in `eos` become **finished** (the `eos` is included in `tokens`) and are not expanded;
* stop when there are no live hypotheses or after `max_len` tokens; live hypotheses still unfinished at
  `max_len` count as finished too;
* score = total log-prob / len(tokens) ** alpha (alpha = 0 → plain total).

With `beam=1` it must equal greedy decoding.
"""

import math

def beam_search(step_fn, beam: int, max_len: int, eos: int, alpha: float = 0.0):
    raise NotImplementedError  # TODO: implement
