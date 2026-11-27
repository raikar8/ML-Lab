"""
title: micrograd — a scalar reverse-mode autodiff engine
week: 5
day: 1
level: medium
time: 45
tags: python, autodiff, classic
core: yes

Implement `Value`, a scalar that records the operations producing it, and `backward()` that fills `.grad` for every
node by reverse-mode autodiff.

Support: `+`, `-`, `*`, `/`, unary `-`, `** k` (k a Python number), `relu()`, `tanh()`, `exp()`, `log()`, and
operations with plain Python numbers on either side (`2 * v`, `v + 1`, `1 - v`). `backward()` must visit nodes in
reverse **topological order** and **accumulate** gradients (`+=`) so values used several times get the sum.
"""

import math

class Value:
    def __init__(self, data, children=(), op=""):
        raise NotImplementedError  # TODO: implement

    def __repr__(self):
        raise NotImplementedError  # TODO: implement

    @staticmethod
    def wrap(x):
        raise NotImplementedError  # TODO: implement

    def __add__(self, other):
        raise NotImplementedError  # TODO: implement

    def __mul__(self, other):
        raise NotImplementedError  # TODO: implement

    def __pow__(self, k):
        raise NotImplementedError  # TODO: implement

    def relu(self):
        raise NotImplementedError  # TODO: implement

    def tanh(self):
        raise NotImplementedError  # TODO: implement

    def exp(self):
        raise NotImplementedError  # TODO: implement

    def log(self):
        raise NotImplementedError  # TODO: implement

    def __neg__(self):
        raise NotImplementedError  # TODO: implement

    def __sub__(self, other):
        raise NotImplementedError  # TODO: implement

    def __truediv__(self, other):
        raise NotImplementedError  # TODO: implement

    def __radd__(self, other):
        raise NotImplementedError  # TODO: implement

    def __rmul__(self, other):
        raise NotImplementedError  # TODO: implement

    def __rsub__(self, other):
        raise NotImplementedError  # TODO: implement

    def __rtruediv__(self, other):
        raise NotImplementedError  # TODO: implement

    def backward(self):
        raise NotImplementedError  # TODO: implement
