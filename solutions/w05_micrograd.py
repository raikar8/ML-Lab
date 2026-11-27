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
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = Value.wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def bw():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = bw
        return out

    def __mul__(self, other):
        other = Value.wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def bw():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = bw
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def bw():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def bw():
            self.grad += (out.data > 0) * out.grad
        out._backward = bw
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def bw():
            self.grad += (1 - t * t) * out.grad
        out._backward = bw
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def bw():
            self.grad += out.data * out.grad
        out._backward = bw
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def bw():
            self.grad += out.grad / self.data
        out._backward = bw
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Value.wrap(other))

    def __truediv__(self, other):
        return self * Value.wrap(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value.wrap(other) - self

    def __rtruediv__(self, other):
        return Value.wrap(other) / self

    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
