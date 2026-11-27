"""
title: Python's object model — a validating descriptor, a lazy attribute, a proper value class
week: 0
day: 2
level: medium
time: 40
tags: python, descriptors, dunder, object-model, elective
core: yes

* `Bounded(lo, hi)` — a **data descriptor** for numeric attributes: `__set_name__` remembers the attribute name;
  `__set__` raises `TypeError` for non-numbers (bools rejected) and `ValueError` outside `[lo, hi]`; values live in
  the instance `__dict__` under the same name; reading an unset attribute raises `AttributeError`; accessing it on
  the class returns the descriptor itself.
* `lazy` — a **non-data descriptor** decorator like `functools.cached_property` (don't use it): the first access calls
  the function and stores the result in the instance `__dict__`, so later accesses never call it again and
  `del obj.attr` resets it.
* `Vec2(x, y)` — an immutable 2-D vector: `__repr__` → `Vec2(1.0, 2.0)`; `__eq__` and `__hash__` consistent; `+` and
  `-` with another `Vec2`; `*` with a number on either side (`__mul__`, `__rmul__`); `abs()` is the length; iteration
  yields x then y (so `x, y = v` works); assigning attributes raises `AttributeError`; arithmetic with unsupported types
  must return `NotImplemented` so Python raises `TypeError` itself. Use `__slots__`.
"""

import math

from numbers import Real

class Bounded:
    def __init__(self, lo, hi):
        raise NotImplementedError  # TODO: implement

    def __set_name__(self, owner, name):
        raise NotImplementedError  # TODO: implement

    def __get__(self, obj, objtype=None):
        raise NotImplementedError  # TODO: implement

    def __set__(self, obj, value):
        raise NotImplementedError  # TODO: implement


class lazy:
    def __init__(self, fn):
        raise NotImplementedError  # TODO: implement

    def __set_name__(self, owner, name):
        raise NotImplementedError  # TODO: implement

    def __get__(self, obj, objtype=None):
        raise NotImplementedError  # TODO: implement


class Vec2:
    __slots__ = ("_x", "_y")
    def __init__(self, x, y):
        raise NotImplementedError  # TODO: implement

    @property
    def x(self):
        raise NotImplementedError  # TODO: implement

    @property
    def y(self):
        raise NotImplementedError  # TODO: implement

    def __setattr__(self, name, value):
        raise NotImplementedError  # TODO: implement

    def __repr__(self):
        raise NotImplementedError  # TODO: implement

    def __eq__(self, other):
        raise NotImplementedError  # TODO: implement

    def __hash__(self):
        raise NotImplementedError  # TODO: implement

    def __iter__(self):
        raise NotImplementedError  # TODO: implement

    def __add__(self, other):
        raise NotImplementedError  # TODO: implement

    def __sub__(self, other):
        raise NotImplementedError  # TODO: implement

    def __mul__(self, k):
        raise NotImplementedError  # TODO: implement

    __rmul__ = __mul__
    def __abs__(self):
        raise NotImplementedError  # TODO: implement
