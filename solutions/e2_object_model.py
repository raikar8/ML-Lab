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
        self.lo, self.hi = lo, hi
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return obj.__dict__[self.name]
        except KeyError:
            raise AttributeError(self.name) from None

    def __set__(self, obj, value):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{self.name} must be a number, got {type(value).__name__}")
        if not self.lo <= value <= self.hi:
            raise ValueError(f"{self.name}={value} outside [{self.lo}, {self.hi}]")
        obj.__dict__[self.name] = value


class lazy:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.__doc__ = fn.__doc__

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.fn(obj)
        obj.__dict__[self.name] = value          # instance dict now shadows this non-data descriptor
        return value


class Vec2:
    __slots__ = ("_x", "_y")

    def __init__(self, x, y):
        object.__setattr__(self, "_x", float(x))
        object.__setattr__(self, "_y", float(y))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __setattr__(self, name, value):
        raise AttributeError("Vec2 is immutable")

    def __repr__(self):
        return f"Vec2({self._x!r}, {self._y!r})"

    def __eq__(self, other):
        if not isinstance(other, Vec2):
            return NotImplemented
        return (self._x, self._y) == (other._x, other._y)

    def __hash__(self):
        return hash((self._x, self._y))

    def __iter__(self):
        yield self._x
        yield self._y

    def __add__(self, other):
        if not isinstance(other, Vec2):
            return NotImplemented
        return Vec2(self._x + other._x, self._y + other._y)

    def __sub__(self, other):
        if not isinstance(other, Vec2):
            return NotImplemented
        return Vec2(self._x - other._x, self._y - other._y)

    def __mul__(self, k):
        if isinstance(k, bool) or not isinstance(k, Real):
            return NotImplemented
        return Vec2(self._x * k, self._y * k)

    __rmul__ = __mul__

    def __abs__(self):
        return math.hypot(self._x, self._y)
