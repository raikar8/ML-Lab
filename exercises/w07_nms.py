"""
title: IoU and non-maximum suppression
week: 7
day: 5
level: easy
time: 20
tags: pytorch, vision, classic
core: no

Boxes are `(N, 4)` tensors `[x1, y1, x2, y2]` with `x2 > x1`, `y2 > y1`.

* `box_iou(a, b)` → `(N, M)` IoU matrix, vectorised (broadcast the intersections).
* `nms(boxes, scores, iou_thr)` → `LongTensor` of kept indices, sorted by decreasing score: repeatedly keep the
  best remaining box and discard boxes with IoU **greater than** `iou_thr` with it.
"""

import torch

def box_iou(a, b):
    raise NotImplementedError  # TODO: implement


def nms(boxes, scores, iou_thr):
    raise NotImplementedError  # TODO: implement
