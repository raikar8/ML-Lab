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
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None] - inter)


def nms(boxes, scores, iou_thr):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)
