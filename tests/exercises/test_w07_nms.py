import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_iou():
    a = torch.tensor([[0.0, 0, 2, 2]])
    b = torch.tensor([[1.0, 1, 3, 3], [2.0, 2, 3, 3], [0.0, 0, 2, 2]])
    torch.testing.assert_close(m.box_iou(a, b), torch.tensor([[1 / 7, 0.0, 1.0]]))


def test_nms():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [21, 21, 29, 29], [0, 0, 9, 10]], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.95, 0.6])
    assert m.nms(boxes, scores, 0.5).tolist() == [3, 0]
    assert m.nms(boxes, scores, 0.99).tolist() == [3, 0, 1, 2, 4]
