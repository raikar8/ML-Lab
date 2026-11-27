import torch
from helpers import key_of, load

m = load(key_of(__file__))


def test_collate_and_masked_mean():
    batch = [(torch.ones(3, 2), 1), (torch.full((5, 2), 2.0), 0)]
    out = m.pad_collate(batch)
    assert out["x"].shape == (2, 5, 2) and out["lengths"].tolist() == [3, 5] and out["y"].tolist() == [1, 0]
    assert out["mask"].tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    torch.testing.assert_close(m.masked_mean(out["x"], out["mask"]), torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_bucket_sampler():
    g = torch.Generator().manual_seed(0)
    lengths = torch.randint(10, 1000, (997,), generator=g).tolist()
    s = m.BucketBatchSampler(lengths, 16, seed=1)
    b = list(s)
    flat = sorted(i for bb in b for i in bb)
    assert flat == list(range(997)) and len(s) == len(b)
    spread = sum(max(lengths[i] for i in bb) - min(lengths[i] for i in bb) for bb in b) / len(b)
    assert spread < 100  # random batches would be ~900
    s.set_epoch(1)
    assert list(s) != b
    ds = [(torch.zeros(l // 100 + 1, 3), 0) for l in lengths]
    dl = torch.utils.data.DataLoader(ds, batch_sampler=s, collate_fn=m.pad_collate)
    assert next(iter(dl))["x"].ndim == 3
