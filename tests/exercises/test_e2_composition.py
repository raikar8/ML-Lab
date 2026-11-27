import inspect

import numpy as np

from helpers import key_of, load

m = load(key_of(__file__))
rng = np.random.default_rng(0)


def test_protocol_without_inheritance():
    objs = [m.Gain(0), m.DCBlock(), m.Clip(1.0), m.Pipeline([])]
    assert all(isinstance(o, m.Stage) for o in objs)
    for cls in (m.Gain, m.DCBlock, m.Clip, m.Pipeline):
        assert cls.__bases__ == (object,), cls


def test_gain_and_clip():
    assert np.allclose(m.Gain(20).process([1.0, -2.0]), [10.0, -20.0])
    c = m.Clip(0.5)
    assert np.allclose(c.process([0.2, 0.7, -0.9]), [0.2, 0.5, -0.5]) and c.clipped == 2
    c.process([1.0]); assert c.clipped == 3
    c.reset(); assert c.clipped == 0


def test_dcblock_streaming_equals_batch_and_removes_dc():
    x = rng.normal(size=4000) + 3.0
    whole = m.DCBlock(0.99).process(x)
    d = m.DCBlock(0.99)
    cuts = np.sort(rng.choice(np.arange(1, 4000), size=15, replace=False))
    streamed = np.concatenate([d.process(f) for f in np.split(x, cuts)])
    assert np.allclose(streamed, whole)
    assert abs(whole[2000:].mean()) < 0.1          # DC removed
    d.reset()
    assert np.allclose(d.process(x[:10]), whole[:10])


def test_pipeline_composite_and_immutable_then():
    p = m.Pipeline([m.DCBlock(0.9), m.Gain(6)])
    q = p.then(m.Clip(0.25))
    assert len(p.stages) == 2 and len(q.stages) == 3
    x = rng.normal(size=1000)
    out1 = q.process(x)
    q.reset()
    out2 = np.concatenate([q.process(f) for f in np.split(x, 10)])
    assert np.allclose(out1, out2) and np.abs(out1).max() <= 0.25
    nested = m.Pipeline([m.Pipeline([m.Gain(6)]), m.Gain(-6)])
    assert np.allclose(nested.process(x), x)
