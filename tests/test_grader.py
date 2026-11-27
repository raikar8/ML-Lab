import json

from mlprep import errorlog, grader


def test_offline_grader_coverage():
    ref = "Reverse mode computes a vector-Jacobian product; one backward pass gives gradients for all parameters."
    good = grader.offline_grade("Why reverse mode?", ref, "Because reverse mode gives a vector-Jacobian product and one backward pass gives gradients of all parameters")
    bad = grader.offline_grade("Why reverse mode?", ref, "Because it is faster I think, probably")
    assert good["score"] > bad["score"] and good["missed"] != bad["missed"]


def test_backends_parse_with_fake_transport(monkeypatch):
    reply = {"score": 2, "covered": ["a"], "missed": ["b"], "errors": [], "vague": [], "follow_up": "why?"}
    def fake(url, payload, headers):
        assert "messages" in payload
        text = "Here you go:\n" + json.dumps(reply)
        return {"content": [{"text": text}]} if "anthropic" in url else {"choices": [{"message": {"content": text}}]}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    r1 = grader.grade("q", "ref", "ans", "claude", transport=fake)
    r2 = grader.grade("q", "ref", "ans", "local", base_url="http://x/v1", transport=fake)
    assert r1["score"] == r2["score"] == 2 and r1["follow_up"] == "why?"


def test_errorlog_cards():
    d = {}
    errorlog.add(d, "Bug I made", "forgot zero_grad", "call opt.zero_grad() each step", 9, ["pytorch"], True)
    errorlog.add(d, "Concept gap", "not in review", "x", None, [], False)
    qs = errorlog.as_questions(d)
    assert len(qs) == 1 and qs[0].qid == "err-1" and "zero_grad" in qs[0].text
