from helpers import key_of, load

m = load(key_of(__file__))
TOOLS = [
    {"name": "set_temperature", "params": {"zone": {"type": "string", "enum": ["driver", "passenger"]},
     "celsius": {"type": "number", "min": 16, "max": 28}}, "required": ["zone", "celsius"], "confirm": False},
    {"name": "send_message", "params": {"to": {"type": "string"}, "text": {"type": "string"}},
     "required": ["to", "text"], "confirm": True},
]


def test_extract():
    out = 'Sure! ```json\n{"name": "set_temperature", "arguments": {"zone": "driver", "celsius": 21}}\n``` done'
    assert m.extract_json(out)["arguments"]["celsius"] == 21
    assert m.extract_json('text {"a": "}"} more')["a"] == "}"
    assert m.extract_json("no json here") is None and m.extract_json("{bad json}") is None


def test_validate():
    ok = {"name": "set_temperature", "arguments": {"zone": "driver", "celsius": 21.5}}
    assert m.validate_call(ok, TOOLS) == []
    bad = {"name": "set_temperature", "arguments": {"zone": "rear", "celsius": True, "fan": 3}}
    errs = m.validate_call(bad, TOOLS)
    assert any("one of" in e for e in errs) and any("number" in e for e in errs) and any("unknown parameter" in e for e in errs)
    assert m.validate_call({"name": "open_sunroof", "arguments": {}}, TOOLS) == ["unknown tool 'open_sunroof'"]
    assert any("missing" in e for e in m.validate_call({"name": "send_message", "arguments": {"to": "x"}}, TOOLS))
    assert m.validate_call({"name": "set_temperature", "arguments": {"zone": "driver", "celsius": 40}}, TOOLS) == ["'celsius' must be <= 28"]
    assert m.needs_confirmation({"name": "send_message"}, TOOLS) and not m.needs_confirmation(ok, TOOLS)
    assert m.needs_confirmation({"name": "unknown"}, TOOLS)
    assert m.error_message(["a", "b"]) == "Tool call invalid: a; b"
