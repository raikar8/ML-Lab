"""
title: Validate and repair LLM tool calls against a JSON schema subset
week: 20
day: 4
level: medium
time: 35
tags: python, agents, tool-use, safety
core: yes

Tool specs are dicts like
`{"name": "set_temperature", "params": {"zone": {"type": "string", "enum": ["driver", "passenger"]},
"celsius": {"type": "number", "min": 16, "max": 28}}, "required": ["zone", "celsius"], "confirm": False}`.

* `extract_json(text)` — find the first balanced `{...}` object in model output (it may be wrapped in prose or a ```json
  fence) and `json.loads` it; return `None` if there's none or it doesn't parse.
* `validate_call(call, tools)` → list of error strings (empty = valid). Check: `call` has `"name"` and `"arguments"` (dict);
  the tool exists; required params present; no unknown params; types (`string`, `number` (int or float, not bool),
  `integer`, `boolean`); `enum`; `min`/`max`.
* `needs_confirmation(call, tools)` → the tool's `confirm` flag (default True for unknown tools — fail safe).
* `error_message(errors)` — a short message to send back to the model for a retry: `"Tool call invalid: " + "; ".join(errors)`.
"""
import json

TYPES = {"string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,)}


def extract_json(text):
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def validate_call(call, tools):
    if not isinstance(call, dict) or "name" not in call or not isinstance(call.get("arguments"), dict):
        return ["call must be an object with 'name' and 'arguments'"]
    spec = next((t for t in tools if t["name"] == call["name"]), None)
    if spec is None:
        return [f"unknown tool '{call['name']}'"]
    args, errs = call["arguments"], []
    for r in spec.get("required", []):
        if r not in args:
            errs.append(f"missing required '{r}'")
    for k, v in args.items():
        p = spec["params"].get(k)
        if p is None:
            errs.append(f"unknown parameter '{k}'")
            continue
        ok = isinstance(v, TYPES[p["type"]]) and not (p["type"] in ("number", "integer") and isinstance(v, bool))
        if not ok:
            errs.append(f"'{k}' must be {p['type']}")
            continue
        if "enum" in p and v not in p["enum"]:
            errs.append(f"'{k}' must be one of {p['enum']}")
        if "min" in p and v < p["min"]:
            errs.append(f"'{k}' must be >= {p['min']}")
        if "max" in p and v > p["max"]:
            errs.append(f"'{k}' must be <= {p['max']}")
    return errs


def needs_confirmation(call, tools):
    spec = next((t for t in tools if t["name"] == call.get("name")), None)
    return True if spec is None else bool(spec.get("confirm", True))


def error_message(errors):
    return "Tool call invalid: " + "; ".join(errors)
