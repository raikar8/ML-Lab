"""Notebooks are valid JSON and every code cell is valid Python (IPython ! and % lines stripped)."""
import ast
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks"


def test_notebooks_parse():
    files = sorted(NB.glob("*.ipynb"))
    assert len(files) == 7
    for f in files:
        nb = json.loads(f.read_text())
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                src = "".join(c["source"])
                src = re.sub(r"^(\s*)(?:[\w, ]+=\s*)?[!%].*$", r"\1pass", src, flags=re.M)
                ast.parse(src)
