"""Generate exercise stubs from reference solutions.

The stub keeps the module docstring (metadata + spec), imports, constants, class and function
signatures and docstrings, and replaces bodies with `raise NotImplementedError`. A function or
class preceded by a `# given` comment is copied unchanged (helpers the student doesn't write).
Exercises with `stub: custom` in their metadata keep a hand-written stub.

    python tools/make_stubs.py            # create missing stubs
    python tools/make_stubs.py --force    # overwrite all generated stubs (loses your work!)
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOL, EX = ROOT / "solutions", ROOT / "exercises"


def stub_source(src: str) -> str:
    tree = ast.parse(src)
    lines = src.splitlines()
    out: list[str] = []

    def given(node) -> bool:
        if node.name.startswith("_") and not node.name.startswith("__"):
            return True  # private helpers are provided
        start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 2
        return start >= 0 and lines[start].strip() == "# given"

    def emit_func(fn, indent: str):
        start = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
        body0 = fn.body[0]
        has_doc = isinstance(body0, ast.Expr) and isinstance(getattr(body0, "value", None), ast.Constant) \
            and isinstance(body0.value.value, str)
        header_end = (body0.end_lineno if has_doc else fn.body[0].lineno - 1)
        out.extend(lines[start:header_end])
        out.append(indent + "    raise NotImplementedError  # TODO: implement")
        out.append("")

    for node in tree.body:
        seg = lines[(node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1:
                    node.end_lineno]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not given(node):
            emit_func(node, "")
            out.append("")
        elif isinstance(node, ast.ClassDef) and not given(node):
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            first = node.body[0]
            out.extend(lines[start:first.lineno - 1] if not isinstance(first, ast.Expr) else lines[start:first.end_lineno])
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if given(item):
                        out.extend(lines[(item.decorator_list[0].lineno if item.decorator_list else item.lineno) - 1:
                                         item.end_lineno])
                        out.append("")
                    else:
                        emit_func(item, "    ")
                elif item is not first or not isinstance(item, ast.Expr):
                    out.extend(lines[item.lineno - 1:item.end_lineno])
            out.append("")
        else:
            out.extend(seg)
            out.append("" if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else "")
    text = "\n".join(out)
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text.rstrip() + "\n"


def main(force: bool) -> None:
    for sol in sorted(SOL.glob("w*.py")) + sorted(SOL.glob("e*.py")) + sorted(SOL.glob("f*.py")) + sorted(SOL.glob("c*.py")):
        dst = EX / sol.name
        src = sol.read_text(encoding="utf-8")
        if "stub: custom" in src.split('"""')[1]:
            if not dst.exists():
                print("missing custom stub:", dst.name)
            continue
        if dst.exists() and not force:
            continue
        dst.write_text(stub_source(src), encoding="utf-8")
        print("wrote", dst.name)


if __name__ == "__main__":
    main("--force" in sys.argv)
