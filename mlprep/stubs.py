"""Turn a reference solution into an exercise stub (shared by tools/make_stubs.py and the app).

The stub keeps the module docstring (metadata + spec), imports, constants, class and function
signatures and docstrings, and replaces bodies with `raise NotImplementedError`. A function or
class preceded by a `# given` comment, or whose name starts with a single underscore, is copied unchanged.
"""
import ast


def _strip_spoilers(src: str) -> str:
    """Interview problems keep hints / follow-ups / complexity in the docstring header; the stub keeps only the
    title and the problem statement (the app reads the metadata from the reference solution)."""
    if not src.lstrip().startswith('"""'):
        return src
    start = src.index('"""') + 3
    end = src.index('"""', start)
    doc = src[start:end]
    head, sep, spec = doc.strip("\n").partition("\n\n")
    if not any(line.startswith(("hint:", "follow-up:", "complexity:")) for line in head.splitlines()):
        return src
    title = next((line.split(":", 1)[1].strip() for line in head.splitlines() if line.startswith("title:")), "")
    new_doc = f"\n{title}\n\n{spec.strip()}\n"
    return src[:start] + new_doc + src[end:]


def stub_source(src: str) -> str:
    src = _strip_spoilers(src)
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
