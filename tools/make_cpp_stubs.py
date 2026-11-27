"""Generate C++ exercise stubs from cpp/solutions/*.hpp.

In a solution header, the code between `// BEGIN SOLUTION` and `// END SOLUTION` is replaced by a TODO, and a following
`// STUB: <code>` line (e.g. `// STUB: return 0;`) becomes real code so the stub still compiles.

    python tools/make_cpp_stubs.py           # create missing stubs
    python tools/make_cpp_stubs.py --force   # overwrite all (loses your work!)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "cpp"


def stub(src: str) -> str:
    src = re.sub(r"^([ \t]*)// BEGIN SOLUTION\n.*?^[ \t]*// END SOLUTION\n", r"\1// TODO: implement\n", src,
                 flags=re.S | re.M)
    return re.sub(r"^([ \t]*)// STUB: (.*)$", r"\1\2", src, flags=re.M)


if __name__ == "__main__":
    force = "--force" in sys.argv
    for sol in sorted((ROOT / "solutions").glob("*.hpp")):
        dst = ROOT / "exercises" / sol.name
        if dst.exists() and not force:
            continue
        dst.write_text(stub(sol.read_text(encoding="utf-8")), encoding="utf-8")
        print("wrote", dst.relative_to(ROOT.parent))
