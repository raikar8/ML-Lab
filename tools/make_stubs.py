"""Generate exercise stubs from reference solutions.

The stub keeps the module docstring (metadata + spec), imports, constants, class and function
signatures and docstrings, and replaces bodies with `raise NotImplementedError`. A function or
class preceded by a `# given` comment is copied unchanged (helpers the student doesn't write).
Exercises with `stub: custom` in their metadata keep a hand-written stub.

    python tools/make_stubs.py            # create missing stubs
    python tools/make_stubs.py --force    # overwrite all generated stubs (loses your work!)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mlprep.stubs import stub_source  # noqa: E402

SOL, EX = ROOT / "solutions", ROOT / "exercises"


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
