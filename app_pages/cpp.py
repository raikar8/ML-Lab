"""C/C++ track: embedded/real-time exercises compiled and tested from the app."""
import re
import shutil
import subprocess

import streamlit as st

from mlprep import progress
from mlprep.content import ROOT
from mlprep.ui import data, md

CPP = ROOT / "cpp"
d = data()
status = d.setdefault("cpp_status", {})


def parse(path):
    src = path.read_text(encoding="utf-8")
    m = re.match(r"\s*/\*(.*?)\*/", src, re.S)
    head, _, spec = (m.group(1).strip("\n") if m else "").partition("\n\n")
    meta = dict(line.split(":", 1) for line in head.splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()} | {"spec": spec.strip(), "key": path.stem}


items = [parse(p) for p in sorted((CPP / "solutions").glob("*.hpp"))]
st.title("C/C++ track")
st.markdown(
    "Embedded and edge-AI interviews almost always include a C/C++ round. These exercises are the classic ones — fixed "
    "point, filters, real-time buffers, allocation-free memory, int8 kernels, streaming convolution, cache-blocked GEMM — "
    "each written from scratch with tests. Edit `cpp/exercises/<name>.hpp`, then **Compile & test**. From a terminal:"
)
st.code("cmake -S cpp -B build && cmake --build build -j && ctest --test-dir build --output-on-failure\n"
        "# check the reference solutions:  cmake -S cpp -B build-sol -DMLPREP_SOLUTIONS=ON", language="bash")
compiler = shutil.which("g++") or shutil.which("clang++")
if not compiler:
    st.warning("No C++ compiler found (g++ or clang++). Install build-essential (Linux), Xcode command-line tools (macOS) "
               "or MSYS2/MinGW (Windows) to compile from the app.")
n_ok = sum(1 for it in items if status.get(it["key"], {}).get("ok"))
st.progress(n_ok / max(1, len(items)), text=f"{n_ok} of {len(items)} passing")

key = st.selectbox("Exercise", [it["key"] for it in items],
                   format_func=lambda k: f"{'✅' if status.get(k, {}).get('ok') else '⬜'} {k} — "
                                         f"{next(i for i in items if i['key'] == k)['title']}")
it = next(i for i in items if i["key"] == key)
st.header(it["title"])
st.caption(f"Week {it.get('week', '?')} · {it.get('level', '')} · ~{it.get('time', '?')} min · {it.get('tags', '')}")
md(it["spec"])

ex_path, sol_path, test_path = CPP / "exercises" / f"{key}.hpp", CPP / "solutions" / f"{key}.hpp", CPP / "tests" / f"test_{key}.cpp"
t_edit, t_test, t_sol = st.tabs(["Edit & run", "Tests", "Reference solution"])


def run():
    exe = CPP / "build-app" / f"test_{key}"
    exe.parent.mkdir(exist_ok=True)
    c = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-pthread", f"-I{CPP / 'exercises'}",
                        f"-I{CPP / 'tests'}", str(test_path), "-o", str(exe)], capture_output=True, text=True, timeout=180)
    if c.returncode:
        return False, "Compilation failed:\n" + c.stderr[-5000:]
    try:
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        return r.returncode == 0, (c.stderr[-2000:] + r.stdout[-5000:] + r.stderr[-1000:])
    except subprocess.TimeoutExpired:
        return False, "Test timed out (30 s) — an infinite loop? (stubs that return 0 can make loops spin)"


with t_edit:
    code = st.text_area("Your code (saved to the exercise header)", ex_path.read_text(encoding="utf-8"), height=440,
                        key=f"cpp-{key}")
    if st.button("Save, compile & test", type="primary", icon=":material/play_arrow:", disabled=not compiler):
        ex_path.write_text(code, encoding="utf-8")
        with st.spinner("Compiling…"):
            ok, out = run()
        status[key] = {"ok": ok}
        progress.save(d)
        (st.success if ok else st.error)("All tests passed." if ok else "Not passing yet — see output.")
        st.code(out, language="text")
with t_test:
    st.code(test_path.read_text(encoding="utf-8"), language="cpp")
    st.caption("`tests/mini_test.hpp` is a 40-line test harness written for this repo (no external dependencies).")
with t_sol:
    if st.toggle("I've made a real attempt — show the reference solution", key=f"cppsol-{key}"):
        st.code(sol_path.read_text(encoding="utf-8"), language="cpp")
