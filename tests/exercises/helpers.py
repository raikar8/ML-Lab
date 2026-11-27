"""Load the module under test: your exercise (default) or the reference solution.

    MLPREP_TARGET=solutions pytest tests/exercises   # check the reference solutions
"""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(key: str):
    target = os.environ.get("MLPREP_TARGET", "exercises")
    path = ROOT / target / f"{key}.py"
    spec = importlib.util.spec_from_file_location(f"{target}_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def key_of(test_file: str) -> str:
    return Path(test_file).stem.removeprefix("test_")
