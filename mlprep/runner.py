"""Run one exercise's pytest file against the learner's code (exercises/), from the app."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .content import ROOT


def run_pytest(test_path: Path, timeout: int = 300) -> tuple[bool, str]:
    env = dict(os.environ, MLPREP_TARGET="exercises")
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider",
                            str(test_path)], cwd=ROOT, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout} s — look for an infinite loop or an exponential algorithm."
    return r.returncode == 0, (r.stdout + r.stderr)[-6000:]
