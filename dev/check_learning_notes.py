"""Check the learning notes against the real code.

1. Every `cell → name` reference: the cell id exists and defines/uses that name.
2. Every other backticked code-like name (snake_case, CamelCase, dotted) exists in the notebook source or the tests.
3. Every ```python block runs, and its printed output (lines joined with " | ") equals the
   `<!-- expect: ... -->` line that follows it. Quoted code that should not run uses ```py instead.

Usage (from the project root):  python dev/check_learning_notes.py [learning/03-*.md ...]
Needs: torch, numpy, nbformat
"""
import contextlib, glob, io, re, sys, traceback
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "dev" / "notebook_source.py").read_text(encoding="utf-8")
TESTS = (ROOT / "dev" / "test_units.py").read_text(encoding="utf-8")
cells = {c.id: c.source for c in nbformat.read(ROOT / "PointNet_ModelNet10.ipynb", as_version=4).cells}

SNIPPET = re.compile(r"```python\n(.*?)```\n(?:<!-- expect: (.*?) -->)?", re.S)   # runnable snippet + its expected output
REF = re.compile(r"`([a-z_]+) → ([\w.]+)`")              # `model → PointNet.forward`
TICK = re.compile(r"`([^`\n]+)`")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
IGNORE = {"keepdims", "drop_last", "num_classes", "kernel_size", "requires_grad", "weight_decay",
          "set_to_none", "zero_grad", "state_dict", "load_state_dict", "no_grad", "draw_plotly", "draw_geometries",
          "run_summary.json"}   # run_summary.json: written by the notebook once Step 3 of the plan lands


def looks_like_code(tok):
    return NAME.match(tok) and ("_" in tok or "." in tok or (tok[0].isupper() and any(c.islower() for c in tok[1:])))


def main(paths):
    problems = 0
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        print(f"\n=== {path}")
        for cell, name in REF.findall(text):
            last = name.split(".")[-1]
            if cell not in cells:
                print(f"  BAD REF  cell '{cell}' does not exist"); problems += 1
            elif last not in cells[cell]:
                print(f"  BAD REF  '{name}' not found in cell '{cell}'"); problems += 1
        for tok in set(TICK.findall(text)):
            tok = tok.strip()
            if not looks_like_code(tok) or tok in IGNORE or tok.startswith(("torch.", "nn.", "np.", "o3d.", "F.")):
                continue
            last = tok.split(".")[-1]
            if last not in SOURCE and last not in TESTS:
                print(f"  UNKNOWN  `{tok}` is not in notebook_source.py or test_units.py"); problems += 1
        for i, (block, expect) in enumerate(SNIPPET.findall(text), 1):
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(block, f"{path}#snippet{i}", "exec"), {"__name__": "__snippet__"})
            except Exception:
                problems += 1
                print(f"  snippet {i}: ERROR\n" + traceback.format_exc()); continue
            got = " | ".join(buf.getvalue().strip().splitlines())
            if not expect:
                problems += 1; print(f"  snippet {i}: NO EXPECT LINE (got: {got})")
            elif got != expect.strip():
                problems += 1; print(f"  snippet {i}: WRONG ANSWER\n    note says: {expect.strip()}\n    real output: {got}")
            else:
                print(f"  snippet {i}: OK -> {got}")
    print(f"\n{'ALL CHECKS PASSED' if not problems else f'{problems} PROBLEM(S)'}")
    return problems


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(glob.glob(str(ROOT / "learning" / "*.md")))
    sys.exit(1 if main(targets) else 0)
