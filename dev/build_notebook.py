"""Convert dev/notebook_source.py (percent format) into PointNet_ModelNet10.ipynb.

Usage:  python dev/build_notebook.py
Needs:  pip install nbformat
"""
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
SRC = HERE / "notebook_source.py"
OUT = HERE.parent / "PointNet_ModelNet10.ipynb"

HEADER = re.compile(r"^# %%(?P<md> \[markdown\])?(?P<attrs>.*)$")
ATTR = re.compile(r'(\w+)="([^"]*)"')


def parse(text):
    cells, cur = [], None
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            if cur:
                cells.append(cur)
            cur = {"kind": "markdown" if m.group("md") else "code",
                   "attrs": dict(ATTR.findall(m.group("attrs") or "")), "lines": []}
        elif cur is not None:                      # anything before the first header is the file's own comment block
            cur["lines"].append(line)
    if cur:
        cells.append(cur)
    return cells


def markdown_source(lines):
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.strip() == "#" or not ln.strip():
            out.append("")
        else:
            raise ValueError(f"markdown line must start with '# ': {ln!r}")
    return "\n".join(out).strip("\n")


def main():
    nb = new_notebook()
    seen = set()
    for c in parse(SRC.read_text(encoding="utf-8")):
        cid = c["attrs"].get("id")
        assert cid and cid not in seen, f"missing or duplicate cell id: {cid!r}"
        seen.add(cid)
        if c["kind"] == "markdown":
            cell = new_markdown_cell(markdown_source(c["lines"]), id=cid)
        else:
            cell = new_code_cell("\n".join(c["lines"]).strip("\n"), id=cid)
            tags = [t for t in c["attrs"].get("tags", "").split(",") if t]
            if tags:
                cell.metadata["tags"] = tags
        nb.cells.append(cell)
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "name": "PointNet_ModelNet10.ipynb"},
    }
    nbformat.validate(nb)
    nbformat.write(nb, OUT)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {OUT}  ({len(nb.cells)} cells: {n_code} code, {len(nb.cells) - n_code} markdown)")


if __name__ == "__main__":
    main()
