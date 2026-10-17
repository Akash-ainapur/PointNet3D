"""Run the notebook's code cells locally (no Colab, no GPU needed) against a TINY sample of real ModelNet10 files.

What this does:
  1. builds a mini ModelNet10.zip from <sample_dir>/ModelNet10 (+ macOS-style junk entries, like the real zip has)
  2. serves it from a local HTTP server and points the notebook's download cell at it, so the notebook's own
     download -> extract -> junk-filter -> count-check code really runs
  3. executes every code cell except those tagged "colab-only", in one shared namespace
  4. saves every matplotlib figure as a PNG; plotly's show() is disabled so no browser opens

What this does NOT prove: Colab behaviour, GPU behaviour, the real 473 MB download, or accuracy on the full dataset.

Usage (from the project root):
  python dev/run_local_test.py --sample-dir <dir containing ModelNet10/> --workdir <empty short dir> [--tnets] [--runs 2]
"""
import argparse, functools, http.server, os, re, shutil, sys, threading, time, traceback, zipfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import nbformat


def build_mini_zip(sample_dir, zip_path):
    root = Path(sample_dir) / "ModelNet10"
    files = sorted(p for p in root.rglob("*.off"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(root.parent).as_posix())
        # junk like the real archive contains: AppleDouble files that END IN .off but are not meshes
        zf.writestr("__MACOSX/ModelNet10/chair/train/._chair_0001.off", b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        \x00")
        zf.writestr("__MACOSX/ModelNet10/bed/test/._bed_0001.off", b"\x00\x05\x16\x07")
        zf.writestr("ModelNet10/.DS_Store", b"\x00\x00\x00\x01Bud1")
    return len(files)


def patch_once(src, pattern, repl):
    new, n = re.subn(pattern, repl, src, count=1, flags=re.M)
    assert n == 1, f"test seam not found in notebook: {pattern}"
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=str(Path(__file__).resolve().parent.parent / "PointNet_ModelNet10.ipynb"))
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--tnets", action="store_true", help="switch both T-Nets on")
    ap.add_argument("--runs", type=int, default=1, help="repeat the whole notebook (2nd run exercises the cache paths)")
    ap.add_argument("--no-open3d", action="store_true", help="make `import open3d` fail to test the matplotlib fallback")
    args = ap.parse_args()
    if args.no_open3d:
        sys.modules["open3d"] = None     # `import open3d` now raises ImportError

    work = Path(args.workdir).resolve()
    if work.exists() and any(work.iterdir()):
        sys.exit(f"workdir {work} is not empty; pass an empty/new directory (this script never deletes anything)")
    work.mkdir(parents=True, exist_ok=True)
    serve = work / "_serve"; serve.mkdir()
    n_files = build_mini_zip(args.sample_dir, serve / "ModelNet10.zip")
    zip_bytes = (serve / "ModelNet10.zip").stat().st_size
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(serve))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/ModelNet10.zip"
    print(f"mini zip: {n_files} real meshes + 3 junk entries, {zip_bytes:,} bytes, served at {url}")

    nb = nbformat.read(args.notebook, as_version=4)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code"), "notebook should ship without stored outputs"

    import matplotlib.pyplot as plt
    fig_dir = work / "figures"; fig_dir.mkdir()
    counter = {"n": 0}
    def save_show(*a, **k):
        counter["n"] += 1
        plt.gcf().savefig(fig_dir / f"fig_{counter['n']:02d}.png", dpi=80)
        plt.close("all")
    plt.show = save_show
    try:
        import plotly.graph_objects as go
        go.Figure.show = lambda self, *a, **k: counter.setdefault("plotly_figs", []).append(len(self.data))
    except ImportError:
        pass

    os.chdir(work)
    for run in range(1, args.runs + 1):
        print(f"\n{'=' * 20} RUN {run}/{args.runs} {'=' * 20}")
        ns = {"__name__": "__main__"}
        t_all = time.time()
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            if "colab-only" in cell.metadata.get("tags", []):
                print(f"--- [{cell.id}] skipped (colab-only)")
                continue
            src = cell.source
            if cell.id == "config" and args.tnets:
                src = patch_once(src, r"^USE_INPUT_TNET\s*=\s*False", "USE_INPUT_TNET = True")
                src = patch_once(src, r"^USE_FEATURE_TNET\s*=\s*False", "USE_FEATURE_TNET = True")
            if cell.id == "download":     # test seams: point at the local mini zip
                src = patch_once(src, r'^MODELNET10_URL\s*=\s*".*"', f'MODELNET10_URL = "{url}"')
                src = patch_once(src, r"^EXPECTED_ZIP_BYTES\s*=\s*[\d_]+", f"EXPECTED_ZIP_BYTES = {zip_bytes}")
                src = patch_once(src, r"^EXPECTED_NUM_MESHES\s*=\s*\d+", f"EXPECTED_NUM_MESHES = {n_files}")
            print(f"--- [{cell.id}]")
            t0 = time.time()
            try:
                exec(compile(src, f"<cell {cell.id}>", "exec"), ns)
            except BaseException:
                traceback.print_exc()
                print(f"\nFAILED in cell [{cell.id}]")
                sys.exit(1)
            print(f"    ({time.time() - t0:.1f}s)")
        print(f"\nRUN {run} OK in {time.time() - t_all:.0f}s | open3d used: {ns.get('HAVE_O3D')} | "
              f"plotly figures shown: {counter.get('plotly_figs')} | matplotlib figures saved: {counter['n']} in {fig_dir}")
        print(f"smoke test passed flag: {ns['SMOKE_TEST_PASSED']} | checkpoint exists: {ns['CKPT_PATH'].exists()}")
    httpd.shutdown()


if __name__ == "__main__":
    main()
