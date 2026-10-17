"""Unit tests for the notebook's core functions (no dataset, no Colab, no GPU needed).

The functions are taken straight out of the built notebook, so this tests what you actually run.
Usage (from the project root):  python dev/test_units.py
Needs: torch, numpy, matplotlib, nbformat
"""
import ast, math, os, sys, tempfile, textwrap, traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import nbformat
import numpy as np
import torch

NB = Path(__file__).resolve().parent.parent / "PointNet_ModelNet10.ipynb"
cells = {c.id: c.source for c in nbformat.read(NB, as_version=4).cells if c.cell_type == "code"}

# the 'config' cell creates folders relative to the working directory, so run in a scratch dir
os.chdir(tempfile.mkdtemp(prefix="pn_unit_"))
ns = {"__name__": "__main__"}
exec(compile(cells["config"], "<config>", "exec"), ns)
ns["NUM_CLASSES"] = 10


def load_defs(cell_id):
    """Execute only the imports / function / class definitions of a cell (not its data-processing code)."""
    tree = ast.parse(cells[cell_id])
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), f"<{cell_id}>", "exec"), ns)


for cid in ("pc_funcs", "preprocess", "split", "train_funcs", "smoke_funcs"):
    load_defs(cid)
exec(compile(cells["model"], "<model>", "exec"), ns)       # pure definitions + MODEL_KWARGS

g = ns
results = []


def test(fn):
    try:
        fn(); results.append((fn.__name__, True, ""))
        print(f"PASS  {fn.__name__}")
    except Exception:
        results.append((fn.__name__, False, traceback.format_exc()))
        print(f"FAIL  {fn.__name__}\n{textwrap.indent(traceback.format_exc(), '      ')}")


def write(tmp, name, text):
    p = Path(tmp) / name; p.write_text(text); return p


@test
def read_off_standard_and_merged_header_and_quads():
    with tempfile.TemporaryDirectory() as d:
        std = write(d, "a.off", "OFF\n4 2 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n3 0 1 2\n3 0 2 3\n")
        v, f = g["read_off"](std)
        assert v.shape == (4, 3) and f.shape == (2, 3) and v.dtype == np.float32 and f.dtype == np.int64
        merged = write(d, "b.off", "OFF4 2 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n3 0 1 2\n3 0 2 3\n")
        v2, f2 = g["read_off"](merged)
        assert np.array_equal(v, v2) and np.array_equal(f, f2), "merged 'OFF4 2 0' header must parse identically"
        quad = write(d, "c.off", "OFF\n# a comment\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n")
        _, fq = g["read_off"](quad)
        assert fq.shape == (2, 3) and fq.tolist() == [[0, 1, 2], [0, 2, 3]], fq


@test
def read_off_rejects_bad_files():
    with tempfile.TemporaryDirectory() as d:
        for name, text in {"notoff": "PLY\n1 1 0\n", "trunc": "OFF\n4 2 0\n0 0 0\n1 0 0\n",
                           "badidx": "OFF\n3 1 0\n0 0 0\n1 0 0\n1 1 0\n3 0 1 7\n"}.items():
            try:
                g["read_off"](write(d, name + ".off", text))
            except ValueError:
                continue
            raise AssertionError(f"{name}: expected ValueError")


@test
def sampling_is_area_weighted_and_on_the_surface():
    # triangle A has area 0.5 (z=0), triangle B has area 1.5 (z=1): expect 25% / 75% of the points
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 1], [0, 1, 1]], np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    pts = g["sample_points_from_mesh"](verts, faces, 40000, np.random.default_rng(0))
    assert pts.shape == (40000, 3) and pts.dtype == np.float32
    frac_b = (pts[:, 2] > 0.5).mean()
    assert abs(frac_b - 0.75) < 0.01, frac_b
    assert np.all(pts[:, 0] >= -1e-6) and np.all(pts[:, 1] >= -1e-6)
    a = pts[pts[:, 2] < 0.5]; assert np.all(a[:, 0] + a[:, 1] <= 1 + 1e-5), "points of triangle A must stay inside it"
    b = pts[pts[:, 2] > 0.5]; assert np.all(b[:, 0] / 3 + b[:, 1] <= 1 + 1e-5), "points of triangle B must stay inside it"
    same = g["sample_points_from_mesh"](verts, faces, 10, np.random.default_rng(5))
    assert np.array_equal(same, g["sample_points_from_mesh"](verts, faces, 10, np.random.default_rng(5))), "same seed -> same points"


@test
def sampling_degenerate_mesh_falls_back_to_vertices():
    verts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], np.float32)       # all on a line -> zero area
    pts = g["sample_points_from_mesh"](verts, np.array([[0, 1, 2]]), 16, np.random.default_rng(0))
    assert pts.shape == (16, 3) and np.isfinite(pts).all()


@test
def normalize_centres_and_scales_to_unit_sphere():
    rng = np.random.default_rng(1)
    pts = (rng.normal(size=(500, 3)) * [5, 1, 20] + [100, -50, 7]).astype(np.float32)
    out = g["normalize_point_cloud"](pts)
    assert np.abs(out.mean(0)).max() < 1e-5 and abs(np.linalg.norm(out, axis=1).max() - 1) < 1e-5
    const = g["normalize_point_cloud"](np.ones((10, 3), np.float32))
    assert np.isfinite(const).all(), "constant cloud must not produce NaN"


@test
def augment_rotates_about_z_and_jitters_within_clip():
    torch.manual_seed(0)
    x = torch.randn(16, 256, 3)
    no_jit = g["augment_batch"](x, jitter_sigma=0.0)
    assert torch.allclose(no_jit[..., 2], x[..., 2], atol=1e-6), "z (up) must be unchanged by a rotation about z"
    assert torch.allclose(no_jit.norm(dim=2), x.norm(dim=2), atol=1e-5), "rotation must preserve distances to origin"
    assert not torch.allclose(no_jit[..., :2], x[..., :2]), "xy must actually change"
    jit = g["augment_batch"](x, jitter_sigma=10.0, jitter_clip=0.05)       # huge sigma: clipping must bound the noise
    torch.manual_seed(1); a = g["augment_batch"](x, jitter_sigma=0.0)
    torch.manual_seed(1); b = g["augment_batch"](x, jitter_sigma=10.0, jitter_clip=0.05)
    assert (b - a).abs().max().item() <= 0.05 + 1e-6, "jitter must be clipped to +-0.05"
    assert jit.shape == x.shape and jit.dtype == x.dtype


@test
def stratified_split_properties():
    labels = np.array([0] * 100 + [1] * 20 + [2] * 2 + [3] * 1)
    tr, va = g["stratified_split"](labels, 0.1, seed=3)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == len(labels)
    cnt = lambda idx, c: int((labels[idx] == c).sum())
    assert cnt(va, 0) == 10 and cnt(va, 1) == 2, (cnt(va, 0), cnt(va, 1))
    assert cnt(va, 2) == 1 and cnt(tr, 2) == 1, "a 2-sample class keeps one sample on each side"
    assert cnt(va, 3) == 0 and cnt(tr, 3) == 1, "a 1-sample class must stay in train"
    tr2, va2 = g["stratified_split"](labels, 0.1, seed=3)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2), "same seed -> same split"


@test
def evaluate_confusion_matrix_and_metrics():
    class Fixed(torch.nn.Module):          # predicts class = (label + 1) % 3 for label 0, else the label itself
        def forward(self, points):
            lab = points[:, 0, 0].long()
            pred = torch.where(lab == 0, torch.ones_like(lab), lab)
            return torch.nn.functional.one_hot(pred, 3).float() * 10, {}
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    pts = torch.zeros(6, 4, 3); pts[:, 0, 0] = labels.float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pts, labels), batch_size=4)
    r = g["evaluate"](Fixed(), loader, torch.device("cpu"), 3)
    assert r["confusion"].tolist() == [[0, 2, 0], [0, 3, 0], [0, 0, 1]], r["confusion"]
    assert abs(r["acc"] - 4 / 6) < 1e-9
    assert np.allclose(r["per_class_acc"], [0, 1, 1]) and abs(r["mean_class_acc"] - 2 / 3) < 1e-9


@test
def regularizer_and_loss():
    I = torch.eye(5).expand(3, 5, 5)
    assert g["feature_transform_regularizer"](I).item() == 0
    th = 0.7; R = torch.eye(5).clone(); R[0, 0], R[0, 1], R[1, 0], R[1, 1] = math.cos(th), -math.sin(th), math.sin(th), math.cos(th)
    assert g["feature_transform_regularizer"](R.expand(3, 5, 5)).item() < 1e-10, "a rotation is orthogonal -> zero penalty"
    # 2*I: m m^T = 4I, diff = -3I, squared-sum = 9*5 = 45 per sample, x3 samples, halved = 67.5
    assert abs(g["feature_transform_regularizer"](2 * I).item() - 67.5) < 1e-4
    logits, labels = torch.randn(4, 10), torch.tensor([1, 2, 3, 4])
    ce = torch.nn.functional.cross_entropy(logits, labels)
    loss, ce_out = g["compute_loss"](logits, labels, {"feature_matrix": None}, 0.001)
    assert torch.allclose(loss, ce) and torch.allclose(ce_out, ce)
    loss2, ce2 = g["compute_loss"](logits, labels, {"feature_matrix": 2 * torch.eye(4).expand(4, 4, 4)}, 0.001)
    assert torch.allclose(ce2, ce) and loss2.item() > ce.item(), "penalty is added to the loss but not to the returned cross-entropy"
    # 2*I, k=4, B=4: 0.5 * 4 * (9*4) = 72  ->  0.001 * 72 = 0.072 extra
    assert abs((loss2 - ce2).item() - 0.072) < 1e-5


@test
def model_shapes_any_num_points_and_all_variants():
    for it in (False, True):
        for ft in (False, True):
            m = g["PointNet"](num_classes=10, input_transform=it, feature_transform=ft).eval()
            for n in (64, 1024):
                with torch.no_grad():
                    lg, aux = m(torch.randn(3, n, 3))
                assert tuple(lg.shape) == (3, 10)
                assert (aux["input_matrix"] is not None) == it and (aux["feature_matrix"] is not None) == ft
                if it: assert tuple(aux["input_matrix"].shape) == (3, 3, 3)
                if ft: assert tuple(aux["feature_matrix"].shape) == (3, 64, 64)


@test
def checkpoint_roundtrip_with_weights_only_load():
    kw = dict(num_classes=10, input_transform=True, feature_transform=False, dropout=0.3)
    m = g["PointNet"](**kw).eval()
    x = torch.randn(2, 128, 3)
    with torch.no_grad():
        ref, _ = m(x)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.pt"
        torch.save({"model_state": m.state_dict(), "epoch": 3, "val_acc": 0.5, "model_kwargs": kw, "class_names": ["a", "b"], "num_points": 1024}, p)
        ck = torch.load(p, map_location="cpu", weights_only=True)       # the strictest loader must accept our checkpoint
    m2 = g["PointNet"](**ck["model_kwargs"]).eval(); m2.load_state_dict(ck["model_state"])
    with torch.no_grad():
        assert torch.allclose(ref, m2(x)[0], atol=1e-6)


@test
def stratified_subset_picker_is_deterministic_and_capped():
    items = [(Path(f"x/{c}_{i}.off"), c) for c in ("a", "b") for i in range(10)]
    s1 = g["pick_subset"](items, 4, 7); s2 = g["pick_subset"](items, 4, 7)
    assert s1 == s2 and sum(1 for _, c in s1 if c == "a") == 4 and sum(1 for _, c in s1 if c == "b") == 4
    assert g["pick_subset"](items, None, 7) == items


bad = [r for r in results if not r[1]]
print(f"\n{len(results) - len(bad)}/{len(results)} unit tests passed")
sys.exit(1 if bad else 0)
