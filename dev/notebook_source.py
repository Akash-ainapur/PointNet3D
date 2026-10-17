# Source of truth for PointNet_ModelNet10.ipynb, in "percent" cell format.
#   "# %%"             starts a code cell
#   "# %% [markdown]"  starts a markdown cell (its lines are written as "# ...")
#   id="..." tags="a,b" are optional cell attributes ("colab-only" cells are skipped by dev/run_local_test.py)
# Rebuild the notebook with:  python dev/build_notebook.py

# %% [markdown] id="title"
# # PointNet: classifying 3D objects (ModelNet10) on Google Colab
#
# **What you will build:** a neural network that looks at a 3D object, stored as a cloud of 1024 `(x, y, z)` points, and says which of 10 furniture categories it is (chair, bed, toilet, ...). The network is **PointNet** (Qi et al., CVPR 2017), written in PyTorch. Open3D is used to look at the data.
#
# **How to run it (about 3 steps):**
# 1. **Runtime → Change runtime type → T4 GPU.** The free tier is enough. Nothing here needs a paid plan.
# 2. Keep `QUICK_MODE = True` (section 1) and **Runtime → Run all.** This is a small first run that proves every stage works. *Accuracy in quick mode is not meaningful.*
# 3. Set `QUICK_MODE = False` and run all again for the real experiment. Then, optionally, switch on the T-Nets (section 6) and compare.
#
# **Honest status of this notebook**
# * No accuracy number appears anywhere in this notebook. Nothing has been trained on Colab yet; every number you see will come from *your* run.
# * The notebook was written and tested outside Colab. See the README for exactly what was and was not tested.
#
# **Sections:** 1 Setup · 2 Dataset · 3 Mesh → point cloud · 4 Visualization · 5 Preprocessing, splits, labels · 6 Model · 7 Smoke test · 8 Training · 9 Evaluation · 10 Limits and next steps

# %% [markdown] id="s1"
# ## 1. Setup
# Three things happen here: install Open3D, import libraries, and set the configuration.
# **The configuration cell is the only place you normally need to edit.**

# %% id="install" tags="colab-only"
# Open3D = the 3D library we use to look at point clouds.
# "open3d-cpu" is the smaller (CPU-only) official wheel; we only need CPU features. Fall back to the full package if it fails.
!pip install -q open3d-cpu || pip install -q open3d

# %% id="config"
import os, sys, time, math, random, zipfile, urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:                      # progress bars are optional
    def tqdm(it, **kwargs):
        return it

# ============================== CONFIG: edit here ==============================
QUICK_MODE = True            # True  = small first run (a subset of the data, few epochs) to prove the pipeline works
                             # False = full ModelNet10 run

NUM_POINTS   = 1024          # points per object (the PointNet paper also uses 1024 for ModelNet40)
VAL_FRACTION = 0.10          # share of the official TRAIN set held back for validation
SEED         = 42

USE_INPUT_TNET   = False     # T-Net #1: 3x3 alignment of the raw xyz points      (baseline: off)
USE_FEATURE_TNET = False     # T-Net #2: 64x64 alignment of the per-point features (baseline: off)

BATCH_SIZE = 32
LR = 1e-3                    # Adam learning rate
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3                # drop probability. The original code states it as keep-probability 0.7.
FEATURE_REG_WEIGHT = 0.001   # only used when USE_FEATURE_TNET = True (value from the original code)

if QUICK_MODE:
    EPOCHS = 5
    MAX_TRAIN_PER_CLASS = 100    # use at most this many official-train meshes per class
    MAX_TEST_PER_CLASS = 30      # ... and this many official-test meshes per class
else:
    EPOCHS = 30
    MAX_TRAIN_PER_CLASS = None   # None = use everything
    MAX_TEST_PER_CLASS = None

USE_GOOGLE_DRIVE = False     # True = keep the cache + best model in your Google Drive so they survive a runtime reset
                             # (Colab will ask you to authorise Drive access when you set this to True)
# ===============================================================================

LOCAL_DIR = (Path("/content") if Path("/content").exists() else Path.cwd()) / "pointnet_modelnet10"
DATA_DIR = LOCAL_DIR / "data"                 # downloaded zip + extracted meshes (extracted size is roughly 2 GB, estimated from a sample)
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK_DIR = Path("/content/drive/MyDrive/pointnet_modelnet10")
else:
    WORK_DIR = LOCAL_DIR / "outputs"          # preprocessed point clouds + best model checkpoint
DATA_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

TAG = "quick" if QUICK_MODE else "full"
CACHE_PATH = WORK_DIR / f"modelnet10_{NUM_POINTS}pts_{TAG}.npz"
CKPT_PATH = WORK_DIR / f"best_pointnet_{TAG}.pt"

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE_TEST_PASSED = False                      # becomes True only if the smoke test in section 7 passes

print(f"PyTorch {torch.__version__} | device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
if device.type == "cpu":
    print("WARNING: no GPU found. Use Runtime -> Change runtime type -> T4 GPU. QUICK_MODE still works on CPU; the full run will be slow.")
print(f"mode: {TAG} | epochs: {EPOCHS} | T-Nets: input={USE_INPUT_TNET}, feature={USE_FEATURE_TNET}")
print(f"data dir: {DATA_DIR}\noutputs : {WORK_DIR}")

try:
    import open3d as o3d
    HAVE_O3D = True
    print("Open3D", o3d.__version__)
except Exception as e:                         # we keep going with matplotlib if Open3D is unavailable
    HAVE_O3D = False
    print("Open3D could not be imported -> visualization falls back to matplotlib.", repr(e))

# %% [markdown] id="s2"
# ## 2. Dataset: ModelNet10
#
# | Fact | Value (checked against the official page / the file itself) |
# |---|---|
# | Source | Princeton ModelNet, `https://modelnet.cs.princeton.edu/download.html` links to the 10-class subset |
# | File used | `ModelNet10.zip`, **473,402,300 bytes (~451 MiB)**, HTTP 200 when this notebook was written |
# | Content | 4,899 CAD models (surface **meshes**, not point clouds) in **OFF** format, orientation manually aligned |
# | Classes (10) | bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet |
# | Official split | 3,991 train / 908 test, encoded in the folder names `ModelNet10/<class>/<train or test>/<class>_NNNN.off` |
# | License | **No open-source license is stated.** The page says the CAD models' original authors hold the copyright and the dataset is "provided for the convenience of academic research only". We do not redistribute anything: the notebook downloads straight from Princeton. |
# | Citation | Z. Wu et al., *3D ShapeNets: A Deep Representation for Volumetric Shapes*, CVPR 2015 |
#
# Two gotchas found while checking the zip:
# * It also contains macOS junk such as `__MACOSX/.../._chair_0001.off`. These end in `.off` but are **not** meshes, so we filter them out.
# * The classes are **imbalanced** (e.g. 889 chair vs 106 bathtub meshes in train). That is why we report *per-class* accuracy later, not only overall accuracy.

# %% id="download"
MODELNET10_URL = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
EXPECTED_ZIP_BYTES = 473_402_300
EXPECTED_NUM_MESHES = 4899
EXPECTED_CLASSES = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

ZIP_PATH = DATA_DIR / "ModelNet10.zip"
MODEL_ROOT = DATA_DIR / "ModelNet10"

def list_off_files(root):
    """All real mesh files under `root` (skips macOS '._*.off' junk)."""
    return sorted(p for p in root.rglob("*.off") if not p.name.startswith("._"))

def download(url, dest, chunk=1 << 20):
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pointnet-colab-notebook"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total, done, next_report = int(r.headers.get("Content-Length", 0)), 0, 0.0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf); done += len(buf)
            if total and done / total >= next_report:
                print(f"  downloaded {done / 1e6:6.1f} / {total / 1e6:.1f} MB"); next_report += 0.1
    tmp.replace(dest)

def ensure_dataset():
    if MODEL_ROOT.exists() and len(list_off_files(MODEL_ROOT)) == EXPECTED_NUM_MESHES:
        print("Dataset already extracted, nothing to do."); return
    if not (ZIP_PATH.exists() and ZIP_PATH.stat().st_size == EXPECTED_ZIP_BYTES):
        print(f"Downloading {MODELNET10_URL}\n(about 473 MB; this usually takes a minute or two on Colab)")
        download(MODELNET10_URL, ZIP_PATH)
    size = ZIP_PATH.stat().st_size
    if size != EXPECTED_ZIP_BYTES:
        print(f"WARNING: zip is {size:,} bytes, expected {EXPECTED_ZIP_BYTES:,}. The server file may have changed.")
    assert zipfile.is_zipfile(ZIP_PATH), "Downloaded file is not a valid zip. Delete it and try again, or download it manually (see README)."
    print("Extracting meshes (skipping macOS junk files) ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [m for m in zf.namelist() if m.startswith("ModelNet10/") and not Path(m).name.startswith("._")]
        zf.extractall(DATA_DIR, members=members)
    n = len(list_off_files(MODEL_ROOT))
    assert n == EXPECTED_NUM_MESHES, f"Expected {EXPECTED_NUM_MESHES} meshes after extraction, found {n}."
    print(f"Done: {n} meshes.")

ensure_dataset()

# %% id="explore"
CLASS_NAMES = sorted(p.name for p in MODEL_ROOT.iterdir() if p.is_dir())
assert CLASS_NAMES == EXPECTED_CLASSES, f"Unexpected classes: {CLASS_NAMES}"
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}       # LABEL ENCODING: alphabetical order -> 0..9
NUM_CLASSES = len(CLASS_NAMES)

def collect_items(split):
    """[(path, class_name), ...] for split 'train' or 'test', sorted so the order is reproducible."""
    return [(p, c) for c in CLASS_NAMES for p in list_off_files(MODEL_ROOT / c / split)]

train_items_all = collect_items("train")
test_items_all = collect_items("test")
print(f"official split: {len(train_items_all)} train meshes, {len(test_items_all)} test meshes\n")
print(f"{'label':>5}  {'class':<12}{'train':>6}{'test':>6}")
for c in CLASS_NAMES:
    n_tr = sum(1 for _, k in train_items_all if k == c); n_te = sum(1 for _, k in test_items_all if k == c)
    print(f"{CLASS_TO_IDX[c]:>5}  {c:<12}{n_tr:>6}{n_te:>6}")

# %% [markdown] id="s3"
# ## 3. From mesh to point cloud
# A ModelNet file is a **mesh**: a list of vertices plus triangles that connect them. PointNet wants a **fixed-size set of `(x, y, z)` points**. So for every object we do three things:
#
# 1. **Read** the OFF file → `vertices (V, 3)` and `faces (F, 3)` (each face = 3 vertex indices). `V` and `F` differ from object to object.
# 2. **Sample** `NUM_POINTS` random points on the surface. We choose triangles with probability proportional to their **area** (otherwise a region cut into many tiny triangles would get too many points), then pick a random spot inside the chosen triangle → `(NUM_POINTS, 3)`.
# 3. **Normalize:** move the centroid to the origin and divide by the farthest point's distance, so every object fits in a **unit sphere**. Raw sizes in the files are all over the place (a monitor is roughly 10x larger than a chair in file units). The network should learn *shape*, not file units. *Trade-off:* absolute size is thrown away.
#
# ```
# OFF file ──► vertices (V,3) + faces (F,3) ──► sampled points (1024,3) ──► normalized (1024,3)
# ```

# %% id="pc_funcs"
def read_off(path):
    """Read an .off mesh. Returns vertices (V,3) float32 and triangle faces (F,3) int64."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    head = lines[0]
    if not head.startswith("OFF"):
        raise ValueError(f"{path}: not an OFF file (first line {head[:20]!r})")
    if head == "OFF":                      # usual layout: the counts are on the next line
        counts, body = lines[1], lines[2:]
    else:                                  # some OFF files glue the counts to the header, e.g. "OFF490 518 0"
        counts, body = head[3:], lines[1:]
    n_verts, n_faces = (int(v) for v in counts.split()[:2])
    if len(body) < n_verts + n_faces:
        raise ValueError(f"{path}: truncated file")
    verts = np.array([ln.split()[:3] for ln in body[:n_verts]], dtype=np.float32)
    rows = [ln.split() for ln in body[n_verts:n_verts + n_faces]]
    if all(len(r) == 4 and r[0] == "3" for r in rows):               # fast path: every face is a triangle
        faces = np.array(rows, dtype=np.int64)[:, 1:4]
    else:                                                            # polygons: cut into triangle fans
        tris = []
        for r in rows:
            k = int(r[0]); idx = [int(v) for v in r[1:1 + k]]
            tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, k - 1)]
        faces = np.array(tris, dtype=np.int64).reshape(-1, 3)
    if faces.size and faces.max() >= n_verts:
        raise ValueError(f"{path}: a face points to a vertex that does not exist")
    return verts, faces

def sample_points_from_mesh(verts, faces, n_points, rng):
    """Pick n_points random points on the mesh surface, area-weighted. Returns (n_points, 3) float32."""
    tri = verts[faces].astype(np.float64)                                   # (F, 3 corners, 3 coords)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)   # (F,)
    total = area.sum()
    if not np.isfinite(total) or total <= 0:                                # degenerate mesh: fall back to its vertices
        return verts[rng.integers(0, len(verts), n_points)].astype(np.float32)
    face_idx = rng.choice(len(faces), size=n_points, p=area / total)        # which triangle each point lands in
    r1, r2 = rng.random(n_points), rng.random(n_points)
    s = np.sqrt(r1)                                                         # uniform point inside a triangle
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2                                #   (barycentric weights, sum to 1)
    t = tri[face_idx]                                                       # (n_points, 3, 3)
    pts = w0[:, None] * t[:, 0] + w1[:, None] * t[:, 1] + w2[:, None] * t[:, 2]
    return pts.astype(np.float32)

def normalize_point_cloud(pts):
    """Centre at the origin and scale so the farthest point has distance 1 (unit sphere)."""
    pts = pts - pts.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(pts, axis=1).max()
    return (pts / max(radius, 1e-8)).astype(np.float32)

def mesh_to_point_cloud(path, n_points, seed):
    verts, faces = read_off(path)
    pts = sample_points_from_mesh(verts, faces, n_points, np.random.default_rng(seed))
    return normalize_point_cloud(pts)

# %% id="pc_demo"
demo_path, demo_class = next((p, c) for p, c in train_items_all if c == "chair")
verts, faces = read_off(demo_path)
print(f"file: {demo_path.name}  (class '{demo_class}', label {CLASS_TO_IDX[demo_class]})")
print(f"1) mesh            vertices {tuple(verts.shape)}  faces {tuple(faces.shape)}")
print(f"   raw size (x,y,z extent): {np.round(np.ptp(verts, axis=0), 2)}")
sampled = sample_points_from_mesh(verts, faces, NUM_POINTS, np.random.default_rng(0))
print(f"2) sampled points  {tuple(sampled.shape)}")
demo_pc = normalize_point_cloud(sampled)
print(f"3) normalized      {tuple(demo_pc.shape)}  centroid ~ {np.round(demo_pc.mean(0), 6)}  max distance from origin = {np.linalg.norm(demo_pc, axis=1).max():.4f}")

# %% [markdown] id="s4"
# ## 4. Visualization (Open3D)
# We wrap the numpy array in an Open3D `PointCloud` object and colour points by height (`z`). ModelNet10 meshes are aligned with **z pointing up**, so a chair should stand upright.
# On Colab there is no desktop window, so the usual `draw_geometries` pop-up cannot open. `draw_plotly` renders an interactive 3D view inside the notebook instead (drag to rotate).
# First we draw the original **mesh**, then the **1024 sampled points** that PointNet will actually see. A gallery with one object per class follows as a sanity check.
# If Open3D is unavailable, a matplotlib view is used instead.

# %% id="viz"
def colors_by_height(points):
    z = points[:, 2]
    t = (z - z.min()) / max(float(np.ptp(z)), 1e-9)
    return plt.get_cmap("viridis")(t)[:, :3]

def show_point_cloud(points, title="point cloud"):
    """Interactive view with Open3D (falls back to matplotlib)."""
    if HAVE_O3D:
        pcd = o3d.geometry.PointCloud()                                   # Open3D's point-cloud container
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))  # (N,3) numpy -> Open3D
        pcd.colors = o3d.utility.Vector3dVector(colors_by_height(points))   # (N,3) RGB in [0,1]
        print(pcd, "| bounding box:", pcd.get_axis_aligned_bounding_box())
        o3d.visualization.draw_plotly([pcd], window_name=title, width=640, height=480)   # default camera is z-up
    else:
        fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(projection="3d")
        ax.scatter(*points.T, c=colors_by_height(points), s=3); ax.set_title(title)
        ax.set_box_aspect((1, 1, 1)); plt.show()

# 4a) the original mesh, then 4b) the 1024 sampled + normalized points PointNet receives
if HAVE_O3D:
    mesh = o3d.io.read_triangle_mesh(str(demo_path))
    print("4a) original mesh:", mesh)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_plotly([mesh], window_name="original mesh", width=640, height=480)
print("4b) sampled point cloud:")
show_point_cloud(demo_pc, f"{demo_class}: {NUM_POINTS} points")

# %% id="gallery"
# One example per class: a quick visual sanity check of the whole preprocessing chain (matplotlib, always works).
fig = plt.figure(figsize=(16, 6.5))
for i, c in enumerate(CLASS_NAMES):
    path = next(p for p, k in train_items_all if k == c)
    pc = mesh_to_point_cloud(path, NUM_POINTS, seed=0)
    ax = fig.add_subplot(2, 5, i + 1, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors_by_height(pc), s=2)
    ax.set_title(f"{i}: {c}", fontsize=10)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=20, azim=45)
    ax.set_axis_off()
plt.suptitle("One normalized point cloud per class (colour = height z)"); plt.tight_layout(); plt.show()

# %% [markdown] id="s5"
# ## 5. Preprocessing, splits and labels
#
# | Step | What we do | Why |
# |---|---|---|
# | Points per object | exactly `NUM_POINTS` = 1024, sampled **once** with a fixed seed and cached | PointNet batches need equal-sized clouds; caching makes later runs instant |
# | Normalization | centroid at origin, farthest point at distance 1 | removes file units and position |
# | Label encoding | class names sorted alphabetically → integers 0..9 (`CLASS_TO_IDX`) | `cross_entropy` wants integer class ids |
# | **Test set** | the **official** ModelNet10 test split, used **only once** at the end | an honest, comparable number |
# | **Train / validation** | the official train split is divided **stratified** (same class mix in both) with a fixed seed: `VAL_FRACTION` goes to validation | validation picks the best epoch, so the test set never influences training decisions |
# | Augmentation (training only) | random rotation around the up (z) axis + tiny random jitter | makes the model less dependent on exact pose; does not change what is in the cache |
#
# Tensor shapes from here on: **`(B, N, 3)`** = batch of B clouds, each with N points of xyz. Labels: **`(B,)`** integers.

# %% id="preprocess"
def pick_subset(items, max_per_class, seed):
    """Keep at most max_per_class random items per class (None = keep all)."""
    if max_per_class is None:
        return items
    rng = random.Random(seed)
    by_class = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    out = []
    for c in sorted(by_class):
        lst = by_class[c][:]; rng.shuffle(lst)
        out += sorted(lst[:max_per_class])
    return out

def build_arrays(items, n_points, seed, desc):
    X = np.zeros((len(items), n_points, 3), np.float32)
    y = np.zeros(len(items), np.int64)
    ok = np.zeros(len(items), bool)
    failed = []
    for i, (path, cls) in enumerate(tqdm(items, desc=desc)):
        try:
            X[i] = mesh_to_point_cloud(path, n_points, seed + i)
            y[i] = CLASS_TO_IDX[cls]; ok[i] = True
        except Exception as e:                                 # report problems instead of hiding them
            failed.append((path.name, repr(e)))
    if failed:
        print(f"WARNING: skipped {len(failed)} unreadable meshes, e.g. {failed[:3]}")
    return X[ok], y[ok]

if CACHE_PATH.exists():
    d = np.load(CACHE_PATH, allow_pickle=False)
    assert list(d["class_names"]) == CLASS_NAMES and d["X_train"].shape[1] == NUM_POINTS, "Cache does not match the config; delete it."
    X_train_all, y_train_all, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    print(f"Loaded cached point clouds from {CACHE_PATH}")
else:
    t0 = time.time()
    train_items = pick_subset(train_items_all, MAX_TRAIN_PER_CLASS, SEED)
    test_items = pick_subset(test_items_all, MAX_TEST_PER_CLASS, SEED)
    X_train_all, y_train_all = build_arrays(train_items, NUM_POINTS, SEED, "train meshes")
    X_test, y_test = build_arrays(test_items, NUM_POINTS, SEED + 100_000, "test meshes")
    np.savez_compressed(CACHE_PATH, X_train=X_train_all, y_train=y_train_all, X_test=X_test, y_test=y_test, class_names=np.array(CLASS_NAMES))
    print(f"Preprocessed {len(X_train_all) + len(X_test)} meshes in {time.time() - t0:.0f}s and cached to {CACHE_PATH}")

print("official-train arrays:", X_train_all.shape, y_train_all.shape, "| test arrays:", X_test.shape, y_test.shape)

# %% id="split"
def stratified_split(labels, val_fraction, seed):
    """Per class, send ~val_fraction of the samples to validation (at least 1 if the class has >= 2 samples)."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c); rng.shuffle(idx)
        n_val = min(max(1, int(round(len(idx) * val_fraction))), len(idx) - 1)
        val_idx += idx[:n_val].tolist(); train_idx += idx[n_val:].tolist()
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))

tr_idx, va_idx = stratified_split(y_train_all, VAL_FRACTION, SEED)
X_train, y_train = X_train_all[tr_idx], y_train_all[tr_idx]
X_val, y_val = X_train_all[va_idx], y_train_all[va_idx]
assert len(set(tr_idx) & set(va_idx)) == 0, "train and validation overlap!"
assert len(X_train) >= BATCH_SIZE, "Fewer training samples than one batch; lower BATCH_SIZE."

g = torch.Generator().manual_seed(SEED)                                   # makes the shuffling reproducible
make_ds = lambda X, y: TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_loader = DataLoader(make_ds(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=g)   # drop_last: BatchNorm needs > 1 sample per batch
val_loader = DataLoader(make_ds(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(make_ds(X_test, y_test), batch_size=64, shuffle=False)

print(f"{'label':>5}  {'class':<12}{'train':>6}{'val':>6}{'test':>6}")
for i, c in enumerate(CLASS_NAMES):
    print(f"{i:>5}  {c:<12}{(y_train == i).sum():>6}{(y_val == i).sum():>6}{(y_test == i).sum():>6}")
print(f"{'':>5}  {'TOTAL':<12}{len(y_train):>6}{len(y_val):>6}{len(y_test):>6}")

xb, yb = next(iter(train_loader))
print(f"\none training batch:  points {tuple(xb.shape)} {xb.dtype} | labels {tuple(yb.shape)} {yb.dtype}")

# %% [markdown] id="s6"
# ## 6. The PointNet model
#
# **Why a special network?** A point cloud is an *unordered set*: shuffling the rows of an `(N, 3)` array describes the same object. A normal network (or an RNN) would care about the order. PointNet is built so the order **cannot** matter:
#
# 1. A **shared MLP** transforms **each point independently with the same weights** (3 → 64 → 64 → 64 → 128 → 1024 features per point). In code this is `Conv1d` with `kernel_size=1`, which is just a `Linear` layer applied to every point.
# 2. A **global max pool** takes, for each of the 1024 feature channels, the maximum over all points. Max does not care about order, and it squeezes the whole cloud into **one 1024-number "shape fingerprint"**.
# 3. A normal **classifier head** (`1024 → 512 → 256 → 10`, with dropout) turns the fingerprint into 10 class scores (*logits*).
#
# ```
# (B,N,3) ─transpose→ (B,3,N) ─MLP 3→64→64→ (B,64,N) ─MLP 64→64→128→1024→ (B,1024,N) ─max over N→ (B,1024) ─FC→ (B,512) → (B,256) → (B,10)
# ```
# Conv1d wants channels first, hence the transpose: `(batch, points, xyz)` → `(batch, xyz, points)`.
#
# ### T-Net (the "transformation network") and what we simplified
# A **T-Net** is a small PointNet that looks at the cloud and predicts a matrix (3×3 for raw points, 64×64 for features) that is multiplied into the data. Idea: **align** every object to a canonical pose before the main network looks at it, so the result depends less on how the object is rotated. The matrix starts as the identity ("do nothing") and is learned. For the 64×64 version the paper adds a penalty `‖I − A·Aᵀ‖²` that keeps the matrix close to a rotation/reflection, because a 64×64 matrix has many more ways to go wrong.
#
# | | Original PointNet (classification) | This notebook |
# |---|---|---|
# | Input T-Net 3×3 | yes | **optional, off in the baseline** (`USE_INPUT_TNET`) |
# | Feature T-Net 64×64 + orthogonality penalty | yes (weight 0.001; the penalty is `tf.nn.l2_loss`, i.e. summed over the batch and halved) | **optional, off in the baseline** (`USE_FEATURE_TNET`); same penalty definition |
# | Shared MLP 3-64-64 and 64-128-1024, max pool, FC 512-256, dropout keep 0.7 | yes | same |
# | Dataset | ModelNet40 (ready-made HDF5 point clouds, first 1024 of 2048 points used) | ModelNet10 meshes; we sample the 1024 points ourselves (smaller, faster) |
# | Augmentation | random rotation about the up axis + jitter (σ 0.01, clipped at 0.05) | same (the up axis is z here) |
# | Optimizer & schedule | Adam, lr 0.001, exponential decay (rate 0.7), batch 32, **250 epochs**, BatchNorm-momentum schedule | Adam + weight decay, cosine LR, batch 32, 5 (quick) or 30 epochs, default BatchNorm momentum |
# | Segmentation branch | yes (a separate task) | omitted |
#
# *The "original" column was checked against the authors' code (`charlesq34/pointnet`: `models/pointnet_cls.py`, `models/transform_nets.py`, `train.py`, `provider.py`), not against memory.*
#
# *Why start without T-Nets?* ModelNet10 objects are already orientation-aligned and we add rotation augmentation, so the baseline is a fair, fast first experiment. T-Nets cost extra parameters and time. Turning them on is a one-line change, so you can measure what they give you instead of taking it on trust.

# %% id="model"
def conv_bn_relu(in_ch, out_ch):
    """One 'shared MLP' layer: the same weights are applied to every point. (B,in,N) -> (B,out,N)"""
    return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))

def fc_bn_relu(in_f, out_f):
    return nn.Sequential(nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.ReLU(inplace=True))

class TNet(nn.Module):
    """Predicts a k x k matrix from a (B, k, N) input. Starts out as the identity matrix."""
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.shared = nn.Sequential(conv_bn_relu(k, 64), conv_bn_relu(64, 128), conv_bn_relu(128, 1024))   # per point
        self.head = nn.Sequential(fc_bn_relu(1024, 512), fc_bn_relu(512, 256))
        self.out = nn.Linear(256, k * k)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)     # so the output starts as exactly "identity"

    def forward(self, x):                                    # x: (B, k, N)
        g = self.shared(x).max(dim=2).values                 # (B, 1024)  global feature
        m = self.out(self.head(g)).view(-1, self.k, self.k)  # (B, k, k)  learned correction
        return m + torch.eye(self.k, device=x.device, dtype=x.dtype)   # identity + correction

class PointNet(nn.Module):
    def __init__(self, num_classes=10, input_transform=False, feature_transform=False, dropout=0.3):
        super().__init__()
        self.input_tnet = TNet(3) if input_transform else None
        self.mlp1 = nn.Sequential(conv_bn_relu(3, 64), conv_bn_relu(64, 64))
        self.feature_tnet = TNet(64) if feature_transform else None
        self.mlp2 = nn.Sequential(conv_bn_relu(64, 64), conv_bn_relu(64, 128), conv_bn_relu(128, 1024))
        self.classifier = nn.Sequential(
            fc_bn_relu(1024, 512), nn.Dropout(dropout),
            fc_bn_relu(512, 256), nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, points, verbose=False):
        """points: (B, N, 3)  ->  logits (B, num_classes), aux dict (T-Net matrices, or None)."""
        say = (lambda name, t: print(f"  {name:<40s}{str(tuple(t.shape))}")) if verbose else (lambda name, t: None)
        aux = {"input_matrix": None, "feature_matrix": None}
        say("input points (B,N,3)", points)
        x = points.transpose(1, 2)                           # channels first for Conv1d
        say("transpose -> (B,3,N)", x)
        if self.input_tnet is not None:
            aux["input_matrix"] = self.input_tnet(x)         # (B,3,3)
            x = torch.bmm(aux["input_matrix"], x)            # align the raw points
            say("after input T-Net (3x3 applied)", x)
        x = self.mlp1(x);                                    say("shared MLP 3->64->64", x)
        if self.feature_tnet is not None:
            aux["feature_matrix"] = self.feature_tnet(x)     # (B,64,64)
            x = torch.bmm(aux["feature_matrix"], x)          # align the 64-d features
            say("after feature T-Net (64x64 applied)", x)
        x = self.mlp2(x);                                    say("shared MLP 64->64->128->1024", x)
        x = x.max(dim=2).values;                             say("global max pool over the N points", x)
        logits = self.classifier(x);                         say("classifier 1024->512->256->classes", logits)
        return logits, aux

def feature_transform_regularizer(m):
    """||I - m m^T||^2: small when m is close to an orthogonal matrix.
    Summed over the whole batch and halved, which is what tf.nn.l2_loss does in the original code."""
    eye = torch.eye(m.size(1), device=m.device, dtype=m.dtype).unsqueeze(0)
    return 0.5 * ((eye - torch.bmm(m, m.transpose(1, 2))) ** 2).sum()

def compute_loss(logits, labels, aux, reg_weight):
    """Returns (loss_to_backprop, cross_entropy).
    loss_to_backprop = cross-entropy (+ the T-Net orthogonality penalty when the feature T-Net is on)."""
    ce = F.cross_entropy(logits, labels)
    loss = ce
    if aux["feature_matrix"] is not None:
        loss = ce + reg_weight * feature_transform_regularizer(aux["feature_matrix"])
    return loss, ce

MODEL_KWARGS = dict(num_classes=NUM_CLASSES, input_transform=USE_INPUT_TNET, feature_transform=USE_FEATURE_TNET, dropout=DROPOUT)

# %% id="shape_trace"
# Follow one real batch through the network and print the tensor shape after every major stage.
trace_model = PointNet(**MODEL_KWARGS).to(device).eval()
with torch.no_grad():
    logits, _ = trace_model(xb[:4].to(device), verbose=True)
print(f"\nparameters: {sum(p.numel() for p in trace_model.parameters()):,}   (T-Nets: input={USE_INPUT_TNET}, feature={USE_FEATURE_TNET})")
del trace_model

# %% [markdown] id="s7"
# ## 7. Smoke test (run this before any training)
# A smoke test is a quick "does the machine turn on?" check. It uses a **fresh copy** of the model (so it never disturbs the real one) and a **real batch** from the training data. It checks:
#
# * input shapes/dtypes and that the data is normalized and finite
# * forward pass gives logits of shape `(B, 10)`
# * the loss is finite and near `ln(10) ≈ 2.30` (an untrained 10-class model is basically guessing)
# * backpropagation gives finite gradients
# * the model can **overfit one small batch** (loss goes down over a few dozen steps; if it can't, something is broken)
# * **permutation invariance**: shuffling the points does not change the output (the key PointNet property)
# * if T-Nets are on: they start as the identity matrix
#
# Training refuses to start unless this passes.

# %% id="smoke_funcs"
def run_smoke_test(model_kwargs, device, batch, reg_weight, overfit_steps=40):
    K = model_kwargs["num_classes"]
    results = []
    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))

    x, y = batch[0][:8].to(device), batch[1][:8].to(device)
    B, N, _ = x.shape
    print(f"smoke test on a batch of {B} clouds x {N} points")

    # --- data checks ---
    check("points shape (B,N,3) & float32", x.ndim == 3 and x.shape[2] == 3 and x.dtype == torch.float32, f"{tuple(x.shape)} {x.dtype}")
    check("points finite", torch.isfinite(x).all().item())
    radius = x.norm(dim=2).max().item(); centroid = x.mean(dim=1).abs().max().item()
    check("clouds normalized (max radius <= 1, centroid ~ 0)", radius <= 1 + 1e-4 and centroid < 1e-3, f"max radius {radius:.4f}, |centroid| {centroid:.1e}")
    check("labels int64 and in [0, K)", y.dtype == torch.long and y.min().item() >= 0 and y.max().item() < K, f"{y.dtype}, range [{y.min().item()}, {y.max().item()}]")

    torch.manual_seed(0)
    model = PointNet(**model_kwargs).to(device)          # fresh model, only used here

    # --- T-Nets start as identity (eval mode, before any training) ---
    model.eval()
    with torch.no_grad():
        logits, aux = model(x)
    for name, k in (("input_matrix", 3), ("feature_matrix", 64)):
        if aux[name] is not None:
            check(f"{name} starts as identity ({k}x{k})", torch.allclose(aux[name], torch.eye(k, device=device).expand(B, k, k), atol=1e-6))

    # --- forward ---
    check("logits shape (B, K)", tuple(logits.shape) == (B, K), f"{tuple(logits.shape)}")
    check("logits finite", torch.isfinite(logits).all().item())

    # --- permutation invariance (eval mode) ---
    perm = torch.randperm(N, device=device)
    with torch.no_grad():
        logits_perm, _ = model(x[:, perm])
    diff = (logits - logits_perm).abs().max().item()
    check("permutation invariance (shuffled points -> same logits)", diff < 1e-4, f"max |diff| = {diff:.2e}")

    # --- loss + backward ---
    model.train()
    logits, aux = model(x)
    loss, ce = compute_loss(logits, y, aux, reg_weight)
    check("loss is a finite scalar", loss.ndim == 0 and torch.isfinite(loss).item(), f"{loss.item():.4f}")
    check("initial cross-entropy near ln(K) (guessing)", 0.3 * math.log(K) < ce.item() < 4 * math.log(K), f"CE {ce.item():.3f}, ln(K) = {math.log(K):.3f}")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    check("every parameter received a gradient", all(g is not None for g in grads))
    check("gradients finite and not all zero", all(torch.isfinite(g).all().item() for g in grads if g is not None) and sum(g.abs().sum().item() for g in grads if g is not None) > 0)

    # --- overfit one batch ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first, history = None, []
    for _ in range(overfit_steps):
        opt.zero_grad()
        lg, ax = model(x)
        l, _ = compute_loss(lg, y, ax, reg_weight); l.backward(); opt.step()
        first = l.item() if first is None else first
        history.append(l.item())
    best_late = min(history[-5:])
    check(f"can overfit one batch ({overfit_steps} steps)", best_late < 0.7 * first, f"loss {first:.3f} -> {best_late:.3f}")

    ok = all(results)
    print("SMOKE TEST " + ("PASSED" if ok else "FAILED") + f" ({sum(results)}/{len(results)} checks)")
    return ok

# %% id="smoke_run"
SMOKE_TEST_PASSED = run_smoke_test(MODEL_KWARGS, device, batch=(xb, yb), reg_weight=FEATURE_REG_WEIGHT)
assert SMOKE_TEST_PASSED, "Smoke test failed: fix the failing check(s) above before training."

# %% [markdown] id="s8"
# ## 8. Training
# One **epoch** = one pass over all training clouds. For every batch of 32 clouds:
#
# 1. **Forward:** `logits, aux = model(points)` → `(32, 10)` raw class scores.
# 2. **Loss:** `cross_entropy(logits, labels)` turns the scores into one number that is small when the correct class gets the highest score. (With the feature T-Net on, the orthogonality penalty is added for the weight update, but the *logged* training loss is the plain cross-entropy so it stays comparable to the validation loss.)
# 3. **Backward:** `loss.backward()` computes how each weight should change (backpropagation).
# 4. **Step:** `optimizer.step()` (Adam) nudges all weights; `optimizer.zero_grad()` clears the old gradients first.
#
# After each epoch we measure **validation** accuracy (no weight updates, `model.eval()` so dropout is off and BatchNorm uses its running statistics). The epoch with the **best validation accuracy** is saved to disk as the "best model". The test set is not used here.
#
# **Metrics:** *overall accuracy* = fraction of correct predictions. *Mean class accuracy* = the average of the 10 per-class accuracies; it matters because the classes are imbalanced.
# **Augmentation (training batches only):** a random spin around the up axis, plus a tiny random jitter of every point.

# %% id="train_funcs"
def augment_batch(points, jitter_sigma=0.01, jitter_clip=0.05):
    """points (B,N,3) on the GPU/CPU -> same shape. Random rotation about z (the up axis) + clipped Gaussian jitter."""
    B = points.size(0)
    theta = torch.rand(B, device=points.device) * 2 * math.pi
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.zeros(B, 3, 3, device=points.device)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    points = torch.bmm(points, R.transpose(1, 2))                       # rotate every point: p' = R p
    return points + (jitter_sigma * torch.randn_like(points)).clamp(-jitter_clip, jitter_clip)

def train_one_epoch(model, loader, optimizer, device, reg_weight, augment=True):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for points, labels in loader:                                        # points (B,N,3), labels (B,)
        points, labels = points.to(device), labels.to(device)
        if augment:
            points = augment_batch(points)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(points)                                      # (B,K)
        loss, ce = compute_loss(logits, labels, aux, reg_weight)
        loss.backward()
        optimizer.step()
        total_loss += ce.item() * labels.size(0)                         # we LOG plain cross-entropy so it is comparable with the validation loss
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total_loss, n = 0.0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)            # rows = true class, columns = predicted class
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        logits, _ = model(points)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        n += labels.size(0)
        np.add.at(cm, (labels.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()), 1)
    support = cm.sum(axis=1)
    per_class = np.diag(cm) / np.maximum(support, 1)
    return {"loss": total_loss / n, "acc": np.trace(cm) / cm.sum(), "mean_class_acc": per_class[support > 0].mean(),
            "per_class_acc": per_class, "support": support, "confusion": cm}

def fit(model, train_loader, val_loader, epochs, lr, weight_decay, reg_weight, device, ckpt_path, meta):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time(); lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, reg_weight)
        val = evaluate(model, val_loader, device, meta["model_kwargs"]["num_classes"])
        scheduler.step()
        for k, v in zip(hist, (tr_loss, tr_acc, val["loss"], val["acc"])):
            hist[k].append(float(v))
        is_best = val["acc"] > best_val
        if is_best:
            best_val = val["acc"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_acc": float(val["acc"]), **meta}, ckpt_path)
        print(f"epoch {epoch:3d}/{epochs} | lr {lr_now:.5f} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val['loss']:.4f} acc {val['acc']:.3f} | {time.time() - t0:5.1f}s" + ("  <- best, saved" if is_best else ""))
    return hist, best_val

# %% id="train_run"
assert SMOKE_TEST_PASSED, "Run the smoke test (section 7) first and make sure it passes."
set_seed(SEED)
model = PointNet(**MODEL_KWARGS).to(device)
print(f"training {sum(p.numel() for p in model.parameters()):,} parameters on {len(X_train)} clouds, validating on {len(X_val)}\n")

history, best_val_acc = fit(
    model, train_loader, val_loader, EPOCHS, LR, WEIGHT_DECAY, FEATURE_REG_WEIGHT, device, CKPT_PATH,
    meta={"model_kwargs": MODEL_KWARGS, "class_names": CLASS_NAMES, "num_points": NUM_POINTS})
print(f"\nbest validation accuracy: {best_val_acc:.3f}  (model saved to {CKPT_PATH})")

epochs_axis = np.arange(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(epochs_axis, history["train_loss"], label="train"); axes[0].plot(epochs_axis, history["val_loss"], label="validation"); axes[0].set_title("cross-entropy loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(epochs_axis, history["train_acc"], label="train"); axes[1].plot(epochs_axis, history["val_acc"], label="validation"); axes[1].set_title("accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown] id="s9"
# ## 9. Evaluation on the test set
# Now (and only now) we load the **best-validation checkpoint** and score it on the official test split. We report overall accuracy, mean class accuracy, per-class accuracy and a confusion matrix (row = true class, column = predicted class; a perfect model has only the diagonal filled).
# If you train several variants, **choose between them by validation accuracy**, and use the test number just to report the final result.

# %% id="eval"
ckpt = torch.load(CKPT_PATH, map_location=device)
best_model = PointNet(**ckpt["model_kwargs"]).to(device)
best_model.load_state_dict(ckpt["model_state"])
print(f"loaded best model: epoch {ckpt['epoch']}, validation accuracy {ckpt['val_acc']:.3f}")

test = evaluate(best_model, test_loader, device, NUM_CLASSES)
print(f"\nTEST  overall accuracy: {test['acc']:.3f} | mean class accuracy: {test['mean_class_acc']:.3f} | loss {test['loss']:.4f}   ({len(y_test)} test clouds, mode={TAG})")
print(f"\n{'class':<12}{'acc':>7}{'n':>6}")
for i, c in enumerate(CLASS_NAMES):
    print(f"{c:<12}{test['per_class_acc'][i]:>7.3f}{test['support'][i]:>6}")

cm = test["confusion"]
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im)
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test)")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
plt.tight_layout(); plt.show()

# a few individual predictions
best_model.eval()
idx = np.random.default_rng(0).choice(len(X_test), size=min(8, len(X_test)), replace=False)
with torch.no_grad():
    pred = best_model(torch.from_numpy(X_test[idx]).to(device))[0].argmax(dim=1).cpu().numpy()
print("\nsample predictions (true -> predicted):")
for i, p in zip(idx, pred):
    print(f"  test #{i:<4d} {CLASS_NAMES[y_test[i]]:<12s} -> {CLASS_NAMES[p]:<12s} {'ok' if p == y_test[i] else 'WRONG'}")

# %% [markdown] id="s10"
# ## 10. Limits and what to try next
# **Be careful with what you conclude**
# * `QUICK_MODE` uses a subset and 5 epochs. Its accuracy only says "the code runs", not "PointNet works".
# * One run with one seed is one data point. Differences of a percent or so between two runs can be noise. For a fair T-Net comparison, repeat each variant with 2-3 different `SEED` values.
# * Point sampling is random but fixed (cached) per mesh. Re-sampling each epoch would be an extra augmentation we did not implement.
# * Because we normalize to a unit sphere, real-world size cues are discarded.
#
# **Suggested experiments, one change at a time**
# 1. Baseline, `QUICK_MODE = False` (this notebook as is).
# 2. `USE_INPUT_TNET = True` only. Then `USE_FEATURE_TNET = True` only. Then both. Compare *validation* accuracy and the time per epoch.
# 3. More epochs (e.g. 60) or `NUM_POINTS = 2048` (delete the cache file first, since it is keyed by point count and mode).
# 4. Look at the confusion matrix: which classes get mixed up (e.g. desk vs table, dresser vs night_stand)? Does that match your intuition?
# 5. Class-weighted loss, because the classes are imbalanced.
