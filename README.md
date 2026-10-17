# PointNet 3D classification on ModelNet10 (Google Colab)

A beginner-friendly, self-contained PyTorch implementation of **PointNet** (Qi et al., CVPR 2017) that classifies 3D objects, stored as point clouds, into 10 furniture categories. The whole project is one notebook: [`PointNet_ModelNet10.ipynb`](PointNet_ModelNet10.ipynb). Open3D is used to look at the data.

> **Read this first: honest status.** Nothing has been trained on a GPU or on the full dataset yet, so **this project contains no accuracy number**. The notebook ships with no stored outputs. Every number you see will come from your own run. What *was* checked, and what was not, is listed in [Status](#status-what-was-and-was-not-tested).

---

## Quick start (Colab)

1. Go to <https://colab.research.google.com> → **File → Upload notebook** → pick `PointNet_ModelNet10.ipynb`.
2. **Runtime → Change runtime type → T4 GPU.** The free tier is enough; **nothing here needs a paid plan or paid compute units.** If Colab offers an upgrade, ignore it.
3. Leave `QUICK_MODE = True` (section 1) and **Runtime → Run all.** This downloads the dataset once (473 MB), uses a subset (about 1,000 train + 300 test meshes) and trains 5 epochs. It proves every stage works. *Its accuracy is not meaningful.*
4. If the smoke test (section 7) passes and the plots look sane, set `QUICK_MODE = False` and run all again for the real experiment (30 epochs on all 4,899 meshes).
5. At the end of every run the notebook saves a **`run_summary.json`** plus `curves.png` and `confusion.png` and (on Colab) downloads them as a small zip. Put them in [`results/`](results/) and commit them. That file is the only source for numbers you quote.
6. Only then try one change at a time (T-Nets, `NUM_POINTS`, seeds); see [Reproducing experiments](#reproducing-experiments).

The notebook prints the time per epoch, so after the first epoch you can extrapolate the total. Colab runtimes are temporary: set `USE_GOOGLE_DRIVE = True` if you want the cache and the best model to survive a reset (Colab will ask you to authorise Drive access).

## What the notebook does, section by section

| # | Section | Output you will see |
|---|---|---|
| 1 | Setup: install Open3D, imports, **config cell** (the only cell you normally edit) | GPU name, config summary |
| 2 | Dataset: download, extract, filter junk files, list classes | class table: label / name / #train / #test |
| 3 | Mesh → point cloud: OFF reader, area-weighted surface sampling, unit-sphere normalization | shapes `(V,3)` → `(F,3)` → `(1024,3)` |
| 4 | Visualization: Open3D `draw_plotly` for one mesh + its point cloud; matplotlib gallery, one object per class | interactive 3D view, 10-class gallery |
| 5 | Preprocess all meshes once (cached), stratified train/val split, label encoding, DataLoaders | per-class train/val/test counts, batch shapes `(32,1024,3)`, `(32,)` |
| 6 | PointNet model with optional T-Nets | shape at every stage, parameter count |
| 7 | **Smoke test** (gates training) | PASS/FAIL list of 12–14 checks |
| 8 | Training loop: cross-entropy, backprop, Adam, best-on-validation checkpoint | one line per epoch, loss/accuracy curves |
| 9 | Evaluation on the official test set, then **save the run** | overall + mean-class accuracy, per-class table, confusion matrix, sample predictions; `run_summary.json` + figures + zip |
| 10 | Limits and suggested experiments | |

### Tensor shapes (B = batch size, N = 1024 points)

```
mesh file → vertices (V,3) + faces (F,3) → sampled (N,3) → normalized (N,3) → batch (B,N,3)
model:  (B,N,3) → transpose (B,3,N) → shared MLP 3→64→64 (B,64,N) → [feature T-Net] → shared MLP 64→64→128→1024 (B,1024,N)
        → global max pool over N (B,1024) → FC 512 → FC 256 → logits (B,10)        labels: (B,) int64
```

## Dataset: ModelNet10 (verified 2026-10-16)

| Fact | Value | How it was checked |
|---|---|---|
| Official page | <https://modelnet.cs.princeton.edu/download.html> | page fetched; it lists `ModelNet10.zip` |
| File | `http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip` (redirects to https) | HTTP HEAD: **200, `application/zip`, 473,402,300 bytes**, Last-Modified 21 Aug 2014 |
| Format | **OFF** triangle meshes (not point clouds); orientation manually aligned (scale is not) | page text + 70 real files parsed |
| Contents | **4,899 meshes**, 3,991 train + 908 test, 10 classes | zip index read via HTTP Range requests: counts per class and split match the official split |
| Classes | bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet | same zip index |
| Up axis | **z** | bounding boxes of 70 real files + the gallery plot (chairs stand on their legs) |
| License | **None stated.** The page says the CAD models' original authors hold the copyright and the dataset is "provided for the convenience of academic research only". | page text |
| Citation | Z. Wu et al., *3D ShapeNets: A Deep Representation for Volumetric Shapes*, CVPR 2015 | page text |

We **do not redistribute** the data; the notebook downloads it directly from Princeton. Use it for academic/learning purposes and cite the paper.

Gotchas found while checking the zip, both handled in the notebook:
* The zip has **9,798 entries ending in `.off`, but only 4,899 are meshes.** The other half are macOS metadata files such as `__MACOSX/ModelNet10/bathtub/test/._bathtub_0107.off`, one per mesh. A naive `glob("*.off")` would try to parse them and crash.
* The classes are imbalanced (train: chair 889, bed 515, monitor 465, sofa 680 … bathtub 106). We therefore also report **mean class accuracy**.

**Alternatives considered:** ModelNet40 (`ModelNet40.zip`, 2,039,180,837 bytes by the same HEAD check) is 4× larger. The ready-made point-cloud HDF5 files used by the original PointNet (`shapenet.cs.stanford.edu/.../modelnet40_ply_hdf5_2048.zip`) could **not** be verified: my connection to that host timed out, so I did not use it. A "ModelNet10 – Scans" record on Zenodo is a 43 GB scanned derivative, far too large for a first experiment.

## Reproducing experiments

Every run is fully described by the config cell. The settings that define a run are `QUICK_MODE`, `NUM_POINTS`, `SEED`, `USE_INPUT_TNET`, `USE_FEATURE_TNET`, `EPOCHS`, `BATCH_SIZE`, `LR`, `WEIGHT_DECAY`, `DROPOUT`, `FEATURE_REG_WEIGHT`, `VAL_FRACTION` and `DETERMINISTIC`, and all of them are written into `run_summary.json`.

* **Seeds:** `SEED` controls point sampling (the cache name includes it), the validation split, model initialization, shuffling and augmentation. `DETERMINISTIC = True` asks cuDNN for deterministic kernels; GPU runs can still differ slightly, so compare variants across 2–3 seeds before claiming a difference.
* **One folder per run:** `outputs/runs/<mode>_<variant>_n<points>_seed<seed>_<time>/` holds `best_model.pt`, `run_summary.json`, `curves.png` and `confusion.png`. Nothing is overwritten.
* **Protocol:** quick mode (wiring only) → full baseline → one change at a time → **choose on validation**, report the official test split once per final model. Never tune on test accuracy.
* **What to commit:** the small zip's contents into `results/<run name>/`. Datasets, caches (`.npz`) and checkpoints (`.pt`) stay out of Git (see `.gitignore`).

## Results

**Not measured yet.** This table is filled only from committed `results/*/run_summary.json` files.

| Run | Variant | Points | Seed | Best val acc | Test acc | Test mean class acc | Time/epoch | GPU |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — |

## Run locally (tests only, CPU is fine)

```bash
python -m pip install -r requirements-dev.txt
python dev/test_units.py                                   # 13 unit tests of the notebook's functions
python dev/fetch_sample.py                                 # once: ~70 real meshes (~6 MB transfer) into dev/sample/ (git-ignored)
python dev/run_local_test.py --sample-dir dev/sample --workdir ../pn_test_run   # all non-Colab cells end to end; needs a NEW empty dir
python dev/check_learning_notes.py                         # learning notes vs real code
```
The local end-to-end run uses 40 training clouds, so its accuracy is chance level by design: it proves the wiring, not the model.

## Model: what is original, what is simplified

The "original" column was checked against the authors' code ([`charlesq34/pointnet`](https://github.com/charlesq34/pointnet): `models/pointnet_cls.py`, `models/transform_nets.py`, `train.py`, `provider.py`), not against memory.

| | Original PointNet (classification) | This project |
|---|---|---|
| Per-point features | shared MLP 3→64→64, feature T-Net, shared MLP 64→64→128→1024 | same layers |
| Aggregation | global max pool | same |
| Head | FC 512 → 256 → classes, BatchNorm, dropout keep-prob 0.7 | same (`DROPOUT = 0.3` = drop-prob) |
| Input T-Net (3×3) | yes | **optional, off in the baseline** |
| Feature T-Net (64×64) + penalty `‖I − AAᵀ‖²`, weight 0.001 | yes (`tf.nn.l2_loss`: summed over the batch, halved) | **optional, off in the baseline**; same penalty definition |
| Data | ModelNet40, ready-made 2048-point clouds (first 1024 used) | ModelNet10 meshes; we sample 1024 points ourselves |
| Augmentation | random rotation about the up axis + jitter (σ 0.01, clip 0.05) | same (up axis = z) |
| Optimization | Adam lr 0.001, exponential decay (0.7), batch 32, 250 epochs, BN-momentum schedule | Adam + weight decay 1e-4, cosine LR, batch 32, 5 or 30 epochs, default BN momentum |
| Segmentation branch | yes | omitted (not needed) |

**What the T-Net is for:** a T-Net is a mini-PointNet that predicts a matrix (3×3 for raw points, 64×64 for features) which is multiplied into the data, to *align* every object to a canonical pose so the classifier depends less on how the object is rotated. It starts as the identity matrix and is learned; the 64×64 version is kept near-orthogonal by the penalty. **What we omit in the baseline:** both T-Nets and, with them, the penalty. Why that is a reasonable first step: ModelNet10 is already orientation-aligned and we augment with rotations about the up axis. Whether the T-Nets help *here* is for you to measure (suggested in section 10 of the notebook). Parameter counts printed by the notebook: baseline **811,914**, both T-Nets on **3,472,339**.

Pipeline choices (all documented in the notebook): 1024 points per object sampled once (area-weighted, fixed seed) and cached; centroid at the origin and farthest point at distance 1; classes encoded alphabetically as 0–9; **official test split used only for the final evaluation**; validation = stratified 10% of the official train split (fixed seed), used to pick the best epoch.

## Status: what was and was not tested

Tested on a Windows PC (CPU only, Python 3.12, PyTorch 2.13, Open3D 0.20.0), using **70 real ModelNet10 files** fetched individually from the official zip with `dev/fetch_sample.py` (5 train + 2 test per class):

| Check | Result |
|---|---|
| Dataset URL, size, class names, per-class train/test counts, license text, OFF format | verified (table above) |
| Notebook code cells executed top to bottom (everything except the Colab-only `pip` cell): the notebook's own download (from a local server standing in for Princeton), extraction, junk-file filtering, parsing, sampling, caching, loaders, model, smoke test, training, evaluation, checkpoint save/load | **ran without errors**, for the baseline and with both T-Nets on, with Open3D and with Open3D blocked (matplotlib fallback), and twice in a row (second run uses the caches) |
| Smoke test | passes (12/12 checks baseline, 14/14 with both T-Nets), and passed on 12 different real batches (6 per variant), so it is not flaky. Deliberately broken variants (non-invariant pooling, no identity init, un-normalized data, unweighted sampling) are **caught** |
| 13 unit tests of the core functions (OFF parser incl. glued headers and quads, area-weighted sampling, normalization, augmentation, stratified split, metrics, regularizer, checkpoint round trip, run summary JSON) | pass |
| Run record: each local run writes its own folder with `best_model.pt`, `run_summary.json` (config, environment, per-epoch history and timing, test metrics), `curves.png`, `confusion.png`, and a zip without the checkpoint | checked by `dev/run_local_test.py` |
| Open3D: `read_triangle_mesh` on `.off`, `PointCloud` creation, `draw_plotly` figure construction | works on Open3D 0.20.0 (Windows wheel) |
| Learning on the 70-file sample | accuracy is chance level (10%): there are only 40 training clouds, so this shows the wiring works, **not** that PointNet works. Do not read anything into it |

**Not tested (be aware):**
* Anything on **Colab**: the `pip install open3d-cpu` cell (I only confirmed on PyPI that `open3d-cpu` is published by the Open3D team, has Linux x86-64 wheels for CPython 3.12 and 3.13 at about 100 MB (the package lists 3.10–3.14), and has no upper-bound dependency pins), the interactive plotly rendering inside Colab, the real 473 MB download from Princeton (only the HEAD request), Drive mounting.
* **GPU** execution (the code is device-aware, but I had no CUDA device).
* Training on the **full or quick-mode dataset**, hence no accuracy, no timing on Colab, and no verification that the QUICK/full settings give sensible results.
* The visual look of `draw_plotly` output (only that the figures build without errors).
* Whether any file in the full 4,899 has an unusual layout beyond the 70 sampled (the parser handles glued headers, polygons and comments, and unreadable files are reported and skipped rather than hidden).

Rough local timings (my CPU; your Colab numbers will differ): point-cloud preprocessing ≈ 17–23 meshes/s, so roughly **4–6 minutes for all 4,899 meshes** (extrapolated); one training step of 32 clouds ≈ 0.8–1 s on CPU for the baseline and ≈ 2.3 s with both T-Nets. A full run on CPU is therefore slow; use the GPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Download fails or times out | Download `ModelNet10.zip` from the official page yourself, upload it to `/content/pointnet_modelnet10/data/`, and re-run the dataset cell (it skips the download when the file size matches). |
| `Open3D could not be imported` | Harmless: visualization falls back to matplotlib. |
| `pip` prints dependency warnings after installing Open3D | Usually harmless. If a later import fails, use **Runtime → Restart session** and run again. |
| "no GPU found" | Runtime → Change runtime type → T4 GPU, then run all again. |
| CUDA out of memory | Lower `BATCH_SIZE` (e.g. 16). |
| `Cache does not match the config` | Delete that `.npz` file in the outputs folder (its name contains points, quick/full mode and seed). |
| Smoke test fails | Read the `[FAIL]` line: each check says what it verifies. Do not train until it passes. |

## Project structure

```
pointnet-modelnet10/
├── PointNet_ModelNet10.ipynb   # the project: upload this to Colab
├── README.md
├── requirements-dev.txt        # only for the local tests (Colab needs nothing extra)
├── learning/                   # 9 study notes: architecture, syntax, training, T-Nets, evaluation, interview prep
├── results/                    # committed run summaries + figures from REAL runs only
└── dev/                        # for maintainers (not needed to run the notebook)
    ├── notebook_source.py      # source of truth for the notebook (readable "percent" format)
    ├── build_notebook.py       # notebook_source.py -> PointNet_ModelNet10.ipynb   (needs: nbformat)
    ├── test_units.py           # unit tests of the notebook's functions
    ├── fetch_sample.py         # downloads ~70 real meshes for the local test (into git-ignored dev/sample/)
    ├── run_local_test.py       # runs the notebook's code cells locally on that sample
    └── check_learning_notes.py # checks the learning notes against the real code
```

To change the notebook, edit `dev/notebook_source.py`, run `python dev/build_notebook.py`, re-run the tests, and commit both files. Never hand-edit the `.ipynb`.

## References (all opened while building this)

* Qi, Su, Mo, Guibas. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR 2017. <https://arxiv.org/abs/1612.00593>
* Authors' code (TensorFlow): <https://github.com/charlesq34/pointnet>
* ModelNet download page: <https://modelnet.cs.princeton.edu/download.html>
* Wu et al., *3D ShapeNets: A Deep Representation for Volumetric Shapes*, CVPR 2015 (cite this when you use ModelNet).
* Open3D `draw_plotly` in notebooks: <https://www.open3d.org/docs/latest/tutorial/visualization/draw_plotly.html>
* Open3D install options incl. `open3d-cpu`: <https://www.open3d.org/html/getting_started.html>
