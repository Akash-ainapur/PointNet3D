# 07 · Code walkthrough: concept → cell → function → shapes

Everything lives in [`dev/notebook_source.py`](../dev/notebook_source.py). The notebook is *built* from it (`python dev/build_notebook.py`), so **never hand-edit the `.ipynb`**. Cells are marked `# %% id="…"`; search for the id to jump there.

## 1. The whole pipeline on one screen

```
config ─▶ download ─▶ explore ─▶ pc_funcs ─▶ preprocess ─▶ split ─▶ model ─▶ smoke_run ─▶ train_run ─▶ eval
 knobs     473 MB zip  labels     mesh→(1024,3)  cache .npz   loaders   PointNet   gate        fit()        test once
```

## 2. Section by section

| § | Cell id → function | Input → output | One line to point at | Note |
|---|---|---|---|---|
| 1 | `config` | — | `QUICK_MODE`, `USE_INPUT_TNET`, `USE_FEATURE_TNET`, `SEED`, `set_seed(SEED)` | 06 |
| 2 | `download → ensure_dataset` | URL → 4,899 `.off` files | `members = [... not Path(m).name.startswith("._")]`: skip macOS junk | 01 |
| 2 | `explore → collect_items` | folders → `[(path, class)]` | `CLASS_TO_IDX = {name: i ...}`: alphabetical label encoding | 01 |
| 3 | `pc_funcs → read_off` | file → `verts (V,3)`, `faces (F,3)` | the glued-header branch `counts, body = head[3:], lines[1:]` | 01 |
| 3 | `pc_funcs → sample_points_from_mesh` | `(V,3),(F,3)` → `(N,3)` | `rng.choice(len(faces), size=n_points, p=area / total)` | 01 |
| 3 | `pc_funcs → normalize_point_cloud` | `(N,3)` → `(N,3)` in unit sphere | `pts - pts.mean(axis=0, keepdims=True)` | 01 |
| 4 | `viz → show_point_cloud`, `gallery` | `(N,3)` → interactive plot | `o3d.utility.Vector3dVector(...)` | 01 |
| 5 | `preprocess → build_arrays` | list of meshes → `X (M,N,3)`, `y (M,)` | `np.savez_compressed(CACHE_PATH, ...)`: preprocess once | 01 |
| 5 | `split → stratified_split` | `y` → train/val indices | `n_val = min(max(1, int(round(len(idx) * val_fraction))), len(idx) - 1)` | 01, 06 |
| 5 | `split` | arrays → `DataLoader`s | `DataLoader(..., shuffle=True, drop_last=True, generator=g)` | 04 |
| 6 | `model → PointNet.forward` | `(B,N,3)` → logits `(B,10)`, `aux` | `x = x.max(dim=2).values`: the symmetric function | 03 |
| 6 | `model → TNet.forward` | `(B,k,N)` → `(B,k,k)` | `return m + torch.eye(...)` | 05 |
| 6 | `model → compute_loss` | logits, labels → `(loss, ce)` | `loss = ce + reg_weight * feature_transform_regularizer(...)` | 04, 05 |
| 7 | `smoke_funcs → run_smoke_test` | one real batch → PASS/FAIL list | `check("permutation invariance ...", diff < 1e-4, ...)` | 03 |
| 8 | `train_funcs → augment_batch` | `(B,N,3)` → `(B,N,3)` | `torch.bmm(points, R.transpose(1, 2))` | 04 |
| 8 | `train_funcs → train_one_epoch` | loader → mean loss, acc | `zero_grad → model → compute_loss → backward → step` | 04 |
| 8 | `train_funcs → evaluate` | loader → dict of metrics | `np.add.at(cm, (labels.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()), 1)`: count (true, predicted) pairs | 06 |
| 8 | `train_funcs → fit` | model + loaders → history, best val | `is_best = val["acc"] > best_val`, then `torch.save({"model_state": model.state_dict(), ...` | 04, 06 |
| 9 | `eval` | best checkpoint → test metrics | `best_model.load_state_dict(ckpt["model_state"])` | 06 |

## 3. Run the real model's shape trace (predict each line first)

Run from the project folder (needs `torch` and `nbformat`). It loads the **actual** `PointNet` class out of the notebook, so this is your code, not a copy.

```python
import ast, nbformat, torch, torch.nn as nn, torch.nn.functional as F
cells = {c.id: c.source for c in nbformat.read("PointNet_ModelNet10.ipynb", as_version=4).cells if c.cell_type == "code"}
defs = [n for n in ast.parse(cells["model"]).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
ns = {"torch": torch, "nn": nn, "F": F}
exec(compile(ast.Module(body=defs, type_ignores=[]), "model", "exec"), ns)       # defines PointNet, TNet, ...
model = ns["PointNet"](num_classes=10, input_transform=True, feature_transform=True).eval()
with torch.no_grad():
    logits, aux = model(torch.randn(2, 1024, 3), verbose=True)
print(tuple(aux["input_matrix"].shape), tuple(aux["feature_matrix"].shape))
```
<!-- expect: input points (B,N,3) (2, 1024, 3) | transpose -> (B,3,N) (2, 3, 1024) | after input T-Net (3x3 applied) (2, 3, 1024) | shared MLP 3->64->64 (2, 64, 1024) | after feature T-Net (64x64 applied) (2, 64, 1024) | shared MLP 64->64->128->1024 (2, 1024, 1024) | global max pool over the N points (2, 1024) | classifier 1024->512->256->classes (2, 10) | (2, 3, 3) (2, 64, 64) -->
<details><summary>Expected output</summary>

```
  input points (B,N,3)                    (2, 1024, 3)
  transpose -> (B,3,N)                    (2, 3, 1024)
  after input T-Net (3x3 applied)         (2, 3, 1024)
  shared MLP 3->64->64                    (2, 64, 1024)
  after feature T-Net (64x64 applied)     (2, 64, 1024)
  shared MLP 64->64->128->1024            (2, 1024, 1024)
  global max pool over the N points       (2, 1024)
  classifier 1024->512->256->classes      (2, 10)
(2, 3, 3) (2, 64, 64)
```
Note the T-Nets don't change the shape: they transform the values, not the size. Try `input_transform=False` and watch those two lines disappear. (The extraction trick is the same one `dev/test_units.py` uses.)
</details>

## 4. Where to change things (and what else must change)

| You want to… | Change | Also do |
|---|---|---|
| run the real experiment | `QUICK_MODE = False` | Restart & run all |
| test T-Nets | `USE_INPUT_TNET` / `USE_FEATURE_TNET` | Compare on validation; keep everything else fixed |
| use more/fewer points | `NUM_POINTS` | A new cache is built automatically (the name contains the point count) |
| change the model | `model → PointNet` | Rerun the smoke test; `python dev/test_units.py` locally |
| change the notebook at all | `dev/notebook_source.py` | `python dev/build_notebook.py`, then commit both files |

## 5. The safety net around the code

| Tool | What it proves | Run it |
|---|---|---|
| smoke test (notebook §7) | shapes, normalized data, loss ≈ ln(10), gradients, can overfit, permutation invariance, identity T-Nets | automatically, before training |
| `dev/test_units.py` | 12 tests of parser, sampling, normalization, augmentation, split, metrics, regularizer, checkpoint | `python dev/test_units.py` |
| `dev/run_local_test.py` | every non-Colab cell end to end on a small real sample | see README (needs a sample; Step 3 adds a fetch script) |
| `dev/check_learning_notes.py` | these notes reference real code and their snippet answers are right | `python dev/check_learning_notes.py` |

## Self-check
1. Name the 3 functions that turn a file into a `(1024,3)` array, in order.
2. Which cell would you open to make the model use mean pooling? Which line?
3. Where is the best epoch chosen, and on which data?
4. What do you run after editing `dev/notebook_source.py`?
5. Which check would fail if someone replaced max pooling with "take the first point"?

## Ask Claude
- "Quiz me: give me a line from `dev/notebook_source.py` and I'll tell you which cell and what it does."
- "Walk me through `fit()` one line at a time, with shapes."
