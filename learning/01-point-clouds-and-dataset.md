# 01 · Point clouds, OFF meshes and ModelNet10

**Code:** `dev/notebook_source.py`, cells `download`, `explore`, `pc_funcs`, `pc_demo`, `viz`, `gallery`, `preprocess`, `split` (notebook sections 2–5).

## 1. Mesh vs point cloud

| | Mesh (what ModelNet gives you) | Point cloud (what PointNet eats) |
|---|---|---|
| Data | vertices `(V, 3)` + triangles `(F, 3)` (3 vertex indices each) | just points `(N, 3)` |
| Size | different V, F per object | **we force N = 1024 for every object** |
| Order | faces reference vertices by index | **no meaningful order** (shuffle rows, same object) |

A LiDAR or depth camera produces point clouds directly; that's why point-cloud models matter. ModelNet gives meshes, so we convert them.

## 2. The OFF file format (`pc_funcs → read_off`)

```
OFF
2382 2234 0          ← n_vertices n_faces n_edges
x y z                ← 2382 vertex lines
...
3 i j k              ← 2234 face lines: "3" = triangle, then 3 vertex indices
```

`read_off(path)` returns `verts (V,3) float32` and `faces (F,3) int64`. It also handles two quirks: a header glued to the counts (`OFF490 518 0`), and polygons with more than 3 corners, which it splits into triangles ("fan").
*Verify:* unit test `read_off_standard_and_merged_header_and_quads` in [`dev/test_units.py`](../dev/test_units.py).

Real example (measured locally on `chair_0001.off`): vertices `(2382, 3)`, faces `(2234, 3)`, raw size (x, y, z extent) ≈ `[20.4, 23.28, 36.72]` file units.

## 3. Mesh → points: area-weighted sampling (`pc_funcs → sample_points_from_mesh`)

1. Triangle area: `0.5 · |(b − a) × (c − a)|` (cross product) → `area (F,)`.
2. Choose a triangle for each point with probability `area / total area`: `rng.choice(F, size=N, p=area/total)`.
3. Choose a random spot *inside* that triangle with barycentric weights `w0, w1, w2` (they sum to 1).

**Why area-weighted?** CAD models use many tiny triangles where detail is high and a few huge ones on flat parts. Picking triangles uniformly would pile points onto detailed areas and leave big flat surfaces (a table top) almost empty. The cloud would no longer describe the true shape.

**Predict, then run:** two triangles, areas 0.5 and 1.5. What fraction of points lands in the big one?
```python
import numpy as np
area = np.array([0.5, 1.5])
idx = np.random.default_rng(0).choice(2, size=40000, p=area / area.sum())
print((idx == 1).mean())   # predict first!
```
<details><summary>Answer</summary>About 0.75 (= 1.5 / 2.0). Unit test <code>sampling_is_area_weighted_and_on_the_surface</code> checks exactly this on the real function. A deliberately broken version (uniform choice) gives about 0.50 and is caught.</details>

## 4. Normalization (`pc_funcs → normalize_point_cloud`)

```py
# excerpt from normalize_point_cloud (not a standalone snippet)
pts = pts - pts.mean(axis=0, keepdims=True)   # centre: centroid → (0,0,0)       shape stays (N,3)
radius = np.linalg.norm(pts, axis=1).max()    # distance of the farthest point   scalar
pts = pts / radius                            # everything fits in a unit sphere  (N,3)
```
`keepdims=True` keeps the mean as shape `(1,3)` so it **broadcasts** (gets subtracted from every row).

**Why?** File units differ wildly between objects (a monitor can be about 10× larger than a chair in raw units). Without normalization the network would learn "big = monitor" instead of shape.
**Trade-off you should mention:** real size information is thrown away, so a small table and a big table look the same.
*Verify:* unit test `normalize_centres_and_scales_to_unit_sphere`. The smoke test also checks every real batch has `max radius ≤ 1` and `centroid ≈ 0`.

## 5. Dataset facts (all checked against the source; see the README)

| | |
|---|---|
| ModelNet10 | 4,899 CAD meshes, 10 classes, OFF format, orientation aligned, **z is up** |
| Official split | 3,991 train / 908 test (folder names `ModelNet10/<class>/<train|test>/`) |
| Labels | class names sorted alphabetically → 0..9 (`explore → CLASS_TO_IDX`): bathtub=0 … toilet=9 |
| Imbalance | e.g. chair 889 vs bathtub 106 train meshes, which is why we also report mean class accuracy (note 06) |
| Gotcha | half of the zip's `.off` entries are macOS junk (`__MACOSX/.../._*.off`); `download → list_off_files` skips names starting with `._` |
| License | none stated; academic research use; cite Wu et al., CVPR 2015 |

## 6. Splits: train / validation / test (`split → stratified_split`)

- **Test** = the official test split, used **once**, at the very end.
- **Validation** = 10% of the official train split, **stratified** (each class keeps its share), fixed `SEED`. Used to pick the best epoch.
- **Train** = the remaining 90%.

Why not pick the best epoch on test? Then the test score is no longer an unbiased estimate. You would have "trained on the test set" through your choices.

## 7. Open3D's role (`viz → show_point_cloud`)

`o3d.geometry.PointCloud()` + `o3d.utility.Vector3dVector(np_array)` wraps a NumPy `(N,3)` array as an Open3D object. `draw_plotly` renders it in a notebook; the usual `draw_geometries` window can't open on Colab, which has no display. Open3D is only used for **looking at** data here, not for training.

## Simplified vs the paper
The paper's ModelNet40 data came as ready-made 2048-point clouds (first 1024 used). **We sample our own 1024 points from ModelNet10 meshes**, once, with a fixed seed, then cache them.

## Self-check (answer out loud)
1. Why can't you feed a mesh's vertex list straight into PointNet?
2. What would go wrong without area weighting? Without normalization?
3. Why is validation carved out of *train* and not *test*?
4. What does `keepdims=True` change in the centring line?
5. Which axis is "up" in ModelNet10, and where does the code rely on it? *(Hint: augmentation, note 04.)*

## Ask Claude
- "Show me the barycentric formula in `sample_points_from_mesh` with a numeric example."
- "What would change if I re-sampled 1024 new points every epoch instead of caching?"
