# 09 · Own these 3 functions

## The smart (and honest) strategy

Don't *pretend* you wrote these. **Make it true**: rebuild each one yourself from a blank skeleton (section 4), check it against the original, and explain it out loud. These are short (7–12 core lines each). After about 3 sessions you can truthfully say:

> "I implemented the PointNet model, the training loop and the mesh-to-point-cloud sampling myself. I used AI to speed up the scaffolding around them (download, plotting, tests), and I verified everything."

That sentence survives any follow-up question. "I wrote everything" does not.

**Why these 3:** together they cover what interviewers ask most: the **model** (the core idea), the **training loop** (asked in every ML interview) and **data preparation** (unique to 3D, and it shows engineering thinking).

| # | Function | Cell id | What it shows |
|---|---|---|---|
| 1 | `PointNet.__init__` + `PointNet.forward` | `model` | you understand the architecture and tensor shapes |
| 2 | `train_one_epoch` | `train_funcs` | you understand how a network actually learns |
| 3 | `sample_points_from_mesh` + `normalize_point_cloud` | `pc_funcs` | you can turn raw 3D data into model input |

Every "if you remove it" result below was **measured** on this project's code (CPU, small inputs), not guessed.

---

## 1. `PointNet` (cell `model`)

**The version to memorize** (baseline path only; skip the `say(...)` logging and the two `if ... tnet` blocks at first):

```py
class PointNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.mlp1 = nn.Sequential(conv_bn_relu(3, 64), conv_bn_relu(64, 64))
        self.mlp2 = nn.Sequential(conv_bn_relu(64, 64), conv_bn_relu(64, 128), conv_bn_relu(128, 1024))
        self.classifier = nn.Sequential(
            fc_bn_relu(1024, 512), nn.Dropout(dropout),
            fc_bn_relu(512, 256), nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, points):            # points: (B, N, 3)
        x = points.transpose(1, 2)        # (B, 3, N)
        x = self.mlp1(x)                  # (B, 64, N)
        x = self.mlp2(x)                  # (B, 1024, N)
        x = x.max(dim=2).values           # (B, 1024)
        return self.classifier(x)         # (B, 10)
```
`conv_bn_relu(a, b)` = `Conv1d(a, b, kernel_size=1) → BatchNorm1d(b) → ReLU`; `fc_bn_relu` is the same with `Linear`.

| Line | Plain English | Shape after (B=32, N=1024) | If you remove or change it (measured) |
|---|---|---|---|
| `super().__init__()` | set up PyTorch's bookkeeping so layers are registered as parameters | — | assigning the first layer raises `AttributeError: cannot assign module before Module.__init__() call` |
| `mlp1`, `mlp2` (Conv1d k=1) | the **shared MLP**: the same small network on every point | `(32,64,1024)` → `(32,1024,1024)` | — (this *is* PointNet's feature extractor; only about 152K parameters, because the weights are shared) |
| `classifier` | normal MLP from the fingerprint to 10 scores; dropout fights overfitting | `(32,10)` | holds **660,234 of the 811,914** parameters |
| `points.transpose(1, 2)` | Conv1d wants channels (xyz) first | `(32,3,1024)` | **crash**: `RuntimeError … expected input[4, 1024, 3] to have 3 channels, but got 1024 channels instead` |
| `x.max(dim=2).values` | for each of 1024 features keep the strongest point → one fingerprint per object | `(32,1024)` | swap for "take point 0": shuffling the points changes the output (difference 0.046). With max: **exactly 0.0**. Mean: also about 0 (3e-8) but usually weaker |
| `return self.classifier(x)` | raw scores (logits), no softmax: `cross_entropy` applies it | `(32,10)` | adding a softmax here would apply it twice → worse gradients |

**Two measured facts to quote:**
- **Dropout:** the same input twice in `train()` mode gave outputs differing by up to 1.57; in `eval()` mode, exactly 0.0. That's why evaluation must call `model.eval()`.
- **Critical points:** in an *untrained* model on random input, only **138–153 of the 1024 points** won at least one of the 1024 max operations (`x.max(dim=2).indices`). The rest didn't affect the output at all. That's the paper's "critical set" idea; re-measure it on your trained model for a stronger story.

**Interview Q&A**
- *Why `Conv1d` with kernel 1?* It's a `Linear` layer applied to every point with shared weights (note 02 proves they're equal).
- *Why does max make it order-independent?* `max` gives the same answer for any order of its inputs, and every point went through identical weights before it.
- *Where are most parameters?* In the classifier head (81%); the point-processing part is small because it's shared.

---

## 2. `train_one_epoch` (cell `train_funcs`)

```py
def train_one_epoch(model, loader, optimizer, device, reg_weight, augment=True):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for points, labels in loader:                          # (B,N,3), (B,)
        points, labels = points.to(device), labels.to(device)
        if augment:
            points = augment_batch(points)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(points)                        # (B,10)
        loss, ce = compute_loss(logits, labels, aux, reg_weight)
        loss.backward()
        optimizer.step()
        total_loss += ce.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n
```

| Line | Plain English | If you remove or change it (measured) |
|---|---|---|
| `model.train()` | dropout on, BatchNorm learns from this batch | after a validation pass the model would stay in eval mode: no dropout, BatchNorm stops adapting |
| `.to(device)` | move the batch to the GPU where the model is | crash with an error like "Expected all tensors to be on the same device" |
| `augment_batch(points)` | random spin about z + tiny noise (training only) | the model sees identical clouds every epoch → more overfitting |
| `optimizer.zero_grad(...)` | erase the previous batch's gradients | gradients **add up**: after 2 backward passes they were exactly **2.0×** larger, so every step is wrong |
| `model(points)` | forward pass | — |
| `compute_loss(...)` | cross-entropy (+ T-Net penalty if on): one number, how wrong we are | — |
| `loss.backward()` | chain rule → fills `.grad` of every weight | nothing learns (no gradients) |
| `optimizer.step()` | Adam moves each weight a little against its gradient | nothing learns (gradients computed but unused) |
| `ce.item() * labels.size(0)` | sum per-sample losses so the epoch average is exact | `.item()` turns a tensor into a Python number; without it you keep the whole graph in memory |
| `logits.argmax(dim=1) == labels` | predicted class = highest score; count matches | e.g. logits for 3 samples → argmax `[0, 2, 1]` vs labels `[0, 1, 1]` → 2 correct |

**The 4-word mantra:** *zero → forward → backward → step.*

**Interview Q&A**
- *What happens if you forget `zero_grad`?* PyTorch accumulates gradients, so they keep growing (measured 2× after two passes) and training goes wrong.
- *Why is augmentation only in training?* Validation and test must measure the model on unchanged data.
- *Why multiply by `labels.size(0)`?* To average per sample, not per batch.

---

## 3. `sample_points_from_mesh` + `normalize_point_cloud` (cell `pc_funcs`)

```py
def sample_points_from_mesh(verts, faces, n_points, rng):
    tri = verts[faces].astype(np.float64)                               # (F, 3, 3): 3 corners per triangle
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)   # (F,)
    total = area.sum()
    if not np.isfinite(total) or total <= 0:                            # broken mesh: use its vertices
        return verts[rng.integers(0, len(verts), n_points)].astype(np.float32)
    face_idx = rng.choice(len(faces), size=n_points, p=area / total)    # (n_points,) which triangle
    r1, r2 = rng.random(n_points), rng.random(n_points)
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2                            # weights of the 3 corners, sum to 1
    t = tri[face_idx]                                                   # (n_points, 3, 3)
    pts = w0[:, None] * t[:, 0] + w1[:, None] * t[:, 1] + w2[:, None] * t[:, 2]   # (n_points, 3)
    return pts.astype(np.float32)

def normalize_point_cloud(pts):
    pts = pts - pts.mean(axis=0, keepdims=True)           # centroid → origin
    radius = np.linalg.norm(pts, axis=1).max()            # farthest point
    return (pts / max(radius, 1e-8)).astype(np.float32)   # fit inside a unit sphere
```

| Line | Plain English | If you remove or change it (measured) |
|---|---|---|
| `verts[faces]` | NumPy "fancy indexing": for each face, fetch its 3 corner coordinates | — |
| `area = 0.5 * ‖(B−A)×(C−A)‖` | triangle area from the cross product | — |
| `p=area / total` | big triangles get proportionally more points | uniform choice: a triangle 3× larger got only **0.50** of the points instead of **0.753** → flat surfaces become sparse |
| `s = np.sqrt(r1)` | makes points uniform *inside* the triangle | without `sqrt`, corner A's average weight is **0.501** instead of the uniform **0.334** → points bunch toward one corner |
| `w0[:, None] * ...` | `[:, None]` turns `(n,)` into `(n,1)` so it broadcasts over xyz | shape error |
| `if total <= 0` fallback | guards against broken meshes | `p` would contain NaN → crash |
| `pts - pts.mean(axis=0, keepdims=True)` | move the object's centre to the origin | raw `chair_0001`: centroid `[-0.06, 3.12, 2.15]` → after: about 1e-7 |
| `/ radius` | scale into a unit sphere | across 70 real meshes, raw radius ranged **10.1 to 1165.2 (115×)**. The model would learn file units instead of shape |
| `max(radius, 1e-8)` | avoid dividing by zero | a single-point "cloud" → NaN |

**Interview Q&A**
- *Why area-weighted?* CAD meshes use many tiny triangles on detailed parts and few big ones on flat parts; picking triangles uniformly would distort the shape.
- *Why `sqrt(r1)`?* Plain uniform weights aren't uniform over the triangle's area; the square root corrects it (measured above).
- *What do you lose by normalizing?* Real size: a small and a big table look identical.

---

## 4. The rebuild drill (3 short sessions)

**Session 1 (20 min): read and predict.** Read this note. Run the snippets below and guess each output first.

```python
import torch, torch.nn as nn
try:
    nn.Conv1d(3, 64, kernel_size=1)(torch.randn(4, 1024, 3))   # forgot the transpose
except RuntimeError as e:
    print("RuntimeError", "but got 1024 channels" in str(e))
```
<!-- expect: RuntimeError True -->

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
lin = nn.Linear(3, 2); x, y = torch.randn(8, 3), torch.randint(0, 2, (8,))
F.cross_entropy(lin(x), y).backward(); g1 = lin.weight.grad.clone()
F.cross_entropy(lin(x), y).backward()                        # no zero_grad in between
print(round((lin.weight.grad.norm() / g1.norm()).item(), 4))
```
<!-- expect: 2.0 -->

```python
import numpy as np
r1 = np.random.default_rng(0).random(200000)
print(round((1 - np.sqrt(r1)).mean(), 3), round((1 - r1).mean(), 3))   # corner weight with vs without sqrt
```
<!-- expect: 0.334 0.501 -->

**Session 2 (30 min): rebuild from skeletons in Colab.** Run the whole notebook first (so `conv_bn_relu`, `train_loader`, etc. exist). In a new cell, fill in the blanks **without looking**, then run the check cell.

```py
class MyPointNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.mlp1 = ___           # 3 -> 64 -> 64, shared
        self.mlp2 = ___           # 64 -> 64 -> 128 -> 1024, shared
        self.classifier = ___     # 1024 -> 512 -> 256 -> num_classes, dropout after the first two
    def forward(self, points):    # (B, N, 3)
        ___                       # 5 lines: transpose, mlp1, mlp2, max over points, classifier

def my_sample(verts, faces, n_points, rng):
    ___                           # triangles, areas, choose faces by area, sqrt trick, weighted corners

def my_train_one_epoch(model, loader, optimizer, device):
    ___                           # train mode, loop: to(device), zero -> forward -> loss -> backward -> step, return avg loss, acc
```

```py
# CHECK CELL: run after your cell (in the Colab notebook, after Run all)
torch.manual_seed(0); mine = MyPointNet().to(device)
assert sum(p.numel() for p in mine.parameters()) == 811_914, "layer sizes differ from the original"
xb4 = xb[:4].to(device); mine.eval()
with torch.no_grad():
    out = mine(xb4); out_perm = mine(xb4[:, torch.randperm(xb4.shape[1], device=device)])
assert tuple(out.shape) == (4, 10) and (out - out_perm).abs().max() < 1e-4, "shape or permutation invariance wrong"
v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[3,0,1],[0,1,1]], np.float32); f = np.array([[0,1,2],[3,4,5]])
p = my_sample(v, f, 40000, np.random.default_rng(0))
assert p.shape == (40000, 3) and abs((p[:, 2] > 0.5).mean() - 0.75) < 0.01, "not area-weighted"
a = p[p[:, 2] < 0.5]                                           # points in the small triangle (0,0),(1,0),(0,1)
assert abs(a[:, 0].mean() - 1/3) < 0.01 and abs(a[:, 1].mean() - 1/3) < 0.01, "not uniform inside the triangle (check the sqrt trick)"
opt = torch.optim.Adam(mine.parameters(), lr=1e-3)
l1, a1 = my_train_one_epoch(mine, train_loader, opt, device); l2, a2 = my_train_one_epoch(mine, train_loader, opt, device)
print(f"epoch losses {l1:.3f} -> {l2:.3f}  (should go down)"); assert l2 < l1
print("ALL CHECKS PASSED: these three are yours now.")
```

**Session 3 (15 min): explain and experiment.** Explain each function out loud in 60 seconds using its table. Then make **one** change and watch the output, e.g. `x.max(dim=2).values` → `x.mean(dim=2)` in your `MyPointNet`, or remove `np.sqrt`. Say what changed and why. That's exactly the "what if" thinking interviewers love.

Repeat session 2 once more after 2–3 days (from memory). If it passes twice, you own it.

## Ask Claude
- "Quiz me on `train_one_epoch`: show me one line and I'll explain it."
- "My check cell failed with this error: … What did I get wrong?"
