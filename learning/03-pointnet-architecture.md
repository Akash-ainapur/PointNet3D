# 03 · The PointNet architecture (the core idea)

**Code:** `dev/notebook_source.py`, cell `model`: `conv_bn_relu`, `fc_bn_relu`, `PointNet.__init__`, `PointNet.forward`. Run cell `shape_trace` to print every shape below.

## 1. The problem PointNet solves

A point cloud is a **set**: rows of `(N, 3)` in any order describe the same chair. With N = 1024 there are 1024! orderings. A model that flattens the points into one long vector (`3072` numbers) and feeds an MLP gives **different outputs for the same object** depending on the order. PointNet is built so that order **cannot** change the output. This is called **permutation invariance**.

## 2. The recipe: shared MLP → symmetric function → classifier

```
points (B,N,3)
  │ transpose                       (B,3,N)      channels first for Conv1d
  │ mlp1: shared 3→64→64            (B,64,N)     same weights for every point
  │ mlp2: shared 64→64→128→1024     (B,1024,N)   each point now has 1024 features
  │ max over N  (dim=2)             (B,1024)     ONE global "shape fingerprint"
  │ classifier 1024→512→256→10      (B,10)       logits (raw class scores)
```
(`model → PointNet.forward`. The optional T-Nets slot in after the transpose and after `mlp1`; see note 05.)

| Step | Code | Why it exists | If you remove it |
|---|---|---|---|
| Shared MLP | `self.mlp1`, `self.mlp2` built from `conv_bn_relu` = `Conv1d(k=1) → BatchNorm1d → ReLU` | Lifts each point from 3 numbers to a rich 1024-dim description; **sharing weights** means each point is processed identically, regardless of position in the array | Per-point weights would depend on index, so order would matter and N would have to be fixed |
| Max pool over points | `x.max(dim=2).values` | A **symmetric** function: `max(a,b,c) = max(c,a,b)`. It merges N point features into one vector, and it works for any N | Replacing it with "take point 0" or flattening breaks invariance. The smoke test's permutation check catches this (a deliberately broken version failed it) |
| Classifier head | `self.classifier`: `fc_bn_relu(1024,512) → Dropout → fc_bn_relu(512,256) → Dropout → Linear(256,10)` | Ordinary MLP on the fingerprint → 10 scores | No class scores |
| Dropout `p=0.3` | `nn.Dropout(dropout)` | Randomly zeroes 30% of units in training, which reduces overfitting (paper: keep-prob 0.7) | More overfitting on a small dataset like ModelNet10 |

**Why max and not mean/sum?** Both are symmetric too. Max lets each of the 1024 channels act as a detector ("is there *any* point with feature X?"), and the paper's ablation found max pooling worked best among the alternatives they tried. Swapping max for mean is a good what-if experiment (note 06).

## 3. Predict, then run: invariance in 10 lines

```python
import torch, torch.nn as nn
torch.manual_seed(0)
shared = nn.Sequential(nn.Conv1d(3, 16, 1), nn.ReLU())        # shared MLP
x = torch.randn(1, 3, 100)                                      # (B=1, xyz, N=100)
perm = torch.randperm(100)
a = shared(x).max(dim=2).values                                 # (1,16)
b = shared(x[:, :, perm]).max(dim=2).values                     # same points, shuffled
print(torch.allclose(a, b))                                     # → ?
flat = nn.Linear(300, 16)                                       # the "naive" approach
print(torch.allclose(flat(x.flatten(1)), flat(x[:, :, perm].flatten(1))))   # → ?
```
<details><summary>Answer</summary><code>True</code>, then <code>False</code>. Shared MLP + max is order-proof; flattening is not. The project tests this on the real model: <code>smoke_funcs → run_smoke_test</code>, check "permutation invariance (shuffled points → same logits)", measured max difference 0.00e+00 locally.</details>

## 4. Where the parameters are (measured from the code)

| Part | Parameters |
|---|---|
| `mlp1` (3→64→64) | 4,672 |
| `mlp2` (64→64→128→1024) | 147,008 |
| `classifier` (1024→512→256→10) | **660,234** (the first `Linear(1024,512)` alone is 524,800) |
| **Baseline total** | **811,914** |
| + input T-Net (3×3) | +803,081 |
| + feature T-Net (64×64) | +1,857,344 |
| Both T-Nets on | 3,472,339 |

Interview-worthy observation: **81% of the baseline's parameters are in the classifier head**, not in the part that looks at points. The per-point extractor is tiny because its weights are shared across all 1024 points. The T-Nets more than quadruple the model size.

## 5. Two properties to mention

- **Works for any N.** Nothing in the model fixes 1024: shared layers don't care about length, and max pools whatever N it gets. Unit test `model_shapes_any_num_points_and_all_variants` runs N = 64 and 1024.
- **Robust to missing points (the "critical set").** The output depends only on the points that win at least one of the 1024 max operations (`max(dim=2).indices`). Deleting other points changes nothing. The paper calls these the critical points. This would make a nice visual experiment.

## 6. What it does *not* capture (a limitation to admit)

Each point is processed **alone** until the max pool, so PointNet has no notion of a local neighbourhood (e.g. "these 20 points form a chair leg"). PointNet++ (the follow-up by the same authors) adds hierarchical local grouping. Saying this in an interview shows you know where the architecture sits.

## Simplified vs the paper
Layers, sizes, max pool, dropout keep 0.7 and BatchNorm: **same as the authors' code**. Differences: T-Nets are **off in our baseline** (note 05); the segmentation branch is omitted; ModelNet10 instead of ModelNet40.

## Self-check
1. Draw the shape pipeline from `(32,1024,3)` to `(32,10)` from memory.
2. Why must the per-point weights be *shared*?
3. Name two other symmetric functions besides max. Why did the authors pick max?
4. Where is most of the model's capacity, and why is the point-processing part so small?
5. What can't PointNet see that PointNet++ can?

## Ask Claude
- "Show me how to extract and plot the critical points of one cloud using `max(dim=2).indices`."
- "If I replace max with mean pooling, which lines change and what do you expect to happen?"
