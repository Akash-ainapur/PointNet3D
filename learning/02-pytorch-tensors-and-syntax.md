# 02 · PyTorch tensors and the syntax this project uses

Only the syntax that appears in the model and training code. Run each snippet in Colab or locally (`pip install torch`). **Predict the printed shape first.**

## 1. Tensors and shapes

A tensor is an n-dimensional array (like NumPy) that can live on a GPU and track gradients. Its **shape** is the most important thing to know at every line.

| Name in this project | Shape | Meaning |
|---|---|---|
| one cloud | `(N, 3)` = `(1024, 3)` | 1024 points, xyz each |
| a batch | `(B, N, 3)` = `(32, 1024, 3)` | what the DataLoader yields (`split → train_loader`) |
| labels | `(B,)` = `(32,)` | integer class ids, dtype `int64` |
| logits | `(B, 10)` | one raw score per class |

**`dim` / axis:** dimensions are numbered from 0. For `(B, N, 3)`: dim 0 = batch, dim 1 = points, dim 2 = coordinates. An operation "over dim 1" collapses or works along the points.

## 2. `transpose` and `permute`

```python
import torch
x = torch.randn(32, 1024, 3)          # (B, N, 3)
print(x.transpose(1, 2).shape)        # swaps dims 1 and 2      → ?
print(x.permute(0, 2, 1).shape)       # lists the new order      → ?
```
<!-- expect: torch.Size([32, 3, 1024]) | torch.Size([32, 3, 1024]) -->
<details><summary>Answer</summary>Both give <code>torch.Size([32, 3, 1024])</code>. <code>transpose</code> swaps exactly two dims; <code>permute</code> reorders all dims at once. Same result here.</details>

**Why the project does it** (`model → PointNet.forward`, `x = points.transpose(1, 2)`): `Conv1d` expects **channels first**: `(batch, channels, length)`. Our "channels" are xyz (3) and our "length" is the points (N).

## 3. `nn.Module`, `__init__`, `forward`

```python
import torch, torch.nn as nn
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()                 # always call this first
        self.fc = nn.Linear(3, 64)         # layers created here are registered as parameters
    def forward(self, x):                  # defines the computation
        return torch.relu(self.fc(x))
m = Tiny()
print(m(torch.randn(5, 3)).shape)          # calling m(x) runs forward(x)  → ?
print(sum(p.numel() for p in m.parameters()))   # → ?
```
<!-- expect: torch.Size([5, 64]) | 256 -->
<details><summary>Answer</summary><code>torch.Size([5, 64])</code> and <code>256</code> (= 3·64 weights + 64 biases). The notebook prints the same kind of count for PointNet (<code>shape_trace</code> cell): 811,914 for the baseline.</details>

- **`nn.Linear(in, out)`:** `y = x·Wᵀ + b`. It works on the **last** dim: `(…, in) → (…, out)`.
- **`nn.ReLU`:** `max(0, x)` element-wise. Without a non-linearity, stacked linear layers collapse into one linear layer.
- **`nn.Sequential(a, b, c)`:** runs a, then b, then c. The project's building block is `model → conv_bn_relu`: `Conv1d → BatchNorm1d → ReLU`.
- **`nn.BatchNorm1d`:** normalizes each channel over the batch, which stabilizes training. It behaves differently in `model.train()` vs `model.eval()` (note 04), and it needs more than 1 sample per batch in training, hence `drop_last=True` in `split → train_loader`.

## 4. The key trick: `Conv1d(kernel_size=1)` == the same `Linear` on every point

```python
import torch, torch.nn as nn
conv = nn.Conv1d(3, 64, kernel_size=1)
lin = nn.Linear(3, 64)
lin.weight.data = conv.weight.data.squeeze(-1)    # (64,3,1) → (64,3): copy the same weights
lin.bias.data = conv.bias.data
x = torch.randn(2, 1024, 3)                       # (B, N, 3)
a = conv(x.transpose(1, 2))                       # (B, 64, N)
b = lin(x).transpose(1, 2)                        # (B, N, 64) → (B, 64, N)
print(a.shape, torch.allclose(a, b, atol=1e-6))   # → ?
```
<!-- expect: torch.Size([2, 64, 1024]) True -->
<details><summary>Answer</summary><code>torch.Size([2, 64, 1024]) True</code>. A 1-wide convolution looks at one point at a time with shared weights. That is exactly what the paper calls a <b>shared MLP</b>.</details>

## 5. Reductions: `max(dim=…)` returns two things

```python
import torch
f = torch.randn(32, 1024, 1024)           # (B, channels, N) after the shared MLP
out = f.max(dim=2)                        # reduce over the points
print(out.values.shape, out.indices.shape)  # → ?
```
<!-- expect: torch.Size([32, 1024]) torch.Size([32, 1024]) -->
<details><summary>Answer</summary>Both <code>torch.Size([32, 1024])</code>. <code>.values</code> are the maxima (used: <code>model → PointNet.forward</code>, <code>x = x.max(dim=2).values</code>); <code>.indices</code> say <i>which point</i> won each channel (the paper calls these points the "critical set"; a nice what-if experiment).</details>

## 6. Batched matrix multiply: `torch.bmm`

`bmm(A, B)` multiplies matrix `A[i]` with `B[i]` for each batch element `i`: `(B, n, m) @ (B, m, p) → (B, n, p)`. In the T-Net code: `torch.bmm(matrix (B,3,3), x (B,3,N)) → (B,3,N)`, which applies a 3×3 transform to every point (note 05).

## 7. Small things that appear everywhere

| Syntax | Meaning | Where |
|---|---|---|
| `.view(-1, k, k)` | reshape; `-1` = "infer this dim" | `TNet.forward` |
| `torch.eye(k)` | k×k identity matrix | `TNet.forward`, regularizer |
| `.to(device)` | move to GPU/CPU | `train_one_epoch`, `evaluate` |
| `.item()` | 1-element tensor → Python number | loss logging |
| `logits.argmax(dim=1)` | predicted class per sample `(B,)` | accuracy |
| `with torch.no_grad():` | don't track gradients (faster, for evaluation) | `evaluate` (decorator `@torch.no_grad()`) |
| `x[:, perm]` | reorder points (dim 1) | permutation test in `run_smoke_test` |

## Self-check
1. Shape after `points.transpose(1, 2)` for a batch of 32 clouds of 1024 points?
2. Why `Conv1d(k=1)` instead of `Linear`, if they compute the same thing? *(Answer: convenience with channels-first data and it mirrors the original code; both are fine.)*
3. What does `max(dim=2)` remove, and what would `max(dim=1)` do instead?
4. Why must BatchNorm see more than one sample per batch in training?
5. `bmm((32,64,64), (32,64,1024))` → shape?

## Ask Claude
- "Walk me through every shape in `PointNet.forward` for B=2, N=5 with tiny numbers."
- "What's the difference between `.view` and `.reshape`, and when does `.view` fail?"
