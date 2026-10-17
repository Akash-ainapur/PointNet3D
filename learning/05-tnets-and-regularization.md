# 05 · T-Nets and the orthogonality penalty

**Code:** `dev/notebook_source.py`, cell `model → TNet`, `model → PointNet.forward` (the two `if self.…_tnet` blocks), `model → feature_transform_regularizer`, `model → compute_loss`. Switches: `USE_INPUT_TNET`, `USE_FEATURE_TNET` in cell `config` (both `False` in the baseline).

## 1. The idea in one sentence

A **T-Net** is a mini-PointNet that looks at the cloud and predicts a **k×k matrix**, which is multiplied into the data to "straighten" it (put it in a canonical pose) before the main network looks at it.

| T-Net | Applied to | Predicts | Applied as |
|---|---|---|---|
| input T-Net (`TNet(3)`) | raw xyz `(B,3,N)` | `(B,3,3)` | `torch.bmm(matrix, x)` → `(B,3,N)` |
| feature T-Net (`TNet(64)`) | 64-d features after `mlp1` `(B,64,N)` | `(B,64,64)` | `torch.bmm(matrix, x)` → `(B,64,N)` |

## 2. Inside `TNet.forward` (shapes for k = 3)

```
x (B,3,N) → shared MLP 3→64→128→1024 (B,1024,N) → max over N (B,1024)
          → FC 1024→512→256 (B,256) → Linear 256→9 (B,9) → view (B,3,3) → + identity
```
It's a PointNet itself: shared MLP + max pool, so the predicted matrix doesn't depend on point order. Invariance is preserved, and the smoke test's permutation check also passes with both T-Nets on (14/14 locally).

## 3. Identity initialization: why the last layer starts at zero

```py
# excerpt from TNet.__init__ / forward
nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
return m + torch.eye(self.k, ...)      # output = learned correction + identity
```
At the start, `m = 0` for **any** input, so the T-Net returns exactly the identity: "do nothing". Training then learns only a *correction*.
**What goes wrong without it:** a random matrix would scramble or squash every cloud from step 1. The main network would start from garbage, and training would be slower and unstable.
*Verified:* smoke check "input_matrix / feature_matrix starts as identity". A version without the zero init failed this check when tested on purpose.

**Predict, then run:**
```python
import torch, torch.nn as nn
out = nn.Linear(256, 9); nn.init.zeros_(out.weight); nn.init.zeros_(out.bias)
m = out(torch.randn(4, 256)).view(-1, 3, 3) + torch.eye(3)
print(torch.allclose(m, torch.eye(3).expand(4, 3, 3)))         # → ?
pts = torch.randn(4, 3, 100)
print(torch.allclose(torch.bmm(m, pts), pts))                  # → ?
```
<!-- expect: True | True -->
<details><summary>Answer</summary><code>True</code> and <code>True</code>: four different random inputs, all mapped to the identity, so the points come out unchanged.</details>

## 4. The orthogonality penalty (feature T-Net only)

An **orthogonal** matrix A has `A·Aᵀ = I`. It only rotates or reflects, so it keeps distances and loses no information. A 64×64 matrix has 4,096 free numbers. Without guidance it can learn to squash features (lose information) and is harder to optimize. So the loss adds a penalty:

```py
# model → feature_transform_regularizer  (same definition as the authors' tf.nn.l2_loss)
penalty = 0.5 * ((I - m @ mᵀ) ** 2).sum()          # summed over the batch, halved
loss = cross_entropy + 0.001 * penalty              # model → compute_loss (FEATURE_REG_WEIGHT = 0.001)
```

```python
import torch, math
def penalty(m):   # same formula as feature_transform_regularizer
    I = torch.eye(m.size(1)).unsqueeze(0)
    return 0.5 * ((I - torch.bmm(m, m.transpose(1, 2))) ** 2).sum()
R = torch.eye(5); t = 0.7
R[0, 0], R[0, 1], R[1, 0], R[1, 1] = math.cos(t), -math.sin(t), math.sin(t), math.cos(t)
print(round(penalty(R.expand(3, 5, 5)).item(), 6), penalty(2 * torch.eye(5).expand(3, 5, 5)).item())   # → ?
```
<!-- expect: 0.0 67.5 -->
<details><summary>Answer</summary><code>0.0 67.5</code>. A rotation is orthogonal, so it costs nothing. 2·I stretches everything: 0.5 × 3 samples × 5 × (1−4)² = 67.5. Unit test <code>regularizer_and_loss</code> checks these exact values on the real function.</details>

Why only for the 64×64 one? The 3×3 has just 9 numbers and is easy to learn; the paper applies the penalty to the feature transform.

## 5. Cost (measured from the code)

| Variant | Parameters |
|---|---|
| baseline | 811,914 |
| + input T-Net | 1,614,995 |
| + feature T-Net | 2,669,258 |
| both | 3,472,339 |

Local CPU timing on the tiny sample: about 2.4 s per 32-cloud step with both T-Nets vs about 0.8–1.3 s baseline. GPU timing is **not measured yet**.

## 6. Why the baseline has them off (a decision you should be able to defend)

- ModelNet10 is **already orientation-aligned**, and we add rotation augmentation about z. Much of what the input T-Net would learn (alignment) is partly handled already.
- T-Nets **quadruple** the parameter count and slow every step.
- It's a **one-flag ablation**, so you can *measure* their value instead of assuming it. The paper reports gains from them on ModelNet40, but whether they help on aligned ModelNet10 with our setup is an open question for your experiments.

## Simplified vs the paper
Same structure (3→64→128→1024, max, 512→256→k²), same identity init, same penalty definition and weight (checked against the authors' `transform_nets.py` / `pointnet_cls.py`). **Difference:** off by default here; the paper always uses both.

## Self-check
1. What does a T-Net output, and how is it applied (which function, which shapes)?
2. Why start at the identity, and how does zero-initializing the last layer achieve it?
3. What does "orthogonal" mean, and why penalize the 64×64 matrix but not the 3×3?
4. Is PointNet still permutation-invariant with T-Nets? Why?
5. Give two reasons the baseline leaves them off, and how you'd test that choice.

## Ask Claude
- "Plot what the learned 3×3 input matrix does to a chair after training. How do I extract `aux['input_matrix']`?"
- "Would random 3D rotations at test time hurt the baseline more than the T-Net version? How would I test it?"
