# 04 · Training: logits, loss, gradients, optimizer

**Code:** `dev/notebook_source.py`, cells `model → compute_loss`, `train_funcs → augment_batch`, `train_funcs → train_one_epoch`, `train_funcs → fit`, `train_run` (notebook section 8). The smoke test in `smoke_funcs → run_smoke_test` checks most of this before training starts.

## 1. One training step (the 4 lines that matter)

```py
# excerpt from train_one_epoch
optimizer.zero_grad(set_to_none=True)                     # 1. forget the previous batch's gradients
logits, aux = model(points)                               # 2. forward: (B,N,3) -> (B,10)
loss, ce = compute_loss(logits, labels, aux, reg_weight)  # 3. one number: how wrong are we?
loss.backward(); optimizer.step()                         # 4. gradients, then update the weights
```

| Thing | Shape | What it is |
|---|---|---|
| `points` | `(32, 1024, 3)` | one batch, after augmentation |
| `labels` | `(32,)` int64 | true class ids 0..9 |
| `logits` | `(32, 10)` | raw scores, any real number, **not** probabilities |
| `loss` | `()` scalar | averaged over the batch |
| each `p.grad` | same shape as `p` | how the loss changes if that weight moves |

**If you forget `zero_grad`**, PyTorch *adds* new gradients to old ones, so updates become wrong and growing.

## 2. Logits → probabilities → cross-entropy

`softmax` turns logits into probabilities that sum to 1. Cross-entropy = `−log(probability given to the true class)`. `F.cross_entropy` does both in one numerically stable step, so **the model outputs raw logits, with no softmax layer**.

**Predict, then run:**
```python
import torch, torch.nn.functional as F, math
logits = torch.zeros(1, 10)                  # an untrained model: all classes equally likely
print(round(F.cross_entropy(logits, torch.tensor([3])).item(), 4), round(math.log(10), 4))   # → ?
good = torch.zeros(1, 10); good[0, 3] = 5.0  # confident and correct
bad = torch.zeros(1, 10); bad[0, 7] = 5.0    # confident and wrong
print(round(F.cross_entropy(good, torch.tensor([3])).item(), 4), round(F.cross_entropy(bad, torch.tensor([3])).item(), 4))  # → ?
```
<!-- expect: 2.3026 2.3026 | 0.0589 5.0589 -->
<details><summary>Answer</summary><code>2.3026 2.3026</code>: guessing among 10 classes costs ln(10). Then <code>0.0589</code> (confident and right: tiny loss) vs <code>5.0589</code> (confident and wrong: large loss). That is why the smoke test checks "initial cross-entropy near ln(K)": a fresh model far from 2.30 means something is off (e.g. wrong labels or exploding outputs).</details>

## 3. Backpropagation and the optimizer, in one paragraph

`loss.backward()` applies the chain rule from the loss back through every layer and fills `p.grad` for every weight. **Adam** (`fit` → `torch.optim.Adam(lr=1e-3, weight_decay=1e-4)`) then moves each weight a small step against its gradient, with a per-weight step size based on recent gradient history. `weight_decay` adds a small pull towards zero, a mild regularizer.
*Verified by the smoke test:* "every parameter received a gradient", "gradients finite and not all zero", and "can overfit one batch". If a single batch of 8 clouds can't be memorized in 40 steps, the training loop is broken. Locally the loss dropped about 2.4 → 0.01.

## 4. Epochs, batches, learning-rate schedule

- **Batch** = 32 clouds per step (`BATCH_SIZE`). **Epoch** = one pass over all training clouds (about 3,592 clouds = 112 steps in the full run).
- `drop_last=True` drops a final incomplete batch. BatchNorm can't train on a batch of 1.
- **Cosine schedule** (`CosineAnnealingLR(T_max=EPOCHS)`, stepped once per epoch): the LR glides from 1e-3 to almost 0. Big steps early, fine-tuning late.

```python
import math
lr = lambda e, T=30: 0.5 * 1e-3 * (1 + math.cos(math.pi * (e - 1) / T))   # LR used in epoch e (1-based)
print([f"{lr(e):.1e}" for e in (1, 10, 15, 20, 30)])                     # → ?
```
<!-- expect: ['1.0e-03', '7.9e-04', '5.5e-04', '3.0e-04', '2.7e-06'] -->
<details><summary>Answer</summary><code>['1.0e-03', '7.9e-04', '5.5e-04', '3.0e-04', '2.7e-06']</code>. The notebook prints the LR at the start of every epoch, so you can check this against your run.</details>

## 5. `model.train()` vs `model.eval()`: a classic interview question

| | `train()` (in `train_one_epoch`) | `eval()` (in `evaluate`) |
|---|---|---|
| Dropout | randomly zeroes 30% of units | off |
| BatchNorm | normalizes with **this batch's** mean/var and updates running averages | uses the **running averages** |

Forgetting `eval()` makes validation noisy and too low. Real observation from the local 70-mesh test: with only one training step per epoch, the running averages barely settle, and the eval-mode model predicted the same class for everything. That's a likely BatchNorm effect of tiny data, and a reason quick-mode accuracy isn't meaningful.

## 6. Data augmentation (`train_funcs → augment_batch`, training batches only)

```py
# simplified pseudo-code of augment_batch: rotation about z (up) + jitter
R = [[cos θ, −sin θ, 0], [sin θ, cos θ, 0], [0, 0, 1]]    # one random θ per cloud, R: (B,3,3)
points = torch.bmm(points, R.transpose(1, 2))           # (B,N,3) @ (B,3,3) -> (B,N,3)
points += clamp(0.01 · randn, −0.05, 0.05)              # tiny noise per coordinate
```
**Why:** the model should recognise a chair whichever way it faces, and not memorise exact coordinates. **Why only about z:** ModelNet10 is upright-aligned (z = up), so we keep "up" meaningful. Paper: rotation about the up axis + jitter σ 0.01, clip 0.05 (same).
*Verified:* unit test `augment_rotates_about_z_and_jitters_within_clip` (z unchanged, distances preserved, noise bounded).

## 7. Picking the best epoch (`train_funcs → fit`)

After every epoch: `evaluate` on **validation** → if `val acc` beats the best so far, `torch.save` a checkpoint ("`<- best, saved`"). The test set is **not** touched during training. Logged "train loss" is plain cross-entropy (the T-Net penalty is used for the update but not logged), so it's comparable with validation loss.

**Reading the curves:** train loss falling while val loss rises means **overfitting**. Both stuck near 2.30 means it isn't learning (check LR and labels).

## Simplified vs the paper
Paper (authors' `train.py`): Adam lr 0.001 with exponential decay, batch 32, **250 epochs**, BatchNorm-momentum schedule. **Ours:** Adam + weight decay, cosine LR, **30 epochs** (5 in quick mode), default BatchNorm momentum. So expect to land somewhat below a fully-tuned 250-epoch run.

## Self-check
1. Why does the model output logits and not probabilities?
2. What value should the first loss be close to, and why?
3. What do `zero_grad`, `backward` and `step` each do? What breaks without `zero_grad`?
4. Two things `model.eval()` changes?
5. Why is augmentation applied to training batches only, and why only rotation about z?

## Ask Claude
- "Show me the chain rule for one `Linear → ReLU → cross-entropy` path with numbers."
- "Adam vs SGD with momentum: what changes and when would I prefer each?"
