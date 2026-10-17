# 08 · Interview preparation

Practise these **out loud**. Short, precise answers beat long ones. If a follow-up goes beyond what you know, say what you *would* check. That is a strong answer.

## 1. Your pitch

**30 seconds.** "I built a 3D object classifier for point clouds using PointNet in PyTorch on ModelNet10, 10 furniture classes. The pipeline converts CAD meshes into 1024-point clouds with area-weighted surface sampling and unit-sphere normalization. The model applies a shared MLP to every point and max-pools across points, so the prediction doesn't depend on point order. I added a smoke test and unit tests that check shapes, gradients and permutation invariance before training, used a held-out validation split for model selection, and evaluated once on the official test set, *[result: overall X% / mean class Y%, from my run]*. I then ran controlled ablations on the T-Net alignment modules *[and found …]*."

**2 minutes:** add (1) the dataset gotchas (macOS junk files, class imbalance, so mean class accuracy), (2) one design decision you can defend (T-Nets off in the baseline: why), (3) one finding from your confusion matrix, (4) one limitation plus what you'd try next.

## 2. Numbers you can quote (and where each comes from)

| Number | Source | Status |
|---|---|---|
| 4,899 meshes, 10 classes, 3,991 train / 908 test | official zip, verified | ✅ now |
| 1024 points per object; 3,592 train / 399 val (stratified 10%) | config + `stratified_split` | ✅ now |
| 811,914 parameters (baseline), 3,472,339 with both T-Nets; 81% of the baseline sits in the classifier head | counted from the model code | ✅ now |
| 12 smoke checks (14 with T-Nets), 12 unit tests | local CPU runs | ✅ now |
| accuracy, per-class accuracy, confusion pairs, epoch time on T4, T-Net effect | **your Colab run** (`run_summary.json`) | ⏳ after your Colab run |

**Never say** a number from the second group until it's in your saved results. Don't quote the paper's ModelNet40 accuracy as yours; it's a different dataset and training budget.

## 3. Likely questions with accurate answers

**Fundamentals**
1. **Why not just flatten the points into an MLP?** Order would matter: the same object in a different row order gives a different output, and N would have to be fixed. *(note 03)*
2. **How does PointNet achieve permutation invariance?** The same MLP on every point (shared weights), then a symmetric aggregation (max over points). *(03; the smoke test checks it)*
3. **Why max pooling instead of mean or sum?** All are symmetric. Max lets each feature channel act as a detector of "is any point like this?", and the paper's ablation found it worked best. I could test mean pooling as an ablation. *(03, 06)*
4. **What's a shared MLP, and how is it implemented?** One small MLP applied to each point independently with the same weights. `Conv1d(kernel_size=1)` on `(B, C, N)` does exactly that; I verified it equals `Linear` per point. *(02)*
5. **Why transpose `(B,N,3)` → `(B,3,N)`?** `Conv1d` expects channels first. *(02)*
6. **Why other 3D representations aren't used here (voxels, multi-view images)?** Voxel grids cost memory cubically in resolution and lose detail to quantization; multi-view renders turn 3D into 2D images. PointNet consumes raw points directly, which is compact and is what LiDAR/depth sensors produce.

**Data**
7. **How did you turn meshes into point clouds?** Area-weighted triangle sampling + barycentric coordinates, 1024 points, fixed seed, cached. Uniform triangle choice would oversample finely-meshed regions. *(01)*
8. **Why normalize to a unit sphere? What do you lose?** File units vary wildly between objects, and the model should learn shape, not scale. The cost is that absolute size is lost. *(01)*
9. **How did you split the data?** Official test split untouched. Validation is a stratified 10% of train with a fixed seed. Best epoch chosen on validation; test used once. *(01, 06)*
10. **Anything surprising in the data?** Half of the zip's `.off` entries are macOS metadata files; a naive glob crashes. The classes are imbalanced, so I report mean class accuracy too. *(01)*

**Training**
11. **What does the model output, and what loss?** Raw logits `(B,10)` → `F.cross_entropy` (log-softmax + negative log-likelihood in one stable step). An untrained model starts near ln(10) ≈ 2.30. *(04)*
12. **What do `zero_grad`, `backward`, `step` do?** Clear old gradients; compute gradients by the chain rule; update weights with Adam. *(04)*
13. **`model.train()` vs `model.eval()`?** Dropout on/off; BatchNorm uses batch statistics vs running averages. *(04)*
14. **What augmentation, and why only about z?** Random rotation about the up axis + small clipped jitter. ModelNet10 is upright-aligned, so "up" is meaningful. *(04)*
15. **How did you know the training code was correct before spending GPU time?** A smoke test: shapes, normalized data, initial loss ≈ ln(10), finite non-zero gradients, can overfit one batch, permutation invariance. Plus 12 unit tests. Deliberately broken versions were caught. *(07)*

**T-Nets**
16. **What is a T-Net?** A mini-PointNet that predicts a k×k matrix (3×3 for points, 64×64 for features) multiplied into the data to align it to a canonical pose. *(05)*
17. **Why initialize to identity?** So it starts as "do nothing" and learns a correction; a random matrix would scramble inputs at step 1. Done by zero-initializing the last layer and adding I. *(05)*
18. **Why the orthogonality penalty?** Orthogonal matrices rotate or reflect without losing information. The 64×64 matrix has 4,096 free values and is hard to optimize, so `‖I − AAᵀ‖²` × 0.001 keeps it near-orthogonal. *(05)*
19. **Why are T-Nets off in your baseline?** ModelNet10 is already aligned and I augment with rotations; T-Nets quadruple the parameter count. I treated them as an ablation to measure instead of assuming *[result]*. *(05, 06)*

**Evaluation**
20. **Overall vs mean class accuracy?** Overall is dominated by big classes; mean class weights every class equally. They differ when small classes do badly. *(06)*
21. **What did the confusion matrix show?** *[From your run: which pairs were confused, and a geometric reason why.]*
22. **How do you know an improvement is real?** Same data, epochs and seed, one change; select on validation; repeat with 2–3 seeds before claiming a difference. *(06)*

## 4. Follow-ups that separate good answers from great ones

| Question | Strong answer direction |
|---|---|
| "What can PointNet not do?" | No local neighbourhood structure. Each point is processed alone until the max. **PointNet++** adds hierarchical local grouping; graph methods like DGCNN build k-NN neighbourhoods. |
| "Is it rotation invariant?" | No. Only z-rotations are seen in training; an arbitrary 3D rotation at test time would likely hurt. T-Nets help only partly. A test-time rotation experiment would show it *(06 what-if menu)*. |
| "Robust to missing points?" | The output depends only on the points that win the max (the "critical set"), so many points can be removed. Test it by subsampling test clouds *(06)*. |
| "What would you do with more time?" | Multi-seed runs; T-Net and point-count ablations; test-time robustness (noise, dropout, rotation); try PointNet++. |
| "Why ModelNet10 and not ModelNet40?" | Smaller (473 MB vs about 2 GB), faster iteration; the same pipeline extends to 40 classes. |
| "How reproducible is it?" | Fixed seeds, cached sampling, saved config + metrics per run; GPU kernels may still differ slightly between runs. Every run writes a `run_summary.json` with config, versions and metrics. |

## 5. Limitations to state before they're pointed out
- One seed per variant (until you run more): small differences are not conclusive.
- 30 epochs vs the paper's 250; simplified LR schedule.
- Absolute size is discarded by normalization.
- Points sampled once (no per-epoch re-sampling).
- ModelNet10 is clean, aligned CAD data. Real LiDAR scans are noisier and partial.

## 6. How to talk about using AI (honest and strong)
"I used an AI assistant to speed up the implementation and boilerplate. I then verified everything: a smoke test, unit tests, deliberately breaking the code to make sure the tests catch it, and running the experiments myself. I can walk through any part of the code." Then prove it with sections 3–4. This is normal now, and far safer than claiming you typed every line: a follow-up like "why did you zero-init that layer?" exposes that claim instantly, while understanding survives any question.

## 7. Resume bullet (fill the brackets **only** from your `run_summary.json`)
> **3D Point-Cloud Classification with PointNet (PyTorch).** Built an end-to-end pipeline converting 4,899 ModelNet10 CAD meshes to 1,024-point clouds (area-weighted sampling, unit-sphere normalization) and trained a PointNet classifier on a Colab T4, reaching **[X]% overall / [Y]% mean-class test accuracy**; ran controlled T-Net ablations ([result]) with validation-based model selection, plus a smoke test and 12 unit tests verifying shapes, gradients and permutation invariance.

## Self-check (final exam)
Answer questions 2, 3, 9, 11, 13, 17 and 19 out loud in under 30 seconds each, without notes. Then ask Claude to play interviewer.

## Ask Claude
- "Mock interview me on this project: 10 questions, increasing difficulty, and grade my answers."
- "Here is my `run_summary.json`. Fill in the pitch and resume bullet with my real numbers only."
