# 00 · Roadmap: learn the 20% that explains 80%

**Goal:** explain every design choice in this project confidently, defend it under follow-up questions, and invent your own experiments. **Non-goal:** retyping the code line by line.

## How to study (the "smart" method)

1. **Read** a note (10–15 min).
2. **Predict, then run** its 1–2 tiny snippets. Guess the printed shape *before* running. A wrong guess is where the learning happens.
3. **Answer the self-check questions out loud** without looking. If you can't, ask Claude using the prompts at the end of each note.
4. **Connect to code:** open the referenced function in [`dev/notebook_source.py`](../dev/notebook_source.py) and find the line the note talks about. You only *read* it; you don't rewrite it.

Where code lives: the notebook is built from `dev/notebook_source.py`. Each notebook cell has an **id** (e.g. `model`, `train_funcs`). The notes refer to `cell id → function`, e.g. `model → PointNet.forward`.

## 90-minute fast track (enough for most interviews)

| Order | Note | Why first |
|---|---|---|
| 1 | [03 PointNet architecture](03-pointnet-architecture.md) | The core idea: shared MLP + max pool = order doesn't matter |
| 2 | [02 PyTorch tensors & syntax](02-pytorch-tensors-and-syntax.md) | Read the model code without getting lost in shapes |
| 3 | [07 Code walkthrough](07-code-walkthrough.md) | Map every concept to a function you can point at |
| 4 | [08 Interview preparation](08-interview-preparation.md) | Rehearse answers, limitations, follow-ups |
| 5 | [09 Own these 3 functions](09-own-these-3-functions.md) | Rebuild the model, training loop and sampling yourself, so you can truthfully say you implemented them |

## Full path (about 2–2.5 hours, spread over a few days)

| # | Note | You will be able to explain… |
|---|---|---|
| 01 | [Point clouds & dataset](01-point-clouds-and-dataset.md) | mesh vs point cloud, OFF files, area-weighted sampling, unit-sphere normalization, the ModelNet10 split |
| 02 | [PyTorch tensors & syntax](02-pytorch-tensors-and-syntax.md) | shapes, `dim`, `nn.Module`/`forward`, `Linear` vs `Conv1d(k=1)`, `transpose`/`permute`, `bmm` |
| 03 | [PointNet architecture](03-pointnet-architecture.md) | shared MLP, global max pool, head, permutation invariance, parameter budget |
| 04 | [Training & backprop](04-training-and-backpropagation.md) | logits → cross-entropy → gradients → Adam; epochs, batches, LR, validation |
| 05 | [T-Nets & regularization](05-tnets-and-regularization.md) | 3×3 / 64×64 transforms, identity init, orthogonality penalty |
| 06 | [Evaluation & experiments](06-evaluation-and-experiments.md) | accuracy vs mean class accuracy, confusion matrix, checkpoints, controlled experiments, "what-if" menu |
| 07 | [Code walkthrough](07-code-walkthrough.md) | every notebook section → function → shapes |
| 08 | [Interview preparation](08-interview-preparation.md) | likely questions, accurate answers, limitations, honest framing |
| 09 | [Own these 3 functions](09-own-these-3-functions.md) | `PointNet`, `train_one_epoch`, `sample_points_from_mesh` line by line, with measured effects and a rebuild drill |

## The 6 ideas everything hangs on

1. A point cloud is an **unordered set** of `(x, y, z)` points → the model must not care about order.
2. **Shared MLP:** the same small network is applied to every point → `(B, N, 3) → (B, N, 1024)` features.
3. **Max pool over points** → one 1024-number "shape fingerprint" per object; max ignores order.
4. A normal **classifier head** turns the fingerprint into 10 class scores (logits).
5. **T-Nets** (optional here) learn a matrix to "straighten" the input before step 2.
6. **Honest evaluation:** pick the best epoch on *validation*, report *test* once.

## What is verified vs not (keep this straight in interviews)

| Claim | Status |
|---|---|
| Code runs end to end; shapes; smoke test (12/12); 13 unit tests | **Verified locally on CPU** with 70 real ModelNet10 meshes |
| Accuracy on ModelNet10 | **Not measured yet.** It will come from your Colab GPU run |
| Anything on Colab / GPU | **Not yet run** |

Never quote an accuracy until it is in your own `run_summary.json` (the notebook writes one per run, in section 9).
