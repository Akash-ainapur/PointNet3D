# 06 · Evaluation, checkpoints and controlled experiments

**Code:** `dev/notebook_source.py`, cells `train_funcs → evaluate`, `train_funcs → fit` (checkpointing), `eval` (test metrics + confusion matrix), markdown cell `s10` (limits).

## 1. Two accuracies, and why you report both

- **Overall accuracy** = correct / total. Dominated by big classes.
- **Mean class accuracy** = average of the 10 per-class accuracies. Every class counts equally.

ModelNet10 is imbalanced (test: 100 chairs but only 50 bathtubs). A model that is great on chairs and bad on bathtubs can have high overall accuracy but lower mean class accuracy. The PointNet paper reports both.

**Predict, then run:**
```python
import numpy as np
cm = np.array([[90, 10],      # rows = TRUE class, columns = PREDICTED class
               [ 8,  2]])     # class 0: 100 samples, class 1: 10 samples
overall = np.trace(cm) / cm.sum()
per_class = np.diag(cm) / cm.sum(axis=1)
print(round(overall, 3), per_class.round(2), round(per_class.mean(), 3))   # → ?
```
<!-- expect: 0.836 [0.9 0.2] 0.55 -->
<details><summary>Answer</summary><code>0.836 [0.9 0.2] 0.55</code>. A decent-looking overall accuracy hides that class 1 is mostly wrong. <code>evaluate</code> computes exactly this from the confusion matrix (<code>np.trace(cm) / cm.sum()</code>, <code>np.diag(cm) / support</code>). Unit test <code>evaluate_confusion_matrix_and_metrics</code> verifies it.</details>

## 2. Confusion matrix (cell `eval`)

Row = true class, column = predicted class. Diagonal = correct. Off-diagonal cells tell you **which** classes get confused. Expect geometrically similar pairs (e.g. desk vs table, dresser vs night_stand), but **check your own matrix**; that is a hypothesis, not a result. This is the best material for an interview story: "the model mostly confused X with Y because…".

## 3. Checkpoints (`fit` → `torch.save`, cell `eval` → `torch.load`)

Saved whenever validation accuracy improves: `model_state` (all weights), `epoch`, `val_acc`, `model_kwargs`, `class_names`, `num_points`.
At the end, cell `eval` rebuilds `PointNet(**ckpt["model_kwargs"])`, loads the weights, and scores the **test** set once.
Why save `model_kwargs`? A checkpoint is useless if you don't know which architecture (T-Nets on/off) it belongs to.
*Verified:* unit test `checkpoint_roundtrip_with_weights_only_load` (save → load with PyTorch's strict loader → identical outputs).
**Run record (cell `save_run`):** every run gets its own folder `runs/<mode>_<variant>_n<points>_seed<seed>_<time>/` with `best_model.pt`, `curves.png`, `confusion.png` and **`run_summary.json`** (config, seed, versions, GPU, per-epoch history + timing, best epoch, test metrics, per-class accuracy, confusion matrix), built by `train_funcs → build_run_summary`. Unit test `run_summary_is_valid_json_with_measured_fields` checks it. Copy the small zip into the repo's `results/` folder.

## 4. The evaluation discipline (memorize this)

1. **Train** on train, **select** (best epoch, best variant) on **validation**.
2. Look at **test** once per final model, to *report*, never to choose.
3. Change **one thing at a time**; keep seed, epochs and data the same.
4. One run is one sample. Differences of about a percent between two runs can be noise. Repeat with 2–3 seeds before claiming "X is better".

**Seeds and the cache:** the point-cloud cache name contains the point count, mode **and seed** (`config → CACHE_PATH`), so a new `SEED` re-samples the points too, not just the model init, shuffling and validation split. `DETERMINISTIC = True` asks cuDNN for deterministic kernels; GPU results can still differ slightly between runs, so judge differences across seeds.

## 5. Experiment protocol for this project

| # | Run | Changes vs baseline | Compare by |
|---|---|---|---|
| 0 | quick mode | — | smoke test passes, curves move: **not** an accuracy result |
| 1 | **baseline** (full, 30 epochs) | — | val + test, per-class, confusion matrix, runtime |
| 2 | + input T-Net | `USE_INPUT_TNET=True` | validation acc, time/epoch |
| 3 | + feature T-Net | `USE_FEATURE_TNET=True` | validation acc, time/epoch |
| 4 | both T-Nets | both `True` | validation acc, time/epoch |
| 5 | point count | `NUM_POINTS` 512 / 2048 (new cache) | validation acc, time/epoch |

Step-by-step Colab instructions with expected outputs: [`COLAB_CHECKLIST.md`](../COLAB_CHECKLIST.md). Record every run (config + numbers) **as measured**. A table where every number came from a saved file is exactly what makes this resume-worthy.

## 6. "What-if" menu: your own experiment ideas

Each is small, and each teaches you something you can talk about:

| What if… | Change | What it tests |
|---|---|---|
| max → mean pooling | one line in `PointNet.forward` | why the paper chose max |
| fewer points at **test** time (train 1024, test 256) | subsample `X_test[:, :256]` in `eval` | robustness to sparse scans |
| randomly drop 50% of test points | random index per cloud | robustness / critical points |
| add noise to test points | `X_test + 0.02·randn` | sensor-noise robustness |
| random **3D** rotations at test time | rotate around x/y too | the limit of z-only augmentation; do T-Nets help here? |
| no augmentation | `augment=False` in `train_one_epoch` | how much augmentation is worth |
| class-weighted loss | `F.cross_entropy(..., weight=w)` | does mean class accuracy improve? |
| plot critical points | `max(dim=2).indices` | which points the model actually uses |

Rule: decide on **validation**, then report test once for the final pick.

## Simplified vs the paper
Same two metrics. The paper's numbers are on ModelNet40 with 250 epochs, so **don't compare your ModelNet10 numbers to the paper's ModelNet40 numbers** as if they were the same benchmark.

## Self-check
1. Overall vs mean class accuracy: when do they differ a lot?
2. Read a confusion matrix: what do rows, columns and the diagonal mean?
3. Why must model selection use validation, not test?
4. What's stored in the checkpoint, and why `model_kwargs`?
5. Design a fair experiment to test whether the feature T-Net helps.

## Ask Claude
- "Help me implement the 'fewer points at test time' experiment without touching training."
- "How many seeds do I need before I can say one variant is better?"
