# Colab execution checklist

Tick each box. If a **STOP** condition happens, don't continue; bring the output back to Claude. Expected values below were computed from the notebook's own code; anything about Colab speed is unknown until you run it.

## 0. Setup (2 min)
- [ ] Use the notebook from the `main` branch: `PointNet_ModelNet10.ipynb`.
- [ ] <https://colab.research.google.com> → **File → Upload notebook**.
- [ ] **Runtime → Change runtime type → T4 GPU** (free tier; no paid plan needed).

## 1. Quick mode: wiring check (`QUICK_MODE = True`, the default)
**Runtime → Run all**, then check each section:

| § | You should see | STOP if |
|---|---|---|
| 1 | `device: cuda (Tesla T4)` (or similar), `mode: quick \| variant: baseline \| points: 1024 \| seed: 42 \| epochs: 5`, an `Open3D 0.x` line | `device: cpu` → fix the runtime type first |
| 2 | download progress up to ~473 MB, `Done: 4899 meshes.`, class table with **3,991 train / 908 test** (this table always lists the full dataset) | assertion about mesh count or classes |
| 3 | `vertices (…, 3) faces (…, 3)`, `sampled points (1024, 3)`, `max distance from origin = 1.0000` | — |
| 4 | interactive mesh + point cloud (drag to rotate); gallery of 10 classes, **all upright** (z up) | objects lying sideways/upside down (a blank 3D view is *not* blocking: it's the one part never tested on Colab; note it and continue) |
| 5 | preprocessing of **1000 + 300** meshes; split table TOTAL **900 / 100 / 300**; batch `(32, 1024, 3) torch.float32 \| (32,) torch.int64` | `WARNING: skipped N unreadable meshes` with N > 0 → note it |
| 6 | shape trace ending `(4, 10)`, `parameters: 811,914` | — |
| 7 | **`SMOKE TEST PASSED (12/12 checks)`** | **any `[FAIL]`** |
| 8 | 5 epoch lines, printed lr `0.00100, 0.00090, 0.00065, 0.00035, 0.00010`, epoch time in seconds; train loss starts near 2.3 and goes down | `nan`, or train loss not decreasing at all |
| 9 | test metrics, confusion matrix, `saved: [...]`, a zip downloads (allow downloads if the browser asks) | — |

- [ ] Note the **epoch time** from §8 (quick mode = 28 steps per epoch). Quick-mode accuracy means nothing; don't record it as a result.

## 2. Full baseline: the real experiment
- [ ] Set `QUICK_MODE = False` (config cell), then **Runtime → Run all**. The download is skipped; the full point-cloud cache is built once (4,899 meshes; took ~4–6 min extrapolated on a laptop CPU; Colab time unknown).
- [ ] Check: `mode: full`, split TOTAL **3,592 / 399 / 908**, smoke test **12/12**, epoch lines **1/30 … 30/30** (112 steps per epoch).
- [ ] After epoch 1: total time ≈ 30 × epoch-1 time. Keep the tab open; free Colab sessions can disconnect when idle.
- [ ] Printed lr falls from `0.00100` to `0.00000` at epoch 30 (it's really 2.7e-6, rounded). That's expected.
- **STOP** if validation accuracy is still about 0.10 after 5 epochs, or any loss is `nan`.
- [ ] §9: the zip `full_baseline_n1024_seed42_<time>.zip` downloads. If it doesn't: Files panel (left) → `pointnet_modelnet10/outputs/runs/` → download it.

## 3. Record it (5 min, on your PC)
- [ ] Unzip into `results/full_baseline_n1024_seed42_<time>/` (`run_summary.json`, `curves.png`, `confusion.png`).
- [ ] Commit:
  ```bash
  git add results
  git commit -m "Add full baseline run (seed 42)"
  ```
- [ ] Send Claude the `run_summary.json`. It will fill the README Results table and the resume bullet / interview answers in `learning/08` **from that file only**.

## 4. Controlled variations, only after step 3
Change **one** setting per run, keep everything else identical, and record each run as in step 3.

| Order | Change | Why |
|---|---|---|
| 1 | `SEED = 43`, then `SEED = 44` (baseline) | how much results vary run to run; without this, no comparison is trustworthy |
| 2 | `USE_INPUT_TNET = USE_FEATURE_TNET = True`, seeds 42/43/44 | do T-Nets help on aligned ModelNet10? (expect slower epochs: about 4× the parameters) |
| 3 | optional: only one T-Net; `NUM_POINTS = 512` or `2048` | finer ablations |

- [ ] Decide between variants by **mean validation accuracy across seeds**. Report their test numbers, but never pick a variant because its *test* score is higher.
- [ ] Differences smaller than the seed-to-seed spread from row 1 are "no clear difference". Say exactly that in interviews.

**Minimum for a strong resume story:** steps 0–3 plus rows 1 and 2 = **6 full runs** (3 seeds × 2 variants).
