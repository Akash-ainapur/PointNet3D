# results/

Evidence for every number quoted in the README, a resume or an interview. **Only real runs go here.**

For each Colab run, unzip the downloaded `<run name>.zip` into `results/<run name>/`:

```
results/
└── full_baseline_n1024_seed42_20261020-153012/
    ├── run_summary.json   # config, seed, versions, GPU, per-epoch history + timing, best epoch, test metrics, confusion matrix
    ├── curves.png
    └── confusion.png
```

Rules:
- Never edit a `run_summary.json` by hand.
- Don't commit checkpoints (`*.pt`) or caches (`*.npz`); `.gitignore` blocks them.
- `quick_*` runs prove the pipeline only. Their summaries say "accuracy NOT meaningful", so don't put them in the README results table.
- Fill the README "Results" table only from files in this folder.
