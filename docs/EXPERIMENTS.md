# Experiment versioning

Every Bridge2893 run uses one immutable directory under `results/`:

```text
results/<experiment_id>/
├── analysis/          # result interpretation and next actions
├── checkpoints/       # trained models or copied input checkpoints
├── evaluation/        # logs, metrics, predictions, and visualizations
├── provenance/        # Git, environment, hardware, and dataset records
└── metrics.json       # machine-readable final metrics
```

Before starting a run:

1. Commit and push all code and configuration changes.
2. Create a new experiment directory with `scripts/init_experiment.sh`.
3. Point training checkpoint and evaluation output paths into that directory.
4. Never overwrite a completed experiment directory.
5. Record validation-based model selection and keep the frozen test split out of tuning decisions.

Example:

```bash
bash scripts/init_experiment.sh bridge2893_adapter_v002_<short-commit>
```

Experiment artifacts and models stay on the server and are ignored by Git.
Source code, scripts, and configuration are tracked in GitHub.
