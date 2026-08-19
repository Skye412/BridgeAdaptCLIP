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

## Original AdaptCLIP supervised baseline

The Bridge2893 supervised baseline trains only the three upstream AdaptCLIP
adapters. It adds no new model module. Run training and validation with:

```bash
EXPERIMENT_DIR=results/<experiment_id> bash scripts/train_bridge2893_original.sh
EXPERIMENT_DIR=results/<experiment_id> bash scripts/validate_bridge2893_original.sh
```

Validation uses one-shot normal prompts for seeds 10, 20, and 30. The selected
checkpoint maximizes mean validation pixel AP, with mean image AUROC as the
tie-breaker. The frozen test split is evaluated only after selection.
