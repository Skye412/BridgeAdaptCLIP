"""Select a Bridge2893 checkpoint from repeated validation logs."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


METRICS = ('I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max')


def parse_log(path):
    for line in path.read_text(encoding='utf-8').splitlines():
        if re.match(r'^\|\s*structural defects\s*\|', line):
            values = [part.strip() for part in line.strip().strip('|').split('|')]
            if len(values) != len(METRICS) + 1:
                raise ValueError(f'Unexpected metric row in {path}: {line}')
            return {name: float(value) for name, value in zip(METRICS, values[1:])}
    raise ValueError(f'No structural defects metric row found in {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_root', type=Path, required=True)
    parser.add_argument('--checkpoint_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k_shots', type=int, default=1)
    parser.add_argument('--metric_resolution', type=int)
    args = parser.parse_args()

    epochs = []
    for epoch_dir in sorted(args.validation_root.glob('epoch_*'), key=lambda p: int(p.name.split('_')[-1])):
        epoch = int(epoch_dir.name.split('_')[-1])
        logs = sorted(epoch_dir.glob(f'bridge2893_*seed_{args.k_shots}shot_test_log.txt'))
        if not logs:
            raise ValueError(f'No validation logs found in {epoch_dir}')
        runs = [parse_log(log) for log in logs]
        summary = {
            metric: {
                'mean': float(np.mean([run[metric] for run in runs])),
                'std': float(np.std([run[metric] for run in runs])),
            }
            for metric in METRICS
        }
        epochs.append({'epoch': epoch, 'runs': runs, 'summary': summary})

    best = max(epochs, key=lambda item: (item['summary']['P-AP']['mean'], item['summary']['I-AUROC']['mean']))
    report = {
        'selection_metric': (
            'validation P-AP for 0-reference'
            if args.k_shots == 0
            else f'mean validation P-AP over {args.k_shots}-reference seeds'
        ),
        'tie_breaker': 'mean validation I-AUROC',
        'k_shots': args.k_shots,
        'metric_resolution': args.metric_resolution,
        'best_epoch': best['epoch'],
        'best_checkpoint': str(args.checkpoint_root / f"epoch_{best['epoch']}.pth"),
        'best_summary': best['summary'],
        'epochs': epochs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('best_epoch', 'best_checkpoint', 'best_summary')}, indent=2))


if __name__ == '__main__':
    main()
