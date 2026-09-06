"""Streaming metric helpers for native-resolution supervised comparisons."""

import numpy as np
import torch


class BinaryProtocolMetrics:
    def __init__(self, bins=2048):
        self.bins = bins
        self.positive = torch.zeros(bins, dtype=torch.float64)
        self.negative = torch.zeros(bins, dtype=torch.float64)
        self.image_targets = []
        self.image_scores = []

    def update(self, probabilities, targets, image_targets, image_scores=None):
        probabilities = probabilities.detach().cpu().float().clamp(0, 1)
        targets = targets.detach().cpu().bool()
        indices = (probabilities * (self.bins - 1)).floor().long()
        self.positive += torch.bincount(
            indices[targets], minlength=self.bins
        ).double()
        self.negative += torch.bincount(
            indices[~targets], minlength=self.bins
        ).double()
        self.image_targets.extend(
            torch.as_tensor(image_targets).detach().cpu().int().tolist()
        )
        if image_scores is None:
            image_scores = probabilities.flatten(1).max(dim=1).values
        self.image_scores.extend(torch.as_tensor(image_scores).detach().cpu().tolist())

    def compute(self):
        from sklearn.metrics import average_precision_score, roc_auc_score

        tp = torch.flip(self.positive, dims=(0,)).cumsum(0)
        fp = torch.flip(self.negative, dims=(0,)).cumsum(0)
        positives = self.positive.sum().clamp_min(1)
        negatives = self.negative.sum().clamp_min(1)
        recall = tp / positives
        precision = tp / (tp + fp).clamp_min(1)
        recall_with_origin = torch.cat([torch.zeros(1, dtype=recall.dtype), recall])
        ap = ((recall_with_origin[1:] - recall_with_origin[:-1]) * precision).sum()
        fpr = fp / negatives
        auroc = torch.trapz(recall, fpr)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
        best_index = int(torch.argmax(f1))
        # Reversed histogram index i corresponds to original bin bins-1-i.
        best_threshold = (self.bins - 1 - best_index) / (self.bins - 1)
        image_targets = np.asarray(self.image_targets, dtype=np.uint8)
        image_scores = np.asarray(self.image_scores, dtype=np.float64)
        return {
            'P-AP': 100.0 * float(ap),
            'P-AUROC': 100.0 * float(auroc),
            'P-F1max': 100.0 * float(f1[best_index]),
            'P-F1max-threshold': float(best_threshold),
            'I-AP': 100.0 * float(average_precision_score(image_targets, image_scores)),
            'I-AUROC': 100.0 * float(roc_auc_score(image_targets, image_scores)),
        }


def fixed_threshold_counts(probabilities, targets, threshold):
    predictions = probabilities >= threshold
    targets = targets.bool()
    return {
        'tp': int((predictions & targets).sum()),
        'fp': int((predictions & ~targets).sum()),
        'fn': int((~predictions & targets).sum()),
    }


def f1_from_counts(counts):
    denominator = 2 * counts['tp'] + counts['fp'] + counts['fn']
    return 100.0 * (2 * counts['tp'] / max(denominator, 1))
