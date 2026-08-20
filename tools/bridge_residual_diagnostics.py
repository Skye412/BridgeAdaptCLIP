"""Streaming statistics for BridgeAdaptCLIP gated-residual diagnostics."""

import math

import torch


MAP_NAMES = ('gate', 'residual', 'correction', 'abs_correction')


class RegionAccumulator:
    """Accumulate exact regional means without retaining full-resolution maps."""

    def __init__(self):
        self.states = {}

    def update(self, name, mask, gate, residual, correction, error):
        mask = mask.bool()
        count = int(mask.sum())
        if count == 0:
            return
        selected_gate = gate[mask].float()
        selected_residual = residual[mask].float()
        selected_correction = correction[mask].float()
        selected_error = error[mask].float()
        state = self.states.setdefault(name, {
            'count': 0,
            'gate_sum': 0.0,
            'residual_sum': 0.0,
            'correction_sum': 0.0,
            'abs_correction_sum': 0.0,
            'positive_correction_sum': 0.0,
            'negative_correction_sum': 0.0,
            'error_sum': 0.0,
        })
        state['count'] += count
        state['gate_sum'] += float(selected_gate.double().sum())
        state['residual_sum'] += float(selected_residual.double().sum())
        state['correction_sum'] += float(selected_correction.double().sum())
        state['abs_correction_sum'] += float(selected_correction.abs().double().sum())
        state['positive_correction_sum'] += float(
            selected_correction.clamp_min(0).double().sum()
        )
        state['negative_correction_sum'] += float(
            selected_correction.clamp_max(0).double().sum()
        )
        state['error_sum'] += float(selected_error.double().sum())

    def finalize(self):
        report = {}
        for name, state in self.states.items():
            count = state['count']
            report[name] = {
                'pixel_count': count,
                'mean_gate': state['gate_sum'] / count,
                'mean_residual': state['residual_sum'] / count,
                'mean_correction': state['correction_sum'] / count,
                'mean_abs_correction': state['abs_correction_sum'] / count,
                'mean_positive_correction': state['positive_correction_sum'] / count,
                'mean_negative_correction': state['negative_correction_sum'] / count,
                'mean_row0_error': state['error_sum'] / count,
            }
        return report


class PearsonAccumulator:
    """Exact streaming Pearson correlation from sufficient statistics."""

    def __init__(self):
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x, y):
        x = x.float().reshape(-1)
        y = y.float().reshape(-1)
        if x.numel() != y.numel():
            raise ValueError('Pearson inputs must contain the same number of values.')
        self.count += x.numel()
        self.sum_x += float(x.double().sum())
        self.sum_y += float(y.double().sum())
        self.sum_x2 += float((x.double() * x.double()).sum())
        self.sum_y2 += float((y.double() * y.double()).sum())
        self.sum_xy += float((x.double() * y.double()).sum())

    def finalize(self):
        if self.count < 2:
            return None
        n = self.count
        covariance_numerator = self.sum_xy - self.sum_x * self.sum_y / n
        variance_x = self.sum_x2 - self.sum_x * self.sum_x / n
        variance_y = self.sum_y2 - self.sum_y * self.sum_y / n
        denominator = math.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
        return covariance_numerator / denominator if denominator else None
