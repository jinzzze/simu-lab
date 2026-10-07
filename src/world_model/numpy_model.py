"""Pure NumPy macro-action inference with fail-closed artifact validation."""
from pathlib import Path
import numpy as np


def _real_finite(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite real numeric values')
    return array


def features(initial_xy, command_xy):
    state = _real_finite(initial_xy, 'State')
    command = _real_finite(command_xy, 'Command')
    if state.shape != command.shape or state.shape[-1:] != (2,) or state.ndim not in (1, 2):
        raise ValueError('State and command must have matching (2,) or (N,2) shapes')
    if state.size == 0:
        raise ValueError('State and command batches must be nonempty')
    with np.errstate(over='ignore', invalid='ignore'):
        state = state.astype(np.float32)
        command = command.astype(np.float32)
        result = np.concatenate([state / .1, (command - state) / .03], axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('State and command exceed finite float32 feature range')
    return result


class MacroOutcomePredictor:
    """Predict one fixed grasp skill from external XY and the actual XY command.

    No simulator handle or truth accessor is accepted. Event probabilities are
    averaged after sigmoid; displacements are averaged across all saved seeds.
    """
    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as data:
            self.weights = {key: data[key].copy() for key in data.files}
        if not {'seeds', 'delta_scale_m'}.issubset(self.weights):
            raise ValueError('Model is missing seeds or displacement scale')
        seeds = self.weights['seeds']
        if seeds.ndim != 1 or seeds.size == 0 or seeds.dtype.kind not in 'iu':
            raise ValueError('Model seeds must be a nonempty integer vector')
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError('Model seeds must be unique')
        self.seeds = seeds.tolist()
        scale = _real_finite(self.weights['delta_scale_m'], 'Displacement scale')
        if scale.shape != () or float(scale) <= 0:
            raise ValueError('Displacement scale must be a finite positive scalar')
        self.delta_scale = float(scale)
        expected = {'seeds', 'delta_scale_m'}
        shapes = {0: ((64, 4), (64,)), 2: ((64, 64), (64,)), 4: ((4, 64), (4,))}
        for seed in self.seeds:
            for layer, (wshape, bshape) in shapes.items():
                for suffix, shape in [('w', wshape), ('b', bshape)]:
                    key = f's{seed}_{suffix}{layer}'
                    expected.add(key)
                    if key not in self.weights:
                        raise ValueError(f'Model is missing {key}')
                    value = _real_finite(self.weights[key], key)
                    if value.shape != shape:
                        raise ValueError(f'{key} must have shape {shape}, received {value.shape}')
                    if value.dtype.kind != 'f':
                        raise ValueError(f'{key} must use floating point weights')
        if set(self.weights) != expected:
            raise ValueError('Unexpected model arrays; model architecture must match the fixed MLP')

    def predict_members(self, initial_xy, command_xy):
        """Per-seed predictions, preserving the saved seed order for export audits."""
        x = features(initial_xy, command_xy)
        outputs = []
        with np.errstate(over='ignore', invalid='ignore'):
            for seed in self.seeds:
                h = x
                for layer in (0, 2, 4):
                    h = h @ self.weights[f's{seed}_w{layer}'].T + self.weights[f's{seed}_b{layer}']
                    if layer != 4:
                        h = np.maximum(h, 0)
                outputs.append(h)
            stacked = np.stack(outputs)
            delta = stacked[..., :2] * self.delta_scale
        if not np.isfinite(stacked).all() or not np.isfinite(delta).all():
            raise ValueError('World model produced nonfinite outputs')
        probabilities = 1 / (1 + np.exp(-np.clip(stacked[..., 2:], -60, 60)))
        return {'delta_xy_m': delta, 'probabilities': probabilities}

    def __call__(self, initial_xy, command_xy):
        members = self.predict_members(initial_xy, command_xy)
        delta = members['delta_xy_m'].mean(axis=0)
        probabilities = members['probabilities'].mean(axis=0)
        final = np.asarray(initial_xy, dtype=np.float32) + delta
        if not np.isfinite(final).all():
            raise ValueError('World model produced nonfinite final coordinates')
        return {'delta_xy_m': delta, 'final_xy_m': final,
                'lift_probability': probabilities[..., 0], 'success_probability': probabilities[..., 1]}
