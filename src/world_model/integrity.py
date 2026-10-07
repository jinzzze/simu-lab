"""Torch-free provenance checks for the immutable world-model inputs.

Checks run before writers touch existing reports. Historical training source
hashes remain historical: this module never replaces the original manifest.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


class IntegrityError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    path = Path(path)
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
        raise IntegrityError(f'Missing artifact or SHA256 anchor: {path}')
    actual = sha256(path)
    if actual != expected:
        raise IntegrityError(f'SHA256 mismatch; preserve prior outputs: {path}')
    return actual


def relative_hashes(root, paths):
    root = Path(root).resolve()
    return {Path(path).resolve().relative_to(root).as_posix(): sha256(path) for path in paths}


def verify_receipt(root, receipt):
    if not isinstance(receipt, dict) or not receipt:
        raise IntegrityError('Missing verified-artifact receipt; run export_grasp_world_model_predictions.py first')
    root = Path(root).resolve()
    for name, expected in receipt.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise IntegrityError('Artifact receipt contains a path outside the project')
        require_hash(path, expected)


def load_training_data(root, metadata):
    """Validate the collection/training hash chain and whole-scene data contract."""
    root = Path(root)
    path = root / 'data/processed/grasp_world_model_v1.npz'
    manifest_path = root / 'data/manifests/grasp_world_model_v1.json'
    require_hash(manifest_path, metadata['dataset_manifest_sha256'])
    manifest = read_json(manifest_path)
    if manifest.get('status') != 'completed' or manifest.get('plan') != metadata['plan']:
        raise IntegrityError('Collection status/plan does not match the training manifest')
    require_hash(path, metadata['dataset_sha256'])
    require_hash(path, manifest['dataset_sha256'])
    checked = [path, manifest_path]
    for rel, field in [('configs/grasp_world_model.json', 'config_sha256'),
                       ('configs/grasp_sim.json', 'sim_config_sha256'),
                       ('src/sim/grasp_env.py', 'sim_source_sha256')]:
        target = root / rel
        require_hash(target, manifest[field]); checked.append(target)
    cfg = metadata['plan']
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key].copy() for key in stored.files}
    n = sum(spec['count'] for spec in cfg['splits'].values()) * cfg['actions_per_scene']
    shapes = {'initial_xy_m': (n, 2), 'command_xy_m': (n, 2), 'delta_xy_m': (n, 2),
              'binary_targets': (n, 2), 'scene_seeds': (n,), 'split': (n,), 'action_index': (n,)}
    for key, shape in shapes.items():
        if key not in data or data[key].shape != shape:
            raise IntegrityError(f'Invalid dataset array {key}; expected {shape}')
        if key != 'split' and (data[key].dtype.kind not in 'fiu' or not np.isfinite(data[key]).all()):
            raise IntegrityError(f'Dataset array {key} must be finite and numeric')
    if not np.isin(data['binary_targets'], [0, 1]).all():
        raise IntegrityError('Event labels must be binary')
    if set(data['split'].tolist()) != set(cfg['splits']):
        raise IntegrityError('Unknown or missing data split')
    rows = []
    for split, spec in cfg['splits'].items():
        source = root / f'data/processed/grasp_world_model_v1/{split}.jsonl'
        require_hash(source, manifest['jsonl_sha256'][split]); checked.append(source)
        part = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        expected = [(split, seed, action) for seed in range(spec['start'], spec['start'] + spec['count'])
                    for action in range(cfg['actions_per_scene'])]
        if [(r['split'], r['scene_seed'], r['action_index']) for r in part] != expected:
            raise IntegrityError(f'Incomplete, reordered or duplicate episodes in {split}')
        rows.extend(part)
    for key in ['initial_xy_m', 'command_xy_m', 'delta_xy_m']:
        if not np.array_equal(data[key], np.asarray([r[key] for r in rows], dtype=data[key].dtype)):
            raise IntegrityError(f'NPZ and source JSONL disagree for {key}')
    for key, field in [('scene_seeds', 'scene_seed'), ('split', 'split'), ('action_index', 'action_index')]:
        if not np.array_equal(data[key], [r[field] for r in rows]):
            raise IntegrityError(f'NPZ and source JSONL disagree for {key}')
    if not np.array_equal(data['binary_targets'], [[r['lifted'], r['success']] for r in rows]):
        raise IntegrityError('NPZ and source JSONL disagree for event labels')
    scene_splits = {}
    for scene, split in zip(data['scene_seeds'], data['split']):
        if scene in scene_splits and scene_splits[scene] != split:
            raise IntegrityError('A scene appears in multiple splits')
        scene_splits[scene] = split
    return data, relative_hashes(root, checked)


def check_ensemble_anchors(root, out, metadata):
    """Use prior completed reports when available; never rewrite their anchors."""
    path = Path(out) / 'ensemble.npz'
    for report in [Path(out) / 'results.json', Path(root) / 'artifacts/reports/grasp_world_model_v1/results.json']:
        if report.exists():
            prior = read_json(report)
            if prior.get('dataset_sha256') != metadata['dataset_sha256']:
                raise IntegrityError(f'Prior report belongs to different training data: {report}')
            require_hash(path, prior['ensemble_sha256'])


def validate_predictions(delta, probabilities, expected_shape):
    if delta.shape != expected_shape or probabilities.shape != expected_shape:
        raise IntegrityError(f'Prediction arrays must both have shape {expected_shape}')
    if not np.isfinite(delta).all() or not np.isfinite(probabilities).all():
        raise IntegrityError('Predictions contain nonfinite values')
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise IntegrityError('Event probabilities must lie in [0, 1]')


def require_prediction_agreement(delta, probabilities, member_predictions):
    delta_diff = float(np.max(np.abs(delta - member_predictions['delta_xy_m'])))
    prob_diff = float(np.max(np.abs(probabilities - member_predictions['probabilities'])))
    if delta_diff >= 1e-6 or prob_diff >= 2e-6:
        raise IntegrityError('Saved per-seed predictions disagree with the preserved NumPy weights')
    return delta_diff, prob_diff
