"""Fixed 32-image follow-up, sharing all adaptation settings within each seed."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.perception.training import load_arrays, provenance, write_json
from adapt_visual_state import adapt_group


def main():
    plan = json.loads((ROOT / 'configs/low_data_plan.json').read_text(encoding='utf-8-sig'))
    config = json.loads((ROOT / 'configs/visual_training.json').read_text())['adaptation']
    source = ROOT / 'data/processed/sim_adaptation_v1.npz'
    with np.load(source, allow_pickle=False) as data:
        chosen = (data['split'] != 'train') | (data['scene_seeds'] < 10032)
        arrays = {key: data[key][chosen] for key in data.files}
    assert np.sum(arrays['split'] == 'train') == plan['training_images']
    assert np.array_equal(arrays['scene_seeds'][arrays['split'] == 'train'], np.arange(10000,10032))
    path = ROOT / 'data/processed/sim_adaptation_32_v1.npz'
    if path.exists():
        raise FileExistsError(f'Preserve existing low-data input: {path}')
    np.savez_compressed(path, **arrays)
    data_manifest = {'plan': plan, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'scene_seeds': arrays['scene_seeds'].tolist(), 'splits': arrays['split'].tolist()}
    write_json(ROOT / 'data/manifests/sim_adaptation_32_v1.json', data_manifest)
    arrays = load_arrays(path, real=False)
    metadata = provenance(ROOT, path, config)
    metadata.update({'scope': plan['scope'], 'low_data_plan': plan,
                     'extra_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    output = ROOT / 'artifacts/runs/visual_low32_state_v1'
    results = []
    for seed in plan['training_seeds']:
        for group in plan['groups']:
            initialization = ROOT / 'artifacts/runs' / ('imagenet_reference_initialization' if group == 'R' else 'visual_pretraining_v1')
            results.append(adapt_group(arrays, group, seed, config, metadata, initialization, output,
                                       'cuda' if torch.cuda.is_available() else 'cpu'))
    write_json(output / 'summary.json', results)


if __name__ == '__main__':
    main()
