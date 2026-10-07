"""Additional diagnostic: skip personal-video pretraining, keep adaptation identical."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.perception.models import make_encoder
from src.perception.training import load_arrays, provenance, seed_everything, write_json, state_digest
from adapt_visual_state import adapt_group


def main():
    plan_path = ROOT / 'configs/imagenet_reference_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    config = json.loads((ROOT / 'configs/visual_training.json').read_text())['adaptation']
    data_path = ROOT / 'data/processed/sim_adaptation_v1.npz'
    arrays = load_arrays(data_path, real=False)
    initial_root = ROOT / 'artifacts/runs/imagenet_reference_initialization'
    output = ROOT / 'artifacts/runs/visual_reference_state_v1'
    metadata = provenance(ROOT, data_path, config)
    metadata.update({'scope': plan['scope'], 'real_video_pretraining': False,
                     'role': 'additional ImageNet-only diagnostic, not one of A/B/C/D',
                     'diagnostic_plan': plan})
    results = []
    for seed in plan['training_seeds']:
        seed_everything(seed)
        encoder = make_encoder(pretrained=True)
        init = initial_root / f'R_seed{seed}'
        init.mkdir(parents=True, exist_ok=True)
        manifest = {'group': 'R', 'seed': seed, 'status': 'initialization_only', 'steps_completed': 0,
                    'real_data_used': False, 'initial_encoder_sha256': state_digest(encoder.state_dict()),
                    'initialization': 'ImageNet ResNet18 IMAGENET1K_V1; no phone-video frames',
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'format_note': 'visual_pretraining_v1 is the encoder checkpoint wire format only; no pretraining occurred'}
        path = init / 'checkpoint.pt'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite reference initialization: {path}')
        torch.save({'kind': 'visual_pretraining_v1', 'encoder_state': encoder.state_dict(), 'manifest': manifest}, path)
        write_json(init / 'manifest.json', manifest)
        del encoder
        results.append(adapt_group(arrays, 'R', seed, config, metadata, initial_root, output,
                                   'cuda' if torch.cuda.is_available() else 'cpu'))
    write_json(output / 'summary.json', results)


if __name__ == '__main__':
    main()
