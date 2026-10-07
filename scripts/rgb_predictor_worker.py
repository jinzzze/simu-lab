"""Persistent RGB-only predictor process; isolates Torch from simulator OpenMP."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.perception import load_state_predictor
from src.perception.training import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    # Match adaptation evaluation: deterministic kernels, TF32 disabled.
    seed_everything(0)
    predict = load_state_predictor(args.checkpoint)
    print(json.dumps({'ready': True, 'inputs': 'RGB uint8 only'}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('command') == 'close':
            break
        # No scene identifiers, labels, task coordinates or outcomes are sent.
        shape = tuple(request['shape'])
        if len(shape) != 3 or shape[-1] != 3 or max(shape) > 4096:
            raise ValueError('Invalid RGB image shape')
        rgb = np.frombuffer(base64.b64decode(request['rgb'], validate=True), dtype=np.uint8).reshape(shape)
        xy = np.asarray(predict(rgb), dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Invalid model prediction')
        print(json.dumps({'xy_m': xy.tolist()}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
