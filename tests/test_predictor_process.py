"""Exercise live Bullet RGB -> isolated real Torch predictor -> finite XY."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predictor_process import PredictorProcess
from src.sim.grasp_env import GraspEnv


def test_live_simulator_rgb_matches_direct_predictor(tmp_path):
    env = GraspEnv(image_size=224)
    try:
        rgb = env.reset(991)['rgb']
        np.save(tmp_path / 'rgb.npy', rgb)
        # Torch never imports into this parent process. The random checkpoint
        # is a transport fixture only; it is not a trained research result.
        code = '''
import json, sys
from pathlib import Path
import numpy as np
import torch
from src.perception.models import StateModel, load_state_predictor
torch.set_num_threads(4)
torch.manual_seed(4)
folder=Path(sys.argv[1])
model=StateModel([-.0675,0.],[.02,.03])
torch.save({'kind':'visual_state_xy_v1','model_state':model.state_dict(),
    'target_mean':[-.0675,0.],'target_scale':[.02,.03]},folder/'fixture.pt')
pred=load_state_predictor(folder/'fixture.pt',device='cpu')(np.load(folder/'rgb.npy'))
(folder/'expected.json').write_text(json.dumps(pred.tolist()))
'''
        result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        expected = np.asarray(json.loads((tmp_path / 'expected.json').read_text()))
        with PredictorProcess(tmp_path / 'fixture.pt', tmp_path / 'predictor_stderr.log') as predict:
            estimated = predict(rgb)
            assert np.allclose(estimated, expected, atol=1e-6, rtol=1e-5)
            assert np.array_equal(estimated, predict(rgb))
            with pytest.raises(ValueError, match='RGB uint8'):
                predict(rgb.astype(float))
        # The simulator remains usable after model loading/inference.
        assert env.reset(992)['rgb'].shape == (224, 224, 3)
    finally:
        env.close()
