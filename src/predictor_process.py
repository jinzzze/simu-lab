"""Process boundary keeps incompatible Torch/Bullet OpenMP runtimes separate."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PredictorProcess:
    """Same RGB predictor as training, with only image bytes crossing the boundary."""
    def __init__(self, checkpoint, stderr_path):
        self._log = Path(stderr_path).open('w', encoding='utf-8')
        self._process = subprocess.Popen(
            [sys.executable, '-u', str(ROOT / 'scripts/rgb_predictor_worker.py'),
             '--checkpoint', str(Path(checkpoint).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, encoding='utf-8', bufsize=1, cwd=ROOT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            if not self._read().get('ready'):
                raise RuntimeError('Predictor worker did not become ready')
        except Exception:
            self.close()
            raise

    def _read(self):
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError('Predictor exited; inspect predictor_stderr.log')
        return json.loads(line)

    def __call__(self, rgb):
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected RGB uint8 HWC image')
        request = {'shape': image.shape, 'rgb': base64.b64encode(image.tobytes()).decode('ascii')}
        self._process.stdin.write(json.dumps(request) + '\n')
        self._process.stdin.flush()
        xy = np.asarray(self._read()['xy_m'], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Worker returned invalid XY')
        return xy

    def close(self):
        if self._process.poll() is None:
            try:
                self._process.stdin.write('{"command":"close"}\n')
                self._process.stdin.flush()
                self._process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait(timeout=10)
        for stream in (self._process.stdin, self._process.stdout):
            if stream:
                stream.close()
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
