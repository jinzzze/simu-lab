"""Isolate Torch from Bullet; never send object labels to the action policy."""
import base64,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


class PolicyProcess:
    def __init__(self,checkpoint,stderr_path):
        self.log=Path(stderr_path).open('w',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/diffusion_policy_worker.py'),
            '--checkpoint',str(checkpoint)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,encoding='utf-8',bufsize=1,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            if not self.read().get('ready'):raise RuntimeError('Policy worker unavailable')
        except Exception:
            self.close();raise

    def read(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Policy worker exited; inspect stderr log')
        return json.loads(line)

    def request(self,value):
        self.process.stdin.write(json.dumps(value,allow_nan=False)+'\n');self.process.stdin.flush()
        return self.read()

    def reset(self,rgb,sampling_seed):
        image=np.asarray(rgb)
        if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('Invalid RGB')
        result=self.request({'command':'reset','shape':image.shape,'rgb':base64.b64encode(image.tobytes()).decode('ascii'),
                             'sampling_seed':int(sampling_seed)})
        xy=np.asarray(result['initial_visual_xy_m'],dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Invalid visual prediction')
        return xy

    def predict(self,history):
        obs=np.asarray(history)
        if obs.shape!=(2,11) or not np.isfinite(obs).all():raise ValueError('Invalid observation history')
        result=np.asarray(self.request({'command':'predict','history':obs.tolist()})['actions'],dtype=float)
        if result.shape!=(16,4) or not np.isfinite(result).all():raise ValueError('Invalid action chunk')
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"close"}\n');self.process.stdin.flush();self.process.wait(timeout=10)
            except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                self.process.kill();self.process.wait(timeout=10)
        for stream in (self.process.stdin,self.process.stdout):
            if stream:stream.close()
        self.log.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
