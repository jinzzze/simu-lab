"""Bullet-side IPC client; never imports Torch or sends object truth/stage."""
from pathlib import Path
import base64,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent


class ImagePolicyProcess:
    def __init__(self,checkpoint,stderr_path):
        self.log=Path(stderr_path).open('w',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,'-u',str(ROOT/'paired_policy_worker.py'),'--checkpoint',str(checkpoint)],
                                      cwd=ROOT.parents[1],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                                      encoding='utf-8',text=True,bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            if not self.read().get('ready'):raise RuntimeError('Worker not ready')
        except Exception:self.close();raise

    def read(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Image-policy worker exited; inspect stderr log')
        return json.loads(line)

    def request(self,value):
        self.process.stdin.write(json.dumps(value,allow_nan=False)+'\n');self.process.stdin.flush();return self.read()

    def reset(self,seed):assert self.request({'command':'reset','sampling_seed':int(seed)})['reset']

    def predict(self,images,vectors,blank_rgb=False):
        images=np.asarray(images,np.uint8);vectors=np.asarray(vectors,np.float32)
        assert images.shape==(2,224,224,3) and vectors.shape==(2,12) and np.isfinite(vectors).all()
        rgb=np.ascontiguousarray(images).tobytes();vec=np.ascontiguousarray(vectors).tobytes()
        reply=self.request({'command':'predict','shape':list(images.shape),'rgb':base64.b64encode(rgb).decode('ascii'),
                            'vectors':vectors.tolist(),'blank_rgb':bool(blank_rgb)})
        assert reply['received_rgb_sha256']==hashlib.sha256(rgb).hexdigest()
        assert reply['received_vector_sha256']==hashlib.sha256(vec).hexdigest()
        actions=np.asarray(reply['actions'],np.float32)
        if actions.shape!=(16,4) or not np.isfinite(actions).all():raise ValueError('Invalid action chunk')
        return actions

    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.write('{"command":"close"}\n');self.process.stdin.flush();self.process.wait(timeout=10)
            except (BrokenPipeError,OSError,subprocess.TimeoutExpired):self.process.kill();self.process.wait(timeout=10)
        for pipe in (self.process.stdin,self.process.stdout):
            if pipe:pipe.close()
        self.log.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
