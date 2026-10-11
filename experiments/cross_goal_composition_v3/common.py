"""Shared provenance and split checks; no Torch/Bullet imported here."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=ROOT.parent/'paired_two_board_v2'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(PROJECT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def allowed_pair(start,goal):
    return [start,goal] in read(ROOT/'protocol.json')['training_pairs']


def verify_training_pairs(entries):
    if not entries:
        raise ValueError('No training episodes')
    for row in entries:
        if row['split']!='train' or not allowed_pair(row['start_id'],row['goal_id']):
            raise ValueError('Held-out pair or non-training split in training data')


def snapshot(out,paths):
    dest=out/'source_snapshot';dest.mkdir()
    hashes={}
    for path in paths:
        key=str(path.relative_to(PROJECT)).replace('\\','/')
        hashes[key]=sha(path)
        target=dest/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    return hashes


def check_sources(hashes):
    for name,digest in hashes.items():
        if sha(PROJECT/name)!=digest:
            raise ValueError('Source changed: '+name)


def source_paths():
    files=[ROOT/n for n in ('common.py','teacher.py','protocol.json')]
    files += [OLD/n for n in ('paired_sim.py','goal_scoring.py','configs/sim_task_v1.json','configs/sim_physics_v1.json')]
    files += [PROJECT/'src/sim/grasp_env.py']
    return files
