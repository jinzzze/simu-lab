"""Preserve the zero-update CUDA failure and register equivalent pooling retry."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
folder=ROOT/'artifacts/runs/paired_image_policy_v1/simulation_only_seed7'
meta=read(folder/'manifest.json');assert not (folder/'checkpoint.pt').exists() and (folder/'losses.jsonl').stat().st_size==0
meta.update(status='failed',steps_completed=0,error='CUDA deterministic backward unavailable for AdaptiveAvgPool2d; first backward failed before optimizer.step',failed_utc=datetime.now(timezone.utc).isoformat())
write(folder/'manifest.json',meta)
config=read(ROOT/'configs/policy_v1.json');config['revision']='paired_image_policy_v1a'
config['implementation_repair']='deterministic fixed AvgPool2d(4,4) is equivalent to adaptive 4x4 pooling on fixed 16x16 maps; no training budget/data/model dimension change'
config['supersedes_failed_execution']='paired_image_policy_v1; zero optimizer updates'
config['created_utc']=datetime.now(timezone.utc).isoformat()
path=ROOT/'configs/policy_v1a.json';assert not path.exists();write(path,config)
receipt=ROOT/'receipts/paired_policy_v1a_cuda_repair';receipt.mkdir();shutil.copy2(path,receipt/'policy_v1a.json')
write(receipt/'manifest.json',{'frozen_utc':config['created_utc'],'config_sha256':sha(path),'previous_failed_run':str(folder.relative_to(ROOT)).replace('\\','/'),
                             'same_budget_and_research_variables':True,'pool_operation_equivalence_test':'tests/test_paired_image_policy.py',
                             'previous_initial_weights_sha256':meta['initial_weights_sha256']})
print('CUDA pooling repair frozen; original zero-update failure preserved')
