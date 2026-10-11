"""Freeze paired input-intervention controls before rollout outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parent
path=ROOT/'configs/evaluation_addendum_v1a.json';assert not path.exists()
payload={'frozen_utc':datetime.now(timezone.utc).isoformat(),
         'policy_config_sha256':hashlib.sha256((ROOT/'configs/policy_v1a.json').read_bytes()).hexdigest(),
         'diagnostic_controls':{'blank_rgb_control':'same seeds 62300-62303 and sampling seeds as blank_rgb, unmodified RGB',
                                'swapped_goal_vector_control':'same seeds 62400-62403 and sampling seeds as swapped_goal_vector, unmodified goal vector'},
         'scope':'seed7 per condition only; diagnostic input corruption, not arbitrary-goal generalization or model selection',
         'swapped_goal_warning':'scoring and original image target remain prescribed Gi; only numeric target input is corrupted to next Gi; conflicting inputs intentionally test sensitivity',
         'no_real_test_data':'S03/S04 uncollected; all rollouts simulator-only',
         'cooperation_skipped':['additional applicant recordings','physical table-to-robot registration','capture timestamps','strict human footprint adjudication']}
path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Paired RGB/goal corruption controls frozen before evaluation')
