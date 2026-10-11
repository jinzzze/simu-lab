"""Fail on split leakage; report unavailable human sessions without inventing data."""
import csv
import json
from collections import Counter
from common import ROOT,read,verify_training_pairs,sha


def validate():
    p=read(ROOT/'protocol.json');held={tuple(x) for x in p['heldout_pairs']};train={tuple(x) for x in p['training_pairs']}
    assert len(train)==12 and len(held)==4 and not train&held
    assert train|held=={(f'P{i}',f'G{j}') for i in range(1,5) for j in range(1,5)}
    assert Counter(x[0] for x in train)=={f'P{i}':3 for i in range(1,5)}
    assert Counter(x[1] for x in train)=={f'G{i}':3 for i in range(1,5)}
    with (ROOT/'capture_checklist.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==44 and len({r['episode_id'] for r in rows})==44
    assert len({r['planned_video_path'] for r in rows})==44
    assert Counter(r['split'] for r in rows)=={'train':16,'val':12,'test':16}
    assert all((r['start_id'],r['goal_id']) in train for r in rows if r['split'] in ('train','val'))
    assert all(len({r['split'] for r in rows if r['session_id']==s})==1 for s in {r['session_id'] for r in rows})
    study=read(ROOT/'configs/study_v1.json')
    assert study['capture_protocol_sha256']==sha(ROOT/'protocol.json')
    assert all((r['start_id'],r['coverage_goal_id']) in train for r in study['train_cases'])
    assert all((r['start_id'],r['goal_id']) in train for r in study['validation_cases'])
    assert not {r['scene_seed'] for r in study['train_cases']}&{r['scene_seed'] for r in study['validation_cases']+study['test_cases']}
    assert not {r['scene_seed'] for r in study['validation_cases']}&{r['scene_seed'] for r in study['test_cases']}
    index=ROOT/'artifacts/robot_demos_v1/training_index.json'
    if index.exists():
        for entries in read(index)['train_groups'].values():
            verify_training_pairs(entries)
    files={s['session_id']:sum(f.suffix.lower() in ('.mp4','.mov','.m4v') for f in (ROOT/'data/raw'/s['folder']).iterdir()) for s in p['new_sessions']}
    return dict(status='passed',planned_new_clips=44,new_video_file_counts=files,
        scope='split and manifest integrity only; file presence is not capture-date, label or video-content validation',
        test_content_opened=False)


if __name__=='__main__':
    print(json.dumps(validate(),ensure_ascii=False,indent=2))
