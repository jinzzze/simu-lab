"""Static paired-task validation; no claim that new videos exist."""
from pathlib import Path
import csv,json,hashlib
from goal_scoring import footprint_inside_goal,strict_goal_success
root=Path(__file__).resolve().parent
plan=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
with (root/'capture_checklist.csv').open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
assert len(rows)==plan['total_episodes_planned']==60
assert len({r['episode_id'] for r in rows})==60
assert {s:sum(r['split']==s for r in rows) for s in ('train','val','test')}=={'train':32,'val':16,'test':12}
splits={s['session_id']:s['split'] for s in plan['sessions']}
for row in rows:
    assert row['split']==splits[row['session_id']]
    assert plan['pairs'][row['start_id']]==row['goal_id']
    assert row['source_board_id']=='source_left' and row['target_board_id']=='target_right'
    assert row['capture_status'] in ('not_collected','received')
    if row['capture_status']=='not_collected':
        assert not row['video_sha256']
    else:
        path=(root/row['source_video_path']).resolve()
        assert path.is_relative_to((root/'data/raw').resolve()) and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['video_sha256']
        label=json.loads((root/'data/processed/S01_intake_v1'/f"{row['episode_id']}.json").read_text(encoding='utf-8'))
        assert label['status']=='received' and label['split']==row['split']
        assert label['intended_start']['start_id']==row['start_id'] and label['intended_goal']['goal_id']==row['goal_id']
        assert label['source']['relative_video_path']==row['source_video_path'] and label['source']['sha256']==row['video_sha256']
        assert label['source']['captured_at'] is None
        assert not label['frames'] and not label['events'] and label['outcome']['success'] is None
        assert label['source']['frame_count']>0 and label['source']['fps']>0
hash_splits={}
for row in rows:
    digest=row['video_sha256']
    if digest:
        assert digest not in hash_splits or hash_splits[digest]==row['split'], 'Raw data crosses splits'
        hash_splits[digest]=row['split']
for session in plan['sessions']:
    expected=3 if session['split']=='test' else 4
    for i in range(1,5):assert sum(r['session_id']==session['session_id'] and r['start_id']==f'P{i}' for r in rows)==expected
pending=json.loads((root/'templates/episode.pending.json').read_text(encoding='utf-8'))
assert pending['schema_version']=='paired-two-board-labels-v2'
assert pending['intended_start']['board_id']=='source_left'
assert pending['intended_goal']['board_id']=='target_right'
assert pending['source']['sha256'] is None and not pending['frames']
assert not pending['intended_goal']['recorded_before_capture']
# Synthetic geometry checks, NOT robot executions.
corners=[[.09,-.01],[.11,-.01],[.09,.01],[.11,.01]]
assert footprint_inside_goal(corners,[.1,0])
assert not footprint_inside_goal(corners,[.14,0])  # wrong goal after crossing line
assert not footprint_inside_goal([[.125,0]],[.1,0]) # center crossed, corner outside
fake={'stability_scope':'designated_goal_containment','final_flags':dict.fromkeys(('in_workspace','on_table','upright','slow','released','withdrawn'),True),'lifted':True,'max_lift_duration_s':.12,'final_stable_duration_s':.5}
assert strict_goal_success(corners,[.1,0],fake)
assert not strict_goal_success(corners,[.14,0],fake)
fake['final_flags']['released']=False
assert not strict_goal_success(corners,[.1,0],fake)
fake.pop('stability_scope')
try:
    strict_goal_success(corners,[.1,0],fake)
except ValueError:
    pass
else:
    raise AssertionError('Historical line timer must not qualify as goal stability')
print(json.dumps({'passed':True,'planned':60,'pairs':plan['pairs'],'new_recordings_observed':sum(r['capture_status']=='received' for r in rows),'checks':'paired allocation, source hashes, received labels, split isolation, pending template and synthetic goal geometry; no trained-policy benchmark'},indent=2))
