"""Validate separate coarse visual review against its actual evidence frames."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np

root=Path(__file__).resolve().parent
report=root/'artifacts/reports/S01_trajectories_v1'
review=json.loads((report/'coarse_visual_review.json').read_text(encoding='utf-8'))
manifest=json.loads((report/'summary.json').read_text(encoding='utf-8'))
episodes={e['episode_id']:e for e in manifest['episodes']}
assert review['reviewed_episodes']==len(review['episodes'])==16
assert review['sampled_raw_frames_inspected']==384 and review['coarse_event_brackets']==96
count=0
for item in review['episodes']:
    episode=episodes[item['episode_id']]
    assert item['source_sha256']==episode['source_sha256'] and item['session_id']=='S01' and item['split']=='train'
    assert item['review_code_sha256']==hashlib.sha256((root/'finalize_s01_trajectories.py').read_bytes()).hexdigest()
    assert item['review_frame_indices']==episode['review_frame_indices'] and len(item['review_frame_indices'])==24
    assert item['review_sheet_sha256']==hashlib.sha256((root/item['review_sheet']).read_bytes()).hexdigest()
    assert item['strict_success'] is None and item['physical_base_footprint_containment'] is None and item['physical_qualified_lift_height_m'] is None
    assert [e['name'] for e in item['events']]==['contact','lift','transport','placement','release','withdraw']
    for event in item['events']:
        lo,hi=event['frame_interval']
        assert lo in item['review_frame_indices'] and hi in item['review_frame_indices'] and 0<=lo<=hi<episode['frames']
        assert np.allclose(event['time_interval_s'],[lo/25,hi/25])
        assert event['qualified_physical_event'] is None and event['annotation_source']=='Codex_visual_inspection'
        count+=1
    endpoint=item['apparent_final_region_diagnostic']
    assert endpoint['goal_definition_frame_index']==0
    margin=endpoint['minimum_proxy_margin_mm']
    if margin is not None:
        expected='inside_proxy' if margin>2.5 else 'outside_proxy' if margin < -2.5 else 'boundary_uncertain'
        assert endpoint['status']==expected
        cal=json.loads((root/f'data/processed/S01_calibration_v2/{item["episode_id"]}.json').read_text(encoding='utf-8'))
        goal=np.array(cal['goals'][cal['prescribed_goal_id']]['center_grid_mm'])
        corners=np.array(endpoint['visible_dark_rectangle_plane_projection_mm'])
        assert abs(float(np.min(20-np.abs(corners-goal)))-margin)<1e-9
assert count==96
# Verify frozen receipts before introducing this extra validation file.
hashes=json.loads((root/'receipts/S01_trajectories_v1/hashes.json').read_text(encoding='utf-8'))
for name,expected in hashes.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print(json.dumps({'passed':True,'reviewed_episodes':16,'coarse_event_brackets':96,'sampled_frames':384,
                  'final_region_proxy_status_counts':dict(Counter(e['apparent_final_region_diagnostic']['status'] for e in review['episodes'])),
                  'minimum_proxy_margin_range_mm':[min(e['apparent_final_region_diagnostic']['minimum_proxy_margin_mm'] for e in review['episodes']),max(e['apparent_final_region_diagnostic']['minimum_proxy_margin_mm'] for e in review['episodes'])],
                  'strict_success_labels':0,'scope':'coarse visual review and projection diagnostic, not certified physical outcome'},indent=2))
