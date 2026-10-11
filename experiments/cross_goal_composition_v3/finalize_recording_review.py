"""Persist actual Codex overview inspections separately from pseudolabels."""
import argparse
from collections import Counter
from pathlib import Path
import json
from common import ROOT,read,write,sha,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',choices=['X01','X02'],required=True)
    p.add_argument('--bounds',required=True);a=p.parse_args()
    bounds_path=ROOT/a.bounds;bounds=read(bounds_path)
    report=ROOT/f'artifacts/reports/{a.session}_trajectories_v1'
    out=ROOT/f'data/processed/{a.session}_trajectories_v1';summary=read(report/'summary.json')
    assert summary['status']=='completed' and summary['session_id']==a.session
    assert set(bounds)=={e['episode_id'] for e in summary['episodes']}
    receipt=report/'coarse_visual_review_v1.json';assert not receipt.exists(),'Immutable review exists'
    reviewed=[]
    for e in summary['episodes']:
        eid=e['episode_id'];rowpath=out/f'{eid}.frames.jsonl'
        rows=[json.loads(line) for line in rowpath.read_text(encoding='utf-8').splitlines()]
        for path,digest in e['files'].items():assert sha(ROOT/path)==digest,'Extraction output changed'
        indices=e['review_frame_indices'];events=[]
        for name,(lo,hi) in zip(('transport','placement'),bounds[eid]):
            assert 0<=lo<hi<24
            a_idx,b_idx=indices[lo],indices[hi]
            events.append({'name':name,'frame_interval':[a_idx,b_idx],
                'time_interval_s':[rows[a_idx]['timestamp_s'],rows[b_idx]['timestamp_s']],
                'annotation_source':'Codex_visual_inspection','reviewer':'Codex; not a human annotator',
                'interval_semantics':'coarse visual search bracket; physical contact/height/onset not measured',
                'qualified_physical_event':None,'evidence_frame_indices':[a_idx,b_idx]})
        assert events[0]['frame_interval'][1]<events[1]['frame_interval'][0]
        record={'episode_id':eid,'session_id':a.session,'split':e['split'],'source_sha256':e['source_sha256'],
            'created_utc':now(),'events':events,'strict_success':None,'robot_action':None,'metric_airborne_xyz':None,
            'review_method':'Codex inspected all 24 annotated overview tiles; not full continuous visual review or independent ground truth',
            'review_frame_indices':indices,'review_sheet_sha256':sha(report/f'{eid}_review.jpg'),
            'frames_sha256':sha(rowpath),'review_bounds_sha256':sha(bounds_path),'code_sha256':sha(Path(__file__)),
            'sampled_object_check':'No obvious wrong-object assignment among displayed valid boxes; missing boxes retained. Not a quantified accuracy audit.',
            'visual_transfer_and_hand_withdraw_observed':True,
            'allowed_use':'offline coarse transfer window and progress guide; never current policy input'}
        assert not (out/f'{eid}.reviewed.json').exists()
        write(out/f'{eid}.reviewed.json',record);reviewed.append(record)
    result={'status':'completed','session_id':a.session,'split':summary['split'],'created_utc':now(),
        'reviewed_episodes':len(reviewed),'overview_frames_inspected':24*len(reviewed),
        'coarse_search_brackets':2*len(reviewed),'strict_success_labels':0,'episodes':reviewed,
        'bounds_sha256':sha(bounds_path),'extraction_manifest_sha256':sha(report/'summary.json')}
    write(receipt,result);print({k:v for k,v in result.items() if k!='episodes'})


if __name__=='__main__':main()
