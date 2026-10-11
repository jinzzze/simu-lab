"""Freeze new personal-timing study before robot collection, training or testing."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from common import ROOT,OLD,VIDEOS,PROJECT,read,write,sha,now


def transport_profile(rows,review,cal,start_id):
    """Same dimensionless projection recipe as S01; no human action recovery."""
    assert review['split']=='train' and all(r['split']=='train' for r in rows),'Training human sources only'
    assert rows[0]['object']['auxiliary_mask_candidate'],'Visible pre-action object required'
    assert all(r['source_sha256']==review['source_sha256'] for r in rows)
    first=np.array(rows[0]['object']['center_px']);goal=np.array(cal['goals'][cal['prescribed_goal_id']]['center_image_pixels'])
    direction=goal-first;norm=float(direction@direction);assert norm>=100,'Degenerate direction'
    events={e['name']:e for e in review['events']}
    start=events['transport']['time_interval_s'][0];end=events['placement']['time_interval_s'][1]
    assert .4<=end-start<=6.,'Unusable coarse duration; do not substitute another clip'
    knots=[[0.,0.]];source_frames=[];raw_progress=[]
    for r in rows:
        o=r['object'];t=r['timestamp_s']
        if start<t<end and o['auxiliary_mask_candidate']:
            progress=float(np.clip((np.array(o['center_px'])-first)@direction/norm,0,1))
            knots.append([(t-start)/(end-start),max(knots[-1][1],progress)])
            source_frames.append(r['frame_index']);raw_progress.append(progress)
    assert len(source_frames)>=3,'Insufficient visible constraints; retain failure, no replacement'
    knots.append([1.,1.]);gap=float(np.max(np.diff(np.array(knots)[:,0])))
    assert gap<=.6,'Unconstrained majority of transfer window; requires review'
    return {'human_episode_id':review['episode_id'],'human_session':review['session_id'],'human_split':'train',
        'human_source_sha256':review['source_sha256'],'start_id':start_id,'goal_id':cal['prescribed_goal_id'],
        'duration_s':end-start,'progress_knots':knots,'valid_source_frame_indices':source_frames,
        'source_progress_before_monotone_envelope':raw_progress,'maximum_constraint_gap_fraction':gap,
        'coarse_time_interval_s':[start,end],'source_event_intervals':{k:events[k]['time_interval_s'] for k in ('transport','placement')},
        'semantics':'offline dimensionless image-progress/time guide; not physical XY/XYZ, lift height or robot action',
        'runtime_observation':False,'human_missing_labels_interpolated':False,'desired_robot_setpoints_interpolated':True,
        'physical_robot_height_source':'shared engineered grasp and transport heights'}


def main():
    for sub in ('configs','data/processed','artifacts','receipts'):(ROOT/sub).mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'configs/study_v1.json').exists(),'Immutable study already frozen'
    originals=[]
    for i in range(1,9):
        eid=f'S01_E{i:02d}';intake_path=OLD/f'data/processed/S01_intake_v1/{eid}.json';intake=read(intake_path)
        originals.append((OLD,eid,'P'+intake['intended_goal']['goal_id'][1],intake['source']['relative_video_path'],intake_path,
            OLD/f'data/processed/S01_calibration_v2/{eid}.json',OLD/f'data/processed/S01_trajectories_v1/{eid}.frames.jsonl',
            OLD/f'data/processed/S01_trajectories_v1/{eid}.reviewed.json'))
    intake_path=VIDEOS/'data/processed/intake/x01_intake_v1/manifest.json'
    for item in read(intake_path)['recordings']:
        eid=item['episode_id']
        assert item['split']=='train' and item['session_id']=='X01'
        originals.append((VIDEOS,eid,item['intended_start_id'],item['relative_video_path'],intake_path,
            VIDEOS/f'data/processed/X01_calibration_v2/{eid}.json',VIDEOS/f'data/processed/X01_trajectories_v1/{eid}.frames.jsonl',
            VIDEOS/f'data/processed/X01_trajectories_v1/{eid}.reviewed.json'))
    sources={};profiles=[];calibrations=[]
    for root,eid,start,raw,intake_path,cal_path,rows_path,review_path in originals:
        review=read(review_path);cal=read(cal_path);rows=[json.loads(s) for s in rows_path.read_text(encoding='utf-8').splitlines()]
        assert sha(root/raw)==review['source_sha256']==cal['raw_sha256']
        assert all(r['session_id']==review['session_id'] for r in rows)
        profile=transport_profile(rows,review,cal,start);profiles.append(profile);calibrations.append(cal)
        provenance={str(f.relative_to(PROJECT)).replace('\\','/'):sha(f) for f in (root/raw,intake_path,cal_path,rows_path,review_path)}
        profile['human_derivation_file_sha256']=provenance;sources.update(provenance)
    assert len(profiles)==24 and len({p['human_source_sha256'] for p in profiles})==24
    protocol=read(VIDEOS/'protocol.json');allowed=protocol['training_pairs']
    assert Counter((p['start_id'],p['goal_id']) for p in profiles)==Counter({tuple(p):2 for p in allowed})
    assert {p['human_session'] for p in profiles}=={'S01','X01'}
    profile_path=ROOT/'data/processed/human_profiles_v1.json';assert not profile_path.exists()
    write(profile_path,{'status':'frozen_before_robot_collection','created_utc':now(),'profiles':profiles,'source_sha256':sources,
        'selection':'S01 E01-E08 = first two predeclared repeats per diagonal; all sixteen X01. No outcome filtering.',
        'technical_gates':{'min_visible_constraints':3,'duration_s':[.4,6.],'max_constraint_gap_fraction':.6},
        'code_sha256':sha(Path(__file__)),'test_or_validation_sources_used':False})
    task=read(OLD/'configs/sim_task_v1.json');task['version']='personal-timing-shared-geometry-v1'
    for goal in task['goals']:
        center=np.mean([c['goals'][goal]['center_grid_mm'] for c in calibrations],axis=0)
        xy=np.array(task['target_grid_top_left_xy_m'])+center*[.001,-.001]
        task['goals'][goal]={'xy_m':xy.tolist(),'human_grid_center_mm':center.tolist(),
            'source':'mean first-frame blue centers in the SAME selected 24 TRAIN clips; engineering-aligned sim grid'}
    task['scope']='Shared simulation geometry for personal-video timing ablation'
    task['human_reference']='S01 selected8 + X01 all16 pre-action calibration. No X02/X03 geometry.'
    task['collection']={'train_per_condition':24,'conditions':['geometry_only_24','video_timing_24'],
        'retention':'all attempts including failures','human_sessions':['S01','X01'],
        'video_limit':'timing and dimensionless image progress only; robot XYZ/width labels from executed sim commands'}
    task['deployment_status']='new models not yet trained';write(ROOT/'configs/sim_task_v1.json',task)
    protocol.update(version='personal-video-timing-v4',scope='two conditions share all 12 training combinations')
    write(ROOT/'protocol.json',protocol)
    train=[{'scene_seed':910000+i,'start_id':p['start_id'],'goal_id':p['goal_id'],'human_episode_id':p['human_episode_id']} for i,p in enumerate(profiles)]
    val=[{'case_id':p+'_'+g+'_dev','scene_seed':920000+i,'start_id':p,'goal_id':g,'heldout_pair':False} for i,(p,g) in enumerate(allowed)]
    test=[];heldout=protocol['heldout_pairs']
    for rep in range(2):
        for i in range(1,5):
            for j in range(1,5):
                p,g=f'P{i}',f'G{j}'
                test.append({'case_id':f'{p}_{g}_r{rep+1}','scene_seed':930000+rep*4+i-1,
                    'start_id':p,'goal_id':g,'heldout_pair':[p,g] in heldout})
    study={'revision':'personal_video_timing_v1','created_utc':now(),'status':'frozen_before_robot_collection_training_and_test',
        'research_question':'Does adding personal coarse timing/image progress improve closed-loop control beyond shared personal geometry, with equal coverage and robot demo count?',
        'conditions':['geometry_only_24','video_timing_24'],'seeds':[7,17,27],
        'train_cases':train,'validation_cases':val,'test_cases':test,'training_pairs':allowed,'heldout_pairs':heldout,'jitter_m':.003,
        'human_profiles_sha256':sha(profile_path),'geometry_sha256':sha(ROOT/'configs/sim_task_v1.json'),
        'teacher':'same legacy initial-XY teacher; profile None versus matched personal timing/progress during transport only',
        'collection_gate':'Each group: all24 qualified lifts and >=20/24 strict successes; retain failures, no replacement or relaxed gate.',
        'selection':'same v6 learner, all3 fixed-final EMA models,2500 updates. Report all development cases and all fresh test cases; no best seed/checkpoint selection.',
        'budget':'24 robot demonstrations each, exact paired scene/P/G/initial RGB, model init, sampling, augmentation, updates. Lengths/frame targets/compute may differ.',
        'information_boundary':'current RGB +12D robot/preprovided goal/past command only; no runtime object truth, clock, phase, or human profile',
        'fresh_test_scope':'32 tasks over8 physical object starts, repeated for3 model seeds; same simulator family, not96 independent environments',
        'limitations':['Prior coverage and S01 timing results informed this design: exploratory new comparison.',
            'Personal labels are RGB candidates and Codex coarse visual brackets, not human-annotated physical actions.',
            'X03 previously viewed for naming/QC; no X03 trajectory extraction, tuning or training in this study.',
            'This comparison tests coarse timing+progress, not a pure timing or pure progress intervention.']}
    write(ROOT/'configs/study_v1.json',study)
    cfg=read(VIDEOS/'configs/policy_v1.json')
    cfg.update(revision='personal-video-timing-bc-v1',scope=study['research_question'],conditions=study['conditions'],
        study_sha256=sha(ROOT/'configs/study_v1.json'),created_utc=now(),revision_reason=study['research_question'],
        hypothesis_status='exploratory new personal-data comparison; fresh robot seeds; all seeds retained',
        collaboration_skipped=['independent human annotation/physical XYZ validation','public release/application submission'])
    cfg['training']['budget_comparison']=study['budget'];cfg['evaluation']['case_source']='configs/study_v1.json'
    write(ROOT/'configs/policy_v1.json',cfg)
    print({'profiles':len(profiles),'pairs':len(allowed),'duration_range_s':[min(p['duration_s'] for p in profiles),max(p['duration_s'] for p in profiles)],
        'max_gap_fraction':max(p['maximum_constraint_gap_fraction'] for p in profiles),'study_sha256':sha(ROOT/'configs/study_v1.json')})


if __name__=='__main__':main()
