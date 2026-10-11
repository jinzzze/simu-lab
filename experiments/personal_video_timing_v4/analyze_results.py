"""Audit the complete frozen experiment and summarize all seeds and failures."""
from collections import Counter
import json
import numpy as np
from common import ROOT,OLD,read,write,sha,now,check_sources,verify_training_pairs


def summary(rows):
    return dict(executions=len(rows),successes=sum(r['success'] for r in rows),
                lifts=sum(r['lifted'] for r in rows),
                final_center_distance_mean_mm=float(np.mean([np.linalg.norm(np.array(r['final_object_xyz_m'])[:2]-r['goal_xy_m'])*1000 for r in rows])) if rows else None,
                failures_by_flag=dict(Counter(k for r in rows if not r['success'] for k,v in r['final_flags'].items() if not v)))


def main():
    out=ROOT/'artifacts/results_v1';out.mkdir(parents=True,exist_ok=False)
    cfg=read(ROOT/'configs/policy_v1.json');study=read(ROOT/'configs/study_v1.json')
    assert sha(ROOT/'configs/study_v1.json')==cfg['study_sha256']
    train_seeds={r['scene_seed'] for r in study['train_cases']}
    validation_seeds={r['scene_seed'] for r in study['validation_cases']}
    test_seeds={r['scene_seed'] for r in study['test_cases']}
    assert not (train_seeds&validation_seeds or train_seeds&test_seeds or validation_seeds&test_seeds)
    index=read(ROOT/cfg['training_index']);dataset=read(ROOT/cfg['dataset_manifest'])
    assert sha(ROOT/cfg['dataset_manifest'])==index['dataset_manifest_sha256']
    assert dataset['training_gate_passed'];check_sources(dataset['source_sha256'])
    sources={cfg['training_index']:sha(ROOT/cfg['training_index']),cfg['dataset_manifest']:sha(ROOT/cfg['dataset_manifest']),
             'configs/study_v1.json':sha(ROOT/'configs/study_v1.json'),'configs/policy_v1.json':sha(ROOT/'configs/policy_v1.json')}
    data_checks=[]
    training_hashes=set();validation_hashes=set()
    for condition,entries in index['train_groups'].items():
        verify_training_pairs(entries)
        assert len(entries)==24
        pairs=Counter((r['start_id'],r['goal_id']) for r in entries)
        assert len(pairs)==12
        assert set(pairs.values())=={2}
        for row in entries:
            path=ROOT/row['file'];assert sha(path)==row['dataset_sha256']
            with np.load(path,allow_pickle=False) as a:
                assert set(a.files)=={'rgb','observations','actions'}
                assert np.array_equal(a['observations'][1:,8:12],a['actions'][:-1])
                assert np.allclose(a['observations'][:,4:6],row['outcome']['goal_xy_m'])
            training_hashes.add(row['initial_rgb_sha256'])
        data_checks.append(dict(condition=condition,episodes=len(entries),pair_counts={a+'_'+b:c for (a,b),c in pairs.items()},
                               causal_previous_command=True,heldout_training_pairs=0,allowed_arrays_only=True))
    validation_hashes={r['initial_rgb_sha256'] for r in index['shared_validation']}
    models={};paired_audit=[]
    for c in cfg['conditions']:
        for seed in cfg['training']['seeds']:
            name=f'{c}_seed{seed}';folder=ROOT/'artifacts/models'/name;m=read(folder/'manifest.json')
            assert m['status']=='completed' and m['config']==cfg
            assert m['steps_completed']==2500 and m['parameters']==227268
            assert m['checkpoint_sha256']==sha(folder/'checkpoint.pt');check_sources(m['source_sha256'])
            assert len(m['episode_sampling_counts'])==24 and min(m['episode_sampling_counts'])>0 and sum(m['episode_sampling_counts'])==5000
            sources[str((folder/'manifest.json').relative_to(ROOT)).replace('\\','/')]=sha(folder/'manifest.json')
            models[name]=m
    for seed in cfg['training']['seeds']:
        a,b=[models[f'{c}_seed{seed}'] for c in cfg['conditions']]
        keys=['initial_weights_sha256','episode_stream_sha256','augmentation_stream_sha256','parameters','steps_completed']
        assert all(a[k]==b[k] for k in keys)
        paired_audit.append(dict(seed=seed,matched_keys=keys,valid_frame_targets=[a['valid_action_targets_seen'],b['valid_action_targets_seen']],
                                 training_seconds=[a['wall_s'],b['wall_s']]))
    totals={};all_rows=[];reference_initial={};overlaps=[];failures=[]
    for suite in ('validation','test'):
        totals[suite]={};cases=study[suite+'_cases']
        for c in cfg['conditions']:
            condition_rows=[];per_seed={}
            for seed in cfg['training']['seeds']:
                folder=ROOT/f'artifacts/evaluation/{suite}/{c}_seed{seed}'
                m=read(folder/'manifest.json');rows=read(folder/'episodes.json')
                assert m['status']=='completed' and m['executions']==len(cases)
                assert m['episodes_sha256']==sha(folder/'episodes.json');check_sources(m['source_sha256'])
                assert [r['case_id'] for r in rows]==[r['case_id'] for r in cases]
                sources[str((folder/'manifest.json').relative_to(ROOT)).replace('\\','/')]=sha(folder/'manifest.json')
                for row,case in zip(rows,cases):
                    assert row['scene_seed']==case['scene_seed'] and row['heldout_pair']==case['heldout_pair']
                    assert row['start_id']==case['start_id'] and row['goal_id']==case['goal_id']
                    assert row['rgb_numerical_and_scoring_goal_agree'] and not row['runtime_object_truth']
                    assert not row['absolute_clock_input'] and not any(row['observation_interventions'].values())
                    ap=folder/(case['case_id']+'_inputs_actions.npz');sp=folder/(case['case_id']+'_scoring.json')
                    assert sha(ap)==row['dataset_sha256'] and sha(sp)==row['scoring_sha256']
                    trace=read(sp);assert len(trace)==384
                    with np.load(ap,allow_pickle=False) as arrays:
                        assert np.array_equal(arrays['inference_history_ticks'].ravel(),np.arange(384))
                        assert np.array_equal(arrays['policy_input_vectors'][:,0,:],arrays['pre_step_vectors'])
                        assert np.array_equal(arrays['pre_step_vectors'][1:,8:12],arrays['effective_commands'][:-1])
                        assert np.allclose(arrays['pre_step_vectors'][:,4:6],row['goal_xy_m'])
                    key=(suite,row['case_id'])
                    initial=(row['initial_rgb_sha256'],trace[0]['object_xyz_m'])
                    if key in reference_initial:
                        assert reference_initial[key][0]==initial[0] and np.allclose(reference_initial[key][1],initial[1],atol=1e-7)
                    else:
                        reference_initial[key]=initial
                    if suite=='test' and seed==7:
                        overlaps.append(dict(condition=c,case_id=row['case_id'],initial_rgb_sha256=row['initial_rgb_sha256'],
                            matches_training=row['initial_rgb_sha256'] in training_hashes,matches_development=row['initial_rgb_sha256'] in validation_hashes))
                    if not row['success']:
                        close=[t for t in trace if t['finger_width_m']<.030]
                        failures.append(dict(suite=suite,condition=c,seed=seed,case_id=row['case_id'],lifted=row['lifted'],
                            first_closed_xy_error_mm=float(np.linalg.norm(np.array(close[0]['ee_xyz_m'])[:2]-np.array(close[0]['object_xyz_m'])[:2])*1000) if close else None,
                            final_flags=row['final_flags']))
                    all_rows.append(dict(suite=suite,**row))
                condition_rows.extend(rows)
                per_seed[str(seed)]=dict(all=summary(rows),same_number=summary([r for r in rows if r['same_number']]),
                    trained_cross=summary([r for r in rows if not r['same_number'] and not r['heldout_pair']]),
                    heldout=summary([r for r in rows if r['heldout_pair']]))
            totals[suite][c]=dict(all=summary(condition_rows),same_number=summary([r for r in condition_rows if r['same_number']]),
                trained_cross=summary([r for r in condition_rows if not r['same_number'] and not r['heldout_pair']]),
                heldout=summary([r for r in condition_rows if r['heldout_pair']]),per_seed=per_seed)
    assert len(all_rows)==264 and sum(r['suite']=='test' for r in all_rows)==192
    overlap_summary=dict(scene_seed_sets_disjoint=True,test_physical_starts=len(test_seeds),
        test_tasks_with_initial_pixels_in_training=sorted({r['case_id'] for r in overlaps if r['matches_training']}),
        test_tasks_with_initial_pixels_in_development=sorted({r['case_id'] for r in overlaps if r['matches_development']}))
    result=dict(status='completed',created_utc=now(),study=study,totals=totals,data_checks=data_checks,paired_training_audit=paired_audit,
                overlap_summary=overlap_summary,
                test_initial_pixel_overlap=overlaps,all_cases=all_rows,failure_diagnostics=failures,source_sha256=sources,
                analysis_sha256=sha(ROOT/'analyze_results.py'),benchmark_executions=len(all_rows),
                scope='384 fixed-step simulated executions.32 test tasks reuse8 starts across3 model seeds. Both groups share personal geometry; one adds S01/X01 coarse timing+image progress. No real-robot evidence.')
    write(out/'results.json',result)
    print(json.dumps({'totals':totals,'executions':len(all_rows),'all_audits':'passed'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
