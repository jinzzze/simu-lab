"""Freeze a coverage-only simulation study before generating its training data."""
import copy
from common import ROOT,OLD,read,write,sha,now


def main():
    target=ROOT/'configs/study_v1.json'
    if target.exists():
        raise FileExistsError(target)
    protocol=read(ROOT/'protocol.json');probe=read(ROOT/'artifacts/teacher_development_v1/manifest.json')
    assert probe['status']=='completed'
    train=[];validation=[];test=[]
    for repeat in range(6):
        for i in range(1,5):
            p=f'P{i}';goals=[g for pp,g in protocol['training_pairs'] if pp==p]
            scene=70000+repeat*4+i-1
            train.append(dict(scene_seed=scene,start_id=p,paired_goal_id=f'G{i}',coverage_goal_id=goals[repeat%3]))
    for i,(p,g) in enumerate(protocol['training_pairs']):
        validation.append(dict(case_id=f'{p}_{g}_dev',scene_seed=71000+i,start_id=p,goal_id=g,heldout_pair=False))
    for repeat in range(2):
        for i in range(1,5):
            for j in range(1,5):
                p,g=f'P{i}',f'G{j}'
                test.append(dict(case_id=f'{p}_{g}_r{repeat+1}',scene_seed=72000+repeat*4+i-1,
                                 start_id=p,goal_id=g,heldout_pair=[p,g] in protocol['heldout_pairs']))
    study=dict(revision='coverage_only_v1',created_utc=now(),status='frozen_before_new_data_training_and_test',
        research_question='Does training coverage across12 combinations improve held-out combinations, with fixed24 executed teacher episodes per condition?',
        conditions=['paired_24','coverage_24'],seeds=[7,17,27],train_cases=train,validation_cases=validation,test_cases=test,
        training_pair_rule=protocol['training_pairs'],heldout_pairs=protocol['heldout_pairs'],
        jitter_m=.003,teacher='unchanged v2 initial-XY privileged execute_expert(profile=None)',
        teacher_development={'legacy':probe['summary']['legacy'],'rejected_quintic':probe['summary']['quintic_0p10'],
            'reason':'Smooth transport failed its predeclared gate; retain legacy teacher to isolate training coverage. All attempts including failures retained.',
            'probe_sha256':sha(ROOT/'artifacts/teacher_development_v1/manifest.json')},
        collection_gate='Both groups must have24 qualified lifts and at least20 strict successes out of24; all failures remain in BC dataset. No test or held-out teacher collection.',
        information_boundary='Current RGB/12D robot-goal-past-command vector only; no online object truth, expert phase or external clock. Goal numeric/RGB/scorer agree.',
        personal_data='Both conditions share existing S01 pre-action goal geometry. No new videos or transport profiles used; this is a coverage experiment, not a personal-video-benefit claim.',
        selection='All3 final EMA checkpoints,2500 updates; no checkpoint or seed selection; development reporting then frozen32-case test regardless of comparative direction.',
        budget='Same24 episodes, paired physical initial scenes, model/init/sampling/augmentation/2500updates; trajectory lengths differ so total frame targets and compute can differ.',
        sample_scope='Test32 tasks reuse8 physical object starts across goals and3 training seeds; not96 independent environments per condition.',
        legacy_configuration_sha256=sha(OLD/'configs/policy_recurrent_goal_rgbaug_v6.json'),
        capture_protocol_sha256=sha(ROOT/'protocol.json'))
    write(target,study)
    cfg=copy.deepcopy(read(OLD/'configs/policy_recurrent_goal_rgbaug_v6.json'))
    cfg.update(revision='cross_goal_coverage_bc_v1',scope=study['research_question'],conditions=study['conditions'],
               training_index='artifacts/robot_demos_v1/training_index.json',dataset_manifest='artifacts/robot_demos_v1/manifest.json',
               dataset_manifest_sha256=None,created_utc=now(),study_sha256=sha(target),
               revision_reason='Matched24-episode paired-only versus12-pair coverage. No added personal timing; retain v6 learner/loss/augmentation/horizon.',
               evaluation_script='evaluate.py')
    cfg['training']['budget_comparison']=study['budget']
    cfg['training']['episodes_per_condition']=24
    cfg['evaluation']={'episode_steps':384,'sampling_seed':8821,'scoring':'unchanged strict v2',
                       'case_source':'configs/study_v1.json','selection':'all final models; no selection on validation/test'}
    cfg['collaboration_skipped']=['X01/X02/X03 uncollected applicant recordings','new-video comparison','capture dates and human labels','external publication/application']
    write(ROOT/'configs/policy_v1.json',cfg)
    print('Frozen',target)


if __name__=='__main__':
    main()
