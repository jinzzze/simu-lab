"""Freeze a development-only label-semantics pilot; leave primary coverage study intact."""
import ast
import copy
from common import ROOT,read,write,sha,now


def main():
    path=ROOT/'configs/waypoint_pilot_study_v1.json'
    if path.exists():
        raise FileExistsError(path)
    original=read(ROOT/'configs/study_v1.json');base=read(ROOT/'configs/policy_v1.json')
    v=read(ROOT/'artifacts/evaluation/validation/coverage_24_seed7/manifest.json')
    assert v['status']=='completed'
    plan=dict(revision='requested_waypoint_development_v1',created_utc=now(),scope='Development only; no held-out test in this pilot',
        hypothesis='BC of near-EE safety-limited commands may be brittle under accumulated state errors. Learn the teacher\'s pre-limit waypoint instead; keep observed histories, architecture, physical limits and scorer.',
        changed_variable='Target actions: teacher pre-limit XYZ+width requests instead of effective safety-limited XYZ+width commands',
        unchanged=['same24 coverage teacher RGB/robot/goal/previous-effective-command histories','same initial weights and sampling recipe',
                   'same causal CNN-GRU and goal-relative XY decoding','same loss scales,2500updates,EMA,augmentation','same runtime8mm limiter and384-step horizon'],
        condition='waypoint_24',pilot_seed=7,additional_seeds=[17,27],
        gate='After all12 developer cases for seed7: expand only if at least6 strict successes and8 qualified lifts. Keep all failures and failed proposals.',
        validation_cases=original['validation_cases'],jitter_m=original['jitter_m'],
        teacher_label_provenance='Re-execute original initial-XY teacher without image capture; compare every effective command, pre-action vector and full scoring trace to original. Store actual pre-limit requests as new BC targets; no observed outcomes define labels.',
        runtime_contract='No oracle phase, target object pose or teacher actions supplied online; model predicts requests, same safety limiter executes.',
        source_index_sha256=sha(ROOT/'artifacts/robot_demos_v1/training_index.json'),
        primary_study_sha256=sha(ROOT/'configs/study_v1.json'),
        evaluation_scope='Known developer cases reused for architecture development. Primary72000-series results not used for checkpoint/seed selection; a future confirmation requires a separately frozen fresh scene list.')
    write(path,plan)
    cfg=copy.deepcopy(base)
    cfg.update(revision='requested_waypoint_bc_pilot_v1',conditions=['waypoint_24'],
        training_index='artifacts/waypoint_demos_v1/training_index.json',dataset_manifest='artifacts/waypoint_demos_v1/manifest.json',
        study_sha256=sha(path),created_utc=now(),scope=plan['scope'],revision_reason=plan['hypothesis'],
        action_semantics='Predicted teacher pre-limit XYZ+width waypoint, encoded XY relative to prescribed goal; current/history previous command remains the physically executed limited command.',
        evaluation_script='evaluate_waypoint.py')
    cfg['training']['budget_comparison']='Same coverage24 RGB/history, initial weights, sample stream and2500updates as primary coverage model; only target-label semantics change.'
    cfg['evaluation']['case_source']='configs/waypoint_pilot_study_v1.json'
    write(ROOT/'configs/waypoint_pilot_v1.json',cfg)
    substitutions=[('configs/policy_v1.json','configs/waypoint_pilot_v1.json'),
                   ('configs/study_v1.json','configs/waypoint_pilot_study_v1.json')]
    generated={}
    for source,target in [('train.py','train_waypoint.py'),('evaluate.py','evaluate_waypoint.py')]:
        text=(ROOT/source).read_text(encoding='utf-8')
        for a,b in substitutions:
            text=text.replace(a,b)
        text=text.replace(source,target)
        if source=='evaluate.py':
            text=text.replace("choices=['validation','test']","choices=['validation']")
            text=text.replace('artifacts/evaluation/','artifacts/waypoint_pilot_evaluation/')
        ast.parse(text)
        (ROOT/target).write_text(text,encoding='utf-8')
        generated[target]=dict(parent=source,parent_sha256=sha(ROOT/source),generated_sha256=sha(ROOT/target))
    write(ROOT/'receipts/waypoint_pilot_freeze.json',dict(created_utc=now(),plan_sha256=sha(path),generated_sources=generated,
        code_generation='Exact parent training/evaluation recipe; config/source names changed, evaluation output isolated and test CLI disabled.'))
    print(plan['gate'])


if __name__=='__main__':
    main()
