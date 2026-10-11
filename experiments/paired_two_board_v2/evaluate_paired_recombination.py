"""Run explicit consistent target cases with frozen policies or labelled oracle."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil
import numpy as np
from paired_sim import ROOT, PROJECT, PairedGoalEnv
import paired_recombination_driver as driver

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--condition', choices=['simulation_only', 'video_guided'])
    parser.add_argument('--seed', type=int); parser.add_argument('--oracle', action='store_true')
    args = parser.parse_args(); plan_path = ROOT / 'configs/consistent_goal_recombination_v1.json'; plan = read(plan_path)
    cfg = read(ROOT / plan['policy_config']); assert sha(ROOT / plan['policy_config']) == plan['policy_config_sha256']
    label = 'privileged_initial_xy_teacher' if args.oracle else f'{args.condition}_seed{args.seed}'
    if not args.oracle: assert args.condition in plan['conditions'] and args.seed in plan['policy_seeds']
    out = ROOT / f'artifacts/reports/{cfg["revision"]}/consistent_goal_recombination_v1/{label}'
    assert not out.exists(); out.mkdir(parents=True)
    names = ['evaluate_paired_recombination.py', 'paired_recombination_driver.py', 'paired_sim.py', 'goal_scoring.py',
             'paired_recurrent_goal_policy.py', 'paired_recurrent_goal_policy_process.py', 'paired_recurrent_goal_policy_worker.py',
             'paired_recurrent_policy.py', 'paired_delta_policy.py', 'paired_image_policy.py', 'collect_paired_demonstrations.py']
    sources = {name: ROOT / name for name in names}
    sources.update({'policy_config.json': ROOT / plan['policy_config'], 'recombination_plan.json': plan_path,
                    'sim_task_v1.json': ROOT / 'configs/sim_task_v1.json', 'sim_physics_v1.json': ROOT / 'configs/sim_physics_v1.json',
                    'grasp_env.py': PROJECT / 'src/sim/grasp_env.py',
                    'training.py': PROJECT / 'src/perception/training.py', 'diffusion.py': PROJECT / 'src/policy/diffusion.py'})
    snap = out / 'source_snapshot'; snap.mkdir()
    for name, path in sources.items(): shutil.copy2(path, snap / name)
    meta = {'status': 'running', 'condition': args.condition, 'seed': args.seed, 'oracle': args.oracle, 'config': cfg,
            'plan': plan, 'started_utc': datetime.now(timezone.utc).isoformat(), 'episodes_planned': 16,
            'source_sha256': {name: sha(path) for name, path in sources.items()}, 'primary_count_addition': 0,
            'runtime_object_truth': args.oracle, 'oracle_truth_scope': 'initial XY only' if args.oracle else None}
    if not args.oracle:
        train = ROOT / f'artifacts/runs/{cfg["revision"]}/{label}'; training = read(train / 'manifest.json')
        checkpoint = train / 'checkpoint.pt'; assert training['status'] == 'completed' and training['config'] == cfg
        assert sha(checkpoint) == training['checkpoint_sha256']; meta['checkpoint_sha256'] = sha(checkpoint)
        for name in ('paired_recurrent_goal_policy.py', 'paired_recurrent_policy.py', 'paired_delta_policy.py', 'paired_image_policy.py'):
            assert sha(ROOT / name) == training['source_sha256'][name]
        for name in ('training.py', 'diffusion.py'):
            assert sha(sources[name]) == training['source_sha256'][name]
    write(out / 'manifest.json', meta); env = PairedGoalEnv(); rows = []
    try:
        if not args.oracle: predictor = driver.ImagePolicyProcess(checkpoint, out / 'policy_stderr.log')
        try:
            for case in plan['cases']:
                if args.oracle:
                    env.task['start_jitter_halfwidth_m'] = plan['jitter_m']
                    obs = env.reset_task(case['scene_seed'], case['start_id'], case['goal_id'])
                    initial_sha = hashlib.sha256(obs['rgb'].tobytes()).hexdigest()
                    initial_xy = env.sim.get_base_position('object')[:2].copy()
                    row = env.execute_expert(initial_xy, record=False)
                    row.update(initial_rgb_sha256=initial_sha, privileged_initial_object_xy=True,
                               online_object_truth_correction=False, controller='fixed physical teacher feasibility diagnostic')
                else:
                    definition = {'start': 68000, 'jitter_m': plan['jitter_m'],
                                  'prescribed_start_id': case['start_id'], 'prescribed_goal_id': case['goal_id']}
                    row, arrays = driver.run_episode(env, predictor, cfg, definition, case['scene_seed'], False)
                    path = out / f'{case["case_id"]}_inputs_actions.npz'; np.savez_compressed(path, **arrays)
                    row['dataset_sha256'] = sha(path)
                    assert arrays['pre_step_vectors'].shape == (384, 12)
                    assert np.allclose(arrays['policy_input_vectors'][:, :, 4:6], row['goal_xy_m'])
                    assert not any(row['observation_interventions'].values())
                assert row['start_id'] == case['start_id'] and row['goal_id'] == case['goal_id']
                path = out / f'{case["case_id"]}_scoring.json'; write(path, env.trace)
                row.update(case_id=case['case_id'], seen_pair=case['seen_pair'], condition=args.condition,
                           policy_seed=args.seed, oracle=args.oracle, scoring_sha256=sha(path),
                           rgb_and_numerical_goal_consistent=True)
                rows.append(row); write(out / 'episodes.json', rows)
                print(json.dumps({'run': label, 'case': case['case_id'], 'success': row['success'], 'lift': row['lifted']}, ensure_ascii=False), flush=True)
        finally:
            if not args.oracle: predictor.close()
        for name, path in sources.items(): assert sha(path) == meta['source_sha256'][name]
        meta.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(), episodes=16,
                    episodes_sha256=sha(out / 'episodes.json'), successes=sum(r['success'] for r in rows),
                    seen_successes=sum(r['success'] for r in rows if r['seen_pair']),
                    unseen_successes=sum(r['success'] for r in rows if not r['seen_pair']))
        write(out / 'manifest.json', meta)
    except Exception as exc:
        meta.update(status='failed', error=str(exc)); write(out / 'manifest.json', meta); raise
    finally:
        env.close()

if __name__ == '__main__':
    main()
