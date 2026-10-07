"""Collect new simulator expert action demonstrations for the independent DP group."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policy.environment import DemonstrationEnv, final_outcome
from src.predictor_process import PredictorProcess


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    cfg = read(ROOT / 'configs/diffusion_policy.json')
    out = ROOT / 'data/processed/diffusion_policy_v1'
    manifest_path = ROOT / 'data/manifests/diffusion_policy_v1.json'
    if manifest_path.exists() or out.exists():
        raise FileExistsError('Preserve existing DP collection; select a fresh version to recollect')
    out.mkdir(parents=True)
    sources = [ROOT / p for p in ['scripts/collect_diffusion_demonstrations.py', 'src/policy/environment.py',
               'src/sim/grasp_env.py', 'configs/grasp_sim.json', 'configs/diffusion_policy.json',
               'src/predictor_process.py', 'scripts/rgb_predictor_worker.py']]
    manifest = {'status': 'collecting', 'started_utc': datetime.now(timezone.utc).isoformat(), 'plan': cfg,
                'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
                'visual_checkpoint_sha256': sha(ROOT / cfg['visual_checkpoint']),
                'provenance': 'New simulated expert commands; applicant videos only supplied upstream visual pretraining',
                'retention': 'All scripted demonstrations retained regardless of success; no outcome-based filtering', 'splits': {}}
    write(manifest_path, manifest)
    env = DemonstrationEnv(image_size=224)
    try:
        with PredictorProcess(ROOT / cfg['visual_checkpoint'], out / 'predictor_stderr.log') as predict:
            for split, spec in cfg['splits'].items():
                observations, actions, images, metadata = [], [], [], []
                for scene in range(spec['start'], spec['start'] + spec['count']):
                    rgb = env.reset(scene)['rgb']
                    visual_xy = predict(rgb)
                    initial_truth = env.get_object_xy_for_labels()
                    env.begin_recording(visual_xy, cfg['episode_steps'])
                    initial_result = env.execute_grasp(initial_truth)
                    expert_length = len(env.actions)
                    if expert_length > cfg['episode_steps']:
                        raise RuntimeError('Expert trajectory exceeds predeclared episode length; preserve partial data')
                    hold = np.r_[env.config['place_xy_m'], env.config['transport_ee_z_m']]
                    while len(env.actions) < cfg['episode_steps']:
                        env._tick(hold, env.config['open_width_m'], 'terminal_hold')
                    outcome = final_outcome(env)
                    observations.append(np.asarray(env.observations)); actions.append(np.asarray(env.actions)); images.append(rgb)
                    metadata.append({'scene_seed': scene, 'initial_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                                     'initial_visual_xy_m': visual_xy.tolist(), 'initial_truth_xy_m': initial_truth.tolist(),
                                     'expert_steps_before_padding': expert_length, 'outcome': outcome,
                                     'initial_controller_success': initial_result['success']})
                    if (scene - spec['start'] + 1) % 8 == 0:
                        print(f'{split}: {scene-spec["start"]+1}/{spec["count"]} demos', flush=True)
                target = out / f'{split}.npz'
                np.savez_compressed(target, observations=np.asarray(observations, np.float32),
                                    actions=np.asarray(actions, np.float32), initial_rgb=np.asarray(images, np.uint8),
                                    scene_seeds=np.asarray([r['scene_seed'] for r in metadata]),
                                    success=np.asarray([r['outcome']['success'] for r in metadata]))
                write(out / f'{split}_episodes.json', metadata)
                manifest['splits'][split] = {'episodes': len(metadata), 'steps': len(metadata)*cfg['episode_steps'],
                    'successful_demonstrations': sum(r['outcome']['success'] for r in metadata),
                    'dataset_sha256': sha(target), 'episodes_sha256': sha(out / f'{split}_episodes.json')}
                write(manifest_path, manifest)
    finally:
        env.close()
    for path in sources:
        assert sha(path) == manifest['source_sha256'][str(path.relative_to(ROOT))], 'Collection source changed during run'
    manifest['status'] = 'completed'
    write(manifest_path, manifest)
    print(json.dumps(manifest['splits'], indent=2), flush=True)


if __name__ == '__main__': main()
