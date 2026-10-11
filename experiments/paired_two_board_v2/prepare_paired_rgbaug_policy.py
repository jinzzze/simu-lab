"""Freeze a single color-augmentation revision after the retained v5 failure."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    failure = ROOT / 'artifacts/reports/paired_recurrent_goal_precision_v5/appearance_analysis_v1/results.json'
    assert (failure.parent.parent / 'appearance_validation.json').exists()
    result = read(failure)
    assert all(result['groups'][c]['successes'] == 0 and result['groups'][c]['episodes'] == 24
               for c in ('simulation_only', 'video_guided'))
    cfg = read(ROOT / 'configs/policy_recurrent_goal_precision_v5.json')
    cfg.update(revision='paired_recurrent_goal_rgbaug_v6', evaluation_script='evaluate_paired_recurrent_rgbaug_policy.py',
               created_utc=datetime.now(timezone.utc).isoformat(),
               revision_reason='v5 retained appearance failures0/24 in both conditions; test stronger episode-consistent RGB color/brightness/contrast augmentation, same learner/data/loss/update budget',
               hypothesis_status='exploratory robustness revision informed by earlier appearance-suite failures; fresh scene seeds and one newly frozen color combination; no outcome-based checkpoint/seed selection')
    cfg['augmentation'].update(brightness=.12, contrast=.20, channel_gain_halfwidth=.25,
                               transform='episode-consistent ((RGB-.5)*contrast+.5+brightness)*per-channel gain, clamp0..1; no geometry, mask, object labels or future-derived parameters')
    cfg['training']['budget_comparison'] = 'same v5 architecture, labels, normalizer, coordinate loss, initialization, episode stream, 2500updates; only the RGB augmentation recipe changes; actual frame/compute counts recorded'
    for name, start in [('nominal', 67000), ('appearance', 67100), ('wider_start', 67200)]:
        cfg['evaluation']['suites'][name]['start'] = start
    cfg['evaluation']['suites']['appearance']['scope'] = 'known color-shift family from v5 failure; new scene IDs, not an unseen appearance family'
    cfg['evaluation']['suites']['novel_appearance'] = {'start': 67120, 'count': 8, 'jitter_m': .003,
        'object_rgba': [.04, .04, .04, 1.], 'table_rgba': [.85, .80, .75, 1.],
        'scope': 'new warm table color combination frozen before v6 training/evaluation; limited appearance shift only'}
    for name, start in [('blank_rgb', 67300), ('swapped_goal_vector', 67400)]:
        cfg['evaluation']['diagnostics'][name]['start'] = start
    path = ROOT / 'configs/policy_recurrent_goal_rgbaug_v6.json'; assert not path.exists(); write(path, cfg)
    out = ROOT / 'receipts/paired_recurrent_goal_rgbaug_v6_plan'; assert not out.exists(); out.mkdir()
    shutil.copy2(path, out / path.name); shutil.copy2(__file__, out / Path(__file__).name)
    write(out / 'manifest.json', {'frozen_utc': cfg['created_utc'], 'config_sha256': sha(path),
          'v5_appearance_failure_source_sha256': sha(failure), 'earlier_models_and_results_preserved': True,
          'same_model_labels_coordinate_loss_and_update_budget': True, 'changed_component': 'RGB augmentation recipe',
          'new_scene_ids_and_warm_color_combination_not_evaluated_before_freeze': True})
    addendum = ROOT / 'configs/evaluation_addendum_rgbaug_v6.json'; assert not addendum.exists()
    write(addendum, {'frozen_utc': cfg['created_utc'], 'config_sha256': sha(path),
        'development_scope': 'shared old8 validation scenes reused; exploratory method development',
        'new_scenes': cfg['evaluation']['suites'],
        'known_appearance_scope': 'earlier color shift informed augmentation development; nominal/known-shift/new-warm-shift/wider scenes frozen before training',
        'diagnostic_controls': 'same scenes and initial RGB, original numerical goal, fixed seed7 each condition',
        'goal_corruption_scope': 'conflicting vector/decoder versus original RGB/scoring goal, not new-goal benchmark',
        'cooperation_skipped': cfg['collaboration_skipped']})
    text = (ROOT / 'train_paired_recurrent_precision_policy.py').read_text(encoding='utf-8')
    text = text.replace('train_paired_recurrent_precision_policy.py', 'train_paired_recurrent_rgbaug_policy.py').replace('configs/policy_recurrent_goal_precision_v5.json', 'configs/policy_recurrent_goal_rgbaug_v6.json')
    anchor = 'from src.perception.training import seed_everything'
    text = text.replace(anchor, anchor + '\nfrom paired_rgb_augmentation import augment_rgb')
    text = text.replace("'paired_image_policy.py')", "'paired_image_policy.py','paired_rgb_augmentation.py')")
    old = "brightness=(torch.rand((len(rgb),1,1,1,1),device=device,generator=aug)*2-1)*cfg['augmentation']['brightness']\n                    contrast=1+(torch.rand((len(rgb),1,1,1,1),device=device,generator=aug)*2-1)*cfg['augmentation']['contrast']\n                    rgb=((rgb-.5)*contrast+.5+brightness).clamp(0,1)"
    assert old in text
    text = text.replace(old, "rgb,augmentation_parameters=augment_rgb(rgb,cfg['augmentation'],aug)\n                    augmentation_stream.update(augmentation_parameters.detach().cpu().numpy().tobytes())")
    text = text.replace('stream=hashlib.sha256();started=', 'stream=hashlib.sha256();augmentation_stream=hashlib.sha256();started=')
    text = text.replace('episode_fraction_stream_sha256=stream.hexdigest(),', 'episode_fraction_stream_sha256=stream.hexdigest(),augmentation_parameter_stream_sha256=augmentation_stream.hexdigest(),')
    target = ROOT / 'train_paired_recurrent_rgbaug_policy.py'; assert not target.exists(); target.write_text(text, encoding='utf-8')
    text = (ROOT / 'evaluate_paired_recurrent_precision_policy.py').read_text(encoding='utf-8')
    text = text.replace('evaluate_paired_recurrent_precision_policy.py', 'evaluate_paired_recurrent_rgbaug_policy.py').replace('configs/policy_recurrent_goal_precision_v5.json', 'configs/policy_recurrent_goal_rgbaug_v6.json').replace('evaluation_addendum_precision_v5.json', 'evaluation_addendum_rgbaug_v6.json')
    target = ROOT / 'evaluate_paired_recurrent_rgbaug_policy.py'; assert not target.exists(); target.write_text(text, encoding='utf-8')
    target = ROOT / 'run_paired_rgbaug_followup.py'; assert not target.exists(); shutil.copy2(ROOT / 'run_paired_precision_followup.py', target)
    print('Frozen RGB augmentation revision; retained v5 failures; no shared runtime/source changes')

if __name__ == '__main__':
    main()
