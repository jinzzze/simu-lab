"""Torch-free export validation, macro-action metrics and plots."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.world_model.numpy_model import MacroOutcomePredictor
from src.world_model.evaluation import metrics,make_report
from src.world_model.integrity import (load_training_data, verify_receipt, check_ensemble_anchors, validate_predictions, require_prediction_agreement)
OUT=ROOT/'artifacts/runs/grasp_world_model_v1'
REPORT=ROOT/'artifacts/reports/grasp_world_model_v1'


def main():
    run=json.loads((OUT/'run_state.json').read_text());metadata=run['metadata'];cfg=metadata['plan']
    verify_receipt(ROOT,run.get('verified_artifacts'))
    data,_=load_training_data(ROOT,metadata)
    check_ensemble_anchors(ROOT,OUT,metadata)
    saved=np.load(OUT/'torch_predictions.npz',allow_pickle=False)
    validate_predictions(saved['delta'],saved['probabilities'],(len(cfg['training']['seeds']),len(data['split']),2))
    delta=saved['delta'].mean(axis=0);prob=saved['probabilities'].mean(axis=0)
    predictor=MacroOutcomePredictor(OUT/'ensemble.npz')
    members=predictor.predict_members(data['initial_xy_m'],data['command_xy_m'])
    require_prediction_agreement(saved['delta'],saved['probabilities'],members)
    actual=predictor(data['initial_xy_m'],data['command_xy_m'])
    delta_difference=float(np.max(np.abs(actual['delta_xy_m']-delta)))
    probability_difference=float(np.max(np.abs(np.column_stack([actual['lift_probability'],actual['success_probability']])-prob)))
    assert delta_difference<1e-6 and probability_difference<2e-6,'NumPy predictions disagree with Torch'
    ids={split:np.flatnonzero(data['split']==split) for split in cfg['splits']}
    means=data['delta_xy_m'][ids['train']].mean(axis=0);priors=data['binary_targets'][ids['train']].mean(axis=0)
    evaluation={}
    for split,index in ids.items():
        constant=np.broadcast_to(priors,(len(index),2))
        evaluation[split]={'ensemble':metrics(delta[index],prob[index],data['delta_xy_m'][index],data['binary_targets'][index]),
            'persistence':metrics(np.zeros((len(index),2)),constant,data['delta_xy_m'][index],data['binary_targets'][index]),
            'train_mean_delta':metrics(np.broadcast_to(means,(len(index),2)),constant,data['delta_xy_m'][index],data['binary_targets'][index]),
            'independent_scenes':len(np.unique(data['scene_seeds'][index]))}
    result={**metadata,'per_seed':run['per_seed'],'evaluations':evaluation,'status':'completed',
        'export_max_delta_difference_m':delta_difference,'export_max_probability_difference':probability_difference,
        'ensemble_sha256':hashlib.sha256((OUT/'ensemble.npz').read_bytes()).hexdigest(),
        'postprocessing_source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
          for path in [Path(__file__),ROOT/'src/world_model/evaluation.py',ROOT/'src/world_model/numpy_model.py',ROOT/'src/world_model/integrity.py']},
        'verified_artifacts':run['verified_artifacts'],
        'recovery_note':run.get('recovery'),
        'limits':['Five actions share each initial scene: 160 test transitions represent 32 initial scenes.',
            'Training and state-prediction evaluation use true initial XY; the separate RGB integration uses estimated state.',
            'Fixed grasp skill, goal, object and physics; no video generation or long-horizon rollout evaluation.',
            'Transition labels come entirely from simulation. Phone videos contribute only through the visual encoder.']}
    for folder in [OUT,REPORT]:
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(REPORT/'predictions.npz',predicted_delta_xy_m=delta,probabilities=prob,
        initial_xy_m=data['initial_xy_m'],command_xy_m=data['command_xy_m'],true_delta_xy_m=data['delta_xy_m'],
        binary_targets=data['binary_targets'],split=data['split'],scene_seeds=data['scene_seeds'])
    make_report(result,data,delta,prob,run['curves'],ids['test'])
    print(json.dumps(evaluation['test'],indent=2))


if __name__=='__main__':main()
