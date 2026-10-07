"""One predeclared RGB-to-world-model prediction demonstration, without replanning."""
from pathlib import Path
import hashlib,json,sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.sim import GraspEnv
from src.predictor_process import PredictorProcess
from src.world_model import MacroOutcomePredictor
from evaluate_visual_grasp import save_video


def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:clean(val) for key,val in value.items()}
    return value


def main():
    cfg=json.loads((ROOT/'configs/grasp_world_model.json').read_text(encoding='utf-8-sig'))
    spec=cfg['integration_demo'];out=ROOT/'artifacts/reports/grasp_world_model_v1/integration'
    if (out/'result.json').exists():raise FileExistsError('Preserve completed integration result')
    out.mkdir(parents=True,exist_ok=True)
    model_path=ROOT/'artifacts/runs/grasp_world_model_v1/ensemble.npz'
    wm=MacroOutcomePredictor(model_path);env=GraspEnv(image_size=224)
    try:
        obs=env.reset(spec['scene_seed'])
        with PredictorProcess(ROOT/spec['visual_model'],out/'predictor_stderr.log') as visual:
            estimated=visual(obs['rgb'])
        command=estimated.copy()
        # The macro model receives only the RGB estimate and the command.
        prediction=clean(wm(estimated,command))
        before={'scene_seed':spec['scene_seed'],'state_estimate_xy_m':estimated.tolist(),'command_xy_m':command.tolist(),
                'world_model_prediction':prediction,'rgb_sha256':hashlib.sha256(obs['rgb'].tobytes()).hexdigest(),
                'visual_checkpoint_sha256':hashlib.sha256((ROOT/spec['visual_model']).read_bytes()).hexdigest(),
                'world_model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
                'ground_truth_supplied_to_model':False,'recorded_before_action_execution':True}
        (out/'input_prediction.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
        truth_initial=env.get_object_xy_for_labels() # scoring diagnostic after prediction has been fixed
        outcome=env.execute_grasp(command,record=True)
        truth_final=env.get_object_xy_for_labels()
        result={**before,'actual_outcome':outcome,'actual_initial_xy_m':truth_initial.tolist(),'actual_final_xy_m':truth_final.tolist(),
                'initial_visual_error_mm':float(np.linalg.norm(estimated-truth_initial)*1000),
                'final_world_model_error_mm':float(np.linalg.norm(np.asarray(prediction['final_xy_m'])-truth_final)*1000),
                'scope':'one preselected integration example; no policy improvement or calibration claim',
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        save_video(out/'rgb_world_model_demo.mp4',env.frames,1/(env.sim.dt*2))
        frames=[cv2.cvtColor(env.frames[index],cv2.COLOR_RGB2BGR) for index in np.linspace(0,len(env.frames)-1,6).round().astype(int)]
        cv2.imwrite(str(out/'contact_sheet.jpg'),np.vstack([np.hstack(frames[:3]),np.hstack(frames[3:])]))
        cap=cv2.VideoCapture(str(out/'rgb_world_model_demo.mp4'));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==len(env.frames)
        result['decoded_video_frames']=count
        (out/'trace.json').write_text(json.dumps(env.trace,indent=2),encoding='utf-8')
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)
    finally:env.close()


if __name__=='__main__':main()
