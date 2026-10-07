"""Robot action interface boundaries; no Torch imports in this physics suite."""
from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import pytest
from src.policy.environment import observation,PolicyGraspEnv
ROOT=Path(__file__).resolve().parents[1]


def test_observation_uses_robot_state_and_history_only():
    robot=SimpleNamespace(get_ee_position=lambda:np.array([.1,.2,.3]),get_fingers_width=lambda:.04)
    env=SimpleNamespace(robot=robot,_step_count=16)
    result=observation(env,np.array([-.05,.01]),np.array([.1,.2,.3,.065]),320)
    np.testing.assert_allclose(result,[-.05,.01,.1,.2,.3,.04,.1,.2,.3,.065,.05])


def test_actions_validate_then_apply_without_stage_controller():
    env=PolicyGraspEnv(image_size=32)
    cfg=json.loads((ROOT/'configs/diffusion_policy.json').read_text(encoding='utf-8-sig'))
    try:
        env.reset(50000);env.begin_policy(cfg)
        def forbidden(*args,**kwargs):raise AssertionError('Scripted grasp controller called')
        env.execute_grasp=forbidden
        with pytest.raises(ValueError):env.step_policy([0,0,float('nan'),0])
        assert env._step_count==0
        actual=env.step_policy([-.2,0,.1,100.])
        assert actual[3]==cfg['safety_limits']['width_high']
        assert env._step_count==1 and env.clipped_actions==1
        np.testing.assert_array_equal(env.previous_command,actual)
    finally:env.close()
