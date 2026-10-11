"""Relative-policy deployment adapter: only robot proprioception is accessed."""
import numpy as np
from paired_sim import PairedGoalEnv


class DeltaGoalEnv(PairedGoalEnv):
    def step_action(self,action):
        action=np.asarray(action,float)
        if action.shape!=(4,) or not np.isfinite(action).all():raise ValueError('Invalid delta action')
        command=np.r_[self.robot.get_ee_position()+action[:3],action[3]]
        return super().step_action(command)
