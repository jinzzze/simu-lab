"""Shared new train-only geometry; legacy robot physics/actions/scoring unchanged."""
import numpy as np
from common import ROOT, OLD, read
from paired_sim import PairedGoalEnv
from src.sim.grasp_env import GraspEnv


class VideoTimingEnv(PairedGoalEnv):
    def __init__(self,image_size=224):
        self.task=read(ROOT/'configs/sim_task_v1.json')
        self.goal_id='G1';self.goal_xy=np.array(self.task['goals']['G1']['xy_m'])
        self.capture_actions=False
        GraspEnv.__init__(self,OLD/self.task['physics_config'],image_size=image_size)
