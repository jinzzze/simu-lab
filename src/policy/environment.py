"""Separate learned-action wrapper; preserves the original grasp environment file."""
from __future__ import annotations
import numpy as np
from src.sim import GraspEnv


def observation(env, initial_visual_xy, previous_command, episode_steps=320):
    """Only initial visual estimate, robot state, past command, and robot clock."""
    result = np.r_[initial_visual_xy, env.robot.get_ee_position(),
                   env.robot.get_fingers_width(), previous_command,
                   env._step_count / episode_steps].astype(np.float32)
    if result.shape != (11,) or not np.isfinite(result).all():
        raise ValueError('Invalid policy observation')
    return result


def final_outcome(env):
    config = env.config
    lifted = env._max_lift_run + 1e-9 >= config['min_lift_duration_s']
    stable = env._stable_run + 1e-9 >= config['stable_duration_s']
    return {'success': bool(lifted and stable and all(env._flags.values())),
            'lifted': bool(lifted), 'scene_seed': env.seed,
            'max_clearance_m': env._max_clearance,
            'max_lift_duration_s': env._max_lift_run,
            'final_stable_duration_s': env._stable_run,
            'final_flags': env._flags.copy(),
            'final_object_xyz_m': env.sim.get_base_position('object').tolist(),
            'simulated_duration_s': env._step_count * env.sim.dt,
            'diagnostic_ground_truth_only': True,
            'object_attached_or_teleported': False}


class DemonstrationEnv(GraspEnv):
    """Record commands BEFORE each expert physics step; no post-motion action labels."""
    def begin_recording(self, visual_xy, episode_steps):
        self.visual_xy = np.asarray(visual_xy)
        self.episode_steps = episode_steps
        self.previous_command = np.r_[self.robot.get_ee_position(), self.config['open_width_m']]
        self.observations, self.actions, self.stages = [], [], []

    def _tick(self, target, width, stage):
        ee = self.robot.get_ee_position()
        delta = np.asarray(target) - ee
        length = np.linalg.norm(delta)
        if length > self.config['max_ee_step_m']:
            delta *= self.config['max_ee_step_m'] / length
        command = np.r_[ee + delta, width]
        self.observations.append(observation(self, self.visual_xy, self.previous_command, self.episode_steps))
        self.actions.append(command.astype(np.float32))
        self.stages.append(stage)
        # Preserve the expert's exact original target and low-level operation.
        super()._tick(target, width, stage)
        self.previous_command = command


class PolicyGraspEnv(GraspEnv):
    def begin_policy(self, config, record=False):
        self.policy_config = config
        self._reset_score()
        self._record = bool(record)
        self.frames = [self.render_oblique()] if record else []
        self.trace, self._step_count = [], 0
        self.previous_command = np.r_[self.robot.get_ee_position(), self.config['open_width_m']]
        self.clipped_actions = 0

    def step_policy(self, action):
        command = np.asarray(action, dtype=float)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError('Policy command must be finite XYZ+finger width')
        limits = self.policy_config['safety_limits']
        clipped = np.clip(command, limits['xyz_low'] + [limits['width_low']],
                           limits['xyz_high'] + [limits['width_high']])
        self.clipped_actions += int(not np.allclose(command, clipped, atol=1e-8))
        # No stage machine, object truth, scripted close/release, or target repair.
        self._tick(clipped[:3], clipped[3], 'diffusion_policy')
        self.previous_command = clipped
        return clipped
