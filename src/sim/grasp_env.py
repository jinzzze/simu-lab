"""Panda physical grasp backend. True object state is restricted to reset, labels and scoring.

The controller gets exactly one external XY estimate, its own proprioception and a
fixed goal. It never uses object feedback to repair a grasp or place action.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

ROOT = Path(__file__).resolve().parents[2]


def box_corners(position, quaternion, size):
    """All eight world-space corners; used only for labels and ground-truth scoring."""
    half = np.asarray(size, dtype=float) / 2
    local = np.array([[x, y, z] for x in (-half[0], half[0])
                      for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    return local @ rotation.T + np.asarray(position)


def score_instantaneous(position, quaternion, linear_velocity, angular_velocity,
                        ee_position, finger_width, table_contact, robot_contact, config):
    """Pure geometric/stability predicate, independent of controller execution."""
    corners = box_corners(position, quaternion, config["object_size_m"])
    low = corners.min(axis=0)
    high = corners.max(axis=0)
    xmin, xmax, ymin, ymax = config["workspace_xy_m"]
    rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    flags = {
        "fully_right": bool(low[0] > config["line_x_m"] + config["line_width_m"] / 2),
        "in_workspace": bool(low[0] >= xmin and high[0] <= xmax and low[1] >= ymin and high[1] <= ymax),
        "on_table": bool(table_contact and abs(low[2]) <= config["table_clearance_tolerance_m"]),
        "upright": bool(rotation[2, 2] >= math.cos(math.radians(config["upright_tolerance_deg"]))),
        "slow": bool(np.linalg.norm(linear_velocity) < config["stable_linear_speed_m_s"]
                     and np.linalg.norm(angular_velocity) < config["stable_angular_speed_rad_s"]),
        "released": bool(finger_width >= 0.04 and not robot_contact),
        "withdrawn": bool(np.linalg.norm(np.asarray(ee_position) - np.asarray(position))
                          >= config["withdraw_distance_m"]),
    }
    return flags, float(low[2])


class GraspEnv:
    """Fixed overhead RGB plus Panda proprioception; meters, +X = screen right.

    reset(seed) returns {rgb, proprio, goal}. execute_grasp(estimated_xy) runs one
    fixed pick/lift/transport/place/release/withdraw sequence. Any ground-truth
    values in its result are terminal scoring diagnostics, never controller input.
    """
    def __init__(self, config_path=None, image_size=224):
        path = Path(config_path) if config_path else ROOT / "configs/grasp_sim.json"
        self.config = json.loads(path.read_text(encoding="utf-8-sig"))
        self.image_size = int(image_size)
        self.sim = PyBullet(render_mode="rgb_array", renderer="Tiny", n_substeps=20,
                            background_color=np.array([245, 245, 245]))
        self.client = self.sim.physics_client
        self.client.setPhysicsEngineParameter(numSolverIterations=100)
        self.robot = Panda(self.sim, block_gripper=False, base_position=np.array([-0.6, 0., 0.]))
        self.robot.joint_forces[-2:] = self.config["finger_force_n"]
        for finger in (9, 10):
            self.sim.set_lateral_friction("panda", finger, self.config["finger_friction"])
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3,
                              lateral_friction=0.6)
        self.sim.create_box("object", half_extents=np.asarray(self.config["object_size_m"]) / 2,
                            mass=self.config["object_mass_kg"], position=np.array([-.06, 0., .005]),
                            rgba_color=np.array([.015, .015, .015, 1]),
                            lateral_friction=self.config["object_friction"], spinning_friction=.002)
        self._make_markings()
        self.view_matrix = self.client.computeViewMatrix(self.config["camera_eye_m"],
                               self.config["camera_target_m"], self.config["camera_up"])
        self.projection_matrix = self.client.computeProjectionMatrixFOV(
            self.config["camera_fov_deg"], 1.0, .01, 2.)
        self.frames = []
        self.trace = []
        self._record = False
        self._step_count = 0
        self._reset_score()

    def _make_markings(self):
        c = self.config
        self.sim.create_box("line", half_extents=np.array([c["line_width_m"] / 2, .105, .00005]),
                            mass=0, ghost=True, position=np.array([c["line_x_m"], 0., .00006]),
                            rgba_color=np.array([.02, .02, .02, 1.]))
        # A fixed start frame around the complete sampling region, including object extent.
        for name, center, half in [
            ("start_left", [-.115, 0., .00006], [.0009, .072, .00005]),
            ("start_right", [-.02, 0., .00006], [.0009, .072, .00005]),
            ("start_top", [-.0675, .072, .00006], [.0475, .0009, .00005]),
            ("start_bottom", [-.0675, -.072, .00006], [.0475, .0009, .00005])]:
            self.sim.create_box(name, np.array(half), 0, np.array(center),
                                rgba_color=np.array([.25, .25, .25, 1.]), ghost=True)

    def _reset_score(self):
        self._lift_run = 0.
        self._max_lift_run = 0.
        self._max_clearance = 0.
        self._stable_run = 0.
        self._flags = {}

    def reset(self, seed=0):
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.robot.reset()
        arm = self.robot.inverse_kinematics(link=self.robot.ee_link, position=np.asarray(self.config["initial_ee_xyz_m"]), orientation=np.array([1., 0., 0., 0.]))[:7]
        joints = np.r_[arm, [0., 0.]]
        joints[-2:] = self.config["open_width_m"] / 2
        self.robot.set_joint_angles(joints)
        self.robot.control_joints(joints)
        xy = rng.uniform(self.config["start_low_xy_m"], self.config["start_high_xy_m"])
        yaw = rng.uniform(*self.config["start_yaw_range_rad"])
        self.sim.set_base_pose("object", np.r_[xy, self.config["object_size_m"][2] / 2 + .001],
                               np.asarray(p.getQuaternionFromEuler([0, 0, yaw])))
        self.client.resetBaseVelocity(self.sim._bodies_idx["object"], [0, 0, 0], [0, 0, 0])
        for _ in range(20):
            self.sim.step()
        self.frames = []
        self.trace = []
        self._step_count = 0
        self._reset_score()
        self._record = False
        return {"rgb": self.render(), "proprio": self.robot.get_obs().astype(np.float32),
                "goal": np.asarray(self.config["place_xy_m"], dtype=np.float32)}

    def render(self):
        _, _, rgba, _, _ = self.client.getCameraImage(
            self.image_size, self.image_size, viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix, renderer=p.ER_TINY_RENDERER, shadow=0)
        return np.asarray(rgba, dtype=np.uint8).reshape(self.image_size, self.image_size, 4)[..., :3].copy()

    def render_oblique(self, size=320):
        view = self.client.computeViewMatrix([.40, -.52, .42], [-.015, 0., .035], [0, 0, 1])
        projection = self.client.computeProjectionMatrixFOV(48, 1, .01, 2.)
        _, _, rgba, _, _ = self.client.getCameraImage(size, size, viewMatrix=view,
                                projectionMatrix=projection, renderer=p.ER_TINY_RENDERER, shadow=1)
        return np.asarray(rgba, dtype=np.uint8).reshape(size, size, 4)[..., :3].copy()

    def get_object_xy_for_labels(self):
        """Explicit privileged accessor. Use only training labels or oracle diagnostics."""
        return self.sim.get_base_position("object")[:2].copy()

    def get_object_label(self):
        """Privileged training label with center projected into the fixed RGB camera."""
        position = self.sim.get_base_position("object")
        view = np.asarray(self.view_matrix).reshape(4, 4, order="F")
        projection = np.asarray(self.projection_matrix).reshape(4, 4, order="F")
        clip = projection @ view @ np.r_[position, 1.]
        ndc = clip[:3] / clip[3]
        uv = np.array([(ndc[0] + 1) / 2, (1 - ndc[1]) / 2])
        return {"xy_m": position[:2].copy(), "uv_normalized": uv}

    def _observe_for_scoring(self, stage):
        """Privileged scorer; its state never influences control decisions."""
        pos = self.sim.get_base_position("object")
        quat = self.sim.get_base_orientation("object")
        vel = self.sim.get_base_velocity("object")
        ang = self.sim.get_base_angular_velocity("object")
        obj = self.sim._bodies_idx["object"]
        table_contact = bool(self.client.getContactPoints(obj, self.sim._bodies_idx["table"]))
        contacts = self.client.getContactPoints(obj, self.sim._bodies_idx["panda"])
        robot_contact = bool(contacts)
        contacting_fingers = {contact[4] for contact in contacts if contact[9] > 1e-5}
        bilateral_grasp_contact = {9, 10}.issubset(contacting_fingers)
        flags, clearance = score_instantaneous(pos, quat, vel, ang,
            self.robot.get_ee_position(), self.robot.get_fingers_width(), table_contact,
            robot_contact, self.config)
        self._max_clearance = max(self._max_clearance, clearance)
        self._lift_run = self._lift_run + self.sim.dt if clearance >= self.config["min_lift_clearance_m"] and bilateral_grasp_contact else 0.
        self._max_lift_run = max(self._max_lift_run, self._lift_run)
        self._stable_run = self._stable_run + self.sim.dt if all(flags.values()) else 0.
        self._flags = flags
        if self._step_count % 5 == 0:
            self.trace.append({"t_s": self._step_count * self.sim.dt, "stage": stage,
                               "object_xyz_m": pos.tolist(), "ee_xyz_m": self.robot.get_ee_position().tolist(),
                               "finger_width_m": float(self.robot.get_fingers_width()),
                               "clearance_m": clearance, "bilateral_grasp_contact": bilateral_grasp_contact, "flags": flags})

    def _tick(self, target, width, stage):
        # Only robot proprioception enters the feedback law.
        delta = np.asarray(target) - self.robot.get_ee_position()
        max_step = self.config["max_ee_step_m"]
        length = np.linalg.norm(delta)
        if length > max_step:
            delta *= max_step / length
        fingers_action = (width - self.robot.get_fingers_width()) / .2
        self.robot.set_action(np.r_[delta / .05, fingers_action])
        self.sim.step()
        self._step_count += 1
        self._observe_for_scoring(stage)
        if self._record and self._step_count % 2 == 0:
            self.frames.append(self.render_oblique())

    def _move(self, target, width, stage, max_steps=180, settle_steps=8):
        reached = False
        for _ in range(max_steps):
            self._tick(target, width, stage)
            if np.linalg.norm(np.asarray(target) - self.robot.get_ee_position()) < .002:
                reached = True
                break
        for _ in range(settle_steps):
            self._tick(target, width, stage)
        return reached

    def execute_grasp(self, estimated_xy, record=False):
        """Execute from predicted XY. No actual object state is used to plan/control."""
        xy = np.asarray(estimated_xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("estimated_xy must contain two finite meter coordinates")
        xmin, xmax, ymin, ymax = self.config["workspace_xy_m"]
        if not (xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax):
            return {"success": False, "failure_reason": "prediction_outside_workspace",
                    "estimated_xy_m": xy.tolist(), "seed": self.seed}
        c = self.config
        zlow, zhigh = c["grasp_ee_z_m"], c["transport_ee_z_m"]
        opened, closed = c["open_width_m"], c["closed_width_m"]
        dest = np.asarray(c["place_xy_m"])
        self._reset_score()
        self._record = bool(record)
        self.frames = [self.render_oblique()] if record else []
        self.trace = []
        self._step_count = 0
        stages = {}
        stages["approach"] = self._move(np.r_[xy, zhigh], opened, "approach")
        stages["descend"] = self._move(np.r_[xy, zlow], opened, "descend")
        stages["close"] = self._move(np.r_[xy, zlow], closed, "close", settle_steps=25)
        stages["lift"] = self._move(np.r_[xy, zhigh], closed, "lift", settle_steps=12)
        stages["transport"] = self._move(np.r_[dest, zhigh], closed, "transport")
        stages["place"] = self._move(np.r_[dest, zlow + .001], closed, "place")
        stages["release"] = self._move(np.r_[dest, zlow + .001], opened, "release", settle_steps=20)
        stages["withdraw"] = self._move(np.r_[dest, zhigh], opened, "withdraw", settle_steps=25)
        lifted = self._max_lift_run + 1e-9 >= c["min_lift_duration_s"]
        stable = self._stable_run + 1e-9 >= c["stable_duration_s"]
        success = bool(lifted and stable and all(self._flags.values()))
        return {"success": success, "seed": self.seed, "estimated_xy_m": xy.tolist(),
                "final_object_xyz_m": self.sim.get_base_position("object").tolist(),
                "max_clearance_m": self._max_clearance, "max_lift_duration_s": self._max_lift_run,
                "final_stable_duration_s": self._stable_run, "lifted": lifted,
                "final_flags": self._flags.copy(), "stages_reached": stages,
                "simulated_duration_s": self._step_count * self.sim.dt,
                "diagnostic_ground_truth_only": True, "object_attached_or_teleported": False}

    def close(self):
        self.sim.close()



