"""Boundary and physical checks for the fixed grasp backend, not learned-policy tests."""
import json
from pathlib import Path

import numpy as np
import pytest
import pybullet as p

from src.sim.grasp_env import GraspEnv, score_instantaneous

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/grasp_sim.json").read_text(encoding="utf-8-sig"))


def flags_at(pos=(.10, 0, .005), quat=(0, 0, 0, 1), **kwargs):
    params = dict(position=pos, quaternion=quat, linear_velocity=(0, 0, 0),
                  angular_velocity=(0, 0, 0), ee_position=(.10, 0, .12),
                  finger_width=.065, table_contact=True, robot_contact=False, config=CONFIG)
    params.update(kwargs)
    return score_instantaneous(**params)[0]


def test_center_crossing_is_not_complete_crossing():
    assert not flags_at((.05, 0, .005))["fully_right"]
    assert flags_at((.06, 0, .005))["fully_right"]


def test_rotated_corners_are_used():
    yaw45 = p.getQuaternionFromEuler([0, 0, np.pi / 4])
    assert flags_at((.058, 0, .005))["fully_right"]
    assert not flags_at((.058, 0, .005), quat=yaw45)["fully_right"]


def test_airborne_or_robot_held_is_not_placed():
    assert not flags_at((.10, 0, .04), table_contact=False)["on_table"]
    assert not flags_at(robot_contact=True)["released"]
    assert not flags_at(finger_width=.02)["released"]
    assert not flags_at(ee_position=(.10, 0, .02))["withdrawn"]


def test_outside_workspace_and_fast_or_toppled_fail():
    assert not flags_at((.158, 0, .005))["in_workspace"]
    assert not flags_at(linear_velocity=(.006, 0, 0))["slow"]
    assert not flags_at(angular_velocity=(0, 0, .11))["slow"]
    assert not flags_at(quat=p.getQuaternionFromEuler([np.pi / 2, 0, 0]))["upright"]


def test_valid_final_pose_is_accepted():
    assert all(flags_at().values())


def test_seeded_rgb_and_camera_direction():
    env = GraspEnv()
    try:
        first = env.reset(100)
        xy = env.get_object_xy_for_labels()
        uv = env.get_object_label()["uv_normalized"]
        assert first["rgb"].shape == (224, 224, 3)
        assert first["rgb"].dtype == np.uint8
        env.reset(100)
        np.testing.assert_allclose(xy, env.get_object_xy_for_labels(), atol=1e-10)
        np.testing.assert_array_equal(first["rgb"], env.render())
        # All start objects are left of world x=0 and must also appear left of image center.
        assert xy[0] < 0 and uv[0] < .5
    finally:
        env.close()


def test_physical_grasp_lifts_and_controller_never_calls_label_accessor():
    env = GraspEnv()
    try:
        env.reset(123)
        oracle_xy = env.get_object_xy_for_labels()
        def forbidden():
            raise AssertionError("Controller must never read the label accessor")
        env.get_object_xy_for_labels = forbidden
        result = env.execute_grasp(oracle_xy)
        assert result["success"], result
        assert result["max_clearance_m"] >= CONFIG["min_lift_clearance_m"]
        assert result["max_lift_duration_s"] >= CONFIG["min_lift_duration_s"]
        assert result["final_stable_duration_s"] >= CONFIG["stable_duration_s"]
        assert env.client.getNumConstraints() == 0
        assert not result["object_attached_or_teleported"]
    finally:
        env.close()


def test_wrong_prediction_is_not_repaired_by_true_state():
    env = GraspEnv()
    try:
        env.reset(123)
        actual = env.get_object_xy_for_labels()
        result = env.execute_grasp(actual + [.055, 0])
        assert not result["success"]
        assert not result["lifted"]
    finally:
        env.close()


def test_invalid_prediction_is_rejected():
    env = GraspEnv()
    try:
        env.reset(1)
        with pytest.raises(ValueError):
            env.execute_grasp([float("nan"), 0])
        result = env.execute_grasp([10, 10])
        assert not result["success"]
        assert result["failure_reason"] == "prediction_outside_workspace"
    finally:
        env.close()
