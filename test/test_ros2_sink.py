"""Layout tests for the mx_mix_command sink - no ROS installation required."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.robot.ros2_sink import (LEFT_MU_IDX, RIGHT_MU_IDX,
                                         build_cartesian_message,
                                         build_joint_message)
from moz1_catch.robot.sink import HandTargets

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class _Point:
    def __init__(self):
        self.x = self.y = self.z = 0.


class _Quaternion:
    def __init__(self):
        self.x = self.y = self.z = 0.
        self.w = 1.


class _Pose:
    def __init__(self):
        self.position = _Point()
        self.orientation = _Quaternion()


class _MechUnitCmd:
    def __init__(self):
        self.mu_idx = -1
        self.jnt_pos = []
        self.end_pose = _Pose()
        self.psi = None
        self.use_jnt = True


class _Header:
    def __init__(self):
        self.stamp = None


class _MechUnitCmdArray:
    def __init__(self):
        self.header = _Header()
        self.cmds = []


def _targets():
    positions = np.array(((-0.30, -0.55, 1.10), (0.30, -0.55, 1.10)))
    rotations = (Rotation.from_euler("xyz", (5., -10., 15.), degrees=True),
                 Rotation.from_euler("xyz", (-5., 10., -15.), degrees=True))
    return HandTargets(positions_m=positions, rotations=rotations)


def test_mech_unit_layout_maps_sides_to_arm_units_in_cartesian_mode():
    message = build_cartesian_message(_MechUnitCmdArray, _targets(),
                                      "mech_unit_cmd_array_left_right",
                                      cmd_cls=_MechUnitCmd)
    assert [cmd.mu_idx for cmd in message.cmds] == [LEFT_MU_IDX, RIGHT_MU_IDX]
    for cmd in message.cmds:
        assert cmd.use_jnt is False and cmd.psi == 0.0 and cmd.jnt_pos == []
    np.testing.assert_allclose(
        (message.cmds[0].end_pose.position.x, message.cmds[0].end_pose.position.y,
         message.cmds[0].end_pose.position.z), (-0.30, -0.55, 1.10))
    # scipy xyzw must land in geometry_msgs wxyz field slots by name.
    quat = _targets().rotations[1].as_quat()
    orientation = message.cmds[1].end_pose.orientation
    assert (orientation.x, orientation.y, orientation.z, orientation.w) == \
        tuple(map(float, quat))


def test_joint_message_sets_joint_mode_for_both_arms():
    left = np.deg2rad((-44.45, -32.26, -33.42, -84.56, 41.84, -27.54, -0.74))
    right = np.deg2rad((44.46, -32.27, 33.41, 84.63, -36.92, -27.64, 0.91))
    message = build_joint_message(_MechUnitCmdArray, left, right, cmd_cls=_MechUnitCmd)
    assert [cmd.mu_idx for cmd in message.cmds] == [LEFT_MU_IDX, RIGHT_MU_IDX]
    for cmd, values in zip(message.cmds, (left, right)):
        assert cmd.use_jnt is True and cmd.psi == 0.0
        np.testing.assert_allclose(cmd.jnt_pos, values)


def test_joint_message_rejects_wrong_joint_count():
    try:
        build_joint_message(_MechUnitCmdArray, np.zeros(6), np.zeros(7),
                            cmd_cls=_MechUnitCmd)
    except ValueError as error:
        assert "7 joint values" in str(error)
    else:
        raise AssertionError("wrong joint count accepted")


def test_unknown_layout_is_rejected():
    try:
        build_cartesian_message(_MechUnitCmdArray, _targets(), "bogus_layout",
                                cmd_cls=_MechUnitCmd)
    except NotImplementedError as error:
        assert "bogus_layout" in str(error)
    else:
        raise AssertionError("unknown layout accepted")


def test_live_config_declares_the_confirmed_controller_interface():
    from moz1_catch.config import load_config
    config = load_config(CONFIG_DIR)
    assert config.ros2.cartesian_topic == "mx_mix_command"
    assert config.ros2.message_type == "mc_core_interface/msg/MechUnitCmdArray"
    assert config.ros2.message_layout == "mech_unit_cmd_array_left_right"
    assert config.ros2.ros_domain_id == 33
