"""ROS2 publisher for the controller's mixed cartesian command topic.

rclpy is imported lazily so the rest of the package (tests, replay, dry runs)
runs on any host without a ROS installation.  The controller interface is
confirmed on the robot host (2026-09-29, ROS_DOMAIN_ID=33): mc_core subscribes
to "mx_mix_command" with mc_core_interface/msg/MechUnitCmdArray - one command
per mechanical unit, whose end_pose is the flange pose in torso_flange.  That
is exactly what palm_targets_to_tcp (robot/sink.py) produces from the
planner's palm targets, so this sink only packages and publishes:

    MechUnitCmdArray { std_msgs/Header header; MechUnitCmd[] cmds }
    MechUnitCmd      { int32 mu_idx; float64[] jnt_pos;
                       geometry_msgs/Pose end_pose; float64 psi; bool use_jnt }

Layout "mech_unit_cmd_array_left_right" maps side 0 -> mu_idx 2 ("left07")
and side 1 -> mu_idx 3 ("right07") in cartesian mode (use_jnt=false,
psi=0 - the controller's IK resolves the arm redundancy; revisit if a
specific elbow configuration is ever required).

On the robot host, create the environment with system site packages so rclpy
resolves (README.md "迁移步骤"), and source the interface package first:
    source /opt/ros/humble/setup.bash
    source ~/ros_pkg/movax_interface/install/setup.bash
    uv venv --system-site-packages && uv pip install -e .
"""

from __future__ import annotations

import os

import numpy as np

from moz1_catch.config import Ros2Config
from moz1_catch.robot.sink import HandTargets

LEFT_MU_IDX = 2     # mc_core mechanical unit "left07"
RIGHT_MU_IDX = 3    # mc_core mechanical unit "right07"


def _import_message(message_type: str):
    from rosidl_runtime_py.utilities import get_message
    return get_message(message_type)


def build_cartesian_message(message_cls, targets: HandTargets, layout: str,
                            cmd_cls=None):
    """Build one MechUnitCmdArray from the two flange-in-torso_flange targets.

    cmd_cls may be injected for ROS-free tests; by default the element type
    mc_core_interface/msg/MechUnitCmd is resolved through rosidl.
    """
    if layout != "mech_unit_cmd_array_left_right":
        raise NotImplementedError(f"unknown message layout {layout!r}")
    message = message_cls()
    if not hasattr(message, "cmds"):
        raise TypeError("message_layout mech_unit_cmd_array_left_right expects "
                        "a MechUnitCmdArray-like type with a cmds sequence")
    if cmd_cls is None:
        cmd_cls = _import_message("mc_core_interface/msg/MechUnitCmd")
    for side, mu_idx in ((0, LEFT_MU_IDX), (1, RIGHT_MU_IDX)):
        cmd = cmd_cls()
        cmd.mu_idx = mu_idx
        cmd.use_jnt = False
        cmd.psi = 0.0
        cmd.end_pose.position.x, cmd.end_pose.position.y, cmd.end_pose.position.z = \
            map(float, targets.positions_m[side])
        quaternion = targets.rotations[side].as_quat()          # scipy xyzw
        cmd.end_pose.orientation.x = float(quaternion[0])
        cmd.end_pose.orientation.y = float(quaternion[1])
        cmd.end_pose.orientation.z = float(quaternion[2])
        cmd.end_pose.orientation.w = float(quaternion[3])
        message.cmds.append(cmd)
    return message


def build_joint_message(message_cls, jnt_pos_left, jnt_pos_right, cmd_cls=None):
    """Build one MechUnitCmdArray carrying joint-space commands for both arms.

    The startup script (scripts/move_to_ready.py) uses this to approach the
    ready posture through joint space - cartesian streaming from an arbitrary
    configuration can drive the controller IK through singularities.  Units are
    radians, URDF/controller joint order LeftArm-0..6 / RightArm-0..6 (the
    /joint_states names and the robot.toml posture lists share this order).
    """
    message = message_cls()
    if not hasattr(message, "cmds"):
        raise TypeError("expected a MechUnitCmdArray-like type with a cmds sequence")
    if cmd_cls is None:
        cmd_cls = _import_message("mc_core_interface/msg/MechUnitCmd")
    for mu_idx, jnt_pos in ((LEFT_MU_IDX, jnt_pos_left), (RIGHT_MU_IDX, jnt_pos_right)):
        values = [float(value) for value in jnt_pos]
        if len(values) != 7:
            raise ValueError(f"expected 7 joint values for mu_idx {mu_idx}, got {len(values)}")
        cmd = cmd_cls()
        cmd.mu_idx = mu_idx
        cmd.use_jnt = True
        cmd.psi = 0.0
        cmd.jnt_pos = values
        message.cmds.append(cmd)
    return message


class Ros2CartesianSink:
    """Publishes flange pose targets on the controller's cartesian topic."""

    def __init__(self, ros2: Ros2Config):
        if "TODO" in ros2.cartesian_topic or "TODO" in ros2.message_type \
                or "TODO" in ros2.message_layout:
            raise RuntimeError(
                "ROS2 interface placeholders are not filled in. Set [ros2] "
                "cartesian_topic / message_type / message_layout in "
                "config/interfaces.toml, then retry.")
        # Must precede node creation: rclpy reads the domain id when the
        # context is created, and the controller lives on this domain.
        os.environ["ROS_DOMAIN_ID"] = str(ros2.ros_domain_id)
        import rclpy  # noqa: WPS433 - deliberate lazy import, see module docstring

        # A launcher may already hold one rclpy context (the joint-space
        # approach in scripts/start_catch.py does); take ownership only when
        # we had to create it ourselves, so close() never tears down a shared
        # context mid-process.
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._message_cls = _import_message(ros2.message_type)
        self._layout = ros2.message_layout
        self._rclpy = rclpy
        self._node = rclpy.create_node(ros2.node_name)
        self._publisher = self._node.create_publisher(
            self._message_cls, ros2.cartesian_topic, ros2.queue_size)

    def send(self, targets: HandTargets, t_host: float) -> None:
        message = build_cartesian_message(self._message_cls, targets, self._layout)
        if hasattr(message, "header"):
            message.header.stamp = self._node.get_clock().now().to_msg()
        self._publisher.publish(message)
        self._rclpy.spin_once(self._node, timeout_sec=0)

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_rclpy:
            self._rclpy.shutdown()
