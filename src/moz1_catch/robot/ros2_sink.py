"""ROS2 publisher for the cartesian target topic.

rclpy is imported lazily so the rest of the package (tests, replay, dry runs)
runs on any host without a ROS installation.  The controller message type is a
config placeholder until the interface is confirmed; this file is the single
place that needs touching once the spec is known:

1. Fill [ros2] message_type / message_layout in config/interfaces.toml.
2. Adjust build_cartesian_message below for the real type (or add a layout).

On the robot host, create the environment with system site packages so rclpy
resolves (README.md "迁移步骤"):
    uv venv --system-site-packages && uv pip install -e .
"""

from __future__ import annotations

import numpy as np

from moz1_catch.config import Ros2Config
from moz1_catch.robot.sink import HandTargets


def _import_message(message_type: str):
    from rosidl_runtime_py.utilities import get_message
    return get_message(message_type)


def build_cartesian_message(message_cls, targets: HandTargets, layout: str):
    """Build one controller message from the two TCP targets.

    PLACEHOLDER layout "left_then_right_pose_array": geometry_msgs/PoseArray in
    torso_flange with pose[0]=left, pose[1]=right.  Replace or extend for the real
    controller message; keep the mapping in this single function.
    """
    if layout != "left_then_right_pose_array":
        raise NotImplementedError(f"unknown message layout {layout!r}")
    message = message_cls()
    if not hasattr(message, "poses"):
        raise TypeError("message_layout left_then_right_pose_array expects a PoseArray-like type")
    for side in range(2):
        pose = message.poses.add()
        pose.position.x, pose.position.y, pose.position.z = map(float, targets.positions_m[side])
        quaternion = targets.rotations[side].as_quat()
        pose.orientation.x, pose.orientation.y = float(quaternion[0]), float(quaternion[1])
        pose.orientation.z, pose.orientation.w = float(quaternion[2]), float(quaternion[3])
    return message


class Ros2CartesianSink:
    """Publishes TCP pose targets on the controller's cartesian topic."""

    def __init__(self, ros2: Ros2Config):
        if "TODO" in ros2.cartesian_topic or "TODO" in ros2.message_type:
            raise RuntimeError(
                "ROS2 interface placeholders are not filled in. Set [ros2] "
                "cartesian_topic / message_type / message_layout in "
                "config/interfaces.toml, then retry.")
        import rclpy  # noqa: WPS433 - deliberate lazy import, see module docstring

        self._message_cls = _import_message(ros2.message_type)
        self._layout = ros2.message_layout
        self._rclpy = rclpy
        self._node = rclpy.create_node(ros2.node_name)
        self._publisher = self._node.create_publisher(
            self._message_cls, ros2.cartesian_topic, ros2.queue_size)

    def send(self, targets: HandTargets, t_host: float) -> None:
        message = build_cartesian_message(self._message_cls, targets, self._layout)
        self._publisher.publish(message)
        self._rclpy.spin_once(self._node, timeout_sec=0)

    def close(self) -> None:
        self._node.destroy_node()
        self._rclpy.shutdown()
