"""URDF forward kinematics: the single source of truth for robot constants.

The ready-posture joint angles in config/robot.toml are the only input that
changes with the robot's setup.  Everything the runtime needs is derived from
them plus the URDF at config-load time:

  * per-hand wait poses  ^base T_palm (the palm target frame at the ready
    posture - what the executor holds and what plan_catch starts from),
  * the fixed legwaist constant ^base T_torso (used at both hardware
    boundaries while legwaist is locked),
  * the joint-independent mounting constant ^palm T_flange per hand (URDF
    fixed joint composed with the tangent-plane-centre offset).

Changing the ready posture therefore means editing the angles in ONE place;
derived values can never go stale.
"""

from __future__ import annotations

import xml.etree.ElementTree
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.calib import as_transform, transform_inverse
from moz1_catch.core.geometry import PALM_CENTER_OFFSETS_BODY_M

HAND_NAMES = ("left", "right")


def parse_joints(urdf_path: Path) -> dict[str, dict]:
    """Joint name -> spec (type, parent, child, fixed origin, axis)."""
    joints = {}
    for joint in xml.etree.ElementTree.parse(urdf_path).getroot().findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        xyz = np.array([float(value) for value in (origin.get("xyz", "0 0 0")).split()])
        rpy = [float(value) for value in (origin.get("rpy", "0 0 0")).split()]
        joints[joint.get("name")] = dict(
            type=joint.get("type"),
            parent=joint.find("parent").get("link"),
            child=joint.find("child").get("link"),
            T_origin=as_transform(Rotation.from_euler("xyz", rpy).as_matrix(), xyz),
            axis=(np.array([float(value) for value in axis.get("xyz").split()])
                  if axis is not None else None),
        )
    return joints


def chain_transform(joints: dict, root: str, leaf: str, angles_rad: dict[str, float]) -> np.ndarray:
    """Compose root -> leaf through the URDF tree; angles keyed by joint name."""
    children = {spec["child"]: (name, spec) for name, spec in joints.items()}
    steps = []
    link = leaf
    while link != root:
        if link not in children:
            raise ValueError(f"{root} is not an ancestor of {leaf} (stuck at {link})")
        name, spec = children[link]
        transform = spec["T_origin"]
        if spec["type"] in ("revolute", "continuous"):
            transform = transform @ as_transform(
                Rotation.from_rotvec(spec["axis"] * angles_rad.get(name, 0.)).as_matrix(),
                np.zeros(3))
        steps.append(transform)
        link = spec["parent"]
    result = np.eye(4)
    for transform in reversed(steps):
        result = result @ transform
    return result


def named_angles(legwaist_deg, left_arm_deg, right_arm_deg) -> dict[str, float]:
    """Map the three posture lists (degrees) onto URDF joint names."""
    groups = (("LegWaist", legwaist_deg, 6), ("LeftArm", left_arm_deg, 7),
              ("RightArm", right_arm_deg, 7))
    angles = {}
    for prefix, values, expected in groups:
        values = [float(value) for value in values]
        if len(values) != expected:
            raise ValueError(f"{prefix} posture needs {expected} joint angles, got {len(values)}")
        angles.update({f"{prefix}-{index}": np.deg2rad(value)
                       for index, value in enumerate(values)})
    return angles


def robot_geometry(joints: dict, legwaist_deg, left_arm_deg, right_arm_deg) -> dict:
    """Derive every robot constant from the posture angles and the URDF."""
    angles = named_angles(legwaist_deg, left_arm_deg, right_arm_deg)
    T_base_torso = chain_transform(joints, "base_link", "torso_flange", angles)
    hands = {}
    for side in HAND_NAMES:
        offset = np.asarray(PALM_CENTER_OFFSETS_BODY_M[0 if side == "left" else 1])
        T_flange_hand = joints[f"{side}_hand_palm_joint"]["T_origin"]
        T_base_hand = chain_transform(joints, "base_link", f"{side}_flange", angles) @ T_flange_hand
        rotation = Rotation.from_matrix(T_base_hand[:3, :3])
        wait_quat = rotation.as_quat()  # xyzw
        if wait_quat[3] < 0:
            wait_quat = -wait_quat
        hands[side] = dict(
            wait_position_m=T_base_hand[:3, 3] + rotation.apply(offset),
            wait_quat_xyzw=wait_quat,
            # ^palm T_flange: hand->flange re-expressed at the tangent-plane centre.
            T_palm_flange=as_transform(np.eye(3), -offset) @ transform_inverse(T_flange_hand),
        )
    return dict(T_base_torso=T_base_torso, hands=hands)
