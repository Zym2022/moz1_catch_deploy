"""FK-derived robot constants: fixture cross-check and posture sensitivity."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moz1_catch.config import load_config
from moz1_catch.kinematics import parse_joints, robot_geometry

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

# Isaac fixture reference (pre-revision, see PROVENANCE.md): the accepted
# nominal catch run's start palm poses in base_link.
FIXTURE_WAIT = {
    "left": (np.array((0.19169, -0.62308, 1.1937)),
             np.array((-0.13962998, -0.10416850, -0.67948160, 0.71271113))),
    "right": (np.array((-0.19171, -0.62064, 1.20004)),
              np.array((0.14528112, 0.12739464, 0.71503513, -0.67185472))),
}


def _geometry(**overrides):
    posture = load_config(CONFIG_DIR).robot.posture
    values = dict(legwaist=list(posture.legwaist_joint_deg),
                  left=list(posture.left_arm_joint_deg),
                  right=list(posture.right_arm_joint_deg))
    values.update(overrides)
    return robot_geometry(parse_joints(posture.urdf),
                          values["legwaist"], values["left"], values["right"])


@pytest.mark.parametrize("side", ("left", "right"))
def test_derived_wait_poses_match_the_isaac_fixture(side):
    """FK at the shipped posture must reproduce the sim reference.

    Rotation within 0.01 deg; position off by ~10 mm along +/-X because the
    fixture predates the outward ready-pose seeding of MozBoxer commit 7231c7d
    (which also carries the palm-target revision these constants include).
    """
    config = load_config(CONFIG_DIR)
    hand = next(item for item in config.robot.hands if item.name == side)
    fixture_position, fixture_quat = FIXTURE_WAIT[side]
    rotation = Rotation.from_quat(hand.wait_quat_xyzw)
    angle = (rotation * Rotation.from_quat(fixture_quat).inv()).magnitude()
    residual = np.linalg.norm(hand.wait_position_m - fixture_position)
    assert np.rad2deg(angle) < 0.01
    assert 9.5 < 1000 * residual < 10.6


def test_ready_posture_is_the_single_source_of_truth():
    """Editing a joint angle moves exactly the derived values it should."""
    base = _geometry()
    bumped_left = _geometry(left=[angle + 5.0 if index == 3 else angle
                                  for index, angle in enumerate(
                                      load_config(CONFIG_DIR).robot.posture.left_arm_joint_deg)])
    assert not np.allclose(bumped_left["hands"]["left"]["wait_position_m"],
                           base["hands"]["left"]["wait_position_m"])
    # The other hand, the legwaist constant and the mounting constant are
    # untouched by a left-arm angle change.
    np.testing.assert_allclose(bumped_left["hands"]["right"]["wait_position_m"],
                               base["hands"]["right"]["wait_position_m"])
    np.testing.assert_allclose(bumped_left["T_base_torso"], base["T_base_torso"])
    np.testing.assert_allclose(bumped_left["hands"]["left"]["T_palm_flange"],
                               base["hands"]["left"]["T_palm_flange"])

    twisted_waist = _geometry(legwaist=[5.0 if index == 1 else angle for index, angle in enumerate(
        load_config(CONFIG_DIR).robot.posture.legwaist_joint_deg)])
    assert not np.allclose(twisted_waist["T_base_torso"], base["T_base_torso"])
    # Arm poses relative to torso_flange are unchanged; the base_link wait
    # poses move with the legwaist constant, as they must.
    assert not np.allclose(twisted_waist["hands"]["left"]["wait_position_m"],
                           base["hands"]["left"]["wait_position_m"])


def test_mounting_constant_is_joint_independent():
    """^palm T_flange depends only on the URDF joint and the tangent offset."""
    base = _geometry()
    other = _geometry(legwaist=[10., -20., 15., -5., 8., 3.],
                      left=[10.] * 7, right=[-10.] * 7)
    for side in ("left", "right"):
        np.testing.assert_allclose(other["hands"][side]["T_palm_flange"],
                                   base["hands"][side]["T_palm_flange"], atol=1e-12)
