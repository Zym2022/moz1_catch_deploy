"""Transform chain, clock mapping and prior rotation contracts."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moz1_catch.calib import (
    ANALYSIS_BASIS_M, ANALYSIS_PRIOR_MPS2, FrameChain, MocapClock,
    as_transform, check_se3, rotate_prior, transform_inverse,
)


def test_se3_checks_reject_broken_transforms():
    check_se3(np.eye(4), "identity")
    with pytest.raises(ValueError, match="SO\\(3\\)"):
        check_se3(np.diag((1., 1., -1., 1.)), "reflected")
    with pytest.raises(ValueError, match="bottom row"):
        check_se3(np.array(((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.),
                            (0., 0., 1., 1.))), "skewed")
    with pytest.raises(ValueError, match="finite"):
        check_se3(np.full((4, 4), np.nan), "nan")


def test_frame_chain_recovers_box_geometry_pose():
    rng = np.random.default_rng(20260928)
    T_FM = as_transform(Rotation.from_rotvec((.3, -.4, 1.1)).as_matrix(), (.7, -1.2, .25))
    T_GD = as_transform(Rotation.from_rotvec((-.2, .5, .3)).as_matrix(), (.01, -.02, .05))
    chain = FrameChain(T_FM=T_FM, T_DG=transform_inverse(T_GD))
    for _ in range(5):
        T_BG = as_transform(Rotation.random(random_state=rng).as_matrix(),
                            rng.uniform(-1., 1., 3))
        T_MD = transform_inverse(T_FM) @ T_BG @ T_GD  # device reports this
        position, rotation = chain.box_geometry_pose(T_MD[:3, 3], Rotation.from_matrix(T_MD[:3, :3]).as_quat())
        np.testing.assert_allclose(position, T_BG[:3, 3], atol=1e-12)
        np.testing.assert_allclose(rotation.as_matrix(), T_BG[:3, :3], atol=1e-12)


def test_frame_chain_composes_the_legwaist_constant():
    """mocap -> torso_flange -> base_link must equal the composed extrinsic."""
    rng = np.random.default_rng(7)
    T_FM = as_transform(Rotation.from_rotvec((.2, .1, -.4)).as_matrix(), (.5, -.3, .8))
    T_base_torso = as_transform(Rotation.from_rotvec((.05, -.02, .01)).as_matrix(),
                                (0., 0.023583, 1.202396))
    chain = FrameChain(T_FM=T_FM, T_DG=np.eye(4), T_base_torso=T_base_torso)
    effective = T_base_torso @ T_FM
    np.testing.assert_allclose(chain.effective, effective, atol=1e-12)
    for _ in range(3):
        T_MD = as_transform(Rotation.random(random_state=rng).as_matrix(), rng.uniform(-1., 1., 3))
        position, rotation = chain.box_geometry_pose(T_MD[:3, 3],
                                                    Rotation.from_matrix(T_MD[:3, :3]).as_quat())
        expected = effective @ T_MD
        np.testing.assert_allclose(position, expected[:3, 3], atol=1e-12)
        np.testing.assert_allclose(rotation.as_matrix(), expected[:3, :3], atol=1e-12)


def test_frame_chain_scales_device_units():
    chain = FrameChain(T_FM=np.eye(4), T_DG=np.eye(4), position_scale=.001)
    position, _ = chain.box_geometry_pose(np.array((1000., -1500., 1200.)), (0., 0., 0., 1.))
    np.testing.assert_allclose(position, (1., -1.5, 1.2), atol=1e-12)


def test_prior_rotation_matches_manual_composition():
    rotation = Rotation.from_rotvec((.3, -.4, 1.1))
    T_FM = as_transform(rotation.as_matrix(), np.zeros(3))
    manual = rotation.as_matrix() @ (ANALYSIS_BASIS_M.T @ ANALYSIS_PRIOR_MPS2)
    np.testing.assert_allclose(rotate_prior(ANALYSIS_PRIOR_MPS2, T_FM), manual, atol=1e-12)
    # Identity extrinsic: the robot-frame prior equals the mocap-frame prior.
    np.testing.assert_allclose(rotate_prior(ANALYSIS_PRIOR_MPS2, np.eye(4)),
                               ANALYSIS_BASIS_M.T @ ANALYSIS_PRIOR_MPS2, atol=1e-12)
    # Rotation preserves the prior magnitude (it is still ~gravity-sized).
    rotated = rotate_prior(ANALYSIS_PRIOR_MPS2, T_FM)
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(ANALYSIS_PRIOR_MPS2))


def test_mocap_clock_offset():
    clock = MocapClock(offset_s=-12.5)
    assert clock.to_host(100.0) == pytest.approx(87.5)
