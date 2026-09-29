"""Joint feedback recording, FK reconstruction and tracking analysis."""

from pathlib import Path

import numpy as np
import pytest

from moz1_catch.config import load_config
from moz1_catch.feedback import JointFeedbackLog, fk_palm_series, tracking_summary
from moz1_catch.kinematics import named_angles, parse_joints

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class FakeClock:
    """Deterministic host clock: 5 ms per call, like a 200 Hz /joint_states."""

    def __init__(self):
        self.t = 1000.

    def __call__(self) -> float:
        self.t += 0.005
        return self.t


def test_snapshot_requires_both_arms_and_merges_partials():
    log = JointFeedbackLog(clock=FakeClock())
    log.on_message([f"LeftArm-{i}" for i in range(7)], [0.1 * i for i in range(7)])
    assert len(log) == 0                       # right arm still unknown
    log.on_message([f"RightArm-{i}" for i in range(7)], [-0.1 * i for i in range(7)])
    assert len(log) == 1
    arrays = log.arrays()
    np.testing.assert_allclose(arrays["feedback_joint_left_rad"],
                               [0.1 * np.arange(7)], atol=1e-12)
    np.testing.assert_allclose(arrays["feedback_joint_right_rad"],
                               [-0.1 * np.arange(7)], atol=1e-12)
    log.on_message(["LeftArm-0"], [1.5])       # partial update re-snapshots merged state
    assert len(log) == 2
    arrays = log.arrays()
    assert arrays["feedback_joint_left_rad"][1, 0] == 1.5
    assert arrays["feedback_joint_left_rad"][1, 1] == pytest.approx(0.1)
    assert arrays["feedback_t_s"][1] - arrays["feedback_t_s"][0] == pytest.approx(0.005)


def test_fk_palm_series_reproduces_wait_pose():
    """FK at the ready-posture angles must be the configured wait pose."""
    config = load_config(CONFIG_DIR, "replay")
    joints = parse_joints(config.robot.posture.urdf)
    posture = config.robot.posture
    positions, quats = fk_palm_series(
        joints, posture.legwaist_joint_deg,
        [np.deg2rad(posture.left_arm_joint_deg)], [np.deg2rad(posture.right_arm_joint_deg)])
    assert positions.shape == (1, 2, 3) and quats.shape == (1, 2, 4)
    for hand in range(2):
        np.testing.assert_allclose(positions[0, hand],
                                   config.robot.hands[hand].wait_position_m, atol=1e-9)
        np.testing.assert_allclose(quats[0, hand],
                                   config.robot.hands[hand].wait_quat_xyzw, atol=1e-9)


def test_fk_palm_series_moves_with_joints():
    config = load_config(CONFIG_DIR, "replay")
    joints = parse_joints(config.robot.posture.urdf)
    posture = config.robot.posture
    ready = np.deg2rad(np.array(posture.left_arm_joint_deg))
    moved = ready.copy()
    moved[3] += 0.2                            # one elbow-ish joint, radians
    positions, _ = fk_palm_series(joints, posture.legwaist_joint_deg,
                                  [ready, moved],
                                  [np.deg2rad(np.array(posture.right_arm_joint_deg))] * 2)
    assert np.linalg.norm(positions[1, 0] - positions[0, 0]) > 1e-3
    np.testing.assert_allclose(positions[0, 1], positions[1, 1], atol=1e-12)  # right untouched
    with pytest.raises(ValueError, match="mismatch"):
        fk_palm_series(joints, posture.legwaist_joint_deg, [ready], [ready, moved])


def _synthetic_series(lag_s: float, feedback_rate: float = 240.):
    """Command = per-hand sine along x; feedback = the same motion trailing by lag_s."""
    t_command = np.arange(0., 2., 1. / 120.)
    amplitudes = (0.15, 0.10)
    command = np.stack([np.stack(
        [amp * np.sin(2 * np.pi * 1.2 * t_command), -0.6 + 0 * t_command,
         1.1 + 0 * t_command], axis=-1) for amp in amplitudes], axis=1)
    t_feedback = np.arange(0.05, 1.95, 1. / feedback_rate)
    feedback = np.stack([np.stack(
        [amp * np.sin(2 * np.pi * 1.2 * (t_feedback - lag_s)), -0.6 + 0 * t_feedback,
         1.1 + 0 * t_feedback], axis=-1) for amp in amplitudes], axis=1)
    return t_command, command, t_feedback, feedback


def test_tracking_summary_detects_constant_lag():
    lag = 0.04
    t_command, command, t_feedback, feedback = _synthetic_series(lag)
    summary = tracking_summary(t_command, command, t_feedback, feedback)
    for hand, amp in zip(("left", "right"), (0.15, 0.10)):
        stats = summary[hand]
        assert abs(stats["lag_s"] - lag) <= 0.005 + 1e-9
        assert stats["compensated_error_mean_mm"] < stats["error_mean_mm"] / 10
        assert stats["error_mean_mm"] > 5.   # a real lag must be visible uncompensated


def test_tracking_summary_zero_error_zero_lag():
    t_command, command, _, _ = _synthetic_series(0.)
    t_feedback = t_command[::2]               # exactly on command samples
    feedback = command[::2]
    summary = tracking_summary(t_command, command, t_feedback, feedback)
    for hand in ("left", "right"):
        assert summary[hand]["lag_s"] == pytest.approx(0., abs=0.005 + 1e-9)
        assert summary[hand]["error_max_mm"] < 1e-6
        assert summary[hand]["compensated_error_mean_mm"] < 1e-6


def test_tracking_summary_rejects_unusable_inputs():
    t_command, command, t_feedback, feedback = _synthetic_series(0.02)
    with pytest.raises(ValueError, match="at least two"):
        tracking_summary(t_command, command, t_feedback[:1], feedback[:1])
    outside = t_feedback + 100.               # entirely outside the command window
    with pytest.raises(ValueError, match="window"):
        tracking_summary(t_command, command, outside, feedback)


def test_tracking_figure_renders(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from moz1_catch.feedback import tracking_figure

    t_command, command, t_feedback, feedback = _synthetic_series(0.03)
    tracking = tracking_summary(t_command, command, t_feedback, feedback)
    figure = tracking_figure(t_command, np.full(len(t_command), "execute"),
                             command, t_feedback, feedback, tracking,
                             contact_after_execute_s=0.5,
                             title="unit test")
    output = tmp_path / "tracking.png"
    figure.savefig(output, dpi=80)
    import matplotlib.pyplot as plt
    plt.close(figure)
    assert output.stat().st_size > 20_000   # a real rendered figure, not a blank
