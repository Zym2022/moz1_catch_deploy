"""Safety clamps, TCP conversion and the executor command stream."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moz1_catch.calib import as_transform, transform_inverse
from moz1_catch.config import load_config
from moz1_catch.core.geometry import PALM_CENTER_OFFSETS_BODY_M
from moz1_catch.core.one_shot import plan_catch
from moz1_catch.executor import Executor
from moz1_catch.robot.mock_sink import MockCartesianSink
from moz1_catch.robot.sink import HandTargets, palm_state_snapshot, palm_targets_to_tcp
from moz1_catch.safety import SafetyEnvelope
from moz1_catch.trace import TraceRecorder

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_workspace_projection_and_speed_cap():
    config = load_config(CONFIG_DIR, "replay")
    safety = SafetyEnvelope(config.safety)
    inside = np.array(((0.1, -0.7, 1.2), (-0.1, -0.7, 1.2)))
    clamped, violations = safety.clamp(inside, inside, 0.01)
    np.testing.assert_allclose(clamped, inside)
    assert violations == []

    far = np.array(((5.0, -0.7, 1.2), (-0.1, -0.7, 1.2)))
    clamped, violations = safety.clamp(far, inside, 0.01)
    # Both protections apply in sequence: projected into the workspace box,
    # then the 0.4 m step is capped to max speed * dt.
    assert "workspace" in violations
    assert clamped[0, 0] == pytest.approx(
        inside[0, 0] + config.safety.max_cartesian_speed_mps * 0.01)
    assert any(item.startswith("speed") for item in violations)

    # A 0.5 m step in one 10 ms period must be capped by max speed * dt.
    jump = np.array(((0.1, -0.7, 1.2), (-0.1, -0.7, 1.7)))
    clamped, violations = safety.clamp(jump, inside, 0.01)
    step = np.linalg.norm(clamped - inside, axis=1)
    assert np.all(step <= config.safety.max_cartesian_speed_mps * 0.01 + 1e-12)
    assert any(item.startswith("speed") for item in violations)


def test_palm_targets_to_tcp_applies_the_installation_transform():
    palm_rotation = Rotation.from_euler("z", -90, degrees=True)
    targets = HandTargets(
        positions_m=np.array(((0.2, -0.6, 1.2), (-0.2, -0.6, 1.2))),
        rotations=(palm_rotation, palm_rotation))
    # T_tcp_palm is stored as ^palm T_flange (palm -> flange) and applied as-is:
    # flange at palm-local (-0.12, 0.06, 0) with the hand->flange rotation.
    T_palm_flange = as_transform(
        Rotation.from_euler("x", 90, degrees=True).as_matrix(), np.array((-0.12, 0.06, 0.0)))
    tcp = palm_targets_to_tcp(targets, (T_palm_flange, T_palm_flange))
    for side in range(2):
        expected_rotation = palm_rotation * Rotation.from_euler("x", 90, degrees=True)
        np.testing.assert_allclose(tcp.positions_m[side],
                                   targets.positions_m[side]
                                   + palm_rotation.apply((-0.12, 0.06, 0.0)), atol=1e-12)
        np.testing.assert_allclose(tcp.rotations[side].as_quat(),
                                   expected_rotation.as_quat(), atol=1e-12)
    with pytest.raises(ValueError):
        palm_targets_to_tcp(targets, (np.eye(4),))


def test_palm_frame_conversion_matches_hand_frame_composition():
    """Direction regression with a non-trivial rotation.

    The planner's pose is the palm frame (hand link translated by the
    tangent-plane centre offset), so converting that pose through the stored
    ^palm T_flange must equal converting the hand-link pose through the URDF
    flange->hand joint.  A pure-translation constant cannot catch a direction
    error; a random rotation can.
    """
    rng = np.random.default_rng(20260928)
    T_flange_hand = as_transform(
        Rotation.from_rotvec((0.12, -0.34, 1.51)).as_matrix(), (0.02, -0.01, 0.03))
    for side in (0, 1):
        offset = np.asarray(PALM_CENTER_OFFSETS_BODY_M[side])
        T_hand_palm = as_transform(np.eye(3), offset)
        T_palm_flange = transform_inverse(T_hand_palm) @ transform_inverse(T_flange_hand)
        T_world_hand = as_transform(
            Rotation.from_rotvec(rng.normal(size=3) * .4).as_matrix(), rng.uniform(-1., 1., 3))
        T_world_palm = T_world_hand @ T_hand_palm
        targets = HandTargets(
            positions_m=np.array((T_world_palm[:3, 3], T_world_palm[:3, 3])),
            rotations=(Rotation.from_matrix(T_world_palm[:3, :3]),) * 2)
        tcp = palm_targets_to_tcp(targets, (T_palm_flange, T_palm_flange))
        expected = T_world_hand @ transform_inverse(T_flange_hand)
        np.testing.assert_allclose(tcp.positions_m[0], expected[:3, 3], atol=1e-12)
        np.testing.assert_allclose(tcp.rotations[0].as_matrix(), expected[:3, :3], atol=1e-12)


def test_executor_streams_plan_and_finishes():
    config = load_config(CONFIG_DIR, "replay")
    sink = MockCartesianSink()
    trace = TraceRecorder(config)
    executor = Executor(sink, SafetyEnvelope(config.safety), config.execution,
                        config.robot, trace)
    # The mock sink records controller-TCP commands in torso_flange: the base_link
    # wait pose goes through the fixed base->torso and hand->TCP transforms.
    T_tcp_palms = tuple(hand.T_tcp_palm for hand in config.robot.hands)
    T_torso_base = transform_inverse(config.robot.T_base_torso)
    wait = np.array([hand.wait_position_m for hand in config.robot.hands])
    wait_rotations = tuple(Rotation.from_quat(hand.wait_quat_xyzw) for hand in config.robot.hands)
    wait_tcp = palm_targets_to_tcp(HandTargets(wait, wait_rotations), T_tcp_palms,
                                   T_torso_base).positions_m
    executor.tick(0.0)
    assert sink.count == 1
    np.testing.assert_allclose(sink.sent_positions[0], wait_tcp, atol=1e-9)

    settings = config.catch_settings
    palms, rotations = palm_state_snapshot(wait, [hand.wait_quat_xyzw for hand in config.robot.hands])
    plan = plan_catch(np.array((0., -1.30, 1.35)), np.array((0.05, 1.7, 1.1)),
                      Rotation.identity(), np.zeros(3), palms, rotations,
                      np.array(((0, -1, 0), (0, 1, 0))), settings)
    exec_start = 10.0
    executor.arm_plan(plan, exec_start)
    # Before the execution start the palms keep holding the wait pose.
    executor.tick(exec_start - 0.05)
    np.testing.assert_allclose(sink.sent_positions[-1], wait_tcp, atol=1e-9)
    # The first stream sample is exactly the held pose (continuous start).
    executor.tick(exec_start + 1e-6)
    np.testing.assert_allclose(sink.sent_positions[-1], wait_tcp, atol=1e-9)
    executor.tick(exec_start + plan.contact_time * 0.5)
    executor.tick(exec_start + plan.contact_time + plan.stop_time
                  + config.execution.settle_margin_s + 0.01)
    assert executor.phase == "done"
    assert sink.count > 3
    # Clamped base_link palm targets stay inside the workspace; the recorded
    # sink commands are torso_flange TCP poses, a frame the box is not defined in.
    palm_targets = np.asarray(trace.commands["palm_position"]).reshape(-1, 2, 3)
    assert np.all(palm_targets >= config.safety.workspace[:, 0] - 1e-9)
    assert np.all(palm_targets <= config.safety.workspace[:, 1] + 1e-9)


def test_executor_reject_blends_back_to_wait():
    config = load_config(CONFIG_DIR, "replay")
    sink = MockCartesianSink()
    trace = TraceRecorder(config)
    executor = Executor(sink, SafetyEnvelope(config.safety), config.execution,
                        config.robot, trace)
    T_torso_base = transform_inverse(config.robot.T_base_torso)
    wait = np.array([hand.wait_position_m for hand in config.robot.hands])
    wait_rotations = tuple(Rotation.from_quat(hand.wait_quat_xyzw) for hand in config.robot.hands)
    wait_tcp = palm_targets_to_tcp(HandTargets(wait, wait_rotations),
                                   tuple(hand.T_tcp_palm for hand in config.robot.hands),
                                   T_torso_base).positions_m
    executor.tick(0.0)
    executor.enter_reject(1.0)
    executor.tick(1.0 + config.execution.reject_duration_s * 0.5)
    executor.tick(1.0 + config.execution.reject_duration_s + 1e-6)
    assert executor.phase == "done"
    np.testing.assert_allclose(sink.sent_positions[-1], wait_tcp, atol=1e-9)
