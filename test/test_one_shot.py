"""One-shot timing, side contact and bounded smooth retreat contracts.

Ported from MozBoxer test_catching_one_shot.py (working tree of 2026-09-28); only the
imports changed.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from moz1_catch.core import one_shot

from moz1_catch.core.geometry import (
    COATING_SPHERE_CENTERS_BODY_M as CENTERS_M, PALM_CENTER_OFFSETS_BODY_M,
)
from moz1_catch.core.one_shot import (
    CatchSettings, _geometry_samples, _geometry_score, estimate_box_state, plan_catch,
)
from moz1_catch.core.prediction import PredictionSettings


def test_pose_only_ballistic_estimate_and_stale_rejection():
    times = np.arange(0.0, 0.049, 0.008)
    now = 0.05
    origin = np.array((0.002, -1.10, 0.82227))
    velocity = np.array((0.01, 1.0, 3.4335))
    spin = np.array((0.0, 0.0, 0.1))
    positions = origin + times[:, None] * velocity + 0.5 * times[:, None]**2 * np.array((0, 0, -9.81))
    quaternions = Rotation.from_rotvec(times[:, None] * spin).as_quat()
    estimated_position, estimated_velocity, estimated_rotation, estimated_spin = estimate_box_state(
        times, np.column_stack((positions, quaternions)), now)
    np.testing.assert_allclose(estimated_position,
                               origin + now * velocity + 0.5 * now**2 * np.array((0, 0, -9.81)), atol=1e-12)
    np.testing.assert_allclose(estimated_velocity, velocity + now * np.array((0, 0, -9.81)), atol=1e-12)
    np.testing.assert_allclose(estimated_spin, spin, atol=1e-12)
    assert (estimated_rotation * Rotation.from_rotvec(now * spin).inv()).magnitude() < 1e-12
    with pytest.raises(ValueError, match="stale"):
        estimate_box_state(times, np.column_stack((positions, quaternions)), now + 0.1)

    long_times = np.arange(16) / 120
    long_positions = origin + long_times[:, None] * velocity + 0.5 * long_times[:, None]**2 * np.array((0, 0, -9.81))
    long_positions[:4] += np.array((0.01, -0.01, 0.01))  # Simulate release-transient pose bias.
    long_quaternions = Rotation.from_rotvec(long_times[:, None] * spin).as_quat()
    final_position, final_velocity, _, _ = estimate_box_state(
        long_times, np.column_stack((long_positions, long_quaternions)), long_times[-1],
        prediction_settings=PredictionSettings(window_s=.08))
    np.testing.assert_allclose(final_position,
                               origin + long_times[-1] * velocity
                               + 0.5 * long_times[-1]**2 * np.array((0, 0, -9.81)), atol=1e-12)
    np.testing.assert_allclose(final_velocity,
                               velocity + long_times[-1] * np.array((0, 0, -9.81)), atol=1e-12)


def test_variable_release_and_smooth_retreat():
    settings = CatchSettings()
    palms = np.array(((0.23, -0.73, settings.center_z), (-0.23, -0.73, settings.center_z)))
    rotations = (Rotation.from_euler("z", -90, degrees=True),) * 2
    normals = np.array(((0, -1, 0), (0, 1, 0)))
    for lateral, speed in ((0.0, 1.0), (0.005, 1.05), (-0.005, 0.95)):
        contact_time = 0.35
        box_z = settings.center_z - 2.97365 * contact_time + 4.905 * contact_time**2
        plan = plan_catch(
            np.array((lateral, settings.plane_y - contact_time * speed, box_z)),
            np.array((0.0, speed, 2.97365)), Rotation.identity(), np.zeros(3),
            palms, rotations, normals, settings,
        )
        normal_contact = plan.target(plan.contact_time - settings.normal_lead)
        during_follow = plan.target(plan.contact_time + .10)
        at_stop = plan.target(plan.contact_time + plan.stop_time)
        np.testing.assert_allclose(
            np.sum((normal_contact[0] - plan.contact_positions) * plan.contact_normals, axis=1),
            0.0, atol=1e-9)
        assert 0 < np.linalg.norm(during_follow[2].mean(axis=0)) < np.linalg.norm(plan.retreat_velocity)
        np.testing.assert_allclose(at_stop[0], plan.stop_positions, atol=1e-9)
        assert np.linalg.norm(plan.tangent_velocity) < speed
        assert plan.relative_tangent_speed > 0
        assert plan.tangent_velocity[2] < 0
        displacement = (plan.stop_positions-plan.contact_positions).mean(axis=0)
        assert abs(displacement[0]) <= settings.max_retreat_lateral+1e-9
        assert displacement[1] <= settings.max_retreat_depth+1e-9
        assert plan.stop_positions[:, 1].mean() <= settings.max_retreat_center_y+1e-9
        if lateral:
            side = 1 if lateral > 0 else 0
            face_x = lateral + (settings.box_half_extents[0] if side == 0 else -settings.box_half_extents[0])
            assert abs(plan.contact_positions[side, 0] - face_x) == pytest.approx(
                settings.lateral_contact_bias_gain * abs(lateral))
        assert plan.settings.max_palm_speed == settings.max_palm_speed
    far_plan = plan_catch(np.array((0, -1.50, settings.center_z)), np.array((0, 1.7, 2.4525)),
                          Rotation.identity(), np.zeros(3), palms, rotations, normals, settings)
    assert far_plan.relative_tangent_speed > 2.0
    np.testing.assert_allclose(far_plan.target(0)[0], palms, atol=1e-12)
    for boundary in (far_plan.contact_time - settings.normal_lead - settings.close_duration,
                     far_plan.contact_time - settings.normal_lead,
                     far_plan.contact_time - settings.tangent_lead,
                     far_plan.contact_time,
                     *(far_plan.contact_time+far_plan.retreat_durations)):
        before, _, before_speed, _ = far_plan.target(boundary - 1e-7)
        after, _, after_speed, _ = far_plan.target(boundary + 1e-7)
        np.testing.assert_allclose(before, after, atol=2e-6)
        np.testing.assert_allclose(before_speed, after_speed, atol=2e-4)
    with pytest.raises(ValueError, match="misses the reachable attempt region"):
        plan_catch(np.array((0.19, settings.plane_y - 0.35, box_z)), np.array((0, 1, 2.97365)),
                   Rotation.identity(), np.zeros(3), palms, rotations, normals, settings)
    faster = plan_catch(np.array((0.11, settings.plane_y - 0.35, box_z)),
                        np.array((0, 1, 2.97365)), Rotation.identity(), np.zeros(3),
                        palms, rotations, normals, settings)
    assert faster.settings.max_palm_speed == settings.fallback_max_palm_speed
    strict = CatchSettings(fallback_max_palm_speed=settings.max_palm_speed,
                           fallback_max_palm_acceleration=settings.max_palm_acceleration)
    with pytest.raises(ValueError, match="speed or acceleration budget"):
        plan_catch(np.array((0.11, settings.plane_y - 0.35, box_z)),
                   np.array((0, 1, 2.97365)), Rotation.identity(), np.zeros(3),
                   palms, rotations, normals, strict)
    plan_catch(np.array((0.101, settings.plane_y - 0.35, box_z)), np.array((-0.29, 1, 2.97365)),
               Rotation.identity(), np.zeros(3), palms, rotations, normals, settings)
    plan_catch(np.array((0, settings.plane_y - 0.35, box_z)), np.array((0, 1, 2.97365)),
               Rotation.from_euler("x", 30, degrees=True), np.zeros(3),
               palms, rotations, normals, settings)
    with pytest.raises(ValueError, match="orientation"):
        plan_catch(np.array((0, settings.plane_y - 0.35, box_z)), np.array((0, 1, 2.97365)),
                   Rotation.from_euler("y", 40, degrees=True), np.zeros(3),
                   palms, rotations, normals, settings)
    for vertical_speed in (2.5, 2.6):
        with pytest.raises(ValueError, match="descending"):
            plan_catch(np.array((0, -0.9, 0.88168)), np.array((0, 1, vertical_speed)),
                       Rotation.identity(), np.zeros(3), palms, rotations, normals, settings)


def test_retreat_keeps_squeeze_and_initial_contact_window_for_a_skew_throw():
    palms = np.array(((.19169, -.62308, 1.1937), (-.19171, -.62064, 1.20004)))
    rotations = (Rotation.from_euler("z", -90, degrees=True),) * 2
    plan = plan_catch(np.array((-.12, -1.35, 1.40)), np.array((.45, 1.75, 1.4)),
                      Rotation.from_euler("xyz", (10, 5, 15), degrees=True), np.zeros(3),
                      palms, rotations, np.array(((0, -1, 0), (0, 1, 0))))
    cfg = plan.settings
    for elapsed in (0., .02, .04, .1, .2, plan.stop_time):
        t = plan.contact_time+elapsed
        positions, _, velocities, _ = plan.target(t)
        travel, rate = plan.retreat(elapsed)
        np.testing.assert_allclose(positions.mean(axis=0), plan.contact_positions.mean(axis=0)+travel)
        np.testing.assert_allclose(velocities.mean(axis=0), rate, atol=1e-12)
        squeeze_time = 2*cfg.grip_compression/cfg.normal_speed
        s = min(elapsed+cfg.normal_lead, squeeze_time)
        expected_gap = -cfg.normal_speed*(s-s*s/(2*squeeze_time))
        np.testing.assert_allclose(np.sum((positions-plan.contact_positions-travel)
                                         * plan.contact_normals, axis=1), expected_gap, atol=1e-12)
    travel, rate = plan.retreat(.04)
    # Early braking preserves the old common path closely, without a follow plateau.
    assert np.linalg.norm(travel-.04*plan.retreat_velocity) < .001
    assert np.linalg.norm(rate) < np.linalg.norm(plan.retreat_velocity)
    np.testing.assert_allclose(plan.target(plan.contact_time+plan.stop_time)[0], plan.stop_positions)
    np.testing.assert_allclose(plan.target(plan.contact_time+plan.stop_time)[2], 0., atol=1e-12)


def test_geometry_search_tracks_lateral_box_motion_and_matches_executed_targets():
    palms = np.array(((0.19169, -0.62308, 1.1937), (-0.19171, -0.62064, 1.20004)))
    rotations = (Rotation.from_euler("z", -90, degrees=True),) * 2
    normals = np.array(((0, -1, 0), (0, 1, 0)))
    spheres = np.asarray(CENTERS_M) - np.asarray(PALM_CENTER_OFFSETS_BODY_M)[:, None, :]
    plan = plan_catch(np.array((0., -1.35, 1.4052)), np.array((0.15, 1.75, 1.574)),
                      Rotation.from_euler("z", 5, degrees=True), np.zeros(3),
                      palms, rotations, normals, palm_sphere_offsets_body=spheres)
    assert plan.lateral_velocity[0] == pytest.approx(0.15)
    np.testing.assert_allclose(plan.target(0.)[0], palms, atol=1e-12)
    np.testing.assert_allclose(plan.target(0.)[2], 0., atol=1e-12)
    times = np.linspace(0., plan.contact_time + plan.stop_time + .05, 41)
    sampled_positions, sampled_rotations = _geometry_samples(plan, times)
    for index, t in enumerate(times):
        positions, orientations, _, _ = plan.target(float(t))
        np.testing.assert_allclose(sampled_positions[index], positions, atol=1e-10)
        for side in range(2):
            np.testing.assert_allclose(sampled_rotations[index, side], orientations[side].as_matrix(), atol=1e-10)
    epsilon = 1e-5
    probes = np.maximum(np.r_[times-epsilon, times+epsilon], 0.)
    positions, _ = _geometry_samples(plan, probes)
    finite_velocities = (positions[len(times):]-positions[:len(times)]) / (2*epsilon)
    np.testing.assert_allclose(finite_velocities, [plan.target(t)[2] for t in times], atol=2e-5)


def test_execution_start_preserves_world_contact_time_and_scores_wrong_face():
    settings = CatchSettings()
    palms = np.array(((.23, -.73, 1.2), (-.23, -.73, 1.2)))
    rotations = (Rotation.from_euler("z", -90, degrees=True),) * 2
    normals = np.array(((0, -1, 0), (0, 1, 0)))
    state = (np.array((0., -1.50, 1.2)), np.array((0., 1.7, 2.4525)),
             Rotation.identity(), np.array((.04, -.03, .1)), palms, rotations, normals)
    immediate = plan_catch(*state)
    delayed = plan_catch(*state, execution_delay_s=.04)
    assert delayed.contact_time + delayed.execution_delay_s == pytest.approx(immediate.contact_time)
    np.testing.assert_allclose(delayed.contact_positions, immediate.contact_positions, atol=1e-12)
    np.testing.assert_allclose(delayed.target(0.)[0], palms, atol=1e-12)
    assert delayed.target(0.)[1][0].as_quat() == pytest.approx(rotations[0].as_quat())
    with pytest.raises(ValueError, match="execution delay"):
        plan_catch(*state, execution_delay_s=-.01)

    # Same instant, same box, but a sphere at a side face versus at the front face.
    half = settings.box_half_extents
    gap = settings.palm_sphere_radius + .003
    good = replace(immediate, contact_positions=np.array(((half[0] + gap, 0., 0.),
                                                         (-half[0] - gap, 0., 0.))))
    wrong = replace(immediate, contact_positions=np.array(((0., half[1] + gap, 0.),
                                                          (0., half[1] + gap, 0.))))
    times = np.array((immediate.contact_time, immediate.contact_time + .005))
    flight = (times, np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)))
    spheres = np.zeros((2, 12, 3))
    good_score, _, _, good_cosines = _geometry_score(good, spheres, flight)
    wrong_score, _, _, wrong_cosines = _geometry_score(wrong, spheres, flight)
    np.testing.assert_allclose(good_cosines, (1., 1.))
    np.testing.assert_allclose(wrong_cosines, (0., 0.))
    assert wrong_score > good_score + .25

    # This moving throw previously returned an unscored fixed fallback with score zero.
    fixture_palms = np.array(((.19169727, -.62306375, 1.19370829),
                             (-.19171417, -.62062025, 1.20005587)))
    fixture_rotations = tuple(Rotation.from_quat((
        (-.13962998, -.10416850, -.67948160, .71271113),
        (.14528112, .12739464, .71503513, -.67185472))))
    fixed = plan_catch(np.array((.13314766, -1.34825134, 1.40700245)),
                       np.array((-.48189086, 1.89688075, 1.38113809)),
                       Rotation.from_quat((.22558701, .02236572, -.09057588, .96974547)),
                       np.array((-.09528373, .36749887, .43188146)),
                       fixture_palms, fixture_rotations, normals,
                       palm_sphere_offsets_body=np.asarray(CENTERS_M)
                       - np.asarray(PALM_CENTER_OFFSETS_BODY_M)[:, None, :])
    assert fixed.geometry_mode == "fixed"
    assert fixed.geometry_score > .1
    assert np.isfinite(fixed.predicted_touch_times_s).all()
    assert np.isfinite(fixed.predicted_touch_normal_cosines).all()


def test_measured_launch_has_no_reserved_wait_and_allows_early_side_contact(monkeypatch):
    palms = np.array(((.23, -.73, 1.2), (-.23, -.73, 1.2)))
    rotations = (Rotation.from_euler("z", -90, degrees=True),) * 2
    normals = np.array(((0, -1, 0), (0, 1, 0)))
    state = (np.array((0., -1.50, 1.2)), np.array((.1, 1.7, 2.4525)),
             Rotation.identity(), np.zeros(3), palms, rotations, normals)
    immediate = plan_catch(*state)
    # Finalization crosses the first available tick and must be included as well.
    clock = iter((0., .0123, .0134, .0134, .0137))
    monkeypatch.setattr(one_shot.time, "perf_counter", lambda: next(clock))
    ready = plan_catch(*state, control_dt_s=.001)
    assert ready.execution_delay_s == pytest.approx(.014)
    assert ready.contact_time + ready.execution_delay_s == pytest.approx(immediate.contact_time)
    np.testing.assert_allclose(ready.target(0.)[0], palms, atol=1e-12)
    np.testing.assert_allclose(ready.target(0.)[2], 0., atol=1e-12)

    half = np.asarray(ready.settings.box_half_extents)
    gap = ready.settings.palm_sphere_radius + .003
    side_positions = np.array(((half[0]+gap, 0., 0.), (-half[0]-gap, 0., 0.)))
    matrices = np.array([rotation.as_matrix() for rotation in rotations])
    monkeypatch.setattr(one_shot, "_geometry_samples", lambda plan, times, rotations=None:
                        (np.tile(side_positions, (len(times), 1, 1)), np.tile(matrices, (len(times), 1, 1, 1))))
    def score(offset):
        times = ready.contact_time + offset + np.array((0., .005))
        return _geometry_score(ready, np.zeros((2, 12, 3)),
                               (times, np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1))))[0]
    assert score(-.08) == pytest.approx(score(0.))


def test_retiming_retains_fast_budget_across_execution_ticks(monkeypatch):
    palms = np.array(((.23, -.73, 1.2), (-.23, -.73, 1.2)))
    rotations = (Rotation.from_euler("z", -90, degrees=True),) * 2
    normals = np.array(((0, -1, 0), (0, 1, 0)))
    state = (np.array((0., -1.50, 1.2)), np.array((.1, 1.7, 2.4525)),
             Rotation.identity(), np.zeros(3), palms, rotations, normals)
    immediate = plan_catch(*state)
    check_speed = one_shot._check_plan_speed
    preferred_rechecks = []

    def needs_fast_budget(plan, *, check_retreat=True):
        if not check_retreat and plan.settings.max_palm_speed == 1.:
            preferred_rechecks.append(plan.contact_time)
            raise ValueError("palm trajectory exceeds speed or acceleration budget")
        check_speed(plan, check_retreat=check_retreat)

    clock = iter((0., .0123, .0134, .0134, .0137))
    monkeypatch.setattr(one_shot.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(one_shot, "_check_plan_speed", needs_fast_budget)
    ready = plan_catch(*state, control_dt_s=.001)
    assert len(preferred_rechecks) == 1
    assert ready.settings.max_palm_speed == ready.settings.fallback_max_palm_speed
    assert ready.execution_delay_s == pytest.approx(.014)
    assert ready.contact_time + ready.execution_delay_s == pytest.approx(immediate.contact_time)
