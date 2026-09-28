"""Causal observation windows, forward crossings and one-shot delay consistency.

Ported from MozBoxer test_catching_prediction.py (branch feat/moz1-swept-geometry-catch,
working tree of 2026-09-28); only the imports changed.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moz1_catch.core import one_shot
from moz1_catch.core.geometry import COATING_SPHERE_CENTERS_BODY_M as CENTERS_M
from moz1_catch.core.geometry import PALM_CENTER_OFFSETS_BODY_M
from moz1_catch.core.one_shot import plan_catch
from moz1_catch.core.prediction import BoxFlight, PredictionSettings, estimate_box_flight


def test_time_window_uses_more_than_nine_poses_and_rejects_future_or_stale_data():
    acceleration = np.array((.1, -.55, -8.7))
    flight = BoxFlight(np.array((.02, -1.5, 1.2)), np.array((.1, 1.8, 2.4)),
                      Rotation.from_euler("xyz", (10, -5, 8), degrees=True),
                      np.array((.1, -.2, .3)), acceleration)
    settings = PredictionSettings(window_s=.10, acceleration_prior_mps2=tuple(acceleration))
    for rate in (120, 200):
        times = np.arange(int(.3*rate)+1)/rate
        poses = np.column_stack((flight.positions(times), flight.rotations(times).as_quat()))
        now = times[-1]+.002
        estimate = estimate_box_flight(times, poses, now, settings)
        expected = flight.at(now)
        np.testing.assert_allclose(estimate.position_m, expected.position_m, atol=1e-12)
        np.testing.assert_allclose(estimate.velocity_mps, expected.velocity_mps, atol=1e-12)
        np.testing.assert_allclose(estimate.acceleration_mps2, acceleration, atol=1e-10)
        np.testing.assert_allclose(estimate.angular_velocity_radps, flight.angular_velocity_radps, atol=1e-12)
        assert (estimate.rotation*expected.rotation.inv()).magnitude() < 1e-12
        ancient = poses.copy(); ancient[times < now-settings.window_s, :3] += 1.
        np.testing.assert_allclose(estimate_box_flight(times, ancient, now, settings).position_m,
                                   expected.position_m, atol=1e-12)
        changed = poses.copy()
        index = np.flatnonzero(times >= now-settings.window_s)[0]
        assert index < len(times)-9
        changed[index, 0] += .01
        assert abs(estimate_box_flight(times, changed, now, settings).position_m[0]-expected.position_m[0]) > 1e-6
        with pytest.raises(ValueError, match="stale"):
            estimate_box_flight(times, poses, times[-2], settings)
        with pytest.raises(ValueError, match="stale"):
            estimate_box_flight(times, poses, now+.04, settings)


def test_nonzero_acceleration_is_shared_by_crossing_geometry_and_execution_delay(monkeypatch):
    flight = BoxFlight(np.array((.002, -1.1, 1.4)), np.array((.03, 1.8, .37)),
                      Rotation.identity(), np.array((0., 0., .1)), np.array((.08, -.6, -8.7)))
    crossing = flight.crossing_time(-.65)
    assert flight.positions(crossing)[1] == pytest.approx(-.65)
    assert flight.at(crossing).velocity_mps[1] > 0
    for ay in (0., 1e-12, -1e-12):
        linear = replace(flight, acceleration_mps2=np.array((0., ay, -9.81)))
        assert linear.crossing_time(-.65) == pytest.approx(.45/1.8, abs=1e-12)
    with pytest.raises(ValueError, match="does not reach"):
        replace(flight, acceleration_mps2=np.array((0., -10., -9.81))).crossing_time(-.65)

    palms = np.array(((.19169, -.62308, 1.1937), (-.19171, -.62064, 1.20004)))
    rotations = (Rotation.from_euler("z", -90, degrees=True),)*2
    normals = np.array(((0, -1, 0), (0, 1, 0)))
    state = (flight.position_m, flight.velocity_mps, flight.rotation, flight.angular_velocity_radps,
             palms, rotations, normals)
    immediate = one_shot.plan_catch(*state, box_acceleration=flight.acceleration_mps2)
    delayed = one_shot.plan_catch(*state, box_acceleration=flight.acceleration_mps2, execution_delay_s=.025)
    assert immediate.contact_time == pytest.approx(crossing)
    assert delayed.contact_time+delayed.execution_delay_s == pytest.approx(crossing)
    np.testing.assert_allclose(delayed.contact_positions, immediate.contact_positions, atol=1e-12)

    scores = []
    original_score = one_shot._geometry_score
    def capture(plan, spheres, sampled, palm_rotations=None):
        scores.append(sampled)
        return original_score(plan, spheres, sampled, palm_rotations)
    monkeypatch.setattr(one_shot, "_geometry_score", capture)
    clock = iter((0., .0123, .0134, .0134, .0137))
    monkeypatch.setattr(one_shot.time, "perf_counter", lambda: next(clock))
    spheres = np.asarray(CENTERS_M)-np.asarray(PALM_CENTER_OFFSETS_BODY_M)[:, None, :]
    ready = one_shot.plan_catch(*state, box_acceleration=flight.acceleration_mps2,
        execution_delay_s=.025, control_dt_s=.001, palm_sphere_offsets_body=spheres)
    times, centers, matrices = scores[-1]
    assert ready.execution_delay_s == pytest.approx(.039)
    assert ready.contact_time+ready.execution_delay_s == pytest.approx(
        flight.crossing_time(ready.settings.plane_y))
    np.testing.assert_allclose(centers, flight.positions(times+ready.execution_delay_s), atol=1e-12)
    np.testing.assert_allclose(matrices, flight.rotations(times+ready.execution_delay_s).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(ready.target(0.)[0], palms, atol=1e-12)
