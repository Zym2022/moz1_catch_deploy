"""SimTracePlan: loading a frozen simulation trace as an Executor-ready plan."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moz1_catch.sim_plan import load_sim_trace_plan

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data" / "sim_plans" / "final_nominal_120hz_200ms.npz"


def test_load_real_trace():
    plan, info = load_sim_trace_plan(TRACE)
    assert info["decision"] == "accept"
    assert info["trace_dt_s"] == pytest.approx(0.001, abs=1e-4)
    assert info["trace_samples"] > 500
    assert 0. < plan.contact_time < 1.
    assert 0. < plan.stop_time < 1.
    assert 0.1 < info["max_palm_speed_mps"] < 3.5


def test_plan_segment_detection():
    """The loaded plan must start at the recorded plan-start pose."""
    plan, info = load_sim_trace_plan(TRACE)
    assert info["plan_start_offset_m"] < 0.001
    positions, _, _, _ = plan.target(0.)
    with np.load(TRACE) as data:
        assert np.allclose(positions, data["start_palm_positions_m"], atol=1e-9)
    # this artifact is observation-relative: 8 ms latency hold, plan starts at sample 7
    assert info["plan_start_index"] == 7


def test_episode_absolute_convention(tmp_path):
    """Prepending the pre-observation hold (episode-absolute) is detected too."""
    hold_samples = 300  # observation at 0.3 s; the file then holds 8 ms more before the plan
    with np.load(TRACE) as data:
        hold = np.repeat(data["target_palm_position"][:1], hold_samples, axis=0)
        positions = np.concatenate([hold, data["target_palm_position"]])
        rotations = np.concatenate(
            [np.repeat(data["target_palm_rotation_xyzw"][:1], hold_samples, axis=0),
             data["target_palm_rotation_xyzw"]])
        velocities = np.concatenate(
            [np.zeros((hold_samples, 2, 3)), data["target_palm_velocity"]])
        time_s = 0.001 + np.arange(len(positions)) * 0.001
        payload = {key: data[key] for key in data.files}
    payload.update(time_s=time_s, target_palm_position=positions,
                   target_palm_rotation_xyzw=rotations,
                   target_palm_velocity=velocities)
    path = tmp_path / "episode.npz"
    np.savez(path, **payload)
    plan, info = load_sim_trace_plan(path)
    assert info["plan_start_index"] == hold_samples + 7  # plan zero lands 8 ms into the file
    reference, _ = load_sim_trace_plan(TRACE)
    got, _, _, _ = plan.target(0.05)
    want, _, _, _ = reference.target(0.05)
    assert np.allclose(got, want, atol=1e-9)


def test_target_hits_recorded_samples():
    with np.load(TRACE) as data:
        time_s = data["time_s"]
        positions = data["target_palm_position"]
        rotations = data["target_palm_rotation_xyzw"]
    plan, info = load_sim_trace_plan(TRACE)
    start_index = info["plan_start_index"]
    for offset_samples in (0, 100, 200):  # node-aligned plan times
        got_pos, got_rot, got_vel, fourth = plan.target(offset_samples * 0.001)
        index = start_index + offset_samples
        assert fourth is None
        assert np.allclose(got_pos, positions[index], atol=1e-9)
        expected = Rotation.from_quat(rotations[index])
        for hand in range(2):
            assert (got_rot[hand] * expected[hand].inv()).magnitude() < 1e-9
        assert got_vel.shape == (2, 3)


def test_speed_scale_stretches_time():
    plan, _ = load_sim_trace_plan(TRACE, speed_scale=0.5)
    plan_full, _ = load_sim_trace_plan(TRACE)
    assert plan.contact_time == pytest.approx(2 * plan_full.contact_time)
    assert (plan.contact_time + plan.stop_time) == pytest.approx(
        2 * (plan_full.contact_time + plan_full.stop_time))
    slow_pos, slow_rot, slow_vel, _ = plan.target(2 * 0.05)
    full_pos, full_rot, _, _ = plan_full.target(0.05)
    assert np.allclose(slow_pos, full_pos, atol=1e-9)
    for hand in range(2):
        assert (slow_rot[hand] * full_rot[hand].inv()).magnitude() < 1e-9


def _copy_trace(tmp_path, **overrides):
    path = tmp_path / "trace.npz"
    with np.load(TRACE) as data:
        payload = {key: data[key] for key in data.files if key not in overrides}
        np.savez(path, **payload, **overrides)
    return path


def test_rejects_non_accept_decision(tmp_path):
    with pytest.raises(ValueError, match="accept"):
        load_sim_trace_plan(_copy_trace(tmp_path, catch_decision=np.asarray("reject")))


def test_rejects_bad_shapes(tmp_path):
    with np.load(TRACE) as data:
        broken = data["target_palm_position"][:, :1]
    with pytest.raises(ValueError, match="target arrays"):
        load_sim_trace_plan(_copy_trace(tmp_path, target_palm_position=broken))


def test_clamps_beyond_end():
    plan, _ = load_sim_trace_plan(TRACE)
    end = plan.contact_time + plan.stop_time
    end_pos, _, _, _ = plan.target(end)
    far_pos, _, _, _ = plan.target(end + 5.)
    assert np.allclose(end_pos, far_pos, atol=1e-12)
