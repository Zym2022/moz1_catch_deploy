"""One-shot box catch targets with a small palm/box geometry search."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.core.geometry import BOX_DIMENSIONS_M
from moz1_catch.core.prediction import BoxFlight, PredictionSettings, estimate_box_flight


@dataclass(frozen=True)
class CatchSettings:
    # Simulation starting values. Measure the hardware limits before deployment.
    plane_y: float = -0.65
    center_z: float = 1.20
    # Broad attempt region; trajectory and joint limits still reject unreachable motions.
    max_lateral_error: float = 0.18
    max_height_error: float = 0.16
    # Side-face geometry follows local X, so roll about X is checked separately.
    max_roll_rad: float = np.deg2rad(45.0)
    max_tilt_rad: float = np.deg2rad(35.0)
    max_roll_speed: float = 1.5
    max_spin_radps: float = 0.8
    max_face_normal_speed: float = 3.0
    max_palm_speed: float = 1.0
    max_palm_acceleration: float = 16.0
    # Only used when the preferred trajectory budget rejects the attempt.
    fallback_max_palm_speed: float = 2.0
    fallback_max_palm_acceleration: float = 32.0
    tangent_speed_cap: float = 0.6
    contact_depth_offset: float = 0.06
    contact_height_offset: float = -0.05
    normal_speed: float = 0.16
    precontact_gap: float = 0.035
    close_duration: float = 0.120
    normal_lead: float = 0.040
    tangent_lead: float = 0.055
    grip_compression: float = 0.010
    # Two-phase retreat: a short lateral/depth cushion plus a long vertical glide
    # that keeps supporting the box, then a lateral-only settle to a carry pose.
    cushion_distance: float = 0.08
    cushion_distance_z: float = 0.20
    settle_duration: float = 0.30
    settle_center_x: float = 0.12
    settle_depth: float = 0.0
    settle_drop: float = 0.0
    min_retreat_center_z: float = 0.85
    max_retreat_lateral: float = 0.08
    max_retreat_depth: float = 0.14
    max_retreat_center_x: float = 0.24
    max_retreat_center_y: float = -0.40
    common_motion_relax_duration: float = 0.080
    lateral_contact_bias_gain: float = 0.0
    box_half_extents: tuple[float, float, float] = tuple(length/2 for length in BOX_DIMENSIONS_M)
    palm_sphere_radius: float = 0.030


# Chosen from paired late-commit Isaac tests with the 301×301×305 mm, 510 g box.
LATE_COMMIT_SETTINGS = CatchSettings(
    plane_y=-0.58, max_height_error=0.28,
    max_palm_speed=3.0, max_palm_acceleration=48.0,
    close_duration=0.08,
    max_retreat_depth=0.10, max_retreat_center_y=-0.45,
    min_retreat_center_z=0.75,
)


def estimate_box_state(times_s: np.ndarray, poses_xyzw: np.ndarray, now_s: float,
                       max_age_s: float = 0.03, *,
                       prediction_settings: PredictionSettings = PredictionSettings()
                       ) -> tuple[np.ndarray, np.ndarray, Rotation, np.ndarray]:
    """Compatibility state view; estimate_box_flight also retains fitted acceleration."""
    flight = estimate_box_flight(times_s, poses_xyzw, now_s, prediction_settings, max_age_s)
    return flight.position_m, flight.velocity_mps, flight.rotation, flight.angular_velocity_radps


def _quintic(start: np.ndarray, end: np.ndarray, end_velocity: np.ndarray, duration: float) -> np.ndarray:
    """Position polynomial with zero start velocity and endpoint accelerations."""
    coefficients = np.zeros((2, 3, 6))
    coefficients[:, :, 0] = start
    matrix = np.array(((duration**3, duration**4, duration**5),
                       (3 * duration**2, 4 * duration**3, 5 * duration**4),
                       (6 * duration, 12 * duration**2, 20 * duration**3)))
    rhs = np.stack((end - start, end_velocity, np.zeros_like(start)), axis=-1)
    coefficients[:, :, 3:] = np.linalg.solve(matrix, rhs.reshape(-1, 3).T).T.reshape(2, 3, 3)
    return coefficients


@dataclass(frozen=True)
class CatchPlan:
    contact_time: float
    contact_positions: np.ndarray
    contact_normals: np.ndarray
    contact_rotations: tuple[Rotation, Rotation]
    start_rotations: tuple[Rotation, Rotation]
    tangent_velocity: np.ndarray
    lateral_velocity: np.ndarray
    relative_tangent_speed: float
    stop_positions: np.ndarray
    stop_time: float
    settings: CatchSettings
    early_coefficients: np.ndarray
    rotation_vectors: np.ndarray
    start_gaps: np.ndarray
    retreat_velocity: np.ndarray
    retreat_durations: np.ndarray
    cushion_displacement: np.ndarray
    settle_displacement: np.ndarray
    settle_start: float
    geometry_score: float = 0.0
    execution_delay_s: float = 0.0
    geometry_mode: str = "fixed"
    predicted_touch_times_s: tuple[float, float] = (float("nan"), float("nan"))
    predicted_touch_normal_cosines: tuple[float, float] = (float("nan"), float("nan"))

    def retreat(self, elapsed: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Cushion along the impact direction, then settle from rest to a safe carry pose."""
        elapsed = np.asarray(elapsed)
        s = np.clip(elapsed[..., None] / self.retreat_durations, 0., 1.)
        travel = self.retreat_velocity * self.retreat_durations * (s - s**3 + .5*s**4)
        velocity = self.retreat_velocity * (1. - 3*s**2 + 2*s**3)
        u = np.clip((elapsed[..., None] - self.settle_start) / self.settings.settle_duration, 0., 1.)
        blend = 10*u**3 - 15*u**4 + 6*u**5
        rate = (30*u**2 - 60*u**3 + 30*u**4) / self.settings.settle_duration
        return travel + self.settle_displacement*blend, velocity + self.settle_displacement*rate

    def target(self, time_s: float) -> tuple[np.ndarray, tuple[Rotation, Rotation], np.ndarray, np.ndarray]:
        """Return palm targets at time relative to the scheduled execution start."""
        cfg = self.settings
        tangent_start = self.contact_time - cfg.tangent_lead
        normal_end = self.contact_time - cfg.normal_lead
        normal_start = normal_end - cfg.close_duration
        angular_velocity = np.zeros((2, 3))
        if time_s < tangent_start:
            t = max(0.0, time_s)
            powers = np.array((1, t, t**2, t**3, t**4, t**5))
            rates = np.array((0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4))
            positions = self.early_coefficients @ powers
            velocities = self.early_coefficients @ rates
            s = t / tangent_start
            blend = 10*s**3 - 15*s**4 + 6*s**5
            blend_rate = (30*s**2 - 60*s**3 + 30*s**4) / tangent_start
            rotations = tuple(Rotation.from_rotvec(blend * vector) * initial
                              for vector, initial in zip(self.rotation_vectors, self.start_rotations))
            angular_velocity = blend_rate * self.rotation_vectors
            face_travel = (time_s - self.contact_time) * self.lateral_velocity
            face_travel_rate = self.lateral_velocity
        elif time_s < self.contact_time:
            positions = self.contact_positions + (time_s - self.contact_time) * self.tangent_velocity
            velocities = np.tile(self.tangent_velocity, (2, 1))
            rotations = self.contact_rotations
            face_travel = (time_s - self.contact_time) * self.lateral_velocity
            face_travel_rate = self.lateral_velocity
        else:
            travel, travel_rate = self.retreat(time_s - self.contact_time)
            positions = self.contact_positions + travel
            velocities = np.tile(travel_rate, (2, 1))
            rotations = self.contact_rotations
            face_travel = travel
            face_travel_rate = travel_rate
        if time_s < normal_start:
            s = max(0.0, time_s) / normal_start
            blend = 10*s**3 - 15*s**4 + 6*s**5
            rate = (30*s**2 - 60*s**3 + 30*s**4) / normal_start
            gap = self.start_gaps + (cfg.precontact_gap - self.start_gaps) * blend
            gap_rate = (cfg.precontact_gap - self.start_gaps) * rate
            # Start from the held hand's zero velocity; smoothly acquire lateral tracking.
            face_travel = (-self.contact_time + normal_start*(6*s**3 - 8*s**4 + 3*s**5)) * self.lateral_velocity
            face_travel_rate = (18*s**2 - 32*s**3 + 15*s**4) * self.lateral_velocity
        elif time_s < normal_end:
            s = (time_s - normal_start) / cfg.close_duration
            gap = (cfg.precontact_gap
                   + (cfg.normal_speed * cfg.close_duration - 3*cfg.precontact_gap) * s**2
                   + (2*cfg.precontact_gap - cfg.normal_speed * cfg.close_duration) * s**3)
            gap_rate = (2*(cfg.normal_speed * cfg.close_duration - 3*cfg.precontact_gap) * s
                        + 3*(2*cfg.precontact_gap - cfg.normal_speed * cfg.close_duration) * s**2
                        ) / cfg.close_duration
        else:
            squeeze_time = 2 * cfg.grip_compression / cfg.normal_speed
            squeeze_t = min(time_s - normal_end, squeeze_time)
            gap = -cfg.normal_speed * (squeeze_t - squeeze_t**2 / (2*squeeze_time))
            gap_rate = (-cfg.normal_speed * (1 - squeeze_t / squeeze_time)
                        if squeeze_t < squeeze_time else 0.0)
        base_gap = np.sum((positions - self.contact_positions - face_travel) * self.contact_normals, axis=1)
        base_rate = np.sum((velocities - face_travel_rate) * self.contact_normals, axis=1)
        positions += (gap - base_gap)[:, None] * self.contact_normals
        velocities += (gap_rate - base_rate)[:, None] * self.contact_normals
        return positions, rotations, velocities, angular_velocity


def _build_plan(
    box_position: np.ndarray,
    box_velocity: np.ndarray,
    box_rotation: Rotation,
    box_angular_velocity: np.ndarray,
    palm_positions: np.ndarray,
    palm_rotations: tuple[Rotation, Rotation],
    palm_normals_body: np.ndarray,
    settings: CatchSettings = CatchSettings(),
    *,
    follow_lateral_box_motion: bool = False,
    check_speed: bool = True,
    box_acceleration: np.ndarray | tuple[float, float, float] = (0., 0., -9.81),
) -> CatchPlan:
    """Plan one catch at the predicted plane crossing, subject to reach and motion checks."""
    cfg = settings
    box_position = np.asarray(box_position, dtype=float)
    box_velocity = np.asarray(box_velocity, dtype=float)
    box_angular_velocity = np.asarray(box_angular_velocity, dtype=float)
    palm_positions = np.asarray(palm_positions, dtype=float)
    palm_normals_body = np.asarray(palm_normals_body, dtype=float)
    box_half_extents = np.asarray(cfg.box_half_extents, dtype=float)
    if (box_position.shape != (3,) or box_velocity.shape != (3,)
            or box_angular_velocity.shape != (3,) or palm_positions.shape != (2, 3)
            or palm_normals_body.shape != (2, 3)
            or not all(np.isfinite(value).all() for value in
                       (box_position, box_velocity, box_angular_velocity, palm_positions, palm_normals_body,
                        box_rotation.as_quat(), *(rotation.as_quat() for rotation in palm_rotations)))):
        raise ValueError("invalid measured box or palm state")
    positive_settings = (cfg.max_lateral_error, cfg.max_height_error, cfg.max_roll_rad,
                         cfg.max_tilt_rad, cfg.max_roll_speed, cfg.max_spin_radps,
                         cfg.max_face_normal_speed,
                         cfg.max_palm_speed, cfg.max_palm_acceleration,
                         cfg.fallback_max_palm_speed, cfg.fallback_max_palm_acceleration,
                         cfg.tangent_speed_cap,
                         cfg.normal_speed, cfg.precontact_gap, cfg.close_duration,
                         cfg.normal_lead, cfg.tangent_lead, cfg.grip_compression,
                         cfg.cushion_distance, cfg.cushion_distance_z, cfg.settle_duration,
                         cfg.max_retreat_lateral, cfg.max_retreat_depth,
                         cfg.max_retreat_center_x, cfg.common_motion_relax_duration, cfg.palm_sphere_radius)
    if (box_velocity[1] <= 0 or not np.isfinite(cfg.plane_y) or not np.isfinite(cfg.center_z)
            or not np.isfinite(cfg.max_retreat_center_y) or not np.isfinite(cfg.min_retreat_center_z)
            or any(not np.isfinite(value) or value < 0
                   for value in (cfg.settle_center_x, cfg.settle_depth, cfg.settle_drop))
            or box_half_extents.shape != (3,) or not np.isfinite(box_half_extents).all()
            or np.any(box_half_extents <= 0)
            or any(not np.isfinite(value) or value <= 0 for value in positive_settings)
            or not np.isfinite(cfg.lateral_contact_bias_gain) or cfg.lateral_contact_bias_gain < 0
            or not np.isfinite(cfg.contact_depth_offset) or cfg.contact_depth_offset < 0
            or not np.isfinite(cfg.contact_height_offset)):
        raise ValueError("invalid incoming direction or catch settings")
    flight = BoxFlight(box_position, box_velocity, box_rotation, box_angular_velocity, box_acceleration)
    contact_time = flight.crossing_time(cfg.plane_y)
    tangent_start = contact_time - cfg.tangent_lead
    normal_start = contact_time - cfg.normal_lead - cfg.close_duration
    if min(tangent_start, normal_start) <= 0:
        raise ValueError("too little time to close the palms")
    contact = flight.at(contact_time)
    contact_box_position = contact.position_m
    contact_box_velocity = contact.velocity_mps
    if contact_box_velocity[2] >= -1e-6:  # Exclude the apex despite floating-point roundoff.
        raise ValueError("box must be descending at the contact plane")
    rotation = contact.rotation
    wait_center = palm_positions.mean(axis=0)
    if (abs(contact_box_position[0] - wait_center[0]) > cfg.max_lateral_error
            or abs(contact_box_position[2] - wait_center[2]) > cfg.max_height_error):
        raise ValueError("predicted box center misses the reachable attempt region")
    orientation_body = rotation.inv().apply(rotation.as_rotvec())
    local_spin = rotation.inv().apply(box_angular_velocity)
    if (abs(orientation_body[0]) > cfg.max_roll_rad
            or np.linalg.norm(orientation_body[1:]) > cfg.max_tilt_rad
            or abs(local_spin[0]) > cfg.max_roll_speed
            or np.linalg.norm(local_spin[1:]) > cfg.max_spin_radps):
        raise ValueError("box orientation or spin exceeds the fixed-face corridor")
    box_axis = rotation.apply((1., 0., 0.))
    if abs(np.dot(contact_box_velocity, box_axis)) > cfg.max_face_normal_speed:
        raise ValueError("box moves too quickly into a side palm")
    tangent = contact_box_velocity - np.dot(contact_box_velocity, box_axis) * box_axis
    tangent_speed = np.linalg.norm(tangent)
    if tangent_speed == 0:
        raise ValueError("box has no useful flight direction")
    commanded_tangent = tangent * min(1., cfg.tangent_speed_cap / tangent_speed)
    lateral_velocity = (np.array((contact_box_velocity[0], 0., 0.)) if follow_lateral_box_motion else np.zeros(3))
    follow_velocity = commanded_tangent.copy()
    if follow_lateral_box_motion:
        follow_velocity[0] = contact_box_velocity[0]
    half_width = box_half_extents[0]
    offsets = np.array(((half_width, 0., 0.), (-half_width, 0., 0.)))
    normals = rotation.apply(np.array(((1., 0., 0.), (-1., 0., 0.))))
    surface_velocities = contact_box_velocity + np.cross(box_angular_velocity, rotation.apply(offsets))
    relative_velocities = surface_velocities - follow_velocity
    relative_tangent = relative_velocities - np.sum(relative_velocities * normals, axis=1)[:, None] * normals
    relative_speed = np.linalg.norm(relative_tangent, axis=1).max()
    # At this new Moz1 pose the left hand is at world +X and the right at -X.
    contacts = contact_box_position + rotation.apply(offsets)
    # Aim within each side face toward the robot and below center to avoid early corner contact.
    forward = np.array((0., 1., 0.))
    forward -= np.dot(forward, normals[0]) * normals[0]
    contacts += cfg.contact_depth_offset * forward / np.linalg.norm(forward)
    up = np.array((0., 0., 1.))
    up -= np.dot(up, normals[0]) * normals[0]
    contacts += cfg.contact_height_offset * up / np.linalg.norm(up)
    # ponytail: zero bias for this pose; calibrate from measured left/right contact timing on hardware.
    biased_side = 1 if contact_box_position[0] > 0 else 0
    contacts[biased_side] += (cfg.lateral_contact_bias_gain * abs(contact_box_position[0])
                              * normals[biased_side])
    # Start braking exactly from the velocity commanded at contact: the normal
    # gap projection in target() cancels any common face-normal component of
    # the retreat anyway, and opening at the approach velocity keeps velocity
    # continuous through the contact instant.
    retreat_velocity = follow_velocity
    center = contacts.mean(axis=0)
    # Phase 1 brakes the lateral and chestward axes over a short, hard-clamped
    # cushion, while the vertical axis keeps a long baseline-like glide that
    # keeps supporting the box while the squeeze builds.
    budgets = np.array((cfg.cushion_distance, cfg.cushion_distance, cfg.cushion_distance_z))
    cushion = budgets / np.linalg.norm(follow_velocity) * retreat_velocity
    lateral_space = cfg.max_retreat_center_x - np.sign(retreat_velocity[0])*center[0]
    cushion[0] = np.sign(cushion[0])*min(abs(cushion[0]), cfg.max_retreat_lateral,
                                         max(0., lateral_space))
    if retreat_velocity[1] > 0:
        cushion[1] = min(cushion[1], cfg.max_retreat_depth,
                         max(0., cfg.max_retreat_center_y-center[1]))
    cushion[2] = max(cushion[2], cfg.min_retreat_center_z-center[2])
    moving = np.abs(retreat_velocity) > 1e-9
    if np.any(moving & (np.abs(cushion) < 1e-9)):
        raise ValueError("no common retreat space at the contact pose")
    retreat_durations = np.ones(3)
    retreat_durations[moving] = 2*cushion[moving]/retreat_velocity[moving]
    if np.linalg.norm(1.5*retreat_velocity/retreat_durations) > cfg.max_palm_acceleration:
        raise ValueError("not enough common retreat space to stop the palms")
    # Phase 2: from lateral rest, recentre the held box toward the body midline.
    # The X pull is a normal push between the palms, so it does not shear a
    # marginal grip the way an in-plane settle would.
    cushioned = center + cushion
    settle_target = np.array((np.sign(cushioned[0])*min(abs(cushioned[0]), cfg.settle_center_x),
                              min(cushioned[1]+cfg.settle_depth, cfg.max_retreat_center_y,
                                  center[1]+cfg.max_retreat_depth),
                              max(cushioned[2]-cfg.settle_drop, cfg.min_retreat_center_z)))
    settle = settle_target - cushioned
    horizontal = moving[:2]
    settle_start = float(retreat_durations[:2][horizontal].max()) if horizontal.any() else 0.
    if np.linalg.norm(np.abs(settle)*(10/np.sqrt(3))/cfg.settle_duration**2) > cfg.max_palm_acceleration:
        raise ValueError("settle motion exceeds the palm acceleration budget")
    stop_time = max(settle_start + cfg.settle_duration,
                    float(retreat_durations[moving].max()) if moving.any() else 0.,
                    2*cfg.grip_compression/cfg.normal_speed-cfg.normal_lead)
    precontact = contacts - cfg.tangent_lead * follow_velocity
    early_coefficients = _quintic(palm_positions, precontact, np.tile(follow_velocity, (2, 1)), tangent_start)
    start_gaps = np.sum((palm_positions - contacts + contact_time * lateral_velocity) * normals, axis=1)
    rotations = []
    vectors = []
    for side in range(2):
        source_normal = palm_rotations[side].apply(palm_normals_body[side])
        destination_normal = -normals[side]
        cross = np.cross(source_normal, destination_normal)
        dot = np.clip(np.dot(source_normal, destination_normal), -1., 1.)
        if dot < -0.999 and np.linalg.norm(cross) < 1e-6:
            raise ValueError("palm normal points away from its target box face")
        correction = Rotation.from_rotvec(cross * np.arccos(dot) / max(np.linalg.norm(cross), 1e-12))
        target_rotation = correction * palm_rotations[side]
        rotations.append(target_rotation)
        vectors.append((target_rotation * palm_rotations[side].inv()).as_rotvec())
    plan = CatchPlan(contact_time, contacts, normals, tuple(rotations), palm_rotations,
                     follow_velocity, lateral_velocity, relative_speed,
                     contacts + cushion + settle - cfg.grip_compression * normals,
                     stop_time, cfg, early_coefficients, np.asarray(vectors), start_gaps,
                     retreat_velocity, retreat_durations, cushion, settle, settle_start)
    if check_speed:
        _check_plan_speed(plan)
    return plan


def _check_plan_speed(plan: CatchPlan, *, check_retreat: bool = True) -> None:
    times = np.linspace(0., plan.contact_time, 49)
    if check_retreat:
        times = np.r_[times, plan.contact_time+np.linspace(0., plan.stop_time, 33)[1:]]
    epsilon = 1e-5
    probes = np.maximum(np.r_[times - epsilon, times + epsilon], 0.)
    positions, _ = _geometry_samples(plan, probes, np.empty((len(probes), 2, 3, 3)))
    velocities = (positions[len(times):] - positions[:len(times)]) / (2*epsilon)
    speeds = np.linalg.norm(velocities, axis=-1)
    accelerations = np.linalg.norm(np.diff(velocities, axis=0) / np.diff(times)[:, None, None], axis=-1)
    if (speeds.max() > plan.settings.max_palm_speed
            or accelerations.max() > plan.settings.max_palm_acceleration):
        raise ValueError("palm trajectory exceeds speed or acceleration budget")


def _geometry_samples(plan: CatchPlan, times: np.ndarray,
                      rotations: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one candidate in a batch, without calling target() at every sample."""
    cfg = plan.settings
    t = np.asarray(times)
    count = len(t)
    positions = np.empty((count, 2, 3))
    if rotations is None:
        rotations = np.empty((count, 2, 3, 3))
        calculate_rotations = True
    else:
        calculate_rotations = False
    face_travel = np.empty((count, 3))
    tangent_start = plan.contact_time - cfg.tangent_lead
    early = t < tangent_start
    if early.any():
        s = t[early]
        powers = np.stack((np.ones_like(s), s, s**2, s**3, s**4, s**5), axis=1)
        positions[early] = np.einsum("ijk,nk->nij", plan.early_coefficients, powers)
        blend = 10*(s/tangent_start)**3 - 15*(s/tangent_start)**4 + 6*(s/tangent_start)**5
        if calculate_rotations:
            for side in range(2):
                rotations[early, side] = (Rotation.from_rotvec(blend[:, None] * plan.rotation_vectors[side])
                                          * plan.start_rotations[side]).as_matrix()
        face_travel[early] = (s - plan.contact_time)[:, None] * plan.lateral_velocity
    follow = ~early & (t < plan.contact_time)
    if follow.any():
        s = t[follow] - plan.contact_time
        positions[follow] = plan.contact_positions + s[:, None, None] * plan.tangent_velocity
        face_travel[follow] = s[:, None] * plan.lateral_velocity
    brake = ~(early | follow)
    if brake.any():
        travel, _ = plan.retreat(t[brake] - plan.contact_time)
        positions[brake] = plan.contact_positions + travel[:, None, :]
        face_travel[brake] = travel
    if calculate_rotations:
        for side in range(2):
            rotations[~early, side] = plan.contact_rotations[side].as_matrix()

    normal_end = plan.contact_time - cfg.normal_lead
    normal_start = normal_end - cfg.close_duration
    gaps = np.empty((count, 2))
    before = t < normal_start
    if before.any():
        s = np.maximum(t[before], 0.) / normal_start
        blend = 10*s**3 - 15*s**4 + 6*s**5
        gaps[before] = plan.start_gaps + (cfg.precontact_gap - plan.start_gaps) * blend[:, None]
        face_travel[before] = (-plan.contact_time + normal_start*(6*s**3 - 8*s**4 + 3*s**5))[:, None] * plan.lateral_velocity
    closing = ~before & (t < normal_end)
    if closing.any():
        s = (t[closing] - normal_start) / cfg.close_duration
        gap = (cfg.precontact_gap + (cfg.normal_speed*cfg.close_duration - 3*cfg.precontact_gap)*s**2
               + (2*cfg.precontact_gap - cfg.normal_speed*cfg.close_duration)*s**3)
        gaps[closing] = gap[:, None]
    squeeze = ~(before | closing)
    if squeeze.any():
        squeeze_time = 2*cfg.grip_compression/cfg.normal_speed
        s = np.minimum(t[squeeze] - normal_end, squeeze_time)
        gaps[squeeze] = (-cfg.normal_speed*(s - s**2/(2*squeeze_time)))[:, None]
    base_gap = np.sum((positions - plan.contact_positions - face_travel[:, None, :])
                      * plan.contact_normals, axis=2)
    positions += (gaps - base_gap)[:, :, None] * plan.contact_normals
    return positions, rotations


def _geometry_score(plan: CatchPlan, sphere_offsets: np.ndarray,
                    flight: tuple[np.ndarray, np.ndarray, np.ndarray],
                    palm_rotations: np.ndarray | None = None
                    ) -> tuple[float, np.ndarray, tuple[float, float], tuple[float, float]]:
    """Allow early side contact; penalize front/end/corner impacts that deflect the box."""
    times, box_centers, box_matrices = flight
    palms, rotations = _geometry_samples(plan, times, palm_rotations)
    sphere_world = palms[:, :, None, :] + np.einsum("tsij,skj->tski", rotations, sphere_offsets)
    local = np.einsum("tji,tskj->tski", box_matrices, sphere_world - box_centers[:, None, None, :])
    half = np.asarray(plan.settings.box_half_extents)
    distance = np.linalg.norm(np.maximum(np.abs(local) - half, 0.), axis=-1) - plan.settings.palm_sphere_radius
    side_distance = distance.min(axis=2)
    first = []
    normal_cosines = []
    edge_penalty = 0.
    contact_penalty = 0.
    for side in range(2):
        hit = np.flatnonzero(side_distance[:, side] <= .005)
        if len(hit):
            index = int(hit[0])
            first.append(times[index])
            sphere = int(distance[index, side].argmin())
            center = local[index, side, sphere]
            point = np.clip(center, -half, half)
            normal = center - point
            length = np.linalg.norm(normal)
            if length > 1e-9:
                normal /= length
            else:
                # A sampled sphere center inside the box uses its nearest exit face.
                axis = int(np.argmin(half - np.abs(center)))
                normal = np.zeros(3)
                normal[axis] = 1. if center[axis] >= 0 else -1.
            expected = np.array((1. if side == 0 else -1., 0., 0.))
            cosine = float(np.dot(normal, expected))
            normal_cosines.append(cosine)
            contact_penalty += .15 * (1. - cosine)
            palm_normal_body = plan.contact_rotations[side].inv().apply(-plan.contact_normals[side])
            palm_normal = box_matrices[index].T @ rotations[index, side] @ palm_normal_body
            alignment_error = max(0., 1. + float(np.dot(palm_normal, normal)))
            contact_penalty += .1 * alignment_error
            palm_contact_normal = rotations[index, side].T @ box_matrices[index] @ normal
            extreme = (sphere_offsets[side, sphere, 0] <= sphere_offsets[side, :, 0].min() + 1e-6
                       or sphere_offsets[side, sphere, 0] >= sphere_offsets[side, :, 0].max() - 1e-6)
            end_cap = max(0., abs(float(palm_contact_normal[0])) - .5) if extreme else 0.
            contact_penalty += .25 * end_cap
            if index:
                relative_velocity = (local[index, side, sphere] - local[index-1, side, sphere]
                                     ) / (times[index] - times[index-1])
                incoming = max(0., -float(np.dot(relative_velocity, normal)))
                contact_penalty += .2 * incoming * (max(0., .9 - cosine)
                                                    + max(0., alignment_error - .1) + end_cap)
            edge_penalty += max(0., abs(point[1]) - (half[1] - .025))
            edge_penalty += max(0., abs(point[2]) - (half[2] - .025))
        else:
            first.append(plan.contact_time + .06)
            normal_cosines.append(0.)
    latest = max(first)
    miss = np.maximum(side_distance.min(axis=0), 0.).sum()
    score = (3*abs(first[0]-first[1])
             + 3*max(0., latest - plan.contact_time) + 5*miss + 2*edge_penalty + contact_penalty
             + .3*max(0., .9 - plan.stop_positions[:, 2].mean()))
    # ponytail: free-flight geometry ends after first contact; coupled dynamics need a separate model.
    return score, rotations, tuple(first), tuple(normal_cosines)


def _plan_catch_once(
    box_position: np.ndarray,
    box_velocity: np.ndarray,
    box_rotation: Rotation,
    box_angular_velocity: np.ndarray,
    palm_positions: np.ndarray,
    palm_rotations: tuple[Rotation, Rotation],
    palm_normals_body: np.ndarray,
    settings: CatchSettings = CatchSettings(),
    palm_sphere_offsets_body: np.ndarray | None = None,
    *,
    box_acceleration: np.ndarray | tuple[float, float, float] = (0., 0., -9.81),
) -> CatchPlan:
    """Commit one catch; use measured palm geometry to avoid an early unilateral strike."""
    args = (box_position, box_velocity, box_rotation, box_angular_velocity,
            palm_positions, palm_rotations, palm_normals_body)
    if palm_sphere_offsets_body is None:
        return _build_plan(*args, settings, box_acceleration=box_acceleration)
    spheres = np.asarray(palm_sphere_offsets_body, dtype=float)
    if spheres.ndim != 3 or spheres.shape[:2] != (2, 12) or spheres.shape[2] != 3 or not np.isfinite(spheres).all():
        raise ValueError("expected twelve finite coating sphere offsets per palm")
    candidates = []
    flight_cache = {}
    palm_rotation_cache = {}
    # ponytail: five approach shapes plus a fixed fallback; expand only for a reproduced miss.
    options = ((0., 0., 0., settings.normal_lead, True),
               (0., 0., 0., .005, True),
               (0., 0., -.03, .005, True),
               (.03, .04, 0., settings.normal_lead, True),
               (.03, .04, -.03, .005, True),
               (0., 0., 0., settings.normal_lead, False))
    for plane_shift, depth_shift, height_shift, lead, lateral in options:
        cfg = replace(settings, plane_y=settings.plane_y + plane_shift,
                      contact_depth_offset=settings.contact_depth_offset + depth_shift,
                      contact_height_offset=settings.contact_height_offset + height_shift,
                      normal_lead=lead)
        try:
            plan = _build_plan(*args, cfg, follow_lateral_box_motion=lateral, check_speed=False,
                               box_acceleration=box_acceleration)
        except ValueError as error:
            last_error = error
            continue
        if plane_shift not in flight_cache:
            times = np.linspace(0., plan.contact_time + .04,
                                max(81, int((plan.contact_time + .04)/.005)))
            flight = BoxFlight(box_position, box_velocity, box_rotation, box_angular_velocity, box_acceleration)
            centers = flight.positions(times)
            rotations = flight.rotations(times).as_matrix()
            flight_cache[plane_shift] = (times, centers, rotations)
        score, sampled_rotations, first, cosines = _geometry_score(
            plan, spheres, flight_cache[plane_shift], palm_rotation_cache.get(plane_shift))
        palm_rotation_cache[plane_shift] = sampled_rotations
        candidates.append(replace(plan, geometry_score=score, geometry_mode="lateral" if lateral else "fixed",
                                  predicted_touch_times_s=first, predicted_touch_normal_cosines=cosines))
    if not candidates:
        raise last_error
    for plan in sorted(candidates, key=lambda item: item.geometry_score):
        try:
            _check_plan_speed(plan)
        except ValueError:
            continue
        return plan
    raise ValueError("palm trajectory exceeds speed or acceleration budget")


def plan_catch(
    box_position: np.ndarray,
    box_velocity: np.ndarray,
    box_rotation: Rotation,
    box_angular_velocity: np.ndarray,
    palm_positions: np.ndarray,
    palm_rotations: tuple[Rotation, Rotation],
    palm_normals_body: np.ndarray,
    settings: CatchSettings = CatchSettings(),
    palm_sphere_offsets_body: np.ndarray | None = None,
    *,
    execution_delay_s: float = 0.0,
    control_dt_s: float | None = None,
    planning_started_s: float | None = None,
    box_acceleration: np.ndarray | tuple[float, float, float] = (0., 0., -9.81),
) -> CatchPlan:
    """Choose geometry once; optionally launch after measured computation, with no reserved wait."""
    started = time.perf_counter() if planning_started_s is None else planning_started_s
    if not np.isfinite(execution_delay_s) or execution_delay_s < 0:
        raise ValueError("invalid execution delay")
    if control_dt_s is not None and (not np.isfinite(control_dt_s) or control_dt_s <= 0):
        raise ValueError("invalid control timestep")
    flight = BoxFlight(box_position, box_velocity, box_rotation, box_angular_velocity, box_acceleration).at(execution_delay_s)
    box_position, box_velocity, box_rotation = flight.position_m, flight.velocity_mps, flight.rotation
    args = (box_position, box_velocity, box_rotation, box_angular_velocity,
            palm_positions, palm_rotations, palm_normals_body)
    try:
        plan = _plan_catch_once(*args, settings, palm_sphere_offsets_body, box_acceleration=flight.acceleration_mps2)
    except ValueError as error:
        if (str(error) != "palm trajectory exceeds speed or acceleration budget"
                or (settings.fallback_max_palm_speed <= settings.max_palm_speed
                    and settings.fallback_max_palm_acceleration <= settings.max_palm_acceleration)):
            raise
        faster = replace(settings, max_palm_speed=max(settings.max_palm_speed, settings.fallback_max_palm_speed),
                         max_palm_acceleration=max(settings.max_palm_acceleration,
                                                   settings.fallback_max_palm_acceleration))
        plan = _plan_catch_once(*args, faster, palm_sphere_offsets_body, box_acceleration=flight.acceleration_mps2)
    if control_dt_s is None:
        return replace(plan, execution_delay_s=execution_delay_s)

    # Same selected contact pose and shape, rebuilt from held palms at actual readiness.
    # No new observation or candidate search. Include finalization in the measured delay.
    for _ in range(32):
        delay = math.ceil((time.perf_counter() - started) / control_dt_s) * control_dt_s
        contact_time = plan.contact_time - delay
        tangent_start = contact_time - plan.settings.tangent_lead
        if min(tangent_start, contact_time - plan.settings.normal_lead - plan.settings.close_duration) <= 0:
            raise ValueError("too little time to close the palms after computation")
        ready = replace(plan, contact_time=contact_time,
                        execution_delay_s=execution_delay_s + delay,
                        early_coefficients=_quintic(np.asarray(palm_positions),
                            plan.contact_positions - plan.settings.tangent_lead * plan.tangent_velocity,
                            np.tile(plan.tangent_velocity, (2, 1)), tangent_start),
                        start_gaps=np.sum((palm_positions - plan.contact_positions
                                           + contact_time * plan.lateral_velocity) * plan.contact_normals, axis=1))
        try:
            # Retiming changes the approach; the selected retreat is already checked.
            _check_plan_speed(ready, check_retreat=False)
        except ValueError:
            ready = replace(ready, settings=replace(ready.settings,
                max_palm_speed=max(settings.max_palm_speed, settings.fallback_max_palm_speed),
                max_palm_acceleration=max(settings.max_palm_acceleration, settings.fallback_max_palm_acceleration)))
            _check_plan_speed(ready, check_retreat=False)
            # Reuse the allowed budget when another execution tick is needed.
            plan = replace(plan, settings=ready.settings)
        if palm_sphere_offsets_body is not None:
            probes = np.linspace(0., contact_time + .04, max(81, int((contact_time + .04)/.005)))
            absolute = probes + delay
            sampled_flight = (probes, flight.positions(absolute), flight.rotations(absolute).as_matrix())
            score, _, touch, cosines = _geometry_score(ready, np.asarray(palm_sphere_offsets_body), sampled_flight)
            ready = replace(ready, geometry_score=score, predicted_touch_times_s=touch,
                            predicted_touch_normal_cosines=cosines)
        if math.ceil((time.perf_counter() - started) / control_dt_s) * control_dt_s == delay:
            return ready
    raise ValueError("trajectory finalization exceeds the control timestep")
