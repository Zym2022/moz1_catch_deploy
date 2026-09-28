"""Pose-only free-flight estimation and the trajectory shared by one-shot planning."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PredictionSettings:
    window_s: float = .18
    angular_window_s: float = .08
    acceleration_prior_mps2: tuple[float, float, float] = (0., 0., -9.81)
    # Mean squared position residual + this squared weight * acceleration residual.
    acceleration_regularization_s2: float = .002

    def __post_init__(self):
        values = (self.window_s, self.angular_window_s, self.acceleration_regularization_s2)
        prior = np.asarray(self.acceleration_prior_mps2, dtype=float)
        if (not np.isfinite(values).all() or min(values[:2]) < .02 or values[2] < 0
                or prior.shape != (3,) or not np.isfinite(prior).all()):
            raise ValueError("invalid flight prediction settings")


@dataclass(frozen=True)
class BoxFlight:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    rotation: Rotation
    angular_velocity_radps: np.ndarray
    acceleration_mps2: np.ndarray

    def __post_init__(self):
        for name in ("position_m", "velocity_mps", "angular_velocity_radps", "acceleration_mps2"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("invalid measured box flight state")
            object.__setattr__(self, name, value.copy())
        if self.rotation.as_quat().shape != (4,) or not np.isfinite(self.rotation.as_quat()).all():
            raise ValueError("expected one finite box rotation")

    def positions(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        if seconds.ndim > 1 or not np.isfinite(seconds).all():
            raise ValueError("invalid flight prediction horizon")
        t = seconds[..., None]
        return self.position_m + t*self.velocity_mps + .5*t**2*self.acceleration_mps2

    def rotations(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        if seconds.ndim > 1 or not np.isfinite(seconds).all():
            raise ValueError("invalid flight prediction horizon")
        # ponytail: constant world angular velocity; revisit if held-out errors require torque dynamics.
        return Rotation.from_rotvec(seconds[..., None]*self.angular_velocity_radps)*self.rotation

    def at(self, seconds: float):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid flight prediction horizon")
        return BoxFlight(self.positions(seconds), self.velocity_mps+seconds*self.acceleration_mps2,
                         self.rotations(seconds), self.angular_velocity_radps, self.acceleration_mps2)

    def crossing_time(self, plane_y: float) -> float:
        """Earliest positive forward crossing, stable for near-zero acceleration."""
        delta, speed = plane_y-self.position_m[1], self.velocity_mps[1]
        if not np.isfinite(plane_y) or delta <= 0 or speed <= 0:
            raise ValueError("invalid incoming direction or contact plane")
        discriminant = speed**2 + 2*self.acceleration_mps2[1]*delta
        if discriminant <= 0:
            raise ValueError("box does not reach the contact plane while moving forward")
        return float(2*delta/(speed+np.sqrt(discriminant)))


def estimate_box_flight(times_s, poses_xyzw, now_s: float,
                        settings: PredictionSettings = PredictionSettings(),
                        max_age_s: float = .03) -> BoxFlight:
    """Fit only past free-flight poses, by elapsed time, with an acceleration prior.

    The caller excludes hand-held/release-transient poses and invalid tracking.
    Normalizing residuals makes the prior weight independent of sample count.
    """
    times = np.asarray(times_s, dtype=float)
    poses = np.asarray(poses_xyzw, dtype=float)
    if (times.ndim != 1 or poses.shape != (len(times), 7) or len(times) < 3
            or not np.isfinite(times).all() or not np.isfinite(poses).all()
            or not np.isfinite(now_s) or not np.isfinite(max_age_s) or max_age_s <= 0
            or np.any(np.diff(times) <= 0) or times[-1] > now_s+1e-10
            or now_s-times[-1] > max_age_s):
        raise ValueError("insufficient or stale box pose samples")
    selected = times >= now_s-settings.window_s-1e-10
    t, p = times[selected], poses[selected, :3]
    if len(t) < 3 or t[-1]-t[0] < .02-1e-10:
        raise ValueError("insufficient or stale box pose samples")
    tau = t-now_s
    design = np.column_stack((np.ones_like(tau), tau, .5*tau**2))/np.sqrt(len(t))
    penalty = settings.acceleration_regularization_s2
    design = np.vstack((design, (0., 0., penalty)))
    targets = np.vstack((p/np.sqrt(len(t)), penalty*np.asarray(settings.acceleration_prior_mps2)))
    position, velocity, acceleration = np.linalg.lstsq(design, targets, rcond=None)[0]

    angular = times >= now_s-settings.angular_window_s-1e-10
    angular_t = times[angular]
    if len(angular_t) < 3 or angular_t[-1]-angular_t[0] < .02-1e-10:
        raise ValueError("insufficient or stale angular pose samples")
    rotations = Rotation.from_quat(poses[angular, 3:])
    last = rotations[-1]
    dt = angular_t-angular_t[-1]
    vectors = (rotations*last.inv()).as_rotvec()
    omega = np.sum(dt[:, None]*vectors, axis=0)/np.dot(dt, dt)
    rotation = Rotation.from_rotvec((now_s-angular_t[-1])*omega)*last
    return BoxFlight(position, velocity, rotation, omega, acceleration)
