"""Pure algorithm layer ported verbatim from MozBoxer; no I/O or ROS/Isaac imports."""

from moz1_catch.core.geometry import (
    BOX_DIMENSIONS_M,
    BOX_HALF_EXTENTS_M,
    BOX_MASS_KG,
    COATING_SPHERE_CENTERS_BODY_M,
    COATING_SPHERE_RADIUS_M,
    PALM_CENTER_OFFSETS_BODY_M,
    PALM_NORMAL_AXES_BODY,
    PALM_SPHERE_OFFSETS_BODY_M,
)
from moz1_catch.core.one_shot import CatchPlan, CatchSettings, LATE_COMMIT_SETTINGS, plan_catch
from moz1_catch.core.prediction import BoxFlight, PredictionSettings, estimate_box_flight

__all__ = [
    "BOX_DIMENSIONS_M", "BOX_HALF_EXTENTS_M", "BOX_MASS_KG",
    "COATING_SPHERE_CENTERS_BODY_M", "COATING_SPHERE_RADIUS_M",
    "PALM_CENTER_OFFSETS_BODY_M", "PALM_NORMAL_AXES_BODY", "PALM_SPHERE_OFFSETS_BODY_M",
    "CatchPlan", "CatchSettings", "LATE_COMMIT_SETTINGS", "plan_catch",
    "BoxFlight", "PredictionSettings", "estimate_box_flight",
]
