"""Geometry constants: the palm target point is the tangent-plane centre."""

import numpy as np
import pytest

from moz1_catch.core.geometry import (
    COATING_SPHERE_CENTERS_BODY_M,
    COATING_SPHERE_RADIUS_M,
    PALM_CENTER_OFFSETS_BODY_M,
    PALM_NORMAL_AXES_BODY,
)


@pytest.mark.parametrize("side", (0, 1), ids=("left", "right"))
def test_palm_target_point_is_the_front_tangent_plane_centre(side):
    centers = np.asarray(COATING_SPHERE_CENTERS_BODY_M[side])
    offset = np.asarray(PALM_CENTER_OFFSETS_BODY_M[side])
    normal = np.asarray(PALM_NORMAL_AXES_BODY[side])
    # The target point lies exactly on the plane tangent to the front sphere
    # layer (front = furthest along the closing normal): left -0.050, right
    # +0.040 in hand-local Y, i.e. centre distance + sphere radius.
    projection = centers @ normal
    assert np.dot(offset, normal) == pytest.approx(projection.max() + COATING_SPHERE_RADIUS_M)
    # ... and centred over the array in the other two hand-local axes.
    assert offset[0] == pytest.approx((centers[:, 0].min() + centers[:, 0].max()) / 2)
    assert offset[2] == pytest.approx((centers[:, 2].min() + centers[:, 2].max()) / 2)
    assert offset[0] == pytest.approx(0.100)
    assert np.dot(offset, normal) == pytest.approx(0.050 if side == 0 else 0.040)
