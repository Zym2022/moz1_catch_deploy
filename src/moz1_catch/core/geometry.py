"""Geometry constants ported from the MozBoxer simulation tree.

Provenance (see PROVENANCE.md at the deploy package root):

- Box: the isolated one-shot catching asset, 301x301x305 mm, 510 g
  (``catching/box_asset.py``, commit 35c5461 + working tree of 2026-09-28).
- Palm geometry follows the 2026-09-28 revision: the 12 coating spheres and the
  steel core moved by -10 mm (left) / +10 mm (right) along the palm-local Y,
  with the target point staying the centre of the front tangent plane.
  ``PALM_CENTER_OFFSETS_BODY_M`` is expressed in the hand-link axes and refers
  to that tangent-plane centre (see the block comment below).
- Coating spheres: 12 spheres per palm, 30 mm radius, 4x3 grid.
  ``PALM_SPHERE_OFFSETS_BODY_M`` is relative to the palm target point so it can
  be passed directly as ``palm_sphere_offsets_body`` of ``plan_catch``.

Hand order is [left, right]; at the Moz1 catch pose the left hand sits at
world +X and the right at -X.
"""

from __future__ import annotations

import numpy as np


BOX_DIMENSIONS_M = (0.301, 0.301, 0.305)
BOX_HALF_EXTENTS_M = tuple(length / 2 for length in BOX_DIMENSIONS_M)
BOX_MASS_KG = 0.510

# The planner's per-hand output pose is the PALM TARGET frame: origin at the
# tangent-plane centre of the 12-sphere coating array, axes along the hand link
# (the custom mounting orientation).  Concretely, for the left palm the front
# sphere layer sits at hand-local y=-0.020 with radius 0.030, so the tangent
# plane is y=-0.050 and the target point is its centre: x=0.100 (midpoint of
# the 0.040..0.160 sphere centres), z=0 (midpoint of -0.030..0.030).  The right
# palm mirrors at y=+0.040.  plan_catch receives and CatchPlan.target() returns
# (target-point position, hand-link orientation) - one consistent rigid frame,
# the hand link translated by PALM_CENTER_OFFSETS_BODY_M.
PALM_CENTER_OFFSETS_BODY_M = (
    (0.100, -0.050, 0.000),
    (0.100, +0.040, 0.000),
)

# Palm closing-face normals in the hand-link axes; left, right.
PALM_NORMAL_AXES_BODY = (
    (0.000, -1.000, 0.000),
    (0.000, +1.000, 0.000),
)

COATING_SPHERE_RADIUS_M = 0.030
COATING_SPHERE_CENTERS_BODY_M = tuple(
    tuple((x, y, z) for x in (.040, .080, .120, .160) for z in (-.030, 0., .030))
    for y in (-.020, .010)
)

# Runtime collision geometry from MozBoxer/tasks/direct/mozboxer/palm_coating.py.
# These are the simulation's design dimensions, not measured physical CAD.
PALM_CORE_CENTERS_BODY_M = ((.100, -.020, 0.), (.100, .010, 0.))
PALM_CORE_SIZE_M = (.140, .020, .070)

PALM_SPHERE_OFFSETS_BODY_M = (np.asarray(COATING_SPHERE_CENTERS_BODY_M)
                              - np.asarray(PALM_CENTER_OFFSETS_BODY_M)[:, None, :])
