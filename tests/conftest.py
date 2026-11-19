"""A synthetic face for tests: no camera, no model.

The face is the generic 68-point face from facecheck.analysis (MediaPipe's
canonical face, in centimetres, x to the subject's left, y up, z towards the
camera). ``project`` photographs it with an ideal pinhole camera and returns
landmarks in the same normalised form the detector produces.
"""

import math

import numpy as np
import pytest

from facecheck.analysis import FACE_MODEL_CM


def canonical_points():
    """(68, 3) points of the generic face, in centimetres."""
    return FACE_MODEL_CM.copy()


def _rotation(yaw, pitch, roll):
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def project(points=None, size=(1280, 720), distance_mm=550.0, hfov_deg=60.0, yaw=0.0, pitch=0.0, roll=0.0):
    """Photograph the face and return (68, 2) normalised landmarks."""
    points = canonical_points() if points is None else np.asarray(points, dtype=np.float64)
    width, height = size
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    head = (points * 10.0) @ _rotation(yaw, pitch, roll).T  # mm, head-centred
    depth = distance_mm - head[:, 2]  # the camera looks along the head's -z axis
    u = width / 2 + focal * head[:, 0] / depth
    v = height / 2 - focal * head[:, 1] / depth  # image y points down
    return np.column_stack([u / width, v / height])


@pytest.fixture
def face():
    return project()


def yunet_row(landmarks=None, size=(1280, 720), score=0.95):
    """The 15 numbers YuNet would report for the synthetic face, in pixels.

    Box from the face outline; then right eye, left eye, nose tip, right and
    left mouth corners (the subject's right is on the image's left).
    """
    from facecheck.analysis import FOREHEAD_TOP, MOUTH_L, MOUTH_R, NOSE_TIP, PUPIL_L, PUPIL_R, extend

    p = extend(project() if landmarks is None else landmarks) * size
    left, right = p[:68, 0].min(), p[:68, 0].max()
    top, bottom = p[FOREHEAD_TOP, 1], p[:68, 1].max()
    points = [p[PUPIL_R], p[PUPIL_L], p[NOSE_TIP], p[MOUTH_R], p[MOUTH_L]]
    return np.array([left, top, right - left, bottom - top, *np.ravel(points), score], dtype=np.float32)
