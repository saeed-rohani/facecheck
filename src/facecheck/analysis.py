"""Landmarks in, numbers out.

This module turns one face's landmarks into measurements, ratios, a head-pose
estimate and a face-shape estimate. It needs only NumPy and OpenCV, so it can
be tested without a camera or a model file.

Landmarks use the common 68-point layout (iBUG 300-W, as produced by OpenCV
Facemark and dlib): a (68, 2) array of ``x, y`` values as fractions of the
image width and height. Point 0 is on the image's left edge of the face,
which is the subject's right side.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np

SCHEMA_VERSION = 2
LANDMARK_COUNT = 68

# --- Landmark indices (R / L are the subject's own right and left) ----------
CHIN = 8
CHEEK_R, CHEEK_L = 1, 15
JAW_R, JAW_L = 4, 12
BROW_OUTER_R, BROW_OUTER_L = 17, 26
BROW_PEAK_R, BROW_PEAK_L = 19, 24
NASION, NOSE_TIP, NOSE_BASE = 27, 30, 33
NOSTRIL_R, NOSTRIL_L = 31, 35
EYE_OUTER_R, EYE_INNER_R, EYE_INNER_L, EYE_OUTER_L = 36, 39, 42, 45
EYE_R, EYE_L = tuple(range(36, 42)), tuple(range(42, 48))
MOUTH_R, MOUTH_L = 48, 54

# Points worked out from the landmarks; extend() appends them after the 68.
BROW_CENTRE, FOREHEAD_TOP, PUPIL_R, PUPIL_L = 68, 69, 70, 71

# --- Measurements ------------------------------------------------------------
# name -> landmark pairs. A measurement with several pairs is their average.
SEGMENTS = {
    "face_length": ((FOREHEAD_TOP, CHIN),),
    "face_width": ((CHEEK_R, CHEEK_L),),
    "forehead_width": ((BROW_OUTER_R, BROW_OUTER_L),),
    "jaw_width": ((JAW_R, JAW_L), (JAW_R + 1, JAW_L - 1)),
    "eye_distance_inner": ((EYE_INNER_R, EYE_INNER_L),),
    "eye_distance_outer": ((EYE_OUTER_R, EYE_OUTER_L),),
    "pupil_distance": ((PUPIL_R, PUPIL_L),),
    "eye_width": ((EYE_OUTER_R, EYE_INNER_R), (EYE_INNER_L, EYE_OUTER_L)),
    "nose_length": ((NASION, NOSE_BASE),),
    "nose_width": ((NOSTRIL_R, NOSTRIL_L),),
    "mouth_width": ((MOUTH_R, MOUTH_L),),
    "lower_face_height": ((NOSE_BASE, CHIN),),
}

# name -> (numerator, denominator)
RATIOS = {
    "length_to_width": ("face_length", "face_width"),
    "forehead_to_width": ("forehead_width", "face_width"),
    "jaw_to_width": ("jaw_width", "face_width"),
    "jaw_to_forehead": ("jaw_width", "forehead_width"),
}

# --- A generic face -----------------------------------------------------------
# The 68 points of an average adult face in centimetres: x towards the
# subject's left, y up, z towards the camera. Taken from MediaPipe's canonical
# face model (Apache-2.0, google-ai-edge/mediapipe, face_geometry/data). It is
# used to estimate head pose and how far back each point lies.
FACE_MODEL_CM = np.array([
    (-7.743, 2.365, -2.005), (-7.664, 0.673, -2.436), (-7.542, -1.049, -2.431), (-7.271, -2.891, -2.252),
    (-6.720, -4.789, -1.745), (-5.085, -7.179, 0.715), (-4.069, -7.993, 1.925), (-2.309, -8.974, 3.609),
    (0.000, -9.403, 4.264), (2.309, -8.974, 3.609), (4.069, -7.993, 1.925), (5.085, -7.179, 0.715),
    (6.720, -4.789, -1.745), (7.271, -2.891, -2.252), (7.542, -1.049, -2.431), (7.664, 0.673, -2.436),
    (7.743, 2.365, -2.005), (-5.721, 4.255, 2.831), (-4.986, 4.802, 3.752), (-3.987, 5.109, 4.466),
    (-2.760, 5.101, 5.016), (-1.396, 5.012, 5.316), (1.396, 5.012, 5.316), (2.760, 5.101, 5.016),
    (3.987, 5.109, 4.466), (4.986, 4.802, 3.752), (5.721, 4.255, 2.831), (0.000, 3.271, 5.236),
    (0.000, 1.728, 6.317), (0.000, 0.366, 7.243), (0.000, -0.463, 7.587), (-1.044, -1.465, 5.662),
    (-0.597, -2.014, 5.866), (0.000, -2.089, 6.058), (0.597, -2.014, 5.866), (1.044, -1.465, 5.662),
    (-4.446, 2.664, 3.173), (-3.670, 2.928, 3.724), (-2.724, 2.962, 3.872), (-1.856, 2.585, 3.758),
    (-2.724, 2.316, 3.777), (-3.670, 2.360, 3.635), (1.856, 2.585, 3.758), (2.724, 2.962, 3.872),
    (3.670, 2.928, 3.724), (4.446, 2.664, 3.173), (3.670, 2.360, 3.635), (2.724, 2.316, 3.777),
    (-2.456, -4.343, 4.284), (-1.432, -3.501, 5.496), (-0.711, -3.329, 5.877), (0.000, -3.406, 5.980),
    (0.711, -3.329, 5.877), (1.432, -3.501, 5.496), (2.456, -4.343, 4.284), (1.325, -5.107, 5.205),
    (0.700, -5.292, 5.448), (0.000, -5.365, 5.535), (-0.700, -5.292, 5.448), (-1.325, -5.107, 5.205),
    (-2.153, -4.276, 4.038), (-0.533, -3.993, 5.138), (0.000, -3.994, 5.219), (0.533, -3.993, 5.138),
    (2.153, -4.276, 4.038), (0.583, -4.518, 5.340), (0.000, -4.542, 5.405), (-0.583, -4.518, 5.340),
])  # fmt: skip

# Pose is fitted on points that stay put when the head turns. The jaw line is
# left out: on a turned head the visible outline slides around the face.
POSE_POINTS = [NASION, NOSE_TIP, NOSE_BASE, EYE_OUTER_R, EYE_INNER_R, EYE_INNER_L, EYE_OUTER_L,
               MOUTH_R, MOUTH_L, CHIN]  # fmt: skip

# --- Scale and pose ----------------------------------------------------------
AVERAGE_IPD_MM = 63.0  # adult average pupil distance; about 61-65 mm for most people
MIN_IPD_PX = 10.0  # eyes closer than this on screen give no usable scale
DEFAULT_HFOV_DEG = 60.0  # typical laptop webcam
FRONTAL_LIMITS_DEG = {"yaw": 12.0, "pitch": 15.0, "roll": 12.0}

# --- Face shape --------------------------------------------------------------
# Each shape is a typical point in feature space; a face gets the label of the
# nearest one. "oval" is the generic face above; the others are hand-set
# offsets from it. Not fitted to labelled faces: tune them here if needed.
SHAPE_FEATURES = ("length_to_width", "forehead_to_width", "jaw_to_width", "jaw_angle_deg")
SHAPE_PROTOTYPES = {
    "oval": (1.42, 0.75, 0.77, 139.0),
    "round": (1.28, 0.74, 0.80, 144.0),
    "square": (1.28, 0.78, 0.86, 131.0),
    "oblong": (1.57, 0.76, 0.79, 137.0),
    "heart": (1.41, 0.80, 0.70, 141.0),
    "diamond": (1.43, 0.68, 0.71, 141.0),
    "triangle": (1.37, 0.69, 0.85, 134.0),
}
SHAPE_SPREAD = (0.075, 0.036, 0.04, 8.0)  # how far each feature usually strays


def extend(points):
    """Return the landmarks with the derived points appended (rows 68 to 71).

    The forehead top is an estimate: by the facial "rule of thirds" the
    forehead is about as tall as the brow-to-nose-base distance.
    """
    p = np.asarray(points, dtype=np.float64)
    brow = (p[BROW_PEAK_R] + p[BROW_PEAK_L]) / 2
    forehead = brow + (brow - p[NOSE_BASE])
    pupil_r, pupil_l = p[list(EYE_R)].mean(axis=0), p[list(EYE_L)].mean(axis=0)
    return np.vstack([p, brow, forehead, pupil_r, pupil_l])


# The generic face in OpenCV camera axes (x right, y down, z away), in mm.
_MODEL = extend(FACE_MODEL_CM * 10.0 * (1, -1, -1))
_MODEL_IPD_MM = float(np.linalg.norm(_MODEL[PUPIL_L] - _MODEL[PUPIL_R]))


def analyze(
    landmarks,
    image_size,
    *,
    ipd_mm=None,
    hfov_deg=DEFAULT_HFOV_DEG,
    label=None,
    source=None,
    frames_averaged=1,
    include_landmarks=False,
):
    """Measure one face and return a JSON-serialisable result dict.

    landmarks: (68, 2) normalised landmarks.
    image_size: (width, height) of the frame the landmarks came from.
    ipd_mm: the person's pupil distance in mm, which sets the millimetre
        scale. None assumes the adult average; 0 gives pixels only.
    hfov_deg: the camera's horizontal field of view, used to undo perspective.
        Pass None for cropped or resized photos, where it cannot be known.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (LANDMARK_COUNT, 2):
        raise ValueError(f"landmarks must be a ({LANDMARK_COUNT}, 2) array")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be (width, height) in pixels")

    pixels = points * (width, height)
    on_image = extend(pixels)
    on_screen = _lengths(on_image)
    if on_screen["face_width"] < 1 or on_screen["forehead_width"] < 1 or on_screen["pupil_distance"] < 1:
        raise ValueError("landmarks are degenerate (no face width)")

    # Pose needs some focal length; for photos a typical one is good enough.
    focal_px = (width / 2) / math.tan(math.radians(hfov_deg or DEFAULT_HFOV_DEG) / 2)
    rotation = _head_rotation(pixels, focal_px, width, height)

    scale = None
    in_space = on_screen
    measured_ipd = AVERAGE_IPD_MM if ipd_mm is None else float(ipd_mm)
    if measured_ipd > 0 and on_screen["pupil_distance"] >= MIN_IPD_PX:
        solid, mm_per_px, distance = _to_3d(
            pixels, rotation, measured_ipd, focal_px if hfov_deg else None, width, height
        )
        in_space = _lengths(extend(solid))  # derive points in 3D, not on the flat image
        scale = {
            "method": "average_ipd" if ipd_mm is None else "given_ipd",
            "ipd_mm": measured_ipd,
            "mm_per_px": round(mm_per_px, 4),
            "camera_distance_mm": round(distance) if distance else None,
            "hfov_deg": float(hfov_deg) if hfov_deg else None,
        }

    ratios = {
        name: round(in_space[top] / in_space[bottom], 4) for name, (top, bottom) in RATIOS.items()
    }
    jaw_angle = round(_jaw_angle(on_image), 1)

    result = {
        "schema_version": SCHEMA_VERSION,
        "id": uuid.uuid4().hex,
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label,
        "source": source,
        "image": {"width": width, "height": height},
        "frames_averaged": int(frames_averaged),
        "pose": _pose(rotation),
        "scale": scale,
        "measurements_px": _rounded(on_screen, 1),
        "measurements_mm": _rounded(in_space, 1) if scale else None,
        "jaw_angle_deg": jaw_angle,
        "ratios": ratios,
        "face_shape": classify_shape(
            ratios["length_to_width"],
            ratios["forehead_to_width"],
            ratios["jaw_to_width"],
            jaw_angle,
        ),
    }
    if include_landmarks:
        result["landmarks"] = np.round(points, 5).tolist()
    return result


def classify_shape(length_to_width, forehead_to_width, jaw_to_width, jaw_angle_deg):
    """Label a face by its nearest shape prototype.

    Returns {"label", "confidence", "scores"}. The scores add up to 1 and say
    how well each shape matches relative to the others; they are not
    calibrated probabilities.
    """
    features = np.array([length_to_width, forehead_to_width, jaw_to_width, jaw_angle_deg])
    names = list(SHAPE_PROTOTYPES)
    offsets = (np.array([SHAPE_PROTOTYPES[n] for n in names]) - features) / SHAPE_SPREAD
    logits = -0.5 * np.sum(offsets**2, axis=1)
    weights = np.exp(logits - logits.max())  # subtracting the max avoids underflow
    weights /= weights.sum()
    best = int(np.argmax(weights))
    return {
        "label": names[best],
        "confidence": round(float(weights[best]), 3),
        "scores": {n: round(float(w), 3) for n, w in zip(names, weights)},
    }


def to_json(result, indent=2):
    """Serialise a result dict to a JSON string."""
    return json.dumps(result, indent=indent)


# --- Internals ---------------------------------------------------------------


def _head_rotation(pixels, focal_px, width, height):
    """Rotation of the head relative to the camera, fitted to the generic face."""
    camera = np.array([[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        _MODEL[POSE_POINTS],
        pixels[POSE_POINTS].reshape(-1, 1, 2),
        camera,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or tvec[2, 0] <= 0:  # no fit, or a "face" behind the camera
        raise ValueError("could not estimate the head pose")
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation


def _pose(rotation):
    """Head rotation in degrees.

    yaw > 0: head turned towards the subject's left.
    pitch > 0: chin tucked in (looking down).
    roll > 0: head tilted towards the subject's left shoulder.
    """
    across = rotation @ (1.0, 0.0, 0.0)  # towards the subject's left
    down = rotation @ (0.0, 1.0, 0.0)  # from forehead to chin
    angles = {
        "yaw": math.degrees(math.atan2(across[2], across[0])),
        "pitch": math.degrees(math.atan2(down[2], down[1])),
        "roll": math.degrees(math.atan2(across[1], across[0])),
    }
    return {  # "+ 0.0" turns -0.0 into 0.0
        "yaw_deg": round(angles["yaw"], 1) + 0.0,
        "pitch_deg": round(angles["pitch"], 1) + 0.0,
        "roll_deg": round(angles["roll"], 1) + 0.0,
        "frontal": all(abs(angles[k]) <= FRONTAL_LIMITS_DEG[k] for k in angles),
    }


def _to_3d(pixels, rotation, ipd_mm, focal_px, width, height):
    """Place the 68 landmarks in millimetres. Returns (points, mm per px, distance).

    The pupil distance sets the scale. With a known focal length each point
    is also pushed back to the depth the generic face gives it, which undoes
    the perspective that makes the far-back cheeks look narrow.
    """
    on_image = extend(pixels)
    ipd_px = float(np.linalg.norm(on_image[PUPIL_L] - on_image[PUPIL_R]))
    eye_axis = rotation @ (_MODEL[PUPIL_L] - _MODEL[PUPIL_R])
    facing = math.hypot(eye_axis[0], eye_axis[1]) / float(np.linalg.norm(eye_axis))  # < 1 when turned
    if not focal_px:
        mm_per_px = ipd_mm * facing / ipd_px
        return pixels * mm_per_px, mm_per_px, None

    distance = focal_px * ipd_mm * facing / ipd_px  # camera to eyes, in mm
    depth = (_MODEL @ rotation.T)[:, 2] * (ipd_mm / _MODEL_IPD_MM)
    depth = distance + depth - (depth[PUPIL_R] + depth[PUPIL_L]) / 2
    depth = depth[:LANDMARK_COUNT]
    x = (pixels[:, 0] - width / 2) * depth / focal_px
    y = (pixels[:, 1] - height / 2) * depth / focal_px
    return np.column_stack([x, y, depth]), distance / focal_px, distance


def _lengths(points):
    return {
        name: float(np.mean([np.linalg.norm(points[a] - points[b]) for a, b in pairs]))
        for name, pairs in SEGMENTS.items()
    }


def _rounded(values, digits):
    return {k: round(v, digits) for k, v in values.items()}


def _angle(vertex, a, b):
    u, v = a - vertex, b - vertex
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) or 1.0))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _jaw_angle(p):
    """Angle at the jaw corner between the cheekbone and the chin, both sides averaged.
    Smaller means a more angular jaw."""
    right = _angle(p[JAW_R], p[CHEEK_R], p[CHIN])
    left = _angle(p[JAW_L], p[CHEEK_L], p[CHIN])
    return (right + left) / 2
