import json

import numpy as np
import pytest
from conftest import canonical_points, project

from facecheck.analysis import (
    PUPIL_L,
    PUPIL_R,
    SEGMENTS,
    SHAPE_PROTOTYPES,
    analyze,
    classify_shape,
    extend,
    to_json,
)


def true_mm(name, points_cm=None):
    """Ground-truth length of a measurement on the synthetic face, in mm."""
    points = extend((canonical_points() if points_cm is None else points_cm) * 10)
    return float(np.mean([np.linalg.norm(points[a] - points[b]) for a, b in SEGMENTS[name]]))


TRUE_IPD = true_mm("pupil_distance")


def test_measurements_match_the_real_face(face):
    result = analyze(face, (1280, 720), ipd_mm=TRUE_IPD)
    for name, measured in result["measurements_mm"].items():
        assert measured == pytest.approx(true_mm(name), rel=0.02), name
    assert result["scale"]["method"] == "given_ipd"
    assert result["scale"]["camera_distance_mm"] == pytest.approx(550 - 37, rel=0.03)  # eyes sit 37 mm forward


def test_default_scale_assumes_the_average_pupil_distance(face):
    result = analyze(face, (1280, 720))
    assert result["scale"]["method"] == "average_ipd"
    assert result["scale"]["ipd_mm"] == 63.0
    assert result["measurements_mm"]["pupil_distance"] == pytest.approx(63.0, abs=0.5)
    assert result["measurements_mm"]["face_width"] == pytest.approx(true_mm("face_width"), rel=0.03)


def test_a_bigger_face_measures_bigger():
    big = canonical_points() * 1.1
    result = analyze(project(big), (1280, 720), ipd_mm=TRUE_IPD * 1.1)
    assert result["measurements_mm"]["face_length"] == pytest.approx(true_mm("face_length", big), rel=0.02)


def test_average_face_is_frontal_and_oval(face):
    result = analyze(face, (1280, 720))
    assert result["pose"] == {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0, "frontal": True}
    assert result["face_shape"]["label"] == "oval"
    assert sum(result["face_shape"]["scores"].values()) == pytest.approx(1, abs=0.01)


def test_ratios_do_not_depend_on_camera_distance():
    near = analyze(project(distance_mm=400), (1280, 720))["ratios"]
    far = analyze(project(distance_mm=900), (1280, 720))["ratios"]
    for name in near:
        assert near[name] == pytest.approx(far[name], rel=0.015), name


def test_perspective_correction_matters():
    """Without it, a close camera makes the far-back cheekbones look narrow."""
    landmarks = project(distance_mm=400)
    corrected = analyze(landmarks, (1280, 720), ipd_mm=TRUE_IPD)["measurements_mm"]["face_width"]
    flat = analyze(landmarks, (1280, 720), ipd_mm=TRUE_IPD, hfov_deg=None)["measurements_mm"]["face_width"]
    truth = true_mm("face_width")
    assert abs(corrected - truth) < 0.02 * truth
    assert flat < 0.92 * truth


@pytest.mark.parametrize("angle", [-25, 25])
def test_turned_head_is_flagged_but_still_measured(angle):
    result = analyze(project(yaw=angle), (1280, 720), ipd_mm=TRUE_IPD)
    assert result["pose"]["frontal"] is False
    assert result["pose"]["yaw_deg"] == pytest.approx(angle, abs=2)
    assert result["measurements_mm"]["face_width"] == pytest.approx(true_mm("face_width"), rel=0.03)


@pytest.mark.parametrize("axis", ["pitch", "roll"])
@pytest.mark.parametrize("angle", [-25, 25])
def test_tilted_head_is_flagged(axis, angle):
    result = analyze(project(**{axis: angle}), (1280, 720))
    assert result["pose"]["frontal"] is False
    assert abs(result["pose"][f"{axis}_deg"]) == pytest.approx(25, abs=2)


def test_pose_signs_follow_the_documentation():
    assert analyze(project(yaw=20), (1280, 720))["pose"]["yaw_deg"] > 0  # turned to own left
    assert analyze(project(pitch=20), (1280, 720))["pose"]["pitch_deg"] > 0  # chin down
    assert analyze(project(roll=-20), (1280, 720))["pose"]["roll_deg"] > 0  # towards left shoulder


def test_pixels_only_when_scale_is_switched_off(face):
    result = analyze(face, (1280, 720), ipd_mm=0)
    assert result["measurements_mm"] is None
    assert result["scale"] is None
    assert result["measurements_px"]["face_width"] > 0
    assert result["face_shape"]["label"] in SHAPE_PROTOTYPES


def test_a_longer_face_becomes_oblong():
    stretched = canonical_points() * (1, 1.15, 1)
    assert analyze(project(stretched), (1280, 720))["face_shape"]["label"] == "oblong"


def test_derived_points():
    p = extend(canonical_points())
    assert p.shape == (72, 3)
    assert p[PUPIL_R][0] < 0 < p[PUPIL_L][0]  # the subject's right eye has negative x
    assert TRUE_IPD == pytest.approx(63.6, abs=0.2)


@pytest.mark.parametrize("name", list(SHAPE_PROTOTYPES))
def test_each_prototype_is_its_own_best_match(name):
    shape = classify_shape(*SHAPE_PROTOTYPES[name])
    assert shape["label"] == name
    assert shape["confidence"] == max(shape["scores"].values())


def test_classifier_survives_extreme_input():
    shape = classify_shape(9.0, 9.0, 9.0, 0.0)
    assert shape["label"] in SHAPE_PROTOTYPES
    assert sum(shape["scores"].values()) == pytest.approx(1, abs=0.01)


def test_result_is_json_serialisable(face):
    result = analyze(face, (1280, 720), label="demo", source="camera:0", include_landmarks=True)
    restored = json.loads(to_json(result))
    assert restored == result
    assert len(restored["landmarks"]) == 68
    assert restored["label"] == "demo"
    assert restored["schema_version"] == 2
    assert "landmarks" not in analyze(face, (1280, 720))


@pytest.mark.parametrize(
    "bad",
    [np.zeros((10, 2)), np.zeros((68, 3)), np.zeros((478, 3)), np.zeros((68, 2))],
    ids=["too few", "wrong width", "old mediapipe layout", "all zero"],
)
def test_bad_landmarks_are_rejected(bad):
    with pytest.raises(ValueError):
        analyze(bad, (1280, 720))


def test_bad_image_size_is_rejected(face):
    with pytest.raises(ValueError):
        analyze(face, (0, 720))
