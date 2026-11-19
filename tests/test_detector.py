import shutil
import ssl
import subprocess
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest

from facecheck import detector
from facecheck.detector import MODELS, FaceDetector, ensure_models


def fill(folder, size=None):
    """Put stand-in model files in a folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for model in MODELS.values():
        (folder / model.filename).write_bytes(b"x" * (size or model.min_bytes))
    return folder


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv(detector.MODEL_DIR_ENV, raising=False)


def test_explicit_folder_is_used(tmp_path):
    paths = ensure_models(fill(tmp_path, size=5))
    assert paths["landmarks"] == tmp_path / "lbfmodel.yaml"
    assert paths["detector"].name.endswith(".onnx")


def test_environment_variable_points_at_the_folder(tmp_path, monkeypatch):
    monkeypatch.setenv(detector.MODEL_DIR_ENV, str(fill(tmp_path, size=5)))
    assert ensure_models()["detector"].parent == tmp_path


def test_missing_files_in_explicit_folder_are_listed_with_urls(tmp_path):
    with pytest.raises(FileNotFoundError, match="lbfmodel.yaml: https://"):
        ensure_models(tmp_path)


def test_cached_models_are_reused_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "default_model_dir", lambda: fill(tmp_path))
    monkeypatch.setattr(detector, "_download", lambda *a: pytest.fail("should not download"))
    assert ensure_models()["landmarks"].parent == tmp_path


def test_partial_cached_file_is_downloaded_again(tmp_path, monkeypatch):
    fill(tmp_path)
    (tmp_path / "lbfmodel.yaml").write_bytes(b"cut short")
    fetched = []
    monkeypatch.setattr(detector, "default_model_dir", lambda: tmp_path)
    monkeypatch.setattr(detector, "_download", lambda model, path: fetched.append(model.filename))
    ensure_models()
    assert fetched == ["lbfmodel.yaml"]


def test_failed_download_explains_what_to_do(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("network is down")

    monkeypatch.setattr(detector, "default_model_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(detector.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="run again") as caught:
        ensure_models()
    assert str(tmp_path / "cache") in str(caught.value)
    assert "Install Certificates" not in str(caught.value)
    assert not list((tmp_path / "cache").glob("*"))  # no half-written file left behind


def certificate_failure(*args, **kwargs):
    """What urllib raises when Python has no trusted certificates."""
    raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))


def test_certificate_failure_falls_back_to_curl(tmp_path, monkeypatch):
    calls = []

    def fake_curl(command, check, timeout):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"x" * 200_000)

    monkeypatch.setattr(detector.urllib.request, "urlopen", certificate_failure)
    monkeypatch.setattr(detector.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(detector.subprocess, "run", fake_curl)
    target = tmp_path / "face_detection_yunet_2023mar.onnx"
    detector._download(MODELS["detector"], target)
    assert target.stat().st_size == 200_000
    command = calls[0]
    assert command[0] == "/usr/bin/curl" and command[-1] == MODELS["detector"].url
    assert "--location" in command  # GitHub redirects model downloads
    assert "--insecure" not in command and "-k" not in command  # certificates stay checked


def test_real_curl_fetches_a_file(tmp_path, monkeypatch):
    """Runs the actual curl program on a local file:// URL."""
    if not shutil.which("curl"):
        pytest.skip("curl is not installed")
    source = tmp_path / "source.bin"
    source.write_bytes(b"model bytes")
    monkeypatch.setattr(detector.urllib.request, "urlopen", certificate_failure)
    detector._fetch(source.as_uri(), tmp_path / "copy.bin")
    assert (tmp_path / "copy.bin").read_bytes() == b"model bytes"


def test_certificate_failure_without_curl_explains_the_mac_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(detector.urllib.request, "urlopen", certificate_failure)
    monkeypatch.setattr(detector.shutil, "which", lambda name: None)
    monkeypatch.setattr(detector.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Install Certificates.command"):
        detector._download(MODELS["detector"], tmp_path / "m.onnx")
    assert not list(tmp_path.glob("*"))


def test_failing_curl_reports_the_certificate_problem(tmp_path, monkeypatch):
    def broken_curl(command, check, timeout):
        raise subprocess.CalledProcessError(60, command)

    monkeypatch.setattr(detector.urllib.request, "urlopen", certificate_failure)
    monkeypatch.setattr(detector.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(detector.subprocess, "run", broken_curl)
    monkeypatch.setattr(detector.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="certificate verify failed"):
        detector._download(MODELS["detector"], tmp_path / "m.onnx")


def test_other_network_errors_do_not_try_curl(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(detector.urllib.request, "urlopen", offline)
    monkeypatch.setattr(detector.subprocess, "run", lambda *a, **k: pytest.fail("should not run curl"))
    with pytest.raises(RuntimeError, match="no route to host"):
        detector._download(MODELS["detector"], tmp_path / "m.onnx")


def test_missing_contrib_module_is_explained(monkeypatch):
    monkeypatch.delattr(cv2, "face", raising=False)
    with pytest.raises(RuntimeError, match="opencv-contrib-python"):
        FaceDetector()


def test_damaged_model_file_is_a_clean_error(tmp_path):
    """Real OpenCV loading a junk detector model must not crash the app."""
    if not hasattr(cv2, "face"):
        pytest.skip("needs opencv-contrib-python")
    with pytest.raises(RuntimeError, match="delete it and run again"):
        FaceDetector(fill(tmp_path))


class FakeFinder:
    def __init__(self, faces):
        self.faces = faces
        self.sizes = []

    def setInputSize(self, size):
        self.sizes.append(size)

    def detect(self, image):
        assert (image.shape[1], image.shape[0]) == self.sizes[-1]
        return 1, self.faces


class FakeFacemark:
    def __init__(self, points=None):
        self.points = points
        self.boxes = []

    def fit(self, image, faces):
        assert image.ndim == 2  # grayscale
        self.boxes.append(faces.tolist())
        if self.points is None:
            return False, ()
        return True, (self.points.reshape(1, -1, 2),)


def make_detector(faces, points=None):
    det = FaceDetector.__new__(FaceDetector)
    det._finder, det._facemark = FakeFinder(faces), FakeFacemark(points)
    return det


def yunet_row(x, y, w, h, score=0.9):
    return [x, y, w, h] + [0.0] * 10 + [score]


def test_largest_face_is_used_with_a_square_box():
    # Detection runs on a 640-wide copy of the 1280-wide frame, so boxes come back at half size.
    faces = np.array([yunet_row(10, 10, 20, 30), yunet_row(200, 100, 80, 100)], np.float32)
    points = np.tile([640.0, 360.0], (68, 1)).astype(np.float32)
    det = make_detector(faces, points)
    landmarks = det.detect(np.zeros((720, 1280, 3), np.uint8))
    assert det._finder.sizes == [(640, 360)]
    assert det._facemark.boxes == [[[380, 200, 200, 200]]]  # centred on (480, 300), side 200
    assert landmarks.shape == (68, 2)
    assert landmarks[0].tolist() == pytest.approx([0.5, 0.5])


def test_small_frames_are_not_resized():
    det = make_detector(np.array([yunet_row(100, 50, 60, 80)], np.float32), np.zeros((68, 2), np.float32))
    det.detect(np.zeros((240, 320, 3), np.uint8))
    assert det._finder.sizes == [(320, 240)]


def test_no_face_gives_none():
    assert make_detector(None).detect(np.zeros((72, 128, 3), np.uint8)) is None
    empty = np.zeros((0, 15), np.float32)
    assert make_detector(empty).detect(np.zeros((72, 128, 3), np.uint8)) is None


def test_failed_landmark_fit_gives_none():
    det = make_detector(np.array([yunet_row(10, 10, 50, 50)], np.float32), points=None)
    assert det.detect(np.zeros((72, 128, 3), np.uint8)) is None


# --- Recognition -----------------------------------------------------------------

from conftest import project  # noqa: E402
from conftest import yunet_row as synthetic_face  # noqa: E402

from facecheck.detector import face_problem, square_box  # noqa: E402


def test_only_the_requested_models_are_downloaded(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(detector, "default_model_dir", lambda: tmp_path)
    monkeypatch.setattr(detector, "_download", lambda model, path: fetched.append(model.filename))
    paths = ensure_models(names=["detector", "recognizer"])
    assert set(paths) == {"detector", "recognizer"}
    assert fetched == ["face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"]


class FakeYuNet:
    @staticmethod
    def create(*args):
        return FakeFinder(None)


def test_recognition_alone_does_not_need_the_contrib_module(tmp_path, monkeypatch):
    monkeypatch.delattr(cv2, "face", raising=False)
    monkeypatch.setattr(cv2, "FaceDetectorYN", FakeYuNet)
    monkeypatch.setattr(cv2, "FaceRecognizerSF", type("SF", (), {"create": staticmethod(lambda *a: "sface")}))
    det = FaceDetector(fill(tmp_path), landmarks=False, recognition=True)
    assert det._recognizer == "sface" and det._facemark is None


def test_damaged_recognizer_model_is_a_clean_error(tmp_path, monkeypatch):
    """Real OpenCV loading a junk SFace model must not crash the app."""
    monkeypatch.setattr(cv2, "FaceDetectorYN", FakeYuNet)
    with pytest.raises(RuntimeError, match="delete it and run again"):
        FaceDetector(fill(tmp_path), landmarks=False, recognition=True)


def test_find_face_returns_frame_coordinates():
    row = np.array([yunet_row(100, 50, 80, 100)], np.float32)
    row[0, 4:14] = np.arange(10)  # five key points
    face = make_detector(row).find_face(np.zeros((720, 1280, 3), np.uint8))
    assert face[:4].tolist() == [200, 100, 160, 200]  # detection ran at half size
    assert face[4:14].tolist() == (np.arange(10) * 2).tolist()
    assert face[14] == pytest.approx(0.9)  # the score is not scaled


class FakeRecognizer:
    def __init__(self):
        self.seen = []

    def alignCrop(self, frame, face):
        self.seen.append((frame.shape, face.tolist()))
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, aligned):
        return np.full((1, 128), 3.0, np.float32)


def test_embedding_is_a_unit_vector_from_the_aligned_face():
    det = make_detector(None)
    det._recognizer = FakeRecognizer()
    face = synthetic_face()
    vector = det.embedding(np.zeros((720, 1280, 3), np.uint8), face)
    assert vector.shape == (128,)
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert det._recognizer.seen == [((720, 1280, 3), face.tolist())]


def test_models_that_were_not_loaded_give_a_clear_error():
    det = make_detector(None, None)
    det._facemark = det._recognizer = None
    frame, face = np.zeros((72, 128, 3), np.uint8), synthetic_face()
    with pytest.raises(RuntimeError, match="recognition=False"):
        det.embedding(frame, face)
    with pytest.raises(RuntimeError, match="landmarks=False"):
        det.landmarks(frame, face)


def test_square_box_is_centred_on_the_face():
    assert square_box([100, 50, 60, 80]) == [90, 50, 80, 80]


def test_a_straight_close_face_is_good():
    assert face_problem(synthetic_face()) is None
    assert face_problem(synthetic_face(project(yaw=8, roll=6))) is None


def test_eye_order_does_not_matter():
    face = synthetic_face()
    face[4:6], face[6:8] = face[6:8].copy(), face[4:6].copy()
    assert face_problem(face) is None


@pytest.mark.parametrize(
    "face, problem",
    [
        (None, "No face detected"),
        (synthetic_face(project(distance_mm=3000)), "Move closer to the camera"),
        (synthetic_face(project(yaw=30)), "Look straight at the camera"),
        (synthetic_face(project(yaw=-30)), "Look straight at the camera"),
        (synthetic_face(project(roll=25)), "Look straight at the camera"),
    ],
    ids=["no face", "far away", "turned left", "turned right", "tilted"],
)
def test_unsuitable_faces_get_an_instruction(face, problem):
    assert face_problem(face) == problem
