"""Face detection, landmarks and identity embeddings with OpenCV only.

1. YuNet (OpenCV's face detector) finds the faces; the largest one is kept.
2. Facemark LBF (from opencv-contrib) places 68 landmarks, for measurements.
3. SFace (OpenCV's face recognizer) turns the face into a 128-number
   embedding, for telling people apart.

OpenCV publishes wheels that work on every Python 3 from 3.7 up, so this
runs wherever OpenCV installs. Each model file is downloaded the first time
a command needs it.
"""

from __future__ import annotations

import math
import os
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .identity import normalize


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    size_text: str
    min_bytes: int  # anything smaller is a failed or partial download


_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "detector": Model(
        "face_detection_yunet_2023mar.onnx",
        _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "about 0.2 MB",
        100_000,
    ),
    "landmarks": Model(
        "lbfmodel.yaml",
        "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
        "about 54 MB",
        20_000_000,
    ),
    "recognizer": Model(
        "face_recognition_sface_2021dec.onnx",
        _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "about 37 MB",
        30_000_000,
    ),
}
MODEL_DIR_ENV = "FACECHECK_MODEL_DIR"
DETECT_WIDTH = 640  # frames are shrunk to this width to find the face quickly

# A face is good enough to recognise when it is this big and this straight.
MIN_FACE_PX = 90
MAX_ROLL_DEG = 15.0
MAX_TURN = 0.18  # nose offset from the eyes' midpoint, as a fraction of eye distance (about 16 degrees)


def default_model_dir():
    """Where the downloaded models are cached on this machine."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "facecheck"


def ensure_models(model_dir=None, names=None):
    """Return {name: path} for the requested models, downloading on first use.

    An explicit folder (argument or the FACECHECK_MODEL_DIR environment
    variable) must already hold the files; only the default cache folder is
    downloaded to.
    """
    wanted = {name: MODELS[name] for name in (names or MODELS)}
    explicit = model_dir or os.environ.get(MODEL_DIR_ENV)
    if explicit:
        folder = Path(explicit).expanduser()
        paths = {name: folder / model.filename for name, model in wanted.items()}
        missing = [wanted[name] for name, path in paths.items() if not path.is_file()]
        if missing:
            listing = "\n".join(f"  {m.filename}: {m.url}" for m in missing)
            raise FileNotFoundError(f"model files missing from {folder}:\n{listing}")
        return paths

    folder = default_model_dir()
    paths = {}
    for name, model in wanted.items():
        path = folder / model.filename
        if not (path.is_file() and path.stat().st_size >= model.min_bytes):
            _download(model, path)
        paths[name] = path
    return paths


def _download(model, destination):
    print(
        f"Downloading {model.filename} ({model.size_text}, first run only) to {destination.parent}",
        file=sys.stderr,
        flush=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        _fetch(model.url, partial)
        if partial.stat().st_size < model.min_bytes:
            raise OSError("the download was incomplete")
        partial.replace(destination)
    except (OSError, subprocess.SubprocessError) as error:  # network, timeout, disk or curl errors
        partial.unlink(missing_ok=True)
        hint = ""
        if _is_certificate_error(error) and sys.platform == "darwin":
            hint = (
                "\nThis Python cannot check HTTPS certificates. To fix it for good, open the "
                "Python folder in Applications and run 'Install Certificates.command'."
            )
        raise RuntimeError(
            f"Could not download {model.filename} ({error}).{hint}\n"
            f"Or download the file yourself from {model.url}\n"
            f"into {destination.parent} and run again."
        ) from error


def _fetch(url, path):
    """Download url to path. Certificates are always checked.

    Some Python installs, notably python.org's on macOS, come without the
    list of trusted certificates, so every HTTPS download fails. The curl
    program that ships with macOS, Windows 10+ and most Linux systems uses
    the system's own list, so it is tried before giving up.
    """
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(path, "wb") as out:
            shutil.copyfileobj(response, out)
        return
    except urllib.error.URLError as error:
        curl = shutil.which("curl")
        if not (_is_certificate_error(error) and curl):
            raise
        first_error = error
    print("Python could not check the HTTPS certificate; trying the system's curl.", file=sys.stderr, flush=True)
    try:
        subprocess.run(
            [curl, "--fail", "--silent", "--show-error", "--location", "--retry", "2",
             "--output", str(path), url],
            check=True,
            timeout=900,
        )  # fmt: skip
    except subprocess.CalledProcessError as error:
        raise first_error from error  # report the original certificate problem


def _is_certificate_error(error):
    while error is not None:
        if isinstance(error, ssl.SSLCertVerificationError) or isinstance(
            getattr(error, "reason", None), ssl.SSLCertVerificationError
        ):
            return True
        error = error.__cause__
    return False


def face_problem(face):
    """None if the face suits recognition, otherwise a short instruction.

    `face` is a row from find_face(). The check uses YuNet's five key points:
    the eye line gives the head tilt, and the nose's offset from the eyes'
    midpoint shows how far the head is turned.
    """
    if face is None:
        return "No face detected"
    if face[2] < MIN_FACE_PX:
        return "Move closer to the camera"
    (ax, ay), (bx, by) = sorted([tuple(face[4:6]), tuple(face[6:8])])  # eyes, left to right in the image
    eye_distance = math.hypot(bx - ax, by - ay)
    if eye_distance < 1:
        return "Look straight at the camera"
    roll = math.degrees(math.atan2(by - ay, bx - ax))
    turn = (face[8] - (ax + bx) / 2) / eye_distance
    if abs(roll) > MAX_ROLL_DEG or abs(turn) > MAX_TURN:
        return "Look straight at the camera"
    return None


def square_box(face):
    """The square box around a face that the landmark model expects."""
    x, y, w, h = (float(v) for v in face[:4])
    side = max(w, h)
    left, top = x + w / 2 - side / 2, y + h / 2 - side / 2
    return [int(round(left)), int(round(top)), int(round(side)), int(round(side))]


class FaceDetector:
    """Finds the most prominent face in a frame; gives its landmarks and embedding.

    landmarks=True loads the 68-point model (for measurements);
    recognition=True loads the identity model (for register and check).
    """

    def __init__(self, model_dir=None, landmarks=True, recognition=False, min_confidence=0.6):
        if landmarks and not hasattr(cv2, "face"):
            raise RuntimeError(
                "OpenCV's face module is missing. It comes with opencv-contrib-python. "
                "If opencv-python or opencv-python-headless is also installed, they hide it: run\n"
                "  pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python\n"
                "  pip install --only-binary=:all: opencv-contrib-python"
            )
        names = ["detector"] + ["landmarks"] * landmarks + ["recognizer"] * recognition
        paths = ensure_models(model_dir, names)
        self._facemark = self._recognizer = None
        try:
            self._finder = cv2.FaceDetectorYN.create(
                str(paths["detector"]), "", (320, 320), min_confidence, 0.3, 5000
            )
            if landmarks:
                self._facemark = cv2.face.createFacemarkLBF()
                self._facemark.loadModel(str(paths["landmarks"]))
            if recognition:
                self._recognizer = cv2.FaceRecognizerSF.create(str(paths["recognizer"]), "")
        except cv2.error as error:
            raise RuntimeError(
                f"could not load the face models in {paths['detector'].parent} ({error}). "
                "If a file is damaged, delete it and run again to download a fresh copy."
            ) from error

    def find_face(self, frame_bgr):
        """Return the largest face as 15 numbers in frame pixels, or None.

        Layout (YuNet's): box x, y, w, h; then x, y of the right eye, left
        eye, nose tip, right and left mouth corners; then the score.
        """
        width = frame_bgr.shape[1]
        scale = min(1.0, DETECT_WIDTH / width)
        small = frame_bgr if scale == 1.0 else cv2.resize(
            frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        self._finder.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._finder.detect(small)
        if faces is None or len(faces) == 0:
            return None
        face = np.array(max(faces, key=lambda f: f[2] * f[3]), dtype=np.float32)
        face[:14] /= scale
        return face

    def landmarks(self, frame_bgr, face):
        """Return (68, 2) normalised landmarks for a face from find_face(), or None."""
        if self._facemark is None:
            raise RuntimeError("this FaceDetector was created with landmarks=False")
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        ok, shapes = self._facemark.fit(gray, np.array([square_box(face)], dtype=np.int32))
        if not ok or len(shapes) == 0:
            return None
        points = np.asarray(shapes[0], dtype=np.float32).reshape(-1, 2)
        if len(points) != 68:
            return None
        return points / np.array([width, height], dtype=np.float32)

    def embedding(self, frame_bgr, face):
        """Return the face's identity embedding: 128 numbers of unit length."""
        if self._recognizer is None:
            raise RuntimeError("this FaceDetector was created with recognition=False")
        aligned = self._recognizer.alignCrop(frame_bgr, face)
        return normalize(self._recognizer.feature(aligned))

    def detect(self, frame_bgr):
        """Find the largest face and return its (68, 2) landmarks, or None."""
        face = self.find_face(frame_bgr)
        return None if face is None else self.landmarks(frame_bgr, face)

    def close(self):
        self._finder = self._facemark = self._recognizer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
