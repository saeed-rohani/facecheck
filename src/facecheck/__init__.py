"""facecheck: register people with a camera, then check who is in front of it.

Only OpenCV and NumPy are needed; the face models download on first use.
"""

from importlib.metadata import version

from .analysis import analyze, classify_shape, to_json
from .detector import FaceDetector, face_problem
from .identity import DEFAULT_THRESHOLD, decide, mean_embedding, rank, similarity
from .storage import Store

try:
    __version__ = version("facecheck")
except Exception:  # not installed (a source checkout), or its metadata is unreadable
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_THRESHOLD",
    "FaceDetector",
    "Store",
    "analyze",
    "classify_shape",
    "decide",
    "face_problem",
    "mean_embedding",
    "rank",
    "similarity",
    "to_json",
    "__version__",
]
