"""Camera frames, the preview window, and drawing on it."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from .analysis import SEGMENTS, extend

WINDOW = "facecheck"
MAX_BAD_READS = 30  # consecutive failed camera reads before giving up

# Colours are BGR.
WHITE, GREEN, AMBER, RED, GREY = (255, 255, 255), (120, 220, 120), (60, 190, 255), (90, 90, 240), (200, 200, 200)
OUTLINES = [list(range(0, 17)), list(range(17, 22)), list(range(22, 27))]  # jaw, brows
GUIDES = {
    "face_length": (255, 200, 80),
    "face_width": (120, 220, 120),
    "forehead_width": (240, 150, 240),
    "jaw_width": (60, 190, 255),
}


def open_camera(index):
    # DirectShow opens much faster than the default backend on Windows.
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # show the newest frame, not a queued one
            return capture
        capture.release()
    raise RuntimeError(
        f"could not open camera {index}. Check that it is connected, not in use by "
        "another app, and that this terminal is allowed to use the camera."
    )


class Camera:
    """Frames from the camera, plus an optional mirrored preview window.

    Use as a context manager. frames() stops after `timeout` seconds and sets
    timed_out; show() returns False once the user presses Q or ESC or closes
    the window.
    """

    def __init__(self, index=0, window=True, mirror=True, timeout=60.0):
        self.index, self.window, self.mirror, self.timeout = index, window, mirror, timeout
        self.timed_out = False
        self._capture = None
        self._was_visible = False

    def __enter__(self):
        self._capture = open_camera(self.index)
        return self

    def __exit__(self, *exc_info):
        self._capture.release()
        if self.window:
            cv2.destroyAllWindows()

    def frames(self):
        started, bad_reads = time.monotonic(), 0
        while time.monotonic() - started <= self.timeout:
            ok, frame = self._capture.read()
            if not ok:
                bad_reads += 1
                if bad_reads > MAX_BAD_READS:
                    raise RuntimeError("the camera stopped delivering frames")
                continue
            bad_reads = 0
            yield frame
        self.timed_out = True

    def show(self, frame, lines, face=None, colour=WHITE, footer="Q quit"):
        """Show one preview frame. Returns False when the user wants to stop."""
        if not self.window:
            return True
        cv2.imshow(WINDOW, draw_face(frame, face, colour, lines, footer, self.mirror))
        key = chr(cv2.waitKey(1) & 0xFF).lower()
        if key in ("q", "\x1b"):  # \x1b is ESC
            return False
        visible = window_visible()
        if visible >= 1:
            self._was_visible = True
        elif visible == 0 and self._was_visible:  # closed with the window's own button
            return False
        return True

    def hold(self, frame, lines, face=None, colour=WHITE, seconds=2.5):
        """Keep a final frame on screen for a moment, or until a key is pressed."""
        if self.window:
            cv2.imshow(WINDOW, draw_face(frame, face, colour, lines, "Press any key", self.mirror))
            cv2.waitKey(int(seconds * 1000))


def draw_face(frame, face, colour, lines, footer="", mirror=True):
    """Draw the face box and its five key points, then the text on top."""
    height, width = frame.shape[:2]
    if face is not None:
        values = np.clip(np.asarray(face[:14], dtype=np.float64), -10 * width, 10 * width)
        x, y, w, h = (int(round(v)) for v in values[:4])
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2, cv2.LINE_AA)
        for px, py in values[4:14].reshape(5, 2).astype(np.int32).tolist():
            cv2.circle(frame, (px, py), 3, colour, -1)
    return _finish(frame, lines, footer, mirror)


def draw_overlay(frame, landmarks, lines, footer="", mirror=True):
    """Draw the 68 landmarks and the measurement guide lines, then the text."""
    height, width = frame.shape[:2]
    if landmarks is not None:
        # Clip so landmarks far off-screen cannot overflow the int32 drawing coordinates.
        pixels = np.clip(extend(landmarks) * (width, height), -10 * width, 10 * width)
        points = pixels.astype(np.int32)
        for x, y in points[:68].tolist():
            cv2.circle(frame, (x, y), 2, GREY, -1)
        cv2.polylines(frame, [points[line] for line in OUTLINES], False, WHITE, 1, cv2.LINE_AA)
        for name, colour in GUIDES.items():
            a, b = SEGMENTS[name][0]
            cv2.line(frame, tuple(points[a].tolist()), tuple(points[b].tolist()), colour, 2, cv2.LINE_AA)
    return _finish(frame, lines, footer, mirror)


def _finish(frame, lines, footer, mirror):
    view = cv2.flip(frame, 1) if mirror else frame  # mirror first so the text stays readable
    for row, (text, colour) in enumerate(lines):
        put_text(view, text, (14, 32 + 30 * row), colour, 0.7)
    if footer:
        put_text(view, footer, (14, view.shape[0] - 16), GREY, 0.5)
    return view


def put_text(image, text, origin, colour, size):
    """Text on a dark label, so it stays readable over any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, size, 1)
    x, y = origin
    cv2.rectangle(image, (x - 8, y - text_height - 8), (x + text_width + 8, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(image, text, origin, font, size, colour, 1, cv2.LINE_AA)


def window_visible():
    """1 if the preview is on screen, 0 if it was closed, -1 if OpenCV cannot tell."""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return -1
