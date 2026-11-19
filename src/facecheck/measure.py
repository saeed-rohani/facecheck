"""The `facecheck measure` command: measure a face without registering anyone."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import camera as cam
from . import detector as detection
from .analysis import DEFAULT_HFOV_DEG, analyze, to_json
from .camera import AMBER, GREEN, WHITE, draw_overlay, window_visible
from .storage import Store

SMOOTH_FRAMES = 8  # landmarks are averaged over this many frames to cut jitter
STEADY_FRAMES = 15  # --once waits for this many steady, front-facing frames


def run(args):
    if args.image:
        return run_image(args)
    # The detector comes first so a missing model fails before the camera turns on.
    with detection.FaceDetector(args.model_dir) as detector:
        capture = cam.open_camera(args.camera)
        with Store(args.db) as store:
            try:
                saved, exit_code = camera_loop(args, detector, capture, store)
            finally:
                capture.release()
                if not args.no_window:
                    cv2.destroyAllWindows()
            if saved:
                export(store, args.csv)
    return exit_code


def run_image(args):
    frame = cv2.imread(args.image)
    if frame is None:
        log(f"Error: could not read image {args.image}")
        return 1
    with detection.FaceDetector(args.model_dir) as detector:
        landmarks = detector.detect(frame)
    if landmarks is None:
        log("No face found in the image.")
        return 1
    height, width = frame.shape[:2]
    result = analyze(
        landmarks,
        (width, height),
        ipd_mm=args.ipd,
        hfov_deg=args.hfov or None,  # a photo's field of view is unknown unless given
        label=args.label,
        source=f"image:{Path(args.image).name}",
        include_landmarks=args.landmarks,
    )
    with Store(args.db) as store:
        save(result, store, args.json_dir)
        export(store, args.csv)
    return 0


def camera_loop(args, detector, capture, store):
    """Read frames until the user quits. Returns (results saved, exit code)."""
    show = not args.no_window
    once = args.once or args.no_window
    hfov = DEFAULT_HFOV_DEG if args.hfov is None else (args.hfov or None)
    recent = deque(maxlen=SMOOTH_FRAMES)
    saved = steady = bad_reads = 0
    was_visible = False
    notice, notice_until = "", 0.0
    fps, started = 0.0, time.monotonic()
    last_frame = started
    log("Hold still and face the camera." if once else "SPACE = save, E = export CSV, Q = quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                bad_reads += 1
                if bad_reads > cam.MAX_BAD_READS:
                    log("Error: the camera stopped delivering frames.")
                    return saved, 1
                continue
            bad_reads = 0
            height, width = frame.shape[:2]

            landmarks = detector.detect(frame)
            smooth = result = None
            if landmarks is None:
                recent.clear()
            else:
                recent.append(landmarks)
                smooth = np.mean(recent, axis=0)
                try:
                    result = analyze(
                        smooth,
                        (width, height),
                        ipd_mm=args.ipd,
                        hfov_deg=hfov,
                        label=args.label,
                        source=f"camera:{args.camera}",
                        frames_averaged=len(recent),
                        include_landmarks=args.landmarks,
                    )
                except ValueError:  # unusable landmarks, or no head pose fits them
                    result = None
            ready = bool(result) and result["pose"]["frontal"] and len(recent) == SMOOTH_FRAMES
            steady = steady + 1 if ready else 0

            if once:
                if steady >= STEADY_FRAMES:
                    save(result, store, args.json_dir)
                    return saved + 1, 0
                if time.monotonic() - started > args.timeout:
                    log("No steady, front-facing face was found in time.")
                    return saved, 1

            if show:
                now = time.monotonic()
                rate = 1.0 / max(now - last_frame, 1e-6)
                fps = rate if fps == 0 else 0.9 * fps + 0.1 * rate
                last_frame = now
                lines = status_lines(result, notice if now < notice_until else "")
                footer = f"SPACE save   E export CSV   Q quit   |   saved {saved}   {fps:.0f} fps"
                cv2.imshow(cam.WINDOW, draw_overlay(frame, smooth, lines, footer, not args.no_mirror))
                key = chr(cv2.waitKey(1) & 0xFF).lower()
                if key in ("q", "\x1b"):  # \x1b is ESC
                    break
                if key in (" ", "s"):
                    if result:
                        save(result, store, args.json_dir)
                        saved += 1
                        notice = f"Saved result {saved}"
                    else:
                        notice = "No face to save"
                    notice_until = now + 1.5
                elif key == "e":
                    export(store, args.csv)
                    notice, notice_until = "CSV exported", now + 1.5

                # Stop when the user closes the window with its own close button.
                visible = window_visible()
                if visible >= 1:
                    was_visible = True
                elif visible == 0 and was_visible:
                    break
    except KeyboardInterrupt:
        pass
    return saved, 0


def save(result, store, json_dir):
    """Store a result in SQLite, write it as a JSON file and print it."""
    store.save(result)
    json_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["captured_at"].replace("-", "").replace(":", "")
    path = json_dir / f"{stamp}_{result['id'][:8]}.json"
    text = to_json(result)
    path.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    shape = result["face_shape"]
    log(f"Saved: {shape['label']} ({shape['confidence']:.0%}) -> {path}")


def export(store, csv_path):
    rows = store.export_csv(csv_path)
    log(f"Exported {rows} measurement(s) to {csv_path}")


def status_lines(result, notice=""):
    """Text for the top-left of the preview, as (text, colour) pairs."""
    if result is None:
        lines = [("No face detected", AMBER)]
    else:
        shape = result["face_shape"]
        values = result["measurements_mm"] or result["measurements_px"]
        unit = "mm" if result["measurements_mm"] else "px"
        lines = [
            (f"Shape: {shape['label']} ({shape['confidence']:.0%})", GREEN),
            (f"Length {values['face_length']:.0f} {unit}   Width {values['face_width']:.0f} {unit}", WHITE),
            (f"Forehead {values['forehead_width']:.0f} {unit}   Jaw {values['jaw_width']:.0f} {unit}", WHITE),
        ]
        if not result["pose"]["frontal"]:
            lines.append(("Face the camera straight on", AMBER))
    if notice:
        lines.append((notice, GREEN))
    return lines


def log(message):
    print(message, file=sys.stderr, flush=True)
