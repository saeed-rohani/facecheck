"""The facecheck command line.

  facecheck register NAME     register a new person with the camera
  facecheck check             who is in front of the camera?
  facecheck check --name NAME is this NAME?
  facecheck list | delete NAME | export | measure
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import __version__, measure
from . import detector as detection
from .analysis import AVERAGE_IPD_MM, DEFAULT_HFOV_DEG, analyze, to_json
from .camera import AMBER, GREEN, RED, WHITE, Camera
from .identity import DEFAULT_THRESHOLD, decide, mean_embedding, rank
from .storage import Store, utc_now

DATA_ENV = "FACECHECK_HOME"
DB_NAME = "facecheck.db"
SAMPLE_EVERY = 3  # while registering, keep every 3rd good frame so samples differ a little
CHECK_SAMPLES = 3  # a check averages this many consecutive good frames
LANDMARK_FRAMES = 8  # landmark sets averaged for the registration measurement
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_MATCH = 0, 1, 2, 3


def default_data_dir():
    """Where the database lives unless --data or FACECHECK_HOME says otherwise."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "facecheck"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "facecheck"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "facecheck"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    validate(parser, args)
    args.data = Path(args.data or os.environ.get(DATA_ENV) or default_data_dir()).expanduser()
    args.db = args.data / DB_NAME
    handlers = {
        "register": run_register,
        "check": run_check,
        "list": run_list,
        "delete": run_delete,
        "export": run_export,
        "measure": run_measure,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        log("Cancelled.")
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        log(f"Error: {error}")
    except cv2.error as error:
        log(f"Error from OpenCV: {error}")
        log("If this is about windows or GUI support, run with --no-window.")
    return EXIT_ERROR


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", metavar="DIR", help=f"folder for the database (default: {default_data_dir()}, "
                        f"or set {DATA_ENV})")  # fmt: skip
    common.add_argument("--model-dir", metavar="DIR", help="folder holding your own copies of the model files "
                        "(default: downloaded once and cached)")  # fmt: skip

    camera = argparse.ArgumentParser(add_help=False)
    camera.add_argument("--camera", type=int, default=0, metavar="N", help="camera index (default: 0)")
    camera.add_argument("--no-window", action="store_true", help="no preview window")
    camera.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    camera.add_argument("--timeout", type=float, default=60.0, metavar="SEC",
                        help="give up after this many seconds without a result (default: 60)")  # fmt: skip

    scale = argparse.ArgumentParser(add_help=False)
    scale.add_argument("--ipd", type=float, metavar="MM", help="the person's pupil distance in mm, which sets "
                       f"the millimetre scale (default: the adult average, {AVERAGE_IPD_MM:g}; 0 = pixels only)")  # fmt: skip
    scale.add_argument("--hfov", type=float, metavar="DEG", help="camera horizontal field of view "
                       f"(default: {DEFAULT_HFOV_DEG:g}; 0 = no perspective correction)")  # fmt: skip

    parser = argparse.ArgumentParser(
        prog="facecheck",
        description="Register people with the camera, then check who is in front of it.",
        epilog="Run 'facecheck COMMAND --help' for a command's options. "
        "Exit codes: 0 = success or match, 1 = error, 2 = usage error, 3 = no match.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    register = commands.add_parser("register", parents=[common, camera, scale],
                                   help="register a new person with the camera")  # fmt: skip
    register.add_argument("name", help="the person's name, e.g. \"Saeed Rohani\"")
    register.add_argument("--replace", action="store_true", help="replace a person already registered under this name")
    register.add_argument("--samples", type=int, default=5, metavar="N",
                          help="face samples to store (default: 5; more makes matching sturdier)")  # fmt: skip

    check = commands.add_parser("check", parents=[common, camera],
                                help="check whether the person in front of the camera is registered")  # fmt: skip
    check.add_argument("--name", help="check against this one person only, instead of everyone")
    check.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, metavar="T",
                       help=f"similarity needed for a match, 0 to 1 (default: {DEFAULT_THRESHOLD})")  # fmt: skip

    commands.add_parser("list", parents=[common], help="list registered people")
    delete = commands.add_parser("delete", parents=[common], help="remove a registered person")
    delete.add_argument("name")
    export = commands.add_parser("export", parents=[common], help="export people, checks and measurements to CSV")
    export.add_argument("--to", default=".", metavar="DIR", help="folder for the CSV files (default: here)")

    m = commands.add_parser("measure", parents=[common, camera, scale],
                            help="measure a face and estimate its shape, without registering anyone")  # fmt: skip
    m.add_argument("--image", metavar="FILE", help="measure a photo instead of the camera")
    m.add_argument("--label", help="free-text label stored with each result")
    m.add_argument("--once", action="store_true", help="save one result automatically when the face is steady")
    m.add_argument("--landmarks", action="store_true", help="include all 68 raw landmarks in the JSON")
    return parser


def validate(parser, args):
    hfov, ipd = getattr(args, "hfov", None), getattr(args, "ipd", None)
    if hfov is not None and not 0 <= hfov < 180:
        parser.error("--hfov must be between 0 and 180 degrees")
    if ipd is not None and not (ipd == 0 or 40 <= ipd <= 85):
        parser.error("--ipd must be between 40 and 85 mm, or 0")
    if getattr(args, "timeout", 1) <= 0:
        parser.error("--timeout must be more than 0")
    if not 1 <= getattr(args, "samples", 1) <= 50:
        parser.error("--samples must be between 1 and 50")
    if not 0 < getattr(args, "threshold", 0.5) < 1:
        parser.error("--threshold must be between 0 and 1")
    if hasattr(args, "name") and args.name is not None:
        args.name = clean_name(args.name, parser)


def clean_name(name, parser):
    name = " ".join(unicodedata.normalize("NFC", name).split())
    if not name:
        parser.error("the name must not be empty")
    if len(name) > 100 or any(unicodedata.category(c).startswith("C") for c in name):
        parser.error("the name must be at most 100 printable characters")
    return name


# --- register ----------------------------------------------------------------


def run_register(args):
    with Store(args.db) as store:
        if store.get_user(args.name) and not args.replace:
            log(f"Error: '{args.name}' is already registered. Add --replace to register them again.")
            return EXIT_ERROR
        others = {n: e for n, e in store.gallery().items() if n.casefold() != args.name.casefold()}

        with detection.FaceDetector(args.model_dir, landmarks=True, recognition=True) as detector, \
                camera_for(args) as camera:  # fmt: skip
            captured = capture_samples(args, detector, camera)
            if captured is None:
                return EXIT_ERROR
            frame, face, samples, landmark_sets = captured
            camera.hold(frame, [(f"Captured {args.name}", GREEN)], face, GREEN, seconds=1.5)

        measurement = measure_registration(args, frame.shape, landmark_sets)
        query = mean_embedding(samples)
        similar = None
        if others:
            best, score = rank(query, others)[0]
            if score >= DEFAULT_THRESHOLD:
                similar = {"name": best, "similarity": round(score, 4)}
                log(f"Warning: this face looks like '{best}', who is already registered (similarity {score:.2f}).")
        user = store.add_user(args.name, samples, measurement, replace=args.replace)

    result = {"registered": True, **user, "similar_to": similar, "measurement": measurement}
    print(to_json(result), flush=True)
    shape = f", face shape {measurement['face_shape']['label']}" if measurement else ""
    log(f"Registered {user['name']} with {user['samples']} face samples{shape}. Database: {args.db}")
    return EXIT_OK


def capture_samples(args, detector, camera):
    """Collect face samples and landmark sets. Returns (frame, face, samples, landmarks) or None."""
    samples, landmark_sets, good = [], deque(maxlen=LANDMARK_FRAMES), 0
    log(f"Registering {args.name}: look straight at the camera and hold still. Q cancels.")
    for frame in camera.frames():
        face = detector.find_face(frame)
        problem = detection.face_problem(face)
        if problem is None:
            if good % SAMPLE_EVERY == 0:
                samples.append(detector.embedding(frame, face))
            good += 1
            points = detector.landmarks(frame, face)
            if points is not None:
                landmark_sets.append(points)
        elif face is None:
            good = 0  # someone left: do not mix their samples with the next person's
            samples.clear()
            landmark_sets.clear()
        status = problem or f"Hold still... {len(samples)}/{args.samples}"
        lines = [(f"Registering {args.name}", WHITE), (status, AMBER if problem else GREEN)]
        if len(samples) >= args.samples:
            return frame, face, samples, list(landmark_sets)
        if not camera.show(frame, lines, face, AMBER if problem else GREEN):
            log("Cancelled; nothing was saved.")
            return None
    log("No usable face was seen in time. Face the camera in good light and try again.")
    return None


def measure_registration(args, frame_shape, landmark_sets):
    """The face measurement saved with a new person, or None if it failed."""
    if not landmark_sets:
        log("Warning: no landmarks were found, so no measurements were saved.")
        return None
    height, width = frame_shape[:2]
    hfov = DEFAULT_HFOV_DEG if args.hfov is None else (args.hfov or None)
    try:
        return analyze(
            np.mean(landmark_sets, axis=0),
            (width, height),
            ipd_mm=args.ipd,
            hfov_deg=hfov,
            label=args.name,
            source=f"register:camera:{args.camera}",
            frames_averaged=len(landmark_sets),
        )
    except ValueError as error:
        log(f"Warning: the measurements failed ({error}), so none were saved.")
        return None


# --- check -------------------------------------------------------------------


def run_check(args):
    if not args.db.is_file():
        log("Error: nobody is registered yet. Register someone first: facecheck register \"NAME\"")
        return EXIT_ERROR
    with Store(args.db) as store:
        claimed = None
        if args.name is not None:
            user = store.get_user(args.name)
            if user is None:
                log(f"Error: '{args.name}' is not registered. See who is: facecheck list")
                return EXIT_ERROR
            claimed = user["name"]
        gallery = store.gallery()
        if not gallery:
            log("Error: nobody is registered yet. Register someone first: facecheck register \"NAME\"")
            return EXIT_ERROR

        with detection.FaceDetector(args.model_dir, landmarks=False, recognition=True) as detector, \
                camera_for(args) as camera:  # fmt: skip
            captured = capture_check(args, detector, camera, claimed)
            if captured is None:
                return EXIT_ERROR
            frame, face, samples = captured
            result = decide(mean_embedding(samples), gallery, args.threshold, claimed=claimed)
            result = {
                "id": uuid.uuid4().hex,
                "checked_at": utc_now(),
                **result,
                "frames_used": len(samples),
                "source": f"camera:{args.camera}",
            }
            camera.hold(frame, [verdict(result)], face, GREEN if result["matched"] else RED)
        store.log_check(result)

    print(to_json(result), flush=True)
    log(verdict(result)[0])
    return EXIT_OK if result["matched"] else EXIT_NO_MATCH


def capture_check(args, detector, camera, claimed):
    """Collect a few consecutive good samples. Returns (frame, face, samples) or None."""
    samples = []
    title = f"Checking: is this {claimed}?" if claimed else "Checking who this is"
    log(f"{title}{'' if title.endswith('?') else '.'} Look straight at the camera. Q cancels.")
    for frame in camera.frames():
        face = detector.find_face(frame)
        problem = detection.face_problem(face)
        if problem is None:
            samples.append(detector.embedding(frame, face))
            if len(samples) >= CHECK_SAMPLES:
                return frame, face, samples
        else:
            samples.clear()  # only consecutive frames of one steady face count
        lines = [(title, WHITE), (problem or "Hold still...", AMBER if problem else GREEN)]
        if not camera.show(frame, lines, face, AMBER if problem else GREEN):
            log("Cancelled.")
            return None
    log("No usable face was seen in time. Face the camera in good light and try again.")
    return None


def verdict(result):
    """A one-line summary of a check, as (text, colour)."""
    score = f"similarity {result['similarity']:.2f}, needed {result['threshold']:.2f}"
    if result["mode"] == "verify":
        if result["matched"]:
            return f"Confirmed: this is {result['claimed_name']} ({score})", GREEN
        return f"Not confirmed: this does not look like {result['claimed_name']} ({score})", RED
    if result["matched"]:
        return f"Match: {result['name']} ({score})", GREEN
    return f"No match: this person is not registered ({score})", RED


def camera_for(args):
    return Camera(args.camera, window=not args.no_window, mirror=not args.no_mirror, timeout=args.timeout)


# --- list, delete, export, measure -------------------------------------------


def run_list(args):
    if not args.db.is_file():
        print("Nobody is registered yet.")
        return EXIT_OK
    with Store(args.db) as store:
        people = store.users()
    if not people:
        print("Nobody is registered yet.")
    for person in people:
        shape = f"   {person['face_shape']}" if person["face_shape"] else ""
        print(f"{person['name']}   registered {person['registered_at']}   {person['samples']} samples{shape}")
    return EXIT_OK


def run_delete(args):
    if args.db.is_file():
        with Store(args.db) as store:
            if store.delete_user(args.name):
                log(f"Deleted {args.name} and their face samples.")
                return EXIT_OK
    log(f"Error: '{args.name}' is not registered.")
    return EXIT_ERROR


def run_export(args):
    if not args.db.is_file():
        log(f"Error: no database at {args.db}")
        return EXIT_ERROR
    folder = Path(args.to)
    with Store(args.db) as store:
        counts = {
            "users.csv": store.export_users_csv(folder / "users.csv"),
            "checks.csv": store.export_checks_csv(folder / "checks.csv"),
            "measurements.csv": store.export_csv(folder / "measurements.csv"),
        }
    for filename, rows in counts.items():
        log(f"Exported {rows} row(s) to {folder / filename}")
    return EXIT_OK


def run_measure(args):
    args.json_dir = args.data / "json"
    args.csv = args.data / "measurements.csv"
    return measure.run(args)


def log(message):
    """Status messages go to stderr so stdout carries only results."""
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
