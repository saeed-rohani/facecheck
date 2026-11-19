"""The app end to end, with the camera, the models and the window faked."""

import csv
import json

import cv2
import numpy as np
import pytest
from conftest import project, yunet_row

from facecheck import camera, cli
from facecheck import detector as detection
from facecheck.identity import normalize

rng = np.random.default_rng(11)
PEOPLE = {name: normalize(rng.normal(size=128)) for name in ("alice", "bob")}
REAL_OPEN_CAMERA = camera.open_camera  # saved before any test patches it


class FakeDetector:
    """Stands in for the real models. `person` is who is in front of the camera."""

    person = "alice"

    def __init__(self, model_dir=None, landmarks=True, recognition=False, **kwargs):
        pass

    def find_face(self, frame):
        return None if FakeDetector.person is None else yunet_row()

    def landmarks(self, frame, face):
        return project()

    def detect(self, frame):
        return None if FakeDetector.person is None else project()

    def embedding(self, frame, face):
        return normalize(PEOPLE[FakeDetector.person] + rng.normal(scale=0.03, size=128))

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass


class FakeCapture:
    def read(self):
        return True, np.full((720, 1280, 3), 90, np.uint8)

    def release(self):
        pass


class Window:
    """Records what the preview showed; `keys` are pressed one per frame."""

    def __init__(self):
        self.frames, self.keys = [], []

    def wait_key(self, delay):
        return ord(self.keys.pop(0)) if self.keys else 255


@pytest.fixture
def window(monkeypatch):
    win = Window()
    monkeypatch.setattr(cv2, "imshow", lambda name, image: win.frames.append(image))
    monkeypatch.setattr(cv2, "waitKey", win.wait_key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return win


@pytest.fixture
def app(tmp_path, monkeypatch, window):
    """Run the CLI with fakes and a fresh data folder; returns a function taking CLI args."""
    FakeDetector.person = "alice"
    monkeypatch.setattr(detection, "FaceDetector", FakeDetector)
    monkeypatch.setattr(camera, "open_camera", lambda index: FakeCapture())
    monkeypatch.setenv(cli.DATA_ENV, str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    return lambda *args: cli.main(list(args))


def output(capsys):
    """The JSON document a command printed (stdout carries nothing else)."""
    return json.loads(capsys.readouterr().out)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_register_then_check_finds_the_person(app, capsys, tmp_path):
    assert app("register", "Alice Smith") == 0
    registered = output(capsys)
    assert registered["registered"] is True
    assert registered["name"] == "Alice Smith"
    assert registered["samples"] == 5
    assert registered["similar_to"] is None
    assert registered["measurement"]["face_shape"]["label"] == "oval"
    assert registered["measurement"]["label"] == "Alice Smith"
    assert (tmp_path / "data" / "facecheck.db").is_file()

    assert app("check") == 0
    checked = output(capsys)
    assert checked["mode"] == "identify"
    assert checked["matched"] is True
    assert checked["name"] == "Alice Smith"
    assert checked["similarity"] > 0.8
    assert checked["frames_used"] == cli.CHECK_SAMPLES


def test_a_stranger_is_not_matched(app, capsys):
    app("register", "Alice")
    capsys.readouterr()
    FakeDetector.person = "bob"
    assert app("check") == cli.EXIT_NO_MATCH
    checked = output(capsys)
    assert checked["matched"] is False
    assert checked["name"] is None
    assert checked["candidates"][0]["name"] == "Alice"


def test_verify_against_one_named_person(app, capsys):
    app("register", "Alice")
    FakeDetector.person = "bob"
    app("register", "Bob")
    capsys.readouterr()

    assert app("check", "--name", "Alice") == cli.EXIT_NO_MATCH  # Bob is in front of the camera
    wrong = output(capsys)
    assert wrong["mode"] == "verify" and wrong["claimed_name"] == "Alice" and wrong["matched"] is False

    assert app("check", "--name", "bob") == 0  # names ignore letter case
    right = output(capsys)
    assert right["matched"] is True and right["name"] == "Bob"


def test_checks_need_registered_people(app, capsys, tmp_path):
    assert app("check") == 1
    assert "nobody is registered" in capsys.readouterr().err
    assert not (tmp_path / "data" / "facecheck.db").exists()  # a failed check leaves no files behind
    app("register", "Alice")
    capsys.readouterr()
    assert app("check", "--name", "Carol") == 1
    assert "'Carol' is not registered" in capsys.readouterr().err


def test_a_name_is_registered_once_unless_replaced(app, capsys):
    assert app("register", "Alice") == 0
    assert app("register", "ALICE") == 1
    assert "--replace" in capsys.readouterr().err
    assert app("register", "Alice", "--replace") == 0
    assert output(capsys)["replaced"] is True
    app("list")
    assert capsys.readouterr().out.count("samples") == 1


def test_registering_the_same_face_twice_warns(app, capsys):
    app("register", "Alice")
    capsys.readouterr()
    assert app("register", "Alicia") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["similar_to"]["name"] == "Alice"
    assert "looks like 'Alice'" in captured.err


def test_register_times_out_without_a_face(app, capsys):
    FakeDetector.person = None
    assert app("register", "Alice", "--no-window", "--timeout", "0.2") == 1
    assert "No usable face" in capsys.readouterr().err
    app("list")
    assert "Nobody is registered" in capsys.readouterr().out


def test_q_cancels_without_saving(app, window, capsys):
    window.keys = ["q"]
    FakeDetector.person = None  # so it is still waiting when Q is pressed
    assert app("register", "Alice") == 1
    assert "nothing was saved" in capsys.readouterr().err
    app("list")
    assert "Nobody is registered" in capsys.readouterr().out


def test_preview_shows_progress_then_the_result(app, window):
    app("register", "Alice")
    assert len(window.frames) >= 13  # 5 samples, one every 3rd good frame, then the final frame
    assert window.frames[0].shape == (720, 1280, 3)
    shown = len(window.frames)
    app("check")
    assert len(window.frames) > shown


def test_list_and_delete(app, capsys):
    app("register", "Alice")
    app("register", "Bob", "--samples", "2")
    capsys.readouterr()
    app("list")
    listing = capsys.readouterr().out.splitlines()
    assert listing[0].startswith("Alice") and "5 samples" in listing[0] and "oval" in listing[0]
    assert listing[1].startswith("Bob") and "2 samples" in listing[1]
    assert app("delete", "alice") == 0
    assert app("delete", "alice") == 1
    app("list")
    assert capsys.readouterr().out.startswith("Bob")


def test_export_writes_three_files(app, tmp_path):
    app("register", "Alice")
    app("check")
    assert app("export", "--to", "out") == 0
    users = read_csv(tmp_path / "out" / "users.csv")
    assert [u["name"] for u in users] == ["Alice"]
    assert not any("embedding" in column for column in users[0])
    assert float(users[0]["face_width_mm"]) > 100
    checks = read_csv(tmp_path / "out" / "checks.csv")
    assert checks[0]["matched"] == "1" and checks[0]["name"] == "Alice"
    assert len(read_csv(tmp_path / "out" / "measurements.csv")) == 1


def test_export_without_a_database(app, capsys):
    assert app("export") == 1
    assert "no database" in capsys.readouterr().err


def test_data_option_overrides_the_environment(app, tmp_path):
    app("register", "Alice", "--data", "elsewhere")
    assert (tmp_path / "elsewhere" / "facecheck.db").is_file()
    assert not (tmp_path / "data" / "facecheck.db").exists()


def test_measure_still_works(app, capsys, tmp_path):
    assert app("measure", "--once", "--no-window", "--label", "session 1") == 0
    printed = output(capsys)
    assert printed["label"] == "session 1" and printed["face_shape"]["label"] == "oval"
    assert len(read_csv(tmp_path / "data" / "measurements.csv")) == 1
    assert len(list((tmp_path / "data" / "json").glob("*.json"))) == 1


def test_measure_a_photo(app, capsys, tmp_path):
    photo = tmp_path / "portrait.png"
    cv2.imwrite(str(photo), np.full((720, 1280, 3), 90, np.uint8))
    assert app("measure", "--image", str(photo), "--landmarks") == 0
    printed = output(capsys)
    assert printed["source"] == "image:portrait.png" and len(printed["landmarks"]) == 68


def test_no_command_prints_help(app, capsys):
    assert app() == cli.EXIT_USAGE
    assert "register" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["", "   ", "x" * 101, "bad\x00name"])
def test_bad_names_are_refused(app, name):
    with pytest.raises(SystemExit) as stop:
        app("register", name)
    assert stop.value.code == 2


@pytest.mark.parametrize("option", [["--threshold", "1.5"], ["--timeout", "0"]])
def test_bad_options_are_refused(app, option):
    with pytest.raises(SystemExit):
        app("check", *option)


def test_names_are_tidied(app, capsys):
    app("register", "  Saeed   Rohani ")
    assert output(capsys)["name"] == "Saeed Rohani"


def test_unopenable_camera_is_a_clean_error(app, monkeypatch, capsys):
    class Closed:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(camera, "open_camera", REAL_OPEN_CAMERA)  # undo the fixture's fake
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Closed())
    assert app("register", "Alice", "--no-window") == 1
    assert "could not open camera 0" in capsys.readouterr().err
