import csv
import json
import sqlite3

from conftest import project

from facecheck.analysis import analyze
from facecheck.storage import COLUMNS, TABLE, Store, flatten


def make_result(**kwargs):
    return analyze(project(), (1280, 720), **kwargs)


def test_save_and_count(tmp_path):
    with Store(tmp_path / "nested" / "results.db") as store:
        assert store.count() == 0
        store.save(make_result(label="a"))
        store.save(make_result(label="b"))
        assert store.count() == 2


def test_results_persist_and_keep_their_json(tmp_path):
    path = tmp_path / "results.db"
    result = make_result(label="kept")
    with Store(path) as store:
        store.save(result)
    with Store(path) as store:
        assert store.count() == 1
    row = sqlite3.connect(path).execute(f"SELECT face_shape, result_json FROM {TABLE}").fetchone()
    assert row[0] == result["face_shape"]["label"]
    assert json.loads(row[1]) == result


def test_csv_export_matches_what_was_saved(tmp_path):
    first, second = make_result(label="one"), make_result(label="two, with comma")
    with Store(tmp_path / "results.db") as store:
        store.save(first)
        store.save(second)
        assert store.export_csv(tmp_path / "out" / "results.csv") == 2

    with (tmp_path / "out" / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(COLUMNS)
    assert {r["label"] for r in rows} == {"one", "two, with comma"}
    by_id = {r["id"]: r for r in rows}
    row = by_id[first["id"]]
    assert row["face_shape"] == first["face_shape"]["label"]
    assert float(row["face_width_mm"]) == first["measurements_mm"]["face_width"]
    assert float(row["length_to_width"]) == first["ratios"]["length_to_width"]
    assert row["frontal"] == "1"
    assert row["schema_version"] == "2"
    assert row["scale_method"] == "average_ipd"


def test_missing_millimetres_become_empty_cells(tmp_path):
    result = analyze(project(), (1280, 720), ipd_mm=0)
    assert flatten(result)["face_width_mm"] is None
    assert flatten(result)["scale_method"] is None
    with Store(tmp_path / "results.db") as store:
        store.save(result)
        store.export_csv(tmp_path / "results.csv")
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["face_width_mm"] == ""
    assert float(row["face_width_px"]) > 0


def test_older_database_gains_new_columns(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(f"CREATE TABLE {TABLE} (id TEXT PRIMARY KEY, captured_at TEXT, result_json TEXT NOT NULL)")
    old.commit()
    old.close()
    with Store(path) as store:
        store.save(make_result())
        assert store.export_csv(tmp_path / "results.csv") == 1


# --- People and checks ------------------------------------------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from facecheck.identity import normalize  # noqa: E402

rng = np.random.default_rng(3)


def faces(n=3):
    return [normalize(rng.normal(size=128)) for _ in range(n)]


def test_add_and_read_a_person(tmp_path):
    samples = faces()
    with Store(tmp_path / "f.db") as store:
        user = store.add_user("Saeed Rohani", samples, make_result(label="Saeed Rohani"))
        assert user["name"] == "Saeed Rohani" and user["samples"] == 3 and user["replaced"] is False
        assert store.get_user("saeed rohani")["face_shape"] == "oval"  # names ignore letter case
        stored = store.gallery()["Saeed Rohani"]
        assert stored.shape == (3, 128)
        assert np.allclose(stored, samples)
        assert store.count() == 1  # the measurement is saved too


def test_people_persist_between_sessions(tmp_path):
    with Store(tmp_path / "f.db") as store:
        store.add_user("A", faces(2))
    with Store(tmp_path / "f.db") as store:
        assert [u["name"] for u in store.users()] == ["A"]
        assert store.gallery("a")["A"].shape == (2, 128)


def test_a_name_can_only_be_registered_once(tmp_path):
    with Store(tmp_path / "f.db") as store:
        store.add_user("Alice", faces(), make_result())
        with pytest.raises(ValueError, match="already registered"):
            store.add_user("ALICE", faces())
        assert store.count() == 1  # the failed attempt left nothing behind


def test_replace_swaps_the_samples_and_measurement(tmp_path):
    new = faces(2)
    with Store(tmp_path / "f.db") as store:
        store.add_user("Alice", faces(4), make_result())
        user = store.add_user("Alice", new, make_result(), replace=True)
        assert user["replaced"] is True
        assert np.allclose(store.gallery()["Alice"], new)
        assert store.count() == 1  # the old measurement went with the old record


def test_delete_removes_everything_about_a_person(tmp_path):
    with Store(tmp_path / "f.db") as store:
        store.add_user("Alice", faces(), make_result())
        store.add_user("Bob", faces())
        assert store.delete_user("alice") is True
        assert store.delete_user("alice") is False
        assert list(store.gallery()) == ["Bob"]
        assert store.count() == 0
        leftover = store._db.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0]
        assert leftover == 3  # Bob's only


def test_empty_registration_is_refused(tmp_path):
    with Store(tmp_path / "f.db") as store, pytest.raises(ValueError):
        store.add_user("Nobody", [])


def test_checks_are_logged_and_exported(tmp_path):
    check = {"id": "c1", "checked_at": "2026-11-18T10:00:00Z", "mode": "verify", "claimed_name": "Alice",
             "matched": True, "name": "Alice", "similarity": 0.71, "threshold": 0.363, "source": "camera:0"}
    with Store(tmp_path / "f.db") as store:
        store.log_check(check)
        assert store.export_checks_csv(tmp_path / "checks.csv") == 1
    with (tmp_path / "checks.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["matched"] == "1" and row["name"] == "Alice" and float(row["similarity"]) == 0.71


def test_users_csv_has_measurements_but_never_embeddings(tmp_path):
    with Store(tmp_path / "f.db") as store:
        store.add_user("Alice", faces(), make_result())
        store.add_user("Bob", faces())  # registered without a measurement
        assert store.export_users_csv(tmp_path / "users.csv") == 2
    with (tmp_path / "users.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert not any("embedding" in column for column in rows[0])
    assert rows[0]["name"] == "Alice" and rows[0]["face_shape"] == "oval" and float(rows[0]["face_width_mm"]) > 100
    assert rows[1]["name"] == "Bob" and rows[1]["face_shape"] == ""
