"""SQLite storage and CSV export.

Tables:
  measurements  one row per face measurement (flat columns + full JSON)
  users         one row per registered person
  face_samples  the identity embeddings of each person (several per person)
  checks        a log of every check: who was claimed, who matched, the score
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analysis import RATIOS, SEGMENTS

TABLE = "measurements"


def _columns():
    columns = {
        "id": "TEXT PRIMARY KEY",
        "schema_version": "INTEGER",
        "captured_at": "TEXT",
        "label": "TEXT",
        "source": "TEXT",
        "image_width": "INTEGER",
        "image_height": "INTEGER",
        "frames_averaged": "INTEGER",
        "face_shape": "TEXT",
        "shape_confidence": "REAL",
        "yaw_deg": "REAL",
        "pitch_deg": "REAL",
        "roll_deg": "REAL",
        "frontal": "INTEGER",
        "scale_method": "TEXT",
        "ipd_mm": "REAL",
        "mm_per_px": "REAL",
        "camera_distance_mm": "REAL",
        "jaw_angle_deg": "REAL",
    }
    columns.update({name: "REAL" for name in RATIOS})
    columns.update({f"{name}_mm": "REAL" for name in SEGMENTS})
    columns.update({f"{name}_px": "REAL" for name in SEGMENTS})
    return columns


COLUMNS = _columns()  # column name -> SQL type, in CSV order
CHECK_COLUMNS = ["id", "checked_at", "mode", "claimed_name", "matched", "name", "similarity", "threshold", "source"]
# users.csv repeats each person's measurement, without the measurement's own bookkeeping
USER_MEASUREMENT_COLUMNS = [c for c in COLUMNS if c not in ("id", "schema_version", "label", "source")]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    registered_at TEXT NOT NULL,
    samples INTEGER NOT NULL,
    measurement_id TEXT
);
CREATE TABLE IF NOT EXISTS face_samples (
    user_id INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS face_samples_user ON face_samples (user_id);
CREATE TABLE IF NOT EXISTS checks (
    id TEXT PRIMARY KEY,
    checked_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    claimed_name TEXT,
    matched INTEGER NOT NULL,
    name TEXT,
    similarity REAL,
    threshold REAL,
    source TEXT,
    result_json TEXT NOT NULL
);
"""


def flatten(result):
    """Flatten a nested measurement result into one row keyed by COLUMNS."""
    scale = result.get("scale") or {}
    millimetres = result.get("measurements_mm") or {}
    row = {
        "id": result["id"],
        "schema_version": result.get("schema_version"),
        "captured_at": result["captured_at"],
        "label": result.get("label"),
        "source": result.get("source"),
        "image_width": result["image"]["width"],
        "image_height": result["image"]["height"],
        "frames_averaged": result.get("frames_averaged"),
        "face_shape": result["face_shape"]["label"],
        "shape_confidence": result["face_shape"]["confidence"],
        "yaw_deg": result["pose"]["yaw_deg"],
        "pitch_deg": result["pose"]["pitch_deg"],
        "roll_deg": result["pose"]["roll_deg"],
        "frontal": int(result["pose"]["frontal"]),
        "scale_method": scale.get("method"),
        "ipd_mm": scale.get("ipd_mm"),
        "mm_per_px": scale.get("mm_per_px"),
        "camera_distance_mm": scale.get("camera_distance_mm"),
        "jaw_angle_deg": result["jaw_angle_deg"],
    }
    row.update({name: result["ratios"][name] for name in RATIOS})
    row.update({f"{name}_mm": millimetres.get(name) for name in SEGMENTS})
    row.update({f"{name}_px": result["measurements_px"].get(name) for name in SEGMENTS})
    return row


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    """The database: measurements, registered people and the check log."""

    def __init__(self, db_path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        definitions = ", ".join(f"{name} {kind}" for name, kind in COLUMNS.items())
        with self._db:
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS {TABLE} ({definitions}, result_json TEXT NOT NULL)"
            )
            # A database made by an older version may lack newer columns.
            existing = {row[1] for row in self._db.execute(f"PRAGMA table_info({TABLE})")}
            for name, kind in COLUMNS.items():
                if name not in existing:
                    self._db.execute(f"ALTER TABLE {TABLE} ADD COLUMN {name} {kind.split()[0]}")
            self._db.executescript(_SCHEMA)

    # --- Measurements --------------------------------------------------------

    def save(self, result):
        """Insert one measurement result."""
        with self._db:
            self._insert_measurement(result)

    def _insert_measurement(self, result):
        row = flatten(result)
        names = [*row, "result_json"]
        placeholders = ", ".join("?" for _ in names)
        self._db.execute(
            f"INSERT INTO {TABLE} ({', '.join(names)}) VALUES ({placeholders})",
            [*row.values(), json.dumps(result)],
        )

    def count(self):
        return self._db.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]

    def export_csv(self, csv_path):
        """Write every measurement to a CSV file. Returns the number of rows."""
        names = list(COLUMNS)
        rows = self._db.execute(f"SELECT {', '.join(names)} FROM {TABLE} ORDER BY captured_at, rowid")
        return _write_csv(csv_path, names, rows)

    # --- People ----------------------------------------------------------------

    def add_user(self, name, embeddings, measurement=None, replace=False):
        """Register a person with their embeddings and, optionally, a measurement.

        Names are matched without regard to case. Raises ValueError if the
        name is taken, unless replace=True, which removes the old record first.
        Returns {"name", "registered_at", "samples", "replaced"}.
        """
        vectors = [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings]
        if not vectors:
            raise ValueError("at least one face sample is needed")
        with self._db:
            replaced = self._remove_user(name) if replace else False
            if not replace and self.get_user(name):
                raise ValueError(f"'{name}' is already registered")
            if measurement is not None:
                self._insert_measurement(measurement)
            registered_at = utc_now()
            cursor = self._db.execute(
                "INSERT INTO users (name, registered_at, samples, measurement_id) VALUES (?, ?, ?, ?)",
                (name, registered_at, len(vectors), measurement["id"] if measurement else None),
            )
            self._db.executemany(
                "INSERT INTO face_samples (user_id, embedding) VALUES (?, ?)",
                [(cursor.lastrowid, v.tobytes()) for v in vectors],
            )
        return {"name": name, "registered_at": registered_at, "samples": len(vectors), "replaced": replaced}

    def get_user(self, name):
        """The person's record as a dict, or None. Includes their face shape if measured."""
        row = self._db.execute(
            f"SELECT u.name, u.registered_at, u.samples, m.face_shape FROM users u "
            f"LEFT JOIN {TABLE} m ON m.id = u.measurement_id WHERE u.name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return dict(zip(("name", "registered_at", "samples", "face_shape"), row))

    def users(self):
        """Every registered person, oldest first."""
        rows = self._db.execute(
            f"SELECT u.name, u.registered_at, u.samples, m.face_shape FROM users u "
            f"LEFT JOIN {TABLE} m ON m.id = u.measurement_id ORDER BY u.registered_at, u.id"
        ).fetchall()
        return [dict(zip(("name", "registered_at", "samples", "face_shape"), row)) for row in rows]

    def delete_user(self, name):
        """Remove a person and their face samples. Returns False if not found."""
        with self._db:
            return self._remove_user(name)

    def _remove_user(self, name):
        row = self._db.execute("SELECT id, measurement_id FROM users WHERE name = ?", (name,)).fetchone()
        if row is None:
            return False
        user_id, measurement_id = row
        self._db.execute("DELETE FROM face_samples WHERE user_id = ?", (user_id,))
        self._db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if measurement_id:
            self._db.execute(f"DELETE FROM {TABLE} WHERE id = ?", (measurement_id,))
        return True

    def gallery(self, name=None):
        """{name: (n, d) array of embeddings} for everyone, or for one person."""
        query = "SELECT u.name, s.embedding FROM face_samples s JOIN users u ON u.id = s.user_id"
        rows = self._db.execute(query + " WHERE u.name = ?", (name,)) if name else self._db.execute(query)
        grouped = {}
        for person, blob in rows:
            grouped.setdefault(person, []).append(np.frombuffer(blob, dtype=np.float32))
        return {person: np.vstack(vectors) for person, vectors in grouped.items()}

    # --- Checks ------------------------------------------------------------------

    def log_check(self, result):
        values = [result.get(c) for c in CHECK_COLUMNS]
        values[CHECK_COLUMNS.index("matched")] = int(bool(result["matched"]))
        with self._db:
            self._db.execute(
                f"INSERT INTO checks ({', '.join(CHECK_COLUMNS)}, result_json) "
                f"VALUES ({', '.join('?' for _ in CHECK_COLUMNS)}, ?)",
                [*values, json.dumps(result)],
            )

    def export_users_csv(self, csv_path):
        """One row per person with their measurement. Embeddings are never exported."""
        names = ["name", "registered_at", "samples", *USER_MEASUREMENT_COLUMNS]
        rows = self._db.execute(
            "SELECT u.name, u.registered_at, u.samples, "
            + ", ".join(f"m.{c}" for c in USER_MEASUREMENT_COLUMNS)
            + f" FROM users u LEFT JOIN {TABLE} m ON m.id = u.measurement_id ORDER BY u.registered_at, u.id"
        )
        return _write_csv(csv_path, names, rows)

    def export_checks_csv(self, csv_path):
        rows = self._db.execute(f"SELECT {', '.join(CHECK_COLUMNS)} FROM checks ORDER BY checked_at, rowid")
        return _write_csv(csv_path, CHECK_COLUMNS, rows)

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _write_csv(csv_path, header, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)
