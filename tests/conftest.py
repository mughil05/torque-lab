import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture
def make_source_db(tmp_path):
    """Return a helper that writes a minimal Dataset A-shaped SQLite file."""
    def build(rows):
        path = tmp_path / "synthetic-cars.db"
        with sqlite3.connect(path) as db:
            db.execute("""
                CREATE TABLE cars (
                    car_id TEXT, sources TEXT, propulsion TEXT, power_hp REAL,
                    torque TEXT, curb_weight_lb REAL, drivetrain TEXT,
                    "0_60_mph_sec" REAL, manufacturer TEXT, model TEXT,
                    year TEXT, engine TEXT
                )
            """)
            db.executemany("""
                INSERT INTO cars VALUES (
                    :car_id, :sources, :propulsion, :power_hp, :torque,
                    :curb_weight_lb, :drivetrain, :acceleration,
                    :manufacturer, :model, :year, :engine
                )
            """, rows)
        return path
    return build


@pytest.fixture
def synthetic_training_data():
    import pandas as pd

    rows = []
    drives = ["FWD", "RWD", "AWD", "4WD"]
    for group in range(24):
        for variant in range(2):
            hp = 110 + group * 11 + variant * 7
            torque = 180 + group * 5 + variant * 9
            weight = 1150 + group * 17 + variant * 35
            rows.append({
                "car_id": f"id-{group}-{variant}",
                "manufacturer": f"Make{group % 4}",
                "model": f"Series{group} Trim{variant}",
                "trim": f"Series{group} Trim{variant}",
                "year": str(2000 + group),
                "drivetrain": drives[group % len(drives)],
                "power_hp": float(hp),
                "torque_nm": float(torque),
                "weight_kg": float(weight),
                "power_hp_per_1000kg": hp / (weight / 1000.0),
                "torque_nm_per_1000kg": torque / (weight / 1000.0),
                "zero_to_sixty_s": float(11 - hp / 240 + weight / 1400 - torque / 750
                                          + (0.25 if drives[group % 4] == "FWD" else 0)),
                "group_key": f"make{group % 4}::series{group}",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def valid_source_row():
    return {
        "car_id": "valid-1",
        "sources": "Car & Driver",
        "propulsion": "Petrol",
        "power_hp": 300.0,
        "torque": "200 lb-ft",
        "curb_weight_lb": 3000.0,
        "drivetrain": "RWD",
        "acceleration": 5.4,
        "manufacturer": "Test Motors",
        "model": "Coupe GT",
        "year": "2022",
        "engine": "V6",
    }
