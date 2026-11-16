#!/usr/bin/env python3
"""Reproducible, read-only audit for the two TorqueLab source datasets.

Run from the project root:
  python3 audit_datasets.py
  python3 audit_datasets.py --a path/to/car-performance-data.db --b path/to/engines.csv

Dataset A's 0_60_mph_sec cannot be treated as uniformly genuine 0-60 mph:
the checked-in ETL also writes Kaggle-estimated 0-60 and Autoevolution 0-62
mph values into it. The script reports the raw complete-case number and a
conservative Car & Driver-only subset. Dataset B's target is 0-100 km/h.
No target conversion is performed.
"""
import argparse
import csv
import re
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A_DEFAULT = ROOT / "sources/car-performance-comparison/public/data/car-performance-data.db"
B_DEFAULT = ROOT / "sources/vehicle-makes-models/data/csv/engines.csv"
NAMEPLATES = {
    "Ford Mustang": lambda make, model: make.lower() == "ford" and "mustang" in model.lower(),
    "Chevrolet Corvette": lambda make, model: make.lower() == "chevrolet" and "corvette" in model.lower(),
    "Chevrolet Camaro": lambda make, model: make.lower() == "chevrolet" and "camaro" in model.lower(),
    "BMW M3": lambda make, model: make.lower() == "bmw" and re.search(r"\bm3\b", model.lower()) is not None,
    "BMW M4": lambda make, model: make.lower() == "bmw" and re.search(r"\bm4\b", model.lower()) is not None,
    "Porsche 911": lambda make, model: make.lower() == "porsche" and "911" in model.lower(),
}


def number(value):
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def gasoline(label):
    text = str(label or "").strip().lower()
    if not text or re.search(r"hybrid|plug.?in|diesel|electric|\bbev\b|\bphev\b", text):
        return False
    return text in {"petrol", "gasoline", "gas", "petrol/ice", "ice"} or bool(
        re.search(r"\b(gasoline|petrol)\b", text)
    )


def complete(row, keys):
    return all(number(row.get(k)) is not None and number(row.get(k)) > 0 for k in keys[:3]) and all(
        str(row.get(k) or "").strip() for k in keys[3:]
    )


def report(label, rows, columns, keys, make_key, model_key, fuel_key, target_key,
           source_key=None, strict_source=None, limits=None):
    total = len(rows)
    duplicates = total - len({tuple(str(r.get(k) or "").strip() for k in columns) for r in rows})
    print(f"\n{label}: {total:,} records; exact duplicate rows: {duplicates:,}")
    print("Missing required values:")
    for key in keys + [fuel_key]:
        print(f"  {key}: {sum(r.get(key) in (None, '') for r in rows):,}")
    comp = [r for r in rows if complete(r, keys)]
    gas = [r for r in rows if gasoline(r.get(fuel_key))]
    gas_comp = [r for r in gas if complete(r, keys)]
    print(f"Complete numeric features + target ({target_key}): {len(comp):,}")
    print(f"Gasoline-only records: {len(gas):,}; gasoline complete: {len(gas_comp):,}")
    if source_key and strict_source:
        strict = [r for r in gas_comp if str(r.get(source_key) or '').strip() == strict_source]
        print(f"Conservative {strict_source}-only gasoline complete target rows: {len(strict):,}")
    print("Requested nameplate coverage (gasoline records / complete):")
    for name, test in NAMEPLATES.items():
        match = [r for r in gas if test(str(r.get(make_key) or ''), str(r.get(model_key) or ''))]
        print(f"  {name}: {len(match):,} / {sum(complete(r, keys) for r in match):,}")
    if limits:
        print("Broad out-of-range review counts (not automatically excluded):")
        for field, (low, high) in limits.items():
            count = sum(number(r.get(field)) is not None and not low <= number(r.get(field)) <= high for r in rows)
            print(f"  {field} outside [{low}, {high}]: {count:,}")


def audit_a(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM cars')]
    keys = ["power_hp", "torque", "curb_weight_lb", "drivetrain", "0_60_mph_sec"]
    report("Dataset A / SQLite cars", rows, list(rows[0]), keys, "manufacturer", "model", "propulsion",
           "0_60_mph_sec", "sources", "Car & Driver",
           {"power_hp": (0, 2500), "torque": (0.000001, 3000), "curb_weight_lb": (1000, 10000), "0_60_mph_sec": (1, 30)})
    paired = [r for r in rows if number(r.get("0_60_mph_sec")) is not None and number(r.get("0_100_kmh_sec")) is not None]
    equal = sum(number(r["0_60_mph_sec"]) == number(r["0_100_kmh_sec"]) for r in paired)
    print(f"Rows with both 0-60 mph and 0-100 km/h values: {len(paired):,}; identical values: {equal:,}")
    sources = Counter(r.get("sources") for r in rows if r.get("0_60_mph_sec") is not None)
    print("0-60 field provenance buckets:")
    for source, count in sources.most_common():
        print(f"  {source}: {count:,}")


def audit_b(path):
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute("SELECT e.*, m.name AS make, mo.name AS model FROM engines e JOIN generations g ON g.id=e.generation_id JOIN models mo ON mo.id=g.model_id JOIN makes m ON m.id=mo.make_id")]
    else:
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    keys = ["power_hp", "torque_nm", "curb_weight_kg", "drivetrain", "zero_to_100_s"]
    report("Dataset B / engine variants", rows, list(rows[0]) if rows else [], keys,
           "make", "model", "fuel_type", "zero_to_100_s",
           limits={"power_hp": (0, 2500), "torque_nm": (0, 3000), "curb_weight_kg": (500, 10000), "zero_to_100_s": (1, 60)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=A_DEFAULT, help="Dataset A SQLite path")
    parser.add_argument("--b", type=Path, default=B_DEFAULT, help="Dataset B engines.csv or vehicles.sqlite path")
    parser.add_argument("--only", choices=("a", "b", "both"), default="both")
    args = parser.parse_args()
    if args.only in ("a", "both"):
        if args.a.exists(): audit_a(args.a)
        else: print(f"Dataset A unavailable: {args.a}")
    if args.only in ("b", "both"):
        if args.b.exists(): audit_b(args.b)
        else: print(f"Dataset B unavailable: {args.b}")


if __name__ == "__main__":
    main()
