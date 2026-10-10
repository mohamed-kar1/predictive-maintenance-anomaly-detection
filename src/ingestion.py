"""Étape 1 : charger le fichier AI4I 2020 dans DuckDB avec dlt.

Lancer depuis la racine du projet :
    python src/ingestion.py
"""
from pathlib import Path

import dlt
import duckdb
import pandas as pd

CSV_PATH = Path("data/raw/ai4i2020.csv")
DB_PATH = "ai4i.duckdb"

# Noms de colonnes simples : sans espaces, sans crochets, sans majuscules.
COLUMNS = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "quality_level",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


@dlt.resource(name="sensor_readings", write_disposition="replace")
def sensor_readings():
    """Lit le CSV et le transmet à dlt (utf-8-sig retire le caractère invisible en début de fichier)."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig").rename(columns=COLUMNS)
    df["source_file"] = CSV_PATH.name
    yield df


def run():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CSV_PATH}. Place ai4i2020.csv dans data/raw/.")

    pipeline = dlt.pipeline(
        pipeline_name="ai4i_ingestion",
        destination=dlt.destinations.duckdb(DB_PATH),
        dataset_name="raw",
    )
    print(pipeline.run(sensor_readings()))


def check():
    """Contrôle rapide de ce qui a été chargé."""
    con = duckdb.connect(DB_PATH, read_only=True)
    lignes = con.sql("SELECT COUNT(*) FROM raw.sensor_readings").fetchone()[0]
    pannes = con.sql("SELECT SUM(machine_failure) FROM raw.sensor_readings").fetchone()[0]
    colonnes = [c[0] for c in con.sql("DESCRIBE raw.sensor_readings").fetchall()]
    con.close()
    print(f"\nLignes chargées : {lignes}")
    print(f"Pannes (machine_failure = 1) : {pannes}")
    print("Colonnes :", ", ".join(colonnes))


if __name__ == "__main__":
    run()
    check()
