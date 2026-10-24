"""Journal des prédictions (F9) : une table DuckDB dans journal.duckdb, séparée de ai4i.duckdb.

Chaque appel de l'API ajoute une ligne par machine analysée (mesures, score, résultat, version du modèle).
Les connexions sont courtes (ouvertes puis fermées à chaque appel) : on peut interroger le journal
depuis un autre terminal pendant que l'API tourne.
"""
import threading
from pathlib import Path

import duckdb
import pandas as pd

CHEMIN = Path(__file__).resolve().parent.parent / "journal.duckdb"
VERROU = threading.Lock()
COLONNES = ["horodatage", "source", "fichier", "identifiant", "quality_level", "air_temperature_k",
            "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min",
            "score", "resultat", "niveau_alerte", "modele_version", "seuil"]

CREATION = """CREATE TABLE IF NOT EXISTS predictions (
    horodatage TIMESTAMP, source VARCHAR, fichier VARCHAR, identifiant VARCHAR, quality_level VARCHAR,
    air_temperature_k DOUBLE, process_temperature_k DOUBLE, rotational_speed_rpm DOUBLE,
    torque_nm DOUBLE, tool_wear_min DOUBLE, score DOUBLE, resultat VARCHAR, niveau_alerte VARCHAR,
    modele_version INTEGER, seuil DOUBLE)"""


def _ouvrir():
    con = duckdb.connect(str(CHEMIN))
    con.execute(CREATION)
    return con


def enregistrer(lot: pd.DataFrame) -> None:
    """Ajoute les lignes du DataFrame (colonnes de COLONNES) au journal."""
    lot = lot.reindex(columns=COLONNES)
    cols = ", ".join(COLONNES)
    with VERROU:
        con = _ouvrir()
        try:
            con.register("lot", lot)
            con.execute(f"INSERT INTO predictions ({cols}) SELECT {cols} FROM lot")
        finally:
            con.close()


def derniers(n: int = 10) -> list:
    with VERROU:
        con = _ouvrir()
        try:
            df = con.execute("SELECT * FROM predictions ORDER BY horodatage DESC, rowid DESC LIMIT ?", [n]).df()
        finally:
            con.close()
    df["horodatage"] = df["horodatage"].dt.strftime("%d/%m %H:%M:%S")
    return df.astype(object).where(df.notna(), None).to_dict("records")


def resume() -> dict:
    with VERROU:
        con = _ouvrir()
        try:
            total, anormales, elevees = con.execute(
                "SELECT COUNT(*), COALESCE(SUM((resultat = 'anormal')::INT), 0), "
                "COALESCE(SUM((niveau_alerte = 'élevée')::INT), 0) FROM predictions").fetchone()
        finally:
            con.close()
    return {"total": int(total), "anormales": int(anormales), "alertes_elevees": int(elevees)}
