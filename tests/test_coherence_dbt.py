"""Les variables calculées en Python (features.py) doivent égaler celles de dbt, machine par machine.

Ce test lit la vraie base ai4i.duckdb : il est ignoré si elle est absente (par exemple sur GitHub)
ou si un autre processus la verrouille (Dagster, dbt).
"""
from pathlib import Path

import duckdb
import numpy as np
import pytest

from features import FEATURES, MESURES, calculer_variables

BASE = Path(__file__).resolve().parent.parent / "ai4i.duckdb"


def test_python_egale_dbt():
    if not BASE.exists():
        pytest.skip("ai4i.duckdb absente : lancer l'ingestion et dbt build")
    try:
        con = duckdb.connect(str(BASE), read_only=True)
    except duckdb.Error as e:
        pytest.skip(f"ai4i.duckdb occupée : {str(e)[:60]}")
    try:
        mesures = ", ".join(f"s.{c}" for c in MESURES)
        dbt = ", ".join(f"m.{c} AS dbt_{c}" for c in FEATURES)
        df = con.execute(f"SELECT {mesures}, {dbt} FROM staging.stg_sensor_readings s "
                         f"JOIN marts.fct_machine_features m USING (udi)").df()
    finally:
        con.close()
    assert len(df) > 9000
    python = calculer_variables(df[MESURES])
    for c in FEATURES:
        assert np.allclose(python[c].values, df[f"dbt_{c}"].values.astype(float), rtol=1e-9, atol=1e-9), c
