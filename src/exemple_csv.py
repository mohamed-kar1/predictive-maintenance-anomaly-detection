"""Crée data/exemple_machines.csv : 10 machines en panne + 10 machines normales du jeu AI4I.

Sert à tester l'API (route /predict_csv) : la colonne machine_failure donne la vraie réponse,
l'API ne s'en sert pas.

    python src/exemple_csv.py
"""
from pathlib import Path

import duckdb

SORTIE = Path("data/exemple_machines.csv")
COLONNES = ("udi, quality_level, air_temperature_k, process_temperature_k, "
            "rotational_speed_rpm, torque_nm, tool_wear_min, machine_failure")

con = duckdb.connect("ai4i.duckdb", read_only=True)
df = con.sql(f"""
    (SELECT {COLONNES} FROM staging.stg_sensor_readings WHERE machine_failure = 1 ORDER BY udi LIMIT 10)
    UNION ALL
    (SELECT {COLONNES} FROM staging.stg_sensor_readings WHERE machine_failure = 0 ORDER BY udi LIMIT 10)
""").df()
con.close()

SORTIE.parent.mkdir(exist_ok=True)
df.to_csv(SORTIE, index=False)
print(f"{len(df)} machines écrites dans {SORTIE}")
print(df.to_string(index=False))
