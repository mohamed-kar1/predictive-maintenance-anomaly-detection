"""Étape 2 : explorer et contrôler la couche raw dans DuckDB.
"""

import duckdb

con = duckdb.connect("ai4i.duckdb", read_only=True)
T = "raw.sensor_readings"


def show(titre, requete):
    print(f"\n=== {titre} ===")
    print(con.sql(requete))


show("Tables du schéma raw",
     "SELECT table_name FROM information_schema.tables WHERE table_schema = 'raw' ORDER BY table_name")

colonnes = [c[0] for c in con.sql(f"DESCRIBE {T}").fetchall()]
manquants = sum(con.sql(f"SELECT COUNT(*) FROM {T} WHERE {c} IS NULL").fetchone()[0] for c in colonnes)
print(f"\n=== Valeurs manquantes ===\nTotal sur toutes les colonnes : {manquants}")

show("Doublons sur l'identifiant (udi)",
     f"SELECT COUNT(*) - COUNT(DISTINCT udi) AS doublons FROM {T}")

show("Plage de chaque mesure",
     f"""SELECT 'température de l''air (K)' AS mesure, MIN(air_temperature_k) AS min, ROUND(AVG(air_temperature_k),1) AS moyenne, MAX(air_temperature_k) AS max FROM {T}
         UNION ALL SELECT 'température du process (K)', MIN(process_temperature_k), ROUND(AVG(process_temperature_k),1), MAX(process_temperature_k) FROM {T}
         UNION ALL SELECT 'vitesse (tr/min)', MIN(rotational_speed_rpm), ROUND(AVG(rotational_speed_rpm),1), MAX(rotational_speed_rpm) FROM {T}
         UNION ALL SELECT 'couple (Nm)', MIN(torque_nm), ROUND(AVG(torque_nm),1), MAX(torque_nm) FROM {T}
         UNION ALL SELECT 'usure de l''outil (min)', MIN(tool_wear_min), ROUND(AVG(tool_wear_min),1), MAX(tool_wear_min) FROM {T}""")

show("Répartition du niveau de qualité",
     f"SELECT quality_level, COUNT(*) AS lignes, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pourcentage FROM {T} GROUP BY quality_level ORDER BY quality_level")

show("Pannes par type",
     f"SELECT SUM(machine_failure) AS pannes, SUM(twf) AS usure_outil, SUM(hdf) AS chaleur, SUM(pwf) AS puissance, SUM(osf) AS surcharge, SUM(rnf) AS aleatoire FROM {T}")

show("Cohérence des étiquettes de panne",
     f"""SELECT
           SUM((machine_failure = 1 AND twf + hdf + pwf + osf + rnf = 0)::INT) AS panne_sans_type,
           SUM((machine_failure = 0 AND twf + hdf + pwf + osf + rnf > 0)::INT) AS type_sans_panne
         FROM {T}""")

con.close()