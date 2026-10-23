"""Vérifie que features.py donne exactement les mêmes variables que dbt.

    python src/check_features.py
"""
import sys

import duckdb
import numpy as np

from features import FEATURES, MESURES, calculer_variables
from train_models import FEATURES as FEATURES_ENTRAINEMENT

# 1. Mêmes variables, dans le même ordre, qu'à l'entraînement.
assert FEATURES == list(FEATURES_ENTRAINEMENT), \
    f"Ordre ou liste différents de train_models.py :\n{FEATURES}\n{list(FEATURES_ENTRAINEMENT)}"
print(f"Liste des variables identique à train_models.py ({len(FEATURES)} variables)")

# 2. Mêmes valeurs que dbt, machine par machine.
con = duckdb.connect("ai4i.duckdb", read_only=True)
cols_dbt = ", ".join(f"m.{c} AS dbt_{c}" for c in FEATURES)
cols_mesures = ", ".join(f"s.{c}" for c in MESURES)
df = con.sql(f"""SELECT {cols_mesures}, {cols_dbt}
                 FROM staging.stg_sensor_readings s
                 JOIN marts.fct_machine_features m USING (udi)""").df()
con.close()

py = calculer_variables(df[MESURES])
ok = True
print(f"\nComparaison sur {len(df)} machines :")
for c in FEATURES:
    ecart = float(np.max(np.abs(py[c].values - df[f"dbt_{c}"].values.astype(float))))
    bon = np.allclose(py[c].values, df[f"dbt_{c}"].values.astype(float), rtol=1e-9, atol=1e-9)
    ok &= bon
    print(f"  {'OK    ' if bon else 'ECART '} {c:24s} écart max = {ecart:.2e}")

print("\nRésultat :", "identique à dbt" if ok else "DIFFÉRENCES, à corriger")
sys.exit(0 if ok else 1)