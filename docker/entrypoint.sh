#!/bin/sh
# Les fichiers d'état (bases DuckDB, registre MLflow) vivent dans le volume /app/state,
# et sont reliés là où le code les attend (racine du projet).
set -e
cd /app
mkdir -p state/mlruns
for f in ai4i.duckdb journal.duckdb mlflow.db; do
  [ -e "$f" ] || ln -s "state/$f" "$f"
done
[ -e mlruns ] || ln -s state/mlruns mlruns

# Seul le service « api » prépare les données et le modèle ; les autres attendent qu'il soit prêt.
if [ "${RUN_BOOTSTRAP:-0}" = "1" ]; then
  python scripts/bootstrap.py
fi
exec "$@"
