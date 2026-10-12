"""Orchestration Dagster : ingestion (dlt) -> dbt (modèles + tests)."""
import subprocess
import sys
from pathlib import Path

from dagster import AssetExecutionContext, AssetKey, Definitions, asset
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = ROOT / "dbt_project"

dbt_project = DbtProject(project_dir=DBT_DIR, profiles_dir=DBT_DIR)
dbt_project.prepare_if_dev()


@asset(key=AssetKey(["raw", "sensor_readings"]), group_name="ingestion",
       description="Charge le CSV AI4I dans DuckDB (schéma raw) avec dlt.")
def raw_sensor_readings(context: AssetExecutionContext):
    subprocess.run([sys.executable, "src/ingestion.py"], cwd=ROOT, check=True)


@dbt_assets(manifest=dbt_project.manifest_path)
def ai4i_dbt(context: AssetExecutionContext, dbt: DbtCliResource):
    # "build" = construit les modèles ET exécute les tests dans l'ordre
    yield from dbt.cli(["build"], context=context).stream()


defs = Definitions(
    assets=[raw_sensor_readings, ai4i_dbt],
    resources={"dbt": DbtCliResource(project_dir=dbt_project)},
)
