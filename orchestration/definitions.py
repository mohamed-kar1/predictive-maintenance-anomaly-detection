"""Orchestration Dagster : ingestion (dlt) -> dbt (modèles + tests) -> modélisation -> MLflow.

    dagster dev -f orchestration/definitions.py

Chaîne complète : raw.sensor_readings -> staging -> marts -> comparaison des 6 modèles -> essais MLflow
-> comparaison du modèle composé -> modèle final enregistré dans le registre MLflow.
Un test dbt qui échoue, ou un nombre de lignes différent de 10 000, arrête la chaîne avant l'entraînement.
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dagster import (AssetCheckResult, AssetExecutionContext, AssetKey, AssetSelection, Definitions,
                     MaterializeResult, MetadataValue, asset, asset_check, define_asset_job)
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = ROOT / "dbt_project"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
NOM_MODELE = "ai4i_stacking_10var"
FEATURES_DBT = AssetKey(["marts", "fct_machine_features"])

dbt_project = DbtProject(project_dir=DBT_DIR, profiles_dir=DBT_DIR)
dbt_project.prepare_if_dev()


def lancer(script: str):
    """Exécute src/<script> avec le Python du venv, depuis la racine du projet."""
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONUTF8": "1"}
    subprocess.run([sys.executable, f"src/{script}"], cwd=ROOT, env=env, check=True)


def tableau_md(df: pd.DataFrame, colonnes: list) -> str:
    lignes = ["| " + " | ".join(colonnes) + " |", "|" + "---|" * len(colonnes)]
    lignes += ["| " + " | ".join(str(v) for v in r) + " |" for r in df[colonnes].round(3).itertuples(index=False)]
    return "\n".join(lignes)


# ---------------------------------------------------------------- données

@asset(key=AssetKey(["raw", "sensor_readings"]), group_name="ingestion",
       description="Charge le CSV AI4I dans DuckDB (schéma raw) avec dlt.")
def raw_sensor_readings(context: AssetExecutionContext):
    subprocess.run([sys.executable, "src/ingestion.py"], cwd=ROOT, check=True)


@asset_check(asset=raw_sensor_readings, blocking=True,
             description="Le jeu AI4I compte exactement 10 000 lignes (indicateur du cahier des charges).")
def raw_has_10000_rows() -> AssetCheckResult:
    import duckdb
    con = duckdb.connect(str(ROOT / "ai4i.duckdb"), read_only=True)
    n = con.execute("SELECT COUNT(*) FROM raw.sensor_readings").fetchone()[0]
    con.close()
    return AssetCheckResult(passed=(n == 10000), metadata={"lignes": n})


@dbt_assets(manifest=dbt_project.manifest_path)
def ai4i_dbt(context: AssetExecutionContext, dbt: DbtCliResource):
    # "build" = construit les modèles ET exécute les tests dans l'ordre
    yield from dbt.cli(["build"], context=context).stream()


# ---------------------------------------------------------------- modélisation
# Chaîne volontairement linéaire : une seule étape écrit dans mlflow.db à la fois.

@asset(deps=[FEATURES_DBT], group_name="modelisation",
       description="Compare 4 détecteurs d'anomalies en validation croisée à 5 plis (train_models.py).")
def model_comparison() -> MaterializeResult:
    lancer("train_models.py")
    df = pd.read_csv(ROOT / "reports" / "model_comparison.csv").sort_values("pr_auc_moy", ascending=False)
    return MaterializeResult(metadata={
        "meilleur_detecteur": str(df.iloc[0]["modele"]),
        "pr_auc_meilleur": float(df.iloc[0]["pr_auc_moy"]),
        "classement": MetadataValue.md(tableau_md(df, ["modele", "pr_auc_moy", "pr_auc_ecart", "f1_moy"]))})


@asset(deps=[model_comparison], group_name="suivi_mlflow",
       description="Enregistre les 4 essais de la comparaison dans MLflow (log_comparison.py).")
def mlflow_runs():
    lancer("log_comparison.py")


@asset(deps=[mlflow_runs], group_name="modelisation",
       description="Évalue les variantes du modèle composé LR -> XGBoost (train_stacking.py).")
def stacking_comparison() -> MaterializeResult:
    lancer("train_stacking.py")
    df = pd.read_csv(ROOT / "reports" / "stacking_comparison.csv").sort_values("pr_auc_moy", ascending=False)
    return MaterializeResult(metadata={
        "classement": MetadataValue.md(tableau_md(df, ["variante", "pr_auc_moy", "pr_auc_ecart", "f1_moy"]))})


@asset(deps=[stacking_comparison], group_name="suivi_mlflow",
       description="Entraîne le modèle final retenu (LR -> XGBoost, 10 variables) et l'enregistre dans le registre MLflow. "
                   "L'alias « champion » utilisé par l'API n'est pas déplacé : à faire dans MLflow après vérification.")
def final_model() -> MaterializeResult:
    lancer("train_final_stacking.py")
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri=TRACKING_URI)
    derniere = max(client.search_model_versions(f"name='{NOM_MODELE}'"), key=lambda v: int(v.version))
    run = client.get_run(derniere.run_id)
    try:
        champion = int(client.get_model_version_by_alias(NOM_MODELE, "champion").version)
    except Exception:
        champion = None
    meta = {"modele_retenu": "LR -> XGBoost (modèle composé, 10 variables)",
            "nouvelle_version": int(derniere.version), "version_champion_utilisee_par_l_api": champion}
    comp = pd.read_csv(ROOT / "reports" / "stacking_comparison.csv")
    if "mode" in comp.columns:
        ligne = comp[comp["mode"] == "lr_et_variables"]
        if not ligne.empty:
            meta["pr_auc_modele_compose"] = float(ligne.iloc[0]["pr_auc_moy"])
    if "seuil_anomalie" in run.data.params:
        meta["seuil_anomalie"] = float(run.data.params["seuil_anomalie"])
    for m in ("pr_auc_cv", "precision_cv", "rappel_cv", "f1_cv"):
        if m in run.data.metrics:
            meta[m] = float(run.data.metrics[m])
    return MaterializeResult(metadata=meta)


# ---------------------------------------------------------------- jobs

pipeline_donnees = define_asset_job(
    "pipeline_donnees", selection=AssetSelection.assets(raw_sensor_readings, ai4i_dbt),
    description="Ingestion + dbt (modèles et tests). Rapide, sans entraînement.")
pipeline_complet = define_asset_job(
    "pipeline_complet", selection=AssetSelection.all(),
    description="Toute la chaîne, de la donnée brute au modèle enregistré dans MLflow.")

defs = Definitions(
    assets=[raw_sensor_readings, ai4i_dbt, model_comparison, mlflow_runs, stacking_comparison, final_model],
    asset_checks=[raw_has_10000_rows],
    jobs=[pipeline_donnees, pipeline_complet],
    resources={"dbt": DbtCliResource(project_dir=dbt_project)},
)