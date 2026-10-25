"""Prépare tout ce dont l'API a besoin, une seule fois : données -> dbt -> modèle final -> alias « champion ».

Idempotent : si le modèle « champion » existe déjà dans le registre MLflow, il ne fait rien.
Lancé automatiquement par docker/entrypoint.sh ; utilisable à la main : python scripts/bootstrap.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOM_MODELE = "ai4i_stacking_10var"
ALIAS = "champion"
TRACKING_URI = "sqlite:///mlflow.db"


def etape(titre, *cmd, cwd=ROOT):
    print(f"\n=== {titre} ===", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def champion_existe() -> bool:
    if not (ROOT / "mlflow.db").exists():
        return False
    from mlflow import MlflowClient
    try:
        MlflowClient(tracking_uri=TRACKING_URI).get_model_version_by_alias(NOM_MODELE, ALIAS)
        return True
    except Exception:
        return False


def poser_alias():
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri=TRACKING_URI)
    derniere = max(client.search_model_versions(f"name='{NOM_MODELE}'"), key=lambda v: int(v.version))
    client.set_registered_model_alias(NOM_MODELE, ALIAS, derniere.version)
    print(f"Alias « {ALIAS} » -> version {derniere.version}")


def main():
    if champion_existe():
        print(f"Modèle « {ALIAS} » déjà enregistré : rien à faire.")
        return
    py = sys.executable
    etape("1/4 Données", py, "scripts/fetch_data.py")
    etape("2/4 Ingestion (dlt -> DuckDB)", py, "src/ingestion.py")
    etape("3/4 dbt build (modèles + tests)", py, "-m", "dbt.cli.main", "build", "--profiles-dir", ".",
          cwd=ROOT / "dbt_project")
    etape("4/4 Entraînement du modèle final (MLflow)", py, "src/train_final_stacking.py")
    poser_alias()
    print("\nPrêt : l'API peut démarrer.")


if __name__ == "__main__":
    main()
