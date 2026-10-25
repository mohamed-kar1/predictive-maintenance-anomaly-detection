"""Étape 5a : enregistrer dans MLflow les résultats de la comparaison des modèles.

Lit reports/model_comparison.csv (produit par train_models.py), sans rien réentraîner.
    python src/log_comparison.py
"""
from pathlib import Path

import mlflow
import pandas as pd

CSV = Path("reports/model_comparison.csv")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIENCE = "ai4i_comparaison_modeles"

PARAMS = {
    "Isolation Forest": {"n_estimators": 200},
    "One-Class SVM": {"nu": 0.05, "gamma": "scale"},
    "Local Outlier Factor": {"n_neighbors": 35},
    "Elliptic Envelope": {"contamination": 0.01},
}


def main():
    tableau = pd.read_csv(CSV)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIENCE)
    for _, l in tableau.iterrows():
        with mlflow.start_run(run_name=l["modele"]):
            mlflow.log_params({"n_plis": 5, "seed": 0, "supervise": bool(l["supervise"]),
                               **PARAMS[l["modele"]]})
            mlflow.log_metrics({c: float(l[c]) for c in tableau.columns if c.endswith(("_moy", "_ecart"))})
            mlflow.log_artifact(str(CSV))
    print(f"{len(tableau)} essais enregistrés dans l'expérience « {EXPERIENCE} »")


if __name__ == "__main__":
    main()