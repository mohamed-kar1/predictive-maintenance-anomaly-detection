"""Modèle final : LR -> XGBoost (stacking) avec 10 variables (dont overstrain_ratio).

1. Choisit le seuil d'anomalie par validation croisée (il maximise le F1) et mesure, à ce seuil,
   précision, rappel, F1, F1 macro, précision macro et accuracy sur les prédictions hors échantillon.
2. Entraîne le modèle sur toutes les données.
3. L'enregistre dans le registre MLflow.

4. Calcule l'importance globale des variables avec SHAP (reports/shap_importance.csv et
   reports/shap_summary.png), jointe à l'essai MLflow.

Lancer depuis la racine du projet :
    python src/train_final_stacking.py
"""
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import explain
from train_models import N_PLIS, SEED, TARGET, charger, metriques_decision
from train_stacking import FEATURES_10, creer_lr, creer_xgb

TRACKING_URI = "sqlite:///mlflow.db"
NOM_MODELE = "ai4i_stacking_10var"
SEUIL_ALERTE_ELEVEE = 0.8   # probabilité à partir de laquelle l'alerte est « élevée »
SHAP_CSV = Path("reports/shap_importance.csv")
SHAP_PNG = Path("reports/shap_summary.png")


def creer_stacking():
    """La LR est entraînée en validation croisée interne : sa sortie est donc hors échantillon
    (pas de fuite de données). XGBoost reçoit les 10 variables + la probabilité de la LR."""
    return StackingClassifier(
        estimators=[("lr", creer_lr())], final_estimator=creer_xgb(),
        stack_method="predict_proba", passthrough=True,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


def main():
    df = charger()
    X, y = df[FEATURES_10].values, df[TARGET].values
    print(f"{len(df)} machines, {int(y.sum())} pannes, {len(FEATURES_10)} variables")

    # Seuil choisi sur des prédictions hors échantillon : il maximise le F1.
    cv = StratifiedKFold(N_PLIS, shuffle=True, random_state=SEED)
    proba = cross_val_predict(creer_stacking(), X, y, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    i = int(np.argmax(f1[:-1]))
    seuil = float(t[i])
    print(f"Seuil d'anomalie : {seuil:.3f}  (précision {p[i]:.3f}, rappel {r[i]:.3f}, F1 {f1[i]:.3f})")

    # Toutes les métriques de décision, au seuil retenu (prédictions hors échantillon).
    m = metriques_decision(y, (proba >= seuil).astype(int))
    print(f"À ce seuil : accuracy {m['accuracy']:.3f}, précision macro {m['precision_macro']:.3f}, "
          f"F1 macro {m['f1_macro']:.3f}")

    # Modèle final : entraîné sur toutes les données.
    modele = creer_stacking().fit(X, y)

    # Importance globale des variables (SHAP) sur l'ensemble des machines.
    entrees, contrib = explain.contributions(explain.creer_explainer(modele), modele, X)
    SHAP_CSV.parent.mkdir(exist_ok=True)
    importance = explain.importance_globale(contrib)
    importance.round(4).to_csv(SHAP_CSV, index=False)
    explain.tracer_resume(contrib, entrees, SHAP_PNG)
    print("Variables les plus influentes (SHAP) : "
          + ", ".join(f"{v} ({p:.0%})" for v, p in zip(importance["variable"][:3], importance["part"][:3])))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("ai4i_modele_final")
    with mlflow.start_run(run_name="stacking_10var_final"):
        mlflow.log_params({"modele": "LR -> XGBoost (stacking)", "n_variables": len(FEATURES_10),
                           "variables": ",".join(FEATURES_10), "seuil_anomalie": round(seuil, 4),
                           "seuil_alerte_elevee": SEUIL_ALERTE_ELEVEE})
        mlflow.log_metrics({"pr_auc_cv": float(average_precision_score(y, proba)),
                            "precision_cv": float(p[i]), "rappel_cv": float(r[i]), "f1_cv": float(f1[i]),
                            "accuracy_cv": float(m["accuracy"]),
                            "precision_macro_cv": float(m["precision_macro"]),
                            "f1_macro_cv": float(m["f1_macro"])})
        mlflow.log_artifact(str(SHAP_CSV), artifact_path="shap")
        mlflow.log_artifact(str(SHAP_PNG), artifact_path="shap")
        info = mlflow.sklearn.log_model(
            modele, name="model", registered_model_name=NOM_MODELE,
            serialization_format="cloudpickle")

    version = getattr(info, "registered_model_version", "voir le registre")
    print(f"Modèle enregistré : {NOM_MODELE}, version {version}")


if __name__ == "__main__":
    main()