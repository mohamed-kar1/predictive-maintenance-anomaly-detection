"""Étape 5b : entraîner le modèle final (XGBoost simple), fixer le seuil, l'enregistrer dans MLflow.

    python src/train_final.py
"""
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from train_models import FEATURES, N_PLIS, SEED, TARGET, charger

TRACKING_URI = "sqlite:///mlflow.db"
NOM_MODELE = "ai4i_xgboost"
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
              eval_metric="logloss", random_state=SEED)
SEUIL_ALERTE_ELEVEE = 0.8


def main():
    df = charger()
    X, y = df[FEATURES].values, df[TARGET].values

    # Seuil choisi sur des prédictions hors échantillon (validation croisée) : il maximise le F1.
    cv = StratifiedKFold(N_PLIS, shuffle=True, random_state=SEED)
    proba = cross_val_predict(XGBClassifier(**PARAMS), X, y, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    i = int(np.argmax(f1[:-1]))
    seuil = float(t[i])
    print(f"Seuil d'anomalie : {seuil:.3f}  (précision {p[i]:.3f}, rappel {r[i]:.3f}, F1 {f1[i]:.3f})")

    # Modèle final : entraîné sur toutes les données.
    modele = XGBClassifier(**PARAMS).fit(X, y)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("ai4i_modele_final")
    with mlflow.start_run(run_name="xgboost_final"):
        mlflow.log_params({**PARAMS, "seuil_anomalie": round(seuil, 4),
                           "seuil_alerte_elevee": SEUIL_ALERTE_ELEVEE,
                           "variables": ",".join(FEATURES)})
        mlflow.log_metrics({"pr_auc_cv": float(average_precision_score(y, proba)),
                            "precision_cv": float(p[i]), "rappel_cv": float(r[i]), "f1_cv": float(f1[i])})
        info = mlflow.xgboost.log_model(modele, name="model", registered_model_name=NOM_MODELE)

    version = getattr(info, "registered_model_version", "voir le registre")
    print(f"Modèle enregistré : {NOM_MODELE}, version {version}")


if __name__ == "__main__":
    main()