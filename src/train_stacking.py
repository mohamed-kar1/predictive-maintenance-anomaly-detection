"""Modèle composé LR -> XGBoost avec les 10 variables (dont overstrain_ratio), deux façons :
  1) XGBoost reçoit uniquement la probabilité donnée par la LR ;
  2) XGBoost reçoit les 10 variables + la probabilité donnée par la LR.
Mêmes plis et mêmes métriques que train_models.py.

Anti-fuite : sur le train du pli, la probabilité de la LR vient de prédictions hors-échantillon
(validation croisée interne) ; sur le test, la LR est entraînée sur tout le train du pli.

    python src/train_stacking.py
"""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from train_models import FEATURES, N_PLIS, SEED, TARGET, charger, mesurer

# Sans doublon, que FEATURES contienne déjà overstrain_ratio ou non.
FEATURES_10 = list(dict.fromkeys(FEATURES + ["overstrain_ratio"]))
assert len(FEATURES_10) == 10, f"10 variables attendues, obtenu {len(FEATURES_10)}"

SORTIE = Path("reports/stacking_comparison.csv")
TRACKING_URI = "sqlite:///mlflow.db"

# nom : mode   ("lr_seule" | "lr_et_variables")
VARIANTES = {
    "LR -> XGBoost : sortie LR seule": "lr_seule",
    "LR -> XGBoost : 10 variables + sortie LR": "lr_et_variables",
}


def creer_lr():
    # Variables standardisées + poids de classes, à cause du déséquilibre.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def creer_xgb():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=SEED)


def main():
    df = charger()
    X, y = df[FEATURES_10].values, df[TARGET].values
    print(f"{len(df)} machines, {int(y.sum())} pannes ({y.mean():.1%}), {len(FEATURES_10)} variables\n")

    resultats = {nom: [] for nom in VARIANTES}
    for tr, te in StratifiedKFold(N_PLIS, shuffle=True, random_state=SEED).split(X, y):
        taux = y[tr].mean()
        p_tr = cross_val_predict(creer_lr(), X[tr], y[tr],
                                 cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                 method="predict_proba")[:, 1]
        p_te = creer_lr().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

        for nom, mode in VARIANTES.items():
            if mode == "lr_seule":
                X_tr, X_te = p_tr.reshape(-1, 1), p_te.reshape(-1, 1)
            else:
                X_tr, X_te = np.column_stack([X[tr], p_tr]), np.column_stack([X[te], p_te])
            score = creer_xgb().fit(X_tr, y[tr]).predict_proba(X_te)[:, 1]
            resultats[nom].append(mesurer(y[te], score, taux))

    lignes = []
    for nom, plis in resultats.items():
        t = pd.DataFrame(plis)
        lignes.append({"variante": nom, "mode": VARIANTES[nom], "n_variables": len(FEATURES_10),
                       **{f"{m}_moy": t[m].mean() for m in t},
                       **{f"{m}_ecart": t[m].std() for m in t}})
    tableau = pd.DataFrame(lignes).sort_values("pr_auc_moy", ascending=False)
    SORTIE.parent.mkdir(exist_ok=True)
    tableau.round(4).to_csv(SORTIE, index=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("ai4i_stacking_final")
    for _, l in tableau.iterrows():
        with mlflow.start_run(run_name=l["variante"]):
            mlflow.log_params({"mode": l["mode"], "n_variables": int(l["n_variables"]),
                               "variables": ",".join(FEATURES_10), "n_plis": N_PLIS, "seed": SEED})
            mlflow.log_metrics({c: float(l[c]) for c in tableau.columns if c.endswith(("_moy", "_ecart"))})
            mlflow.log_artifact(str(SORTIE))

    print(f"{'Variante':44s} {'PR-AUC':>13s} {'F1':>13s}")
    for _, l in tableau.iterrows():
        print(f"{l['variante']:44s} {l['pr_auc_moy']:.3f} ± {l['pr_auc_ecart']:.3f}  "
              f"{l['f1_moy']:.3f} ± {l['f1_ecart']:.3f}")
    print(f"\nRésultats enregistrés dans {SORTIE} et dans MLflow (expérience ai4i_stacking_final)")


if __name__ == "__main__":
    main()