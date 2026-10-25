"""Étape 4 : entraîner et comparer quatre détecteurs d'anomalies sur la table de variables (marts).
Le modèle composé LR -> XGBoost est évalué à part (train_stacking.py).

Lancer depuis la racine du projet :
    python src/train_models.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

DB_PATH = "ai4i.duckdb"
SORTIE = Path("reports/model_comparison.csv")

FEATURES = [
    "quality_level_ord", "air_temperature_k", "process_temperature_k",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    "temp_diff_k", "power_w", "wear_x_torque", "overstrain_ratio",
]
# Elliptic Envelope exige une covariance inversible : on retire temp_diff_k,
# qui vaut exactement « température du process − température de l'air ».
SANS_COLINEARITE = [i for i, f in enumerate(FEATURES) if f != "temp_diff_k"]

TARGET = "machine_failure"
EXCLURE_ETIQUETTES_INCOHERENTES = True   # retire les 27 lignes aux étiquettes contradictoires
N_PLIS = 5
SEED = 0


def charger():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.sql("SELECT * FROM marts.fct_machine_features").df()
    con.close()
    if EXCLURE_ETIQUETTES_INCOHERENTES:
        df = df[df["is_label_consistent"]]
    return df.reset_index(drop=True)


# Détecteurs non supervisés : entraînés uniquement sur les machines normales.
DETECTEURS = {
    "Isolation Forest": lambda: IsolationForest(n_estimators=200, random_state=SEED),
    "One-Class SVM": lambda: OneClassSVM(nu=0.05, gamma="scale"),
    "Local Outlier Factor": lambda: LocalOutlierFactor(n_neighbors=35, novelty=True),
    "Elliptic Envelope": lambda: EllipticEnvelope(contamination=0.01, random_state=SEED),
}


def mesurer(y_test, score, taux):
    """PR-AUC, puis précision / rappel / F1 en signalant la part `taux` de machines les plus suspectes."""
    pr_auc = average_precision_score(y_test, score)
    signale = (score >= np.quantile(score, 1 - taux)).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, signale, average="binary", zero_division=0)
    return {"pr_auc": pr_auc, "precision": p, "rappel": r, "f1": f1}


def main():
    df = charger()
    X, y = df[FEATURES].values, df[TARGET].values
    print(f"{len(df)} machines, {int(y.sum())} pannes ({y.mean():.1%}), {len(FEATURES)} variables\n")

    resultats = {nom: [] for nom in list(DETECTEURS)}
    for tr, te in StratifiedKFold(N_PLIS, shuffle=True, random_state=SEED).split(X, y):
        taux = y[tr].mean()

        normales = tr[y[tr] == 0]
        scaler = StandardScaler().fit(X[normales])
        X_normales, X_test = scaler.transform(X[normales]), scaler.transform(X[te])
        for nom, creer in DETECTEURS.items():
            cols = SANS_COLINEARITE if nom == "Elliptic Envelope" else slice(None)
            modele = creer().fit(X_normales[:, cols])
            resultats[nom].append(mesurer(y[te], -modele.decision_function(X_test[:, cols]), taux))

    lignes = []
    for nom, plis in resultats.items():
        t = pd.DataFrame(plis)
        lignes.append({"modele": nom, "supervise": False,
                       **{f"{m}_moy": t[m].mean() for m in t}, **{f"{m}_ecart": t[m].std() for m in t}})
    tableau = pd.DataFrame(lignes).sort_values("pr_auc_moy", ascending=False)

    SORTIE.parent.mkdir(exist_ok=True)
    tableau.round(4).to_csv(SORTIE, index=False)

    print(f"{'Modèle':28s} {'PR-AUC':>13s} {'F1':>13s} {'Précision':>10s} {'Rappel':>8s}")
    for _, l in tableau.iterrows():
        print(f"{l['modele']:28s} {l['pr_auc_moy']:.3f} ± {l['pr_auc_ecart']:.3f}  "
              f"{l['f1_moy']:.3f} ± {l['f1_ecart']:.3f}  {l['precision_moy']:10.3f} {l['rappel_moy']:8.3f}")

    print(f"\nMeilleur détecteur non supervisé (PR-AUC) : {tableau.iloc[0]['modele']}")
    print(f"Résultats enregistrés dans {SORTIE}")


if __name__ == "__main__":
    main()