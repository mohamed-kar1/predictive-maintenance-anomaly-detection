"""Analyse du modèle composé étage par étage : matrice de confusion + SHAP.

Le modèle final est composé de deux étages :

  Étage 1 : régression logistique (LR)  : 10 variables -> proba_lr
  Étage 2 : XGBoost                     : 10 variables + proba_lr -> score final

Pour chaque étage, ce script produit une matrice de confusion et une explication SHAP :

  - étage 1 : matrice de confusion de proba_lr (seuil 0,5, la décision naturelle d'une LR) et SHAP
              de la LR (LinearExplainer, sur les 10 variables standardisées, en log-odds de la LR) ;
  - étage 2 : matrice de confusion du score final (au seuil d'anomalie du modèle) et SHAP d'XGBoost
              (TreeExplainer, sur les 11 entrées dont proba_lr, voir explain.py).

Toutes les matrices sont calculées sur des prédictions hors échantillon (validation croisée) : ce sont
des estimations honnêtes de ce que le modèle fera sur de nouvelles machines, pas un score sur ses
propres données d'entraînement. Les SHAP sont calculés sur les modèles entraînés sur toutes les données.

Lancer depuis la racine du projet :
    python src/analyse_etages.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import explain
from train_models import N_PLIS, SEED, TARGET, charger, metriques_decision
from train_stacking import FEATURES_10, creer_lr

DOSSIER = Path("reports")
SEUIL_LR = 0.5            # décision naturelle d'une régression logistique
CLASSES = ["normal", "anormal"]


# ----------------------------------------------------------------- matrice de confusion

def seuil_max_f1(y, proba) -> float:
    """Seuil qui maximise le F1 de la classe « panne » (même règle que train_final_stacking.py)."""
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[int(np.argmax(f1[:-1]))])


def analyser_matrice(y, proba, seuil: float) -> dict:
    """Matrice de confusion au seuil donné, et les taux qui l'interprètent."""
    pred = (proba >= seuil).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    vn, fp, fn, vp = int(vn), int(fp), int(fn), int(vp)
    return {
        "seuil": round(float(seuil), 4),
        "vrais_negatifs": vn, "faux_positifs": fp, "faux_negatifs": fn, "vrais_positifs": vp,
        "pannes_detectees_%": round(100 * vp / max(vp + fn, 1), 1),            # rappel
        "fausses_alertes_sur_normales_%": round(100 * fp / max(vn + fp, 1), 2),
        "alertes_justifiees_%": round(100 * vp / max(vp + fp, 1), 1),          # précision
        **{k: round(float(v), 4) for k, v in metriques_decision(y, pred).items()
           if k in ("f1", "f1_macro", "precision_macro", "accuracy")},
    }


def tracer_matrice(m: dict, titre: str, chemin) -> None:
    """Matrice de confusion : effectifs, et part de chaque classe réelle entre parenthèses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[m["vrais_negatifs"], m["faux_positifs"]], [m["faux_negatifs"], m["vrais_positifs"]]])
    part = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(part, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({part[i, j]:.1%})", ha="center", va="center", fontsize=12,
                    color="white" if part[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], [f"prédit\n{c}" for c in CLASSES])
    ax.set_yticks([0, 1], [f"réel\n{c}" for c in CLASSES])
    ax.set_title(f"{titre}\nseuil {m['seuil']:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- SHAP de la régression logistique

def contributions_lr(lr, X: np.ndarray):
    """SHAP de la LR (pipeline StandardScaler + LogisticRegression) en log-odds de la LR.

    Renvoie (X standardisé, contributions), deux tableaux de forme (n, 10).
    """
    scaler, regression = lr[0], lr[-1]
    Xs = scaler.transform(X)
    explainer = shap.LinearExplainer(regression, shap.maskers.Independent(Xs, max_samples=len(Xs)))
    return Xs, np.asarray(explainer.shap_values(Xs))


def importance_lr(contrib: np.ndarray) -> pd.DataFrame:
    t = pd.DataFrame({"variable": FEATURES_10, "shap_abs_moyen": np.abs(contrib).mean(axis=0)})
    t["part"] = t["shap_abs_moyen"] / t["shap_abs_moyen"].sum()
    return t.sort_values("shap_abs_moyen", ascending=False).reset_index(drop=True)


def tracer_shap_lr(contrib: np.ndarray, X: np.ndarray, chemin) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(contrib, X, feature_names=FEATURES_10, show=False, max_display=len(FEATURES_10))
    plt.title("Étage 1 : contribution des variables à proba_lr (SHAP, log-odds de la LR)")
    plt.tight_layout()
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------- programme principal

def main():
    from train_final_stacking import creer_stacking   # importé ici : évite de charger MLflow pour les tests

    df = charger()
    X, y = df[FEATURES_10].values, df[TARGET].values
    DOSSIER.mkdir(exist_ok=True)
    print(f"{len(df)} machines, {int(y.sum())} pannes ({y.mean():.1%}), {len(FEATURES_10)} variables\n")
    cv = StratifiedKFold(N_PLIS, shuffle=True, random_state=SEED)

    # ---- Étage 1 : la régression logistique, qui produit proba_lr
    proba_lr = cross_val_predict(creer_lr(), X, y, cv=cv, method="predict_proba")[:, 1]
    m1 = analyser_matrice(y, proba_lr, SEUIL_LR)
    tracer_matrice(m1, "Étage 1 : régression logistique (proba_lr)", DOSSIER / "confusion_etage1_lr.png")

    lr = creer_stacking().fit(X, y).estimators_[0]   # la LR exacte dont la sortie alimente XGBoost
    Xs, contrib_lr = contributions_lr(lr, X)
    imp_lr = importance_lr(contrib_lr)
    imp_lr.round(4).to_csv(DOSSIER / "shap_etage1_lr_importance.csv", index=False)
    tracer_shap_lr(contrib_lr, Xs, DOSSIER / "shap_etage1_lr_summary.png")

    # ---- Étage 2 : XGBoost, qui produit le score final
    proba_final = cross_val_predict(creer_stacking(), X, y, cv=cv, method="predict_proba")[:, 1]
    seuil = seuil_max_f1(y, proba_final)
    m2 = analyser_matrice(y, proba_final, seuil)
    tracer_matrice(m2, "Étage 2 : XGBoost (score final)", DOSSIER / "confusion_etage2_xgboost.png")

    modele = creer_stacking().fit(X, y)
    entrees, contrib_xgb = explain.contributions(explain.creer_explainer(modele), modele, X)
    imp_xgb = explain.importance_globale(contrib_xgb)
    imp_xgb.round(4).to_csv(DOSSIER / "shap_etage2_xgboost_importance.csv", index=False)
    explain.tracer_resume(contrib_xgb, entrees, DOSSIER / "shap_etage2_xgboost_summary.png")

    # ---- Tableau comparatif des deux étages
    tableau = pd.DataFrame([{"etage": "1 : LR (proba_lr)", **m1}, {"etage": "2 : XGBoost (score final)", **m2}])
    tableau.to_csv(DOSSIER / "analyse_etages.csv", index=False)

    # ---- Lecture
    for nom, m, imp in (("Étage 1 : régression logistique", m1, imp_lr), ("Étage 2 : XGBoost", m2, imp_xgb)):
        print(f"{nom}  (seuil {m['seuil']})")
        print(f"  vrais négatifs {m['vrais_negatifs']:5d}   faux positifs {m['faux_positifs']:4d}")
        print(f"  faux négatifs  {m['faux_negatifs']:5d}   vrais positifs {m['vrais_positifs']:4d}")
        print(f"  pannes détectées {m['pannes_detectees_%']} %, fausses alertes {m['fausses_alertes_sur_normales_%']} % des "
              f"machines normales, alertes justifiées {m['alertes_justifiees_%']} %")
        print(f"  F1 {m['f1']}, F1 macro {m['f1_macro']}, accuracy {m['accuracy']}")
        print("  SHAP, 3 premières : " + ", ".join(f"{v} ({p:.0%})" for v, p in
                                                    zip(imp['variable'][:3], imp['part'][:3])))
        print()
    gain = m2["vrais_positifs"] - m1["vrais_positifs"]
    evite = m1["faux_positifs"] - m2["faux_positifs"]
    print(f"Passage de l'étage 1 à l'étage 2 : {gain:+d} pannes détectées, {-evite:+d} fausses alertes "
          "(XGBoost corrige surtout les fausses alertes de la LR).")
    print(f"Fichiers dans {DOSSIER}/ : confusion_etage1_lr.png, confusion_etage2_xgboost.png, "
          "shap_etage1_lr_summary.png, shap_etage2_xgboost_summary.png, analyse_etages.csv, "
          "shap_etage*_importance.csv")


if __name__ == "__main__":
    main()
