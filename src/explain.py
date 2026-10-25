"""Explications SHAP du modèle composé LR -> XGBoost.

Le modèle est un StackingClassifier avec passthrough : XGBoost (l'estimateur final) ne voit pas
seulement les 10 variables, il voit aussi la probabilité calculée par la régression logistique.
Les explications portent donc sur 11 entrées : « proba_lr » + les 10 variables de features.py.

Les contributions sont en log-odds (échelle interne de XGBoost) :
  - positive : la variable pousse vers « anormal » ;
  - négative : la variable pousse vers « normal ».
La somme des contributions + la valeur de base donne le log-odds du score final.

Utilisé par train_final_stacking.py (importance globale) et par api.py (explication d'une machine).
"""
import numpy as np
import shap

from features import FEATURES

PROBA_LR = "proba_lr"
NOMS_ENTREES = [PROBA_LR] + FEATURES   # ordre du passthrough de StackingClassifier : prédictions, puis X


def creer_explainer(modele) -> "shap.TreeExplainer":
    """Explainer d'arbres sur l'estimateur final (XGBoost) du modèle composé."""
    return shap.TreeExplainer(modele.final_estimator_)


def entrees_xgboost(modele, X: np.ndarray) -> np.ndarray:
    """Ce que XGBoost reçoit réellement : [proba_lr, 10 variables]."""
    entrees = modele.transform(X)
    if entrees.shape[1] != len(NOMS_ENTREES):
        raise ValueError(f"{len(NOMS_ENTREES)} entrées attendues pour XGBoost, obtenu {entrees.shape[1]}")
    return entrees


def contributions(explainer, modele, X: np.ndarray):
    """Renvoie (entrées de XGBoost, contributions SHAP), deux tableaux de forme (n, 11)."""
    entrees = entrees_xgboost(modele, X)
    return entrees, np.asarray(explainer.shap_values(entrees))


def top_contributions(entrees_ligne, contrib_ligne, n: int = 5) -> list:
    """Les n entrées qui pèsent le plus (en valeur absolue) sur le score d'une machine."""
    ordre = np.argsort(-np.abs(contrib_ligne))[:n]
    return [{"variable": NOMS_ENTREES[i], "valeur": round(float(entrees_ligne[i]), 4),
             "contribution": round(float(contrib_ligne[i]), 4)} for i in ordre]


def importance_globale(contrib: np.ndarray):
    """Importance moyenne (|SHAP| moyen) par entrée, triée de la plus à la moins importante."""
    import pandas as pd
    t = pd.DataFrame({"variable": NOMS_ENTREES, "shap_abs_moyen": np.abs(contrib).mean(axis=0)})
    t["part"] = t["shap_abs_moyen"] / t["shap_abs_moyen"].sum()
    return t.sort_values("shap_abs_moyen", ascending=False).reset_index(drop=True)


def tracer_resume(contrib: np.ndarray, entrees: np.ndarray, chemin) -> None:
    """Enregistre le graphique SHAP « beeswarm » (une couleur = valeur de la variable)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(contrib, entrees, feature_names=NOMS_ENTREES, show=False, max_display=len(NOMS_ENTREES))
    plt.title("Contribution des variables au score d'anomalie (SHAP, log-odds)")
    plt.tight_layout()
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
