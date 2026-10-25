"""Tests de l'analyse par étage : matrice de confusion et SHAP de la régression logistique."""
import numpy as np
import pytest

import analyse_etages as ae
from conftest import construire_modele_reel
from train_stacking import FEATURES_10


def test_matrice_de_confusion_effectifs_et_taux():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.7, 0.3, 0.9, 0.8, 0.4, 0.6])
    m = ae.analyser_matrice(y, proba, 0.5)
    assert (m["vrais_negatifs"], m["faux_positifs"], m["faux_negatifs"], m["vrais_positifs"]) == (3, 1, 1, 3)
    assert m["pannes_detectees_%"] == 75.0 and m["alertes_justifiees_%"] == 75.0
    assert m["accuracy"] == pytest.approx(0.75)


def test_seuil_max_f1_separe_deux_groupes_parfaits():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert 0.3 < ae.seuil_max_f1(y, proba) <= 0.8


def test_tracer_matrice_ecrit_un_png(tmp_path):
    m = ae.analyser_matrice(np.array([0, 1, 0, 1]), np.array([0.2, 0.9, 0.6, 0.4]), 0.5)
    ae.tracer_matrice(m, "test", tmp_path / "m.png")
    assert (tmp_path / "m.png").stat().st_size > 0


def test_shap_lr_additivite_et_importance():
    """Valeur de base + somme des contributions = log-odds de la LR (donc proba_lr)."""
    modele, X = construire_modele_reel()
    lr = modele.estimators_[0]
    Xs, contrib = ae.contributions_lr(lr, X[:80])
    assert contrib.shape == (80, 10)
    p = lr.predict_proba(X[:80])[:, 1]
    logit = np.log(p / (1 - p))
    base = logit - contrib.sum(axis=1)
    np.testing.assert_allclose(base, base[0], atol=1e-6)   # même valeur de base pour toutes les machines
    imp = ae.importance_lr(contrib)
    assert list(imp.columns) == ["variable", "shap_abs_moyen", "part"]
    assert set(imp["variable"]) == set(FEATURES_10) and imp["part"].sum() == pytest.approx(1.0)
