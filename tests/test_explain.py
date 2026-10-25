"""Tests des explications SHAP, sur un petit modèle composé réel (même structure que le modèle final)."""
import numpy as np
import pytest

import explain
from conftest import construire_modele_reel


@pytest.fixture(scope="module")
def modele_et_donnees():
    return construire_modele_reel()


def test_entrees_de_xgboost_sont_proba_lr_puis_les_10_variables(modele_et_donnees):
    modele, X = modele_et_donnees
    entrees = explain.entrees_xgboost(modele, X)
    assert entrees.shape == (len(X), 11) and len(explain.NOMS_ENTREES) == 11
    assert explain.NOMS_ENTREES[0] == "proba_lr"
    np.testing.assert_allclose(entrees[:, 0], modele.estimators_[0].predict_proba(X)[:, 1])
    np.testing.assert_allclose(entrees[:, 1:], X)


def test_additivite_des_contributions(modele_et_donnees):
    """Valeur de base + somme des contributions = log-odds du score du modèle."""
    modele, X = modele_et_donnees
    explainer = explain.creer_explainer(modele)
    entrees, contrib = explain.contributions(explainer, modele, X[:50])
    assert contrib.shape == (50, 11)
    p = modele.predict_proba(X[:50])[:, 1]
    logit = np.log(p / (1 - p))
    np.testing.assert_allclose(float(np.ravel(explainer.expected_value)[0]) + contrib.sum(axis=1), logit, atol=1e-3)


def test_top_contributions_tri_par_poids_absolu(modele_et_donnees):
    modele, X = modele_et_donnees
    entrees, contrib = explain.contributions(explain.creer_explainer(modele), modele, X[:1])
    top = explain.top_contributions(entrees[0], contrib[0], 3)
    assert len(top) == 3
    poids = [abs(c["contribution"]) for c in top]
    assert poids == sorted(poids, reverse=True)
    assert all(c["variable"] in explain.NOMS_ENTREES for c in top)


def test_importance_globale_somme_a_un(modele_et_donnees):
    modele, X = modele_et_donnees
    _, contrib = explain.contributions(explain.creer_explainer(modele), modele, X)
    imp = explain.importance_globale(contrib)
    assert len(imp) == 11 and imp["part"].sum() == pytest.approx(1.0)
    assert imp["shap_abs_moyen"].is_monotonic_decreasing
