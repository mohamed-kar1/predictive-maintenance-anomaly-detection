"""Tests des métriques de décision (accuracy, précision, F1, versions macro)."""
import numpy as np
import pytest

from train_models import mesurer, metriques_decision


def test_valeurs_calculees_a_la_main():
    # 10 machines dont 2 pannes : 1 panne trouvée, 1 manquée, 1 fausse alerte, 7 normales reconnues.
    y = np.array([0] * 8 + [1, 1])
    pred = np.array([0] * 7 + [1, 1, 0])
    m = metriques_decision(y, pred)
    assert m["accuracy"] == pytest.approx(0.8)
    assert (m["precision"], m["rappel"], m["f1"]) == (pytest.approx(0.5),) * 3
    # Classe normale : précision = rappel = F1 = 7/8. Macro = moyenne des deux classes.
    assert m["precision_macro"] == pytest.approx((7 / 8 + 0.5) / 2)
    assert m["f1_macro"] == pytest.approx((7 / 8 + 0.5) / 2)


def test_accuracy_trompeuse_mais_f1_macro_honnete():
    y = np.array([0] * 96 + [1] * 4)
    m = metriques_decision(y, np.zeros(100, dtype=int))   # « jamais de panne »
    assert m["accuracy"] == pytest.approx(0.96)
    assert m["f1"] == 0 and m["rappel"] == 0
    assert m["f1_macro"] < 0.5   # la classe rare, jamais trouvée, tire la moyenne vers le bas


def test_mesurer_renvoie_toutes_les_metriques():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    score = rng.random(100) + y   # les pannes ont des scores plus élevés
    m = mesurer(y, score, taux=0.1)
    assert set(m) == {"pr_auc", "precision", "rappel", "f1", "accuracy", "precision_macro", "f1_macro"}
    assert all(0 <= v <= 1 for v in m.values())
    # On signale 10 % des machines pour 10 % de pannes : précision = rappel.
    assert m["precision"] == pytest.approx(m["rappel"])
