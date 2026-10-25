"""Outils communs : un modèle factice et un journal temporaire, pour tester l'API sans MLflow."""
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
import explain
import journal
from features import calculer_variables

# Une machine de la couche « normale » du jeu AI4I, et deux cas de surcharge (qualité L).
MACHINE_NORMALE = dict(quality_level="L", air_temperature_k=298.1, process_temperature_k=308.6,
                       rotational_speed_rpm=1551, torque_nm=42.8, tool_wear_min=0)
SURCHARGE = dict(MACHINE_NORMALE, torque_nm=57, tool_wear_min=200)        # usure x couple = 11 400
SURCHARGE_FORTE = dict(MACHINE_NORMALE, torque_nm=60, tool_wear_min=250)  # usure x couple = 15 000


class ModeleFactice:
    """Imite le modèle MLflow : le score dépend du seul overstrain_ratio (dixième variable)."""

    def predict_proba(self, X):
        assert X.shape[1] == 10, f"10 variables attendues, reçu {X.shape[1]}"
        p = 1 / (1 + np.exp(-10 * (X[:, 9] - 1)))
        return np.column_stack([1 - p, p])


def _charger_modele_factice():
    api.ETAT.update(modele=ModeleFactice(), explainer=None, version=99, seuil=0.575, seuil_eleve=0.8)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Client de test : modèle factice, et journal dans un dossier temporaire (le vrai n'est pas touché)."""
    monkeypatch.setattr(journal, "CHEMIN", tmp_path / "journal.duckdb")
    monkeypatch.setattr(api, "charger_modele", _charger_modele_factice)
    with TestClient(api.app) as c:
        yield c


def construire_modele_reel(n=600):
    """Petit modèle composé réel (LR -> XGBoost) sur des mesures simulées : la panne = surcharge."""
    from train_final_stacking import creer_stacking
    rng = np.random.default_rng(0)
    import pandas as pd
    mesures = pd.DataFrame({
        "quality_level": rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1]),
        "air_temperature_k": rng.normal(300, 2, n), "process_temperature_k": rng.normal(310, 1, n),
        "rotational_speed_rpm": rng.normal(1500, 150, n), "torque_nm": rng.normal(40, 10, n).clip(3, 80),
        "tool_wear_min": rng.uniform(0, 250, n)})
    X = calculer_variables(mesures)
    y = (X["overstrain_ratio"] > 0.9).astype(int).values
    return creer_stacking().fit(X.values, y), X.values


@pytest.fixture(scope="session")
def modele_reel():
    return construire_modele_reel()[0]


@pytest.fixture
def client_shap(monkeypatch, tmp_path, modele_reel):
    """Client de test avec un vrai modèle composé et son explainer SHAP."""
    def charger():
        api.ETAT.update(modele=modele_reel, explainer=explain.creer_explainer(modele_reel),
                        version=98, seuil=0.5, seuil_eleve=0.8)
    monkeypatch.setattr(journal, "CHEMIN", tmp_path / "journal.duckdb")
    monkeypatch.setattr(api, "charger_modele", charger)
    with TestClient(api.app) as c:
        yield c
