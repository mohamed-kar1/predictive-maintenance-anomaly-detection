"""Outils communs : un modèle factice et un journal temporaire, pour tester l'API sans MLflow."""
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
import journal

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
    api.ETAT.update(modele=ModeleFactice(), version=99, seuil=0.575, seuil_eleve=0.8)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Client de test : modèle factice, et journal dans un dossier temporaire (le vrai n'est pas touché)."""
    monkeypatch.setattr(journal, "CHEMIN", tmp_path / "journal.duckdb")
    monkeypatch.setattr(api, "charger_modele", _charger_modele_factice)
    with TestClient(api.app) as c:
        yield c
