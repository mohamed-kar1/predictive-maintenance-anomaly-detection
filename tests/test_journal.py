"""Tests du journal des prédictions (journal temporaire)."""
from datetime import datetime

import pandas as pd
import pytest

import journal


@pytest.fixture(autouse=True)
def journal_temporaire(monkeypatch, tmp_path):
    monkeypatch.setattr(journal, "CHEMIN", tmp_path / "journal.duckdb")


def lot(resultats):
    n = len(resultats)
    return pd.DataFrame({
        "horodatage": [datetime(2026, 10, 24, 12, 0, i) for i in range(n)], "source": ["csv"] * n,
        "fichier": ["a.csv"] * n, "identifiant": [str(i) for i in range(n)], "quality_level": ["L"] * n,
        "air_temperature_k": [298.1] * n, "process_temperature_k": [308.6] * n,
        "rotational_speed_rpm": [1551.0] * n, "torque_nm": [42.8] * n, "tool_wear_min": [0.0] * n,
        "score": [0.9 if r == "anormal" else 0.1 for r in resultats], "resultat": list(resultats),
        "niveau_alerte": ["élevée" if r == "anormal" else "aucune" for r in resultats],
        "modele_version": [2] * n, "seuil": [0.5748] * n})


def test_journal_vide():
    assert journal.resume() == {"total": 0, "anormales": 0, "alertes_elevees": 0}
    assert journal.derniers() == []


def test_enregistrer_et_resumer():
    journal.enregistrer(lot(["normal", "anormal", "anormal"]))
    assert journal.resume() == {"total": 3, "anormales": 2, "alertes_elevees": 2}


def test_les_ajouts_s_accumulent():
    journal.enregistrer(lot(["normal"]))
    journal.enregistrer(lot(["anormal", "normal"]))
    assert journal.resume()["total"] == 3


def test_derniers_du_plus_recent_au_plus_ancien():
    journal.enregistrer(lot(["normal", "anormal", "normal"]))
    d = journal.derniers(2)
    assert len(d) == 2
    assert d[0]["identifiant"] == "2" and d[1]["identifiant"] == "1"
    assert d[0]["horodatage"] == "24/10 12:00:02"


def test_colonnes_facultatives_absentes():
    journal.enregistrer(lot(["normal"]).drop(columns=["fichier", "identifiant"]))
    d = journal.derniers(1)[0]
    assert d["fichier"] is None and d["identifiant"] is None
