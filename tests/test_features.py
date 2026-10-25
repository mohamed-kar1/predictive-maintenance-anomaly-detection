"""Tests du calcul des variables (features.py), qui doit être identique à dbt."""
import math

import pandas as pd
import pytest

from features import FEATURES, MESURES, calculer_variables

MACHINE = {"quality_level": "L", "air_temperature_k": 298.1, "process_temperature_k": 308.6,
           "rotational_speed_rpm": 1551, "torque_nm": 42.8, "tool_wear_min": 0}


def une_machine(**changements) -> pd.DataFrame:
    return pd.DataFrame([{**MACHINE, **changements}])


def test_ordre_des_dix_variables():
    # Le modèle a été entraîné sur des tableaux sans noms : l'ordre ne doit jamais changer.
    assert FEATURES == ["quality_level_ord", "air_temperature_k", "process_temperature_k",
                        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
                        "temp_diff_k", "power_w", "wear_x_torque", "overstrain_ratio"]
    assert list(calculer_variables(une_machine()).columns) == FEATURES


def test_variables_calculees():
    v = calculer_variables(une_machine(torque_nm=57, tool_wear_min=200)).iloc[0]
    assert v["temp_diff_k"] == pytest.approx(10.5)
    assert v["power_w"] == pytest.approx(57 * 1551 * 2 * math.pi / 60)
    assert v["wear_x_torque"] == pytest.approx(11400)
    assert v["overstrain_ratio"] == pytest.approx(11400 / 11000)


def test_puissance_connue():
    # Valeur calculée à part : 42,8 Nm x 1 551 tr/min x 2 pi / 60.
    assert calculer_variables(une_machine()).iloc[0]["power_w"] == pytest.approx(6951.59056, rel=1e-6)


@pytest.mark.parametrize("qualite, code, seuil", [("L", 0, 11000), ("M", 1, 12000), ("H", 2, 13000)])
def test_qualite_et_seuil_de_surcharge(qualite, code, seuil):
    v = calculer_variables(une_machine(quality_level=qualite, torque_nm=57, tool_wear_min=200)).iloc[0]
    assert v["quality_level_ord"] == code
    assert v["overstrain_ratio"] == pytest.approx(11400 / seuil)


def test_meme_valeur_surcharge_en_L_pas_en_H():
    ratio = lambda q: calculer_variables(une_machine(quality_level=q, torque_nm=57, tool_wear_min=200)).iloc[0]["overstrain_ratio"]
    assert ratio("L") > 1 > ratio("H")


@pytest.mark.parametrize("saisie", ["l", " M ", "h"])
def test_qualite_tolerante(saisie):
    assert calculer_variables(une_machine(quality_level=saisie)).iloc[0]["quality_level_ord"] in (0, 1, 2)


def test_qualite_inconnue_refusee():
    with pytest.raises(ValueError, match="qualité"):
        calculer_variables(une_machine(quality_level="X"))


def test_mesure_absente_refusee():
    with pytest.raises(ValueError, match="manquantes"):
        calculer_variables(une_machine().drop(columns=["torque_nm"]))


def test_valeur_manquante_refusee():
    with pytest.raises(ValueError, match="manquantes"):
        calculer_variables(une_machine(torque_nm=None))


def test_plusieurs_machines_et_index_conserve():
    df = pd.DataFrame([MACHINE, {**MACHINE, "quality_level": "H"}], index=[10, 20])
    out = calculer_variables(df)
    assert list(out.index) == [10, 20] and out.shape == (2, 10)
