"""Calcul des variables du modèle, identique à dbt (marts.fct_machine_features).

Le modèle est entraîné sur les 10 variables de FEATURES, dans cet ordre. L'API doit appeler
calculer_variables() puis passer le résultat au modèle : aucun calcul de variable ailleurs.

Exemple :
    import pandas as pd
    from features import calculer_variables
    X = calculer_variables(pd.DataFrame([{
        "quality_level": "L", "air_temperature_k": 298.1, "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551, "torque_nm": 42.8, "tool_wear_min": 0}]))
"""
import numpy as np
import pandas as pd

# Les 6 mesures fournies par le technicien.
MESURES = ["quality_level", "air_temperature_k", "process_temperature_k",
           "rotational_speed_rpm", "torque_nm", "tool_wear_min"]

QUALITE_ORD = {"L": 0, "M": 1, "H": 2}
SEUIL_SURCHARGE = {"L": 11000.0, "M": 12000.0, "H": 13000.0}   # usure x couple, par qualité

# Ordre exact des variables vues par le modèle à l'entraînement.
FEATURES = ["quality_level_ord", "air_temperature_k", "process_temperature_k",
            "rotational_speed_rpm", "torque_nm", "tool_wear_min",
            "temp_diff_k", "power_w", "wear_x_torque", "overstrain_ratio"]


def calculer_variables(mesures: pd.DataFrame) -> pd.DataFrame:
    """Reçoit les 6 mesures, renvoie les 10 variables du modèle (colonnes dans l'ordre FEATURES)."""
    manquantes = [c for c in MESURES if c not in mesures.columns]
    if manquantes:
        raise ValueError(f"Mesures manquantes : {manquantes}")
    if mesures[MESURES].isna().any().any():
        raise ValueError("Valeurs manquantes dans les mesures")

    qualite = mesures["quality_level"].astype(str).str.strip().str.upper()
    inconnues = sorted(set(qualite) - set(QUALITE_ORD))
    if inconnues:
        raise ValueError(f"Niveau de qualité inconnu : {inconnues} (attendu : L, M ou H)")

    air = mesures["air_temperature_k"].astype(float)
    process = mesures["process_temperature_k"].astype(float)
    vitesse = mesures["rotational_speed_rpm"].astype(float)
    couple = mesures["torque_nm"].astype(float)
    usure = mesures["tool_wear_min"].astype(float)

    out = pd.DataFrame(index=mesures.index)
    out["quality_level_ord"] = qualite.map(QUALITE_ORD)
    out["air_temperature_k"] = air
    out["process_temperature_k"] = process
    out["rotational_speed_rpm"] = vitesse
    out["torque_nm"] = couple
    out["tool_wear_min"] = usure
    out["temp_diff_k"] = process - air
    out["power_w"] = couple * vitesse * 2 * np.pi / 60
    out["wear_x_torque"] = usure * couple
    out["overstrain_ratio"] = usure * couple / qualite.map(SEUIL_SURCHARGE)
    return out[FEATURES].astype(float)