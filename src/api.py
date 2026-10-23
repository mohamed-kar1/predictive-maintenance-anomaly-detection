"""API de prédiction de maintenance prédictive (FastAPI).

Au démarrage, charge le modèle portant l'alias « champion » dans le registre MLflow, ainsi que
son seuil d'anomalie (paramètres de l'essai qui l'a produit). Les variables sont calculées par
features.py, le même code qu'à l'entraînement.

Lancer depuis la racine du projet :
    python src/api.py
puis ouvrir http://127.0.0.1:8000/docs
"""
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from mlflow import MlflowClient
from pydantic import BaseModel, Field, field_validator

from features import FEATURES, MESURES, calculer_variables

log = logging.getLogger("api")

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
NOM_MODELE = "ai4i_stacking_10var"
ALIAS = "champion"
MAX_LIGNES_CSV = 50_000

# Bornes de plausibilité (larges : le jeu AI4I reste très en dessous).
BORNES = {
    "air_temperature_k": (250.0, 350.0),
    "process_temperature_k": (250.0, 350.0),
    "rotational_speed_rpm": (1.0, 10000.0),
    "torque_nm": (0.0, 200.0),
    "tool_wear_min": (0.0, 1000.0),
}

# En-têtes du fichier AI4I d'origine, acceptés en plus de nos noms de colonnes.
ALIAS_COLONNES = {
    "udi": "udi", "product id": "product_id", "type": "quality_level",
    "air temperature [k]": "air_temperature_k", "process temperature [k]": "process_temperature_k",
    "rotational speed [rpm]": "rotational_speed_rpm", "torque [nm]": "torque_nm",
    "tool wear [min]": "tool_wear_min",
}

ETAT = {}   # modèle, version et seuils, remplis au démarrage


def charger_modele():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(NOM_MODELE, ALIAS)
    if not version.run_id:
        raise RuntimeError("La version du modèle n'est liée à aucun essai : seuil introuvable.")
    params = client.get_run(version.run_id).data.params
    if params.get("variables", "").split(",") != FEATURES:
        raise RuntimeError(
            f"Le modèle « {ALIAS} » n'a pas été entraîné sur les variables de features.py.\n"
            f"  modèle     : {params.get('variables')}\n  features.py: {','.join(FEATURES)}")
    modele = mlflow.sklearn.load_model(f"models:/{NOM_MODELE}@{ALIAS}")
    ETAT.update(modele=modele, version=int(version.version),
                seuil=float(params["seuil_anomalie"]),
                seuil_eleve=float(params.get("seuil_alerte_elevee", 0.8)))
    log.warning("Modèle chargé : %s version %s, seuil %.3f", NOM_MODELE, version.version, ETAT["seuil"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    charger_modele()
    yield


app = FastAPI(title="Maintenance prédictive : détection d'anomalies", version="1.0", lifespan=lifespan)


class Machine(BaseModel):
    """Les six mesures d'une machine."""
    quality_level: Literal["L", "M", "H"] = Field(description="Niveau de qualité : L, M ou H")
    air_temperature_k: float = Field(ge=BORNES["air_temperature_k"][0], le=BORNES["air_temperature_k"][1],
                                     description="Température de l'air (K)")
    process_temperature_k: float = Field(ge=BORNES["process_temperature_k"][0], le=BORNES["process_temperature_k"][1],
                                         description="Température du process (K)")
    rotational_speed_rpm: float = Field(ge=BORNES["rotational_speed_rpm"][0], le=BORNES["rotational_speed_rpm"][1],
                                        description="Vitesse de rotation (tr/min)")
    torque_nm: float = Field(ge=BORNES["torque_nm"][0], le=BORNES["torque_nm"][1], description="Couple (Nm)")
    tool_wear_min: float = Field(ge=BORNES["tool_wear_min"][0], le=BORNES["tool_wear_min"][1],
                                 description="Usure de l'outil (min)")

    @field_validator("quality_level", mode="before")
    @classmethod
    def _qualite(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


class Prediction(BaseModel):
    resultat: Literal["normal", "anormal"]
    score: float
    niveau_alerte: Literal["aucune", "modérée", "élevée"]
    seuil: float


def decision(score: float) -> dict:
    anormal = score >= ETAT["seuil"]
    if not anormal:
        niveau = "aucune"
    elif score >= ETAT["seuil_eleve"]:
        niveau = "élevée"
    else:
        niveau = "modérée"
    return {"resultat": "anormal" if anormal else "normal",
            "score": round(float(score), 4), "niveau_alerte": niveau}


def scorer(mesures: pd.DataFrame):
    X = calculer_variables(mesures).values   # tableau nu : comme à l'entraînement
    return ETAT["modele"].predict_proba(X)[:, 1]


@app.get("/health")
def health():
    return {"statut": "ok", "modele": NOM_MODELE, "version": ETAT["version"],
            "seuil_anomalie": ETAT["seuil"], "seuil_alerte_elevee": ETAT["seuil_eleve"]}


@app.post("/predict", response_model=Prediction)
def predict(machine: Machine):
    """Une machine : renvoie normal ou anormal, le score et le niveau d'alerte."""
    score = scorer(pd.DataFrame([machine.model_dump()]))[0]
    return {**decision(score), "seuil": ETAT["seuil"]}


def lire_csv(contenu: bytes) -> pd.DataFrame:
    # Séparateur « ; » et décimale « , » (Excel en français) ou « , » et « . » (format standard).
    premiere_ligne = contenu.decode("utf-8-sig", errors="replace").splitlines()[0] if contenu.strip() else ""
    sep = ";" if premiere_ligne.count(";") > premiere_ligne.count(",") else ","
    try:
        df = pd.read_csv(io.BytesIO(contenu), sep=sep, decimal="," if sep == ";" else ".",
                         encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(422, f"Fichier CSV illisible : {e}")
    df = df.rename(columns=lambda c: ALIAS_COLONNES.get(str(c).strip().lower(), str(c).strip().lower()))
    manquantes = [c for c in MESURES if c not in df.columns]
    if manquantes:
        raise HTTPException(422, f"Colonnes manquantes : {manquantes}. Colonnes attendues : {MESURES}")
    if df.empty:
        raise HTTPException(422, "Le fichier ne contient aucune machine.")
    if len(df) > MAX_LIGNES_CSV:
        raise HTTPException(422, f"Trop de lignes ({len(df)}), maximum {MAX_LIGNES_CSV}.")

    erreurs = []   # (numéro de ligne dans le fichier, message)
    qualite = df["quality_level"].astype(str).str.strip().str.upper()
    for i in df.index[~qualite.isin(["L", "M", "H"])]:
        erreurs.append((i, f"ligne {i + 2} : qualité « {df.at[i, 'quality_level']} » invalide (L, M ou H)"))
    for col, (mini, maxi) in BORNES.items():
        v = pd.to_numeric(df[col], errors="coerce")
        for i in df.index[v.isna() | (v < mini) | (v > maxi)]:
            erreurs.append((i, f"ligne {i + 2} : {col} = {df.at[i, col]} hors de [{mini}, {maxi}]"))
        df[col] = v
    if erreurs:
        erreurs.sort(key=lambda e: e[0])
        raise HTTPException(422, {"message": f"{len(erreurs)} erreur(s) dans le fichier",
                                  "exemples": [m for _, m in erreurs[:10]]})
    return df


@app.post("/predict_csv")
async def predict_csv(fichier: UploadFile = File(..., description="CSV : une machine par ligne")):
    """Plusieurs machines : un résultat par ligne du fichier."""
    df = lire_csv(await fichier.read())
    scores = scorer(df)
    id_col = next((c for c in ("udi", "product_id", "id") if c in df.columns), None)
    resultats = []
    for pos, score in enumerate(scores):
        r = {"ligne": pos + 1, **decision(score)}
        if id_col:
            r["identifiant"] = str(df.iloc[pos][id_col])
        resultats.append(r)
    return {"n_machines": len(resultats),
            "n_anormales": sum(r["resultat"] == "anormal" for r in resultats),
            "n_alerte_elevee": sum(r["niveau_alerte"] == "élevée" for r in resultats),
            "seuil": ETAT["seuil"], "resultats": resultats}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
