# Predictive Maintenance Anomaly Detection

[![CI](https://github.com/mohamed-kar1/predictive-maintenance-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamed-kar1/predictive-maintenance-anomaly-detection/actions/workflows/ci.yml)

Maintenance prédictive par détection d'anomalies multivariées sur le jeu de données public **AI4I 2020** (10 000 machines, 6 mesures, pannes étiquetées). Le projet couvre toute la chaîne : donnée brute, entrepôt testé, modèle suivi dans MLflow, API de prédiction avec interface web.

## Démarrage rapide (une seule commande)

Prérequis : [Docker](https://docs.docker.com/get-docker/) avec Compose.

```bash
git clone https://github.com/mohamed-kar1/predictive-maintenance-anomaly-detection.git
cd predictive-maintenance-anomaly-detection
docker compose up --build
```

Au premier démarrage, le conteneur prépare tout seul : jeu de données, ingestion, `dbt build` (modèles et tests), entraînement du modèle final, enregistrement dans MLflow (alias `champion`). Compter quelques minutes. Les démarrages suivants sont immédiats : l'état est conservé dans le volume Docker `state`.

| Service | URL |
|---|---|
| Interface web de prédiction | http://localhost:8000 |
| Documentation de l'API (Swagger) | http://localhost:8000/docs |

Outils optionnels (suivi des essais et orchestration) :

```bash
docker compose --profile tools up --build
```

| Service | URL |
|---|---|
| MLflow (essais, registre de modèles) | http://localhost:5000 |
| Dagster (pipeline complet, 4 détecteurs + modèle composé) | http://localhost:3000 |

Repartir de zéro : `docker compose down -v` (supprime le volume d'état).

## Architecture

```
data/raw/ai4i2020.csv
   │  dlt                      src/ingestion.py
   ▼
DuckDB  raw.sensor_readings
   │  dbt (staging → marts)    dbt_project/   (16 tests de données)
   ▼
marts.fct_machine_features     10 variables dérivées (puissance, surcharge, écart de température…)
   │  scikit-learn / XGBoost   src/train_*.py
   ▼
MLflow  (essais + registre)    modèle « ai4i_stacking_10var », alias champion
   │
   ▼
API FastAPI + interface web    src/api.py   (journal des prédictions : journal.duckdb)

Orchestration de l'ensemble : Dagster (orchestration/definitions.py)
```

Points de conception :

- **Une seule définition des variables.** `src/features.py` calcule en Python les mêmes variables que dbt ; un test (`test_coherence_dbt.py`) vérifie qu'elles sont égales machine par machine.
- **Évaluation sans fuite de données.** Validation croisée à 5 plis ; le modèle composé LR → XGBoost entraîne sa régression logistique en validation croisée interne ; le seuil d'anomalie est choisi sur des prédictions hors échantillon (maximisation du F1).
- **Qualité des données.** Les 27 lignes aux étiquettes contradictoires du jeu d'origine sont repérées par dbt et exclues de l'entraînement. Un contrôle bloquant Dagster exige exactement 10 000 lignes avant tout entraînement.
- **Modèle final** : empilement LR → XGBoost sur 10 variables. Les comparaisons chiffrées sont dans `reports/`.

## API

| Route | Rôle |
|---|---|
| `GET /health` | Version du modèle chargé et seuils |
| `POST /predict` | Une machine (6 mesures) → résultat, score, niveau d'alerte |
| `POST /predict_csv` | Un fichier CSV, une machine par ligne |
| `GET /historique` | Dernières prédictions du journal |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "quality_level": "L", "air_temperature_k": 298.1, "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1551, "torque_nm": 57, "tool_wear_min": 200
}'
```

## Installation locale (sans Docker)

Python 3.12 recommandé. Les versions des dépendances sont figées dans `requirements.txt`.

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS : source venv/bin/activate)
pip install -r requirements.txt

python scripts/bootstrap.py      # données -> dbt -> modèle final -> alias champion
python src/api.py                # http://127.0.0.1:8000
```

Pipeline complet avec Dagster (comparaison des modèles, essais MLflow) :

```bash
dagster dev -f orchestration/definitions.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Tests et vérification automatique

```bash
python src/ingestion.py
cd dbt_project && dbt build --profiles-dir . && cd ..
python -m pytest
```

39 tests pytest (variables, API, journal, cohérence Python/dbt) et le `dbt build` sont rejoués à chaque `git push` par GitHub Actions (`.github/workflows/ci.yml`), qui vérifie aussi que l'image Docker se construit.

## Structure

```
src/             ingestion, variables, entraînement, API, interface web
dbt_project/     modèles staging et marts, tests de données
orchestration/   définitions Dagster
tests/           tests pytest
scripts/         préparation des données et démarrage à froid
docker/          entrypoint du conteneur
reports/         comparaisons de modèles (CSV)
```

## Données

AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository (jeu synthétique public). Le fichier `data/raw/ai4i2020.csv` est versionné ; s'il est absent, `scripts/fetch_data.py` le télécharge depuis UCI.
