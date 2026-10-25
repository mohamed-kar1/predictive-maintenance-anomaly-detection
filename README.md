# Predictive Maintenance Anomaly Detection

[![CI](https://github.com/mohamed-kar1/predictive-maintenance-anomaly-detection/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mohamed-kar1/predictive-maintenance-anomaly-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)

Maintenance prédictive par détection d'anomalies multivariées sur le jeu de données public **AI4I 2020** (10 000 machines, 6 mesures, pannes étiquetées).

Le projet couvre toute la chaîne d'un projet data de bout en bout : donnée brute → entrepôt testé → comparaison de modèles → suivi des essais → API de prédiction avec interface web → vérification automatique à chaque `git push`. Un technicien saisit les mesures d'une machine (ou charge un fichier CSV) et obtient un diagnostic : **normal** ou **anormal**, avec un score et un niveau d'alerte.

---

## Sommaire

1. [Démarrage rapide](#1-démarrage-rapide-une-seule-commande)
2. [Architecture](#2-architecture)
3. [Les données](#3-les-données)
4. [Variables du modèle](#4-variables-du-modèle)
5. [Modélisation et résultats](#5-modélisation-et-résultats)
6. [API et interface web](#6-api-et-interface-web)
7. [Orchestration, suivi des essais, journal](#7-orchestration-suivi-des-essais-journal)
8. [Tests et intégration continue](#8-tests-et-intégration-continue)
9. [Installation locale (sans Docker)](#9-installation-locale-sans-docker)
10. [Structure du dépôt](#10-structure-du-dépôt)
11. [Choix de conception](#11-choix-de-conception)
12. [Limites connues](#12-limites-connues)
13. [Dépannage](#13-dépannage)

---

## 1. Démarrage rapide (une seule commande)

Prérequis : [Docker](https://docs.docker.com/get-docker/) avec Compose (Docker Desktop sous Windows et macOS). **Docker Desktop doit être démarré.**

```bash
git clone https://github.com/mohamed-kar1/predictive-maintenance-anomaly-detection.git
cd predictive-maintenance-anomaly-detection
docker compose up --build
```

Au **premier démarrage**, le conteneur prépare tout lui-même, dans cet ordre :

1. données : utilise `data/raw/ai4i2020.csv` (versionné, sinon téléchargé depuis UCI) ;
2. ingestion avec dlt vers DuckDB ;
3. `dbt build` : modèles et tests de données ;
4. entraînement du modèle final et enregistrement dans MLflow (alias `champion`) ;
5. démarrage de l'API.

Compter **10 à 25 minutes** la toute première fois (téléchargement des dépendances selon la connexion, puis quelques minutes de préparation). Ensuite, l'image est en cache et l'état est conservé dans le volume Docker `state` : un redémarrage prend quelques secondes et affiche « Modèle "champion" déjà enregistré : rien à faire ».

L'application est prête quand les logs affichent `Uvicorn running on http://0.0.0.0:8000`.

| Service | URL |
|---|---|
| Interface web du technicien | http://localhost:8000 |
| Documentation interactive de l'API (Swagger) | http://localhost:8000/docs |
| État du modèle chargé | http://localhost:8000/health |

**Outils optionnels** (suivi des essais et orchestration) :

```bash
docker compose --profile tools up --build
```

| Service | URL |
|---|---|
| MLflow : essais et registre de modèles | http://localhost:5000 |
| Dagster : pipeline complet | http://localhost:3000 |

**Commandes utiles**

| Action | Commande |
|---|---|
| Arrêter | `Ctrl+C`, puis `docker compose down` |
| Repartir de zéro (efface modèle, bases et journal) | `docker compose down -v` |
| Relancer en arrière-plan | `docker compose up -d` |
| Voir les logs | `docker compose logs -f api` |

---

## 2. Architecture

```
data/raw/ai4i2020.csv
        │   dlt                         src/ingestion.py
        ▼
DuckDB · schéma raw                     raw.sensor_readings
        │   dbt                         dbt_project/
        ▼
staging.stg_sensor_readings             nettoyage + indicateur de cohérence des étiquettes
        │   dbt
        ▼
marts.fct_machine_features              10 variables prêtes pour l'entraînement
        │   scikit-learn · XGBoost      src/train_*.py
        ▼
MLflow (SQLite)                         essais + registre « ai4i_stacking_10var » (alias champion)
        │   chargement au démarrage
        ▼
API FastAPI + interface web             src/api.py · src/interface.html
        │
        ▼
Journal des prédictions                 journal.duckdb

Orchestration de l'ensemble : Dagster   orchestration/definitions.py
Vérification automatique : GitHub Actions   .github/workflows/ci.yml
```

| Brique | Rôle | Outil |
|---|---|---|
| Ingestion | Charge le CSV dans la couche brute | dlt |
| Entrepôt | Stockage analytique local, un seul fichier | DuckDB |
| Transformation et qualité | Modèles SQL et tests de données | dbt (dbt-duckdb) |
| Modélisation | Détecteurs d'anomalies et modèle composé | scikit-learn, XGBoost |
| Suivi | Essais, métriques, registre de modèles | MLflow |
| Service | API de prédiction et interface | FastAPI, Uvicorn |
| Orchestration | Chaîne complète avec contrôles bloquants | Dagster |
| Reproductibilité | Environnement figé, une commande | Docker, Compose |
| Vérification | Tests et `dbt build` à chaque push | GitHub Actions |

---

## 3. Les données

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository) : jeu **synthétique** public qui imite des données de machines-outils.

| Colonne d'origine | Nom dans le projet | Sens |
|---|---|---|
| `UDI`, `Product ID` | `udi`, `product_id` | Identifiants |
| `Type` | `quality_level` | Qualité du produit : L (faible), M (moyenne), H (haute) |
| `Air temperature [K]` | `air_temperature_k` | Température de l'air |
| `Process temperature [K]` | `process_temperature_k` | Température du process |
| `Rotational speed [rpm]` | `rotational_speed_rpm` | Vitesse de rotation |
| `Torque [Nm]` | `torque_nm` | Couple |
| `Tool wear [min]` | `tool_wear_min` | Usure de l'outil |
| `Machine failure` | `machine_failure` | **Cible** : 1 si la machine est en panne |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | `twf`, `hdf`, `pwf`, `osf`, `rnf` | Type de panne : usure, dissipation de chaleur, puissance, surcharge, aléatoire |

Chiffres clés : **10 000 machines**, **339 pannes** (environ 3,4 %), classe très déséquilibrée.

### Qualité des données (dbt)

16 tests de données s'exécutent à chaque `dbt build` :

- unicité et non-nullité de `udi` ;
- valeurs acceptées pour `quality_level` (L, M, H) et `machine_failure` (0, 1) ;
- absence de valeurs manquantes sur les cinq mesures et sur les variables calculées ;
- **plausibilité physique** (`assert_measures_plausible`) : échec si une température, une vitesse, un couple ou une usure est impossible ;
- **cohérence des étiquettes** (`assert_labels_consistent`, en avertissement) : repère les machines dont `machine_failure` contredit les types de panne. **27 lignes** du jeu d'origine sont dans ce cas. Elles sont signalées, puis **exclues de l'entraînement** (9 973 machines utilisées).

Un contrôle bloquant Dagster exige en plus exactement 10 000 lignes brutes avant tout entraînement.

---

## 4. Variables du modèle

Le modèle reçoit **10 variables** : les 6 mesures du technicien (qualité encodée 0, 1, 2) et 4 variables calculées.

| Variable | Formule | Sens |
|---|---|---|
| `quality_level_ord` | L=0, M=1, H=2 | Qualité encodée |
| `air_temperature_k`, `process_temperature_k` | mesure | Températures |
| `rotational_speed_rpm`, `torque_nm`, `tool_wear_min` | mesure | Vitesse, couple, usure |
| `temp_diff_k` | process − air | Écart de température (dissipation de chaleur) |
| `power_w` | couple × vitesse × 2π / 60 | Puissance mécanique |
| `wear_x_torque` | usure × couple | Contrainte cumulée sur l'outil |
| `overstrain_ratio` | usure × couple / seuil de la qualité | Rapport de surcharge : **> 1 signifie surcharge** |

Les seuils de surcharge dépendent de la qualité : **L = 11 000, M = 12 000, H = 13 000** (min·Nm). Une même valeur d'usure × couple peut donc être une surcharge en qualité L et rester normale en qualité H.

Ces variables sont calculées **deux fois, de manière identique** : en SQL par dbt (`marts.fct_machine_features`) pour l'entraînement, et en Python par `src/features.py` pour l'API. Un test vérifie qu'elles sont égales machine par machine, pour éviter tout écart entre l'entraînement et la production.

---

## 5. Modélisation et résultats

### Protocole d'évaluation

- **Validation croisée stratifiée à 5 plis**, graine fixe (`SEED = 0`).
- **Métrique principale : PR-AUC** (aire sous la courbe précision-rappel), adaptée à une classe rare. L'exactitude (accuracy) serait trompeuse : prédire « jamais de panne » donnerait déjà environ 96,6 %.
- Précision, rappel et F1 sont aussi calculés.
- Les détecteurs non supervisés sont entraînés **uniquement sur les machines normales** du pli d'entraînement. Pour eux, la décision signale la part de machines les plus suspectes égale au taux de pannes (d'où précision = rappel).

### Résultats (6 approches, moyenne sur 5 plis)

| Rang | Modèle | Type | PR-AUC | F1 |
|---|---|---|---|---|
| 1 | **LR → XGBoost : 10 variables + sortie LR** | composé, supervisé | **0,894** ± 0,035 | **0,830** ± 0,050 |
| 2 | Local Outlier Factor | non supervisé | 0,466 ± 0,093 | 0,446 ± 0,072 |
| 3 | One-Class SVM | non supervisé | 0,460 ± 0,111 | 0,458 ± 0,087 |
| 4 | LR → XGBoost : sortie LR seule | composé, supervisé | 0,413 ± 0,106 | 0,457 ± 0,066 |
| 5 | Isolation Forest | non supervisé | 0,360 ± 0,085 | 0,409 ± 0,074 |
| 6 | Elliptic Envelope | non supervisé | 0,189 ± 0,080 | 0,158 ± 0,060 |

Fichiers sources : `reports/comparaison_6_modeles.csv`, `reports/model_comparison.csv`, `reports/stacking_comparison.csv`.

**Lecture des résultats**

- Le modèle composé **utilisant les 10 variables en plus de la sortie de la régression logistique** domine nettement. Utiliser la sortie de la régression logistique seule ne suffit pas (0,413) : c'est la combinaison avec les 10 variables qui apporte l'essentiel du gain.
- Parmi les détecteurs non supervisés, aucun ne dépasse une PR-AUC de 0,47 : les pannes de ce jeu suivent des **règles par combinaisons** (surcharge, puissance, chaleur) que des détecteurs fondés sur la distance au comportement normal semblent capturer mal (hypothèse cohérente avec les résultats, non testée séparément).
- La comparaison est à lire avec une réserve honnête : le modèle composé est **supervisé** (il utilise les étiquettes de panne), les quatre détecteurs ne le sont pas. Le but est de mesurer ce que chaque famille d'approches apporte, pas de les opposer à armes égales.

### Modèle final (en production)

**Empilement LR → XGBoost sur les 10 variables.**

- Une régression logistique (variables standardisées, poids de classes équilibrés) est entraînée en validation croisée interne : sa sortie est donc **hors échantillon**, sans fuite de données.
- XGBoost (300 arbres, profondeur 4, taux d'apprentissage 0,1) reçoit les 10 variables **et** la probabilité de la régression logistique.
- Le **seuil d'anomalie** est choisi sur des prédictions hors échantillon, en maximisant le F1. Sur le jeu AI4I complet, l'entraînement a donné : **seuil 0,575, précision 0,922, rappel 0,788, F1 0,850**.
- Le modèle final est entraîné sur toutes les données, enregistré dans le registre MLflow sous le nom `ai4i_stacking_10var`, et l'alias `champion` désigne la version servie par l'API.

> Les chiffres du tableau (moyennes par pli, seuil dérivé du taux de pannes) et ceux du modèle final (seuil optimisé sur l'ensemble des prédictions hors échantillon) ne se calculent pas de la même façon : leurs F1 (0,830 et 0,850) sont proches sans être identiques.

---

## 6. API et interface web

### Interface

`http://localhost:8000` ouvre le **poste de maintenance** : deux modes de saisie (une machine par formulaire, ou un fichier CSV), un panneau **Diagnostic** et la liste des **dernières analyses**. Mode clair et mode sombre.

### Niveaux d'alerte

| Score | Résultat | Niveau d'alerte |
|---|---|---|
| < seuil d'anomalie (0,575 sur AI4I) | normal | aucune |
| ≥ seuil et < 0,8 | anormal | modérée |
| ≥ 0,8 | anormal | élevée |

### Routes

| Route | Méthode | Rôle |
|---|---|---|
| `/` | GET | Interface web |
| `/health` | GET | Statut, version du modèle chargé, seuils |
| `/predict` | POST | Une machine (JSON, 6 mesures) → résultat, score, niveau d'alerte |
| `/predict_csv` | POST | Un fichier CSV, une machine par ligne → un résultat par ligne et des totaux |
| `/historique` | GET | Résumé du journal et dernières prédictions (`?n=10`, max 200) |
| `/docs` | GET | Documentation Swagger |

### Exemple

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "quality_level": "L",
  "air_temperature_k": 298.1,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1551,
  "torque_nm": 57,
  "tool_wear_min": 200
}'
```

Réponse (exemple) :

```json
{"resultat": "anormal", "score": 0.9984, "niveau_alerte": "élevée", "seuil": 0.575}
```

### Validation des entrées

- Qualité : L, M ou H (casse et espaces tolérés).
- Bornes de plausibilité : températures 250–350 K, vitesse 1–10 000 tr/min, couple 0–200 Nm, usure 0–1 000 min. Toute valeur hors bornes est refusée (code 422) avec un message clair.

### Fichiers CSV

- Une machine par ligne, **50 000 lignes maximum**.
- Colonnes attendues : `quality_level`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`. Les **en-têtes du fichier AI4I d'origine** sont aussi acceptés.
- Deux formats reconnus automatiquement : standard (`,` et `.`) ou Excel français (`;` et `,`).
- Une colonne `udi`, `product_id` ou `id`, si présente, est reprise dans la réponse.
- En cas d'erreur, la réponse liste jusqu'à 10 problèmes **avec le numéro de ligne** du fichier.

Pour tester rapidement : `python src/exemple_csv.py` crée `data/exemple_machines.csv` (10 pannes + 10 machines normales), puis `python src/tester_csv.py` l'envoie à l'API et compare avec la vraie colonne `machine_failure`.

---

## 7. Orchestration, suivi des essais, journal

### Dagster (pipeline complet)

```bash
dagster dev -f orchestration/definitions.py     # http://localhost:3000
```

| Étape (asset) | Contenu |
|---|---|
| `raw_sensor_readings` | Ingestion dlt, avec contrôle bloquant « 10 000 lignes » |
| `ai4i_dbt` | `dbt build` : modèles et tests. Un test qui échoue arrête la chaîne |
| `model_comparison` | Compare les 4 détecteurs non supervisés |
| `mlflow_runs` | Enregistre les 4 essais dans MLflow |
| `stacking_comparison` | Évalue les 2 variantes du modèle composé |
| `final_model` | Entraîne le modèle final et l'enregistre dans le registre |

Deux jobs : `pipeline_donnees` (ingestion + dbt, rapide) et `pipeline_complet` (de la donnée brute au modèle enregistré).

**Important** : `final_model` enregistre une **nouvelle version** mais ne déplace volontairement **pas** l'alias `champion`. Après vérification des métriques dans MLflow, la promotion se fait à la main (voir ci-dessous).

### MLflow

Base SQLite `mlflow.db`. Trois expériences :

- `ai4i_comparaison_modeles` : les 4 détecteurs ;
- `ai4i_stacking_final` : les 2 variantes du modèle composé ;
- `ai4i_modele_final` : le modèle livré, avec son seuil d'anomalie en paramètre (l'API le lit au démarrage).

Promouvoir une version en production :

```python
from mlflow import MlflowClient
MlflowClient(tracking_uri="sqlite:///mlflow.db").set_registered_model_alias("ai4i_stacking_10var", "champion", 2)
```

Puis redémarrer l'API. Au chargement, elle **refuse de démarrer** si les variables du modèle ne correspondent pas à celles de `features.py`.

### Journal des prédictions

Chaque appel de l'API ajoute une ligne par machine analysée dans `journal.duckdb` (table `predictions`) : horodatage, source (formulaire ou CSV), fichier, identifiant, les 6 mesures, score, résultat, niveau d'alerte, version du modèle, seuil. Une panne du journal **ne bloque jamais** une prédiction. Consultation : route `/historique`.

---

## 8. Tests et intégration continue

```bash
python src/ingestion.py
cd dbt_project && dbt build --profiles-dir . && cd ..
python -m pytest
```

**39 tests pytest** :

| Fichier | Ce qui est vérifié |
|---|---|
| `test_features.py` | Ordre des 10 variables, formules (puissance, surcharge), tolérance sur la casse, refus des entrées invalides |
| `test_api.py` | Routes, cas normal / surcharge / alerte élevée, validation, CSV (formats, erreurs avec numéro de ligne, fichier vide), journalisation |
| `test_journal.py` | Création, ajout, ordre, colonnes facultatives |
| `test_coherence_dbt.py` | Variables Python = variables dbt, machine par machine (ignoré si `ai4i.duckdb` est absente) |

Les tests de l'API utilisent un **modèle factice** et un journal temporaire : ils n'ont besoin ni de MLflow ni d'entraînement.

### GitHub Actions

À chaque `git push` et pull request, `.github/workflows/ci.yml` exécute deux jobs :

1. **dbt build + pytest** : installation des dépendances figées, ingestion, `dbt build`, puis les 39 tests (y compris la cohérence Python/dbt).
2. **Image Docker** : vérifie que l'image se construit.

---

## 9. Installation locale (sans Docker)

Python 3.12 recommandé. Les versions sont figées dans `requirements.txt`.

```bash
python -m venv venv
venv\Scripts\activate              # Windows   (Linux/macOS : source venv/bin/activate)
pip install -r requirements.txt

python scripts/bootstrap.py        # données → dbt → modèle final → alias champion
python src/api.py                  # http://127.0.0.1:8000
```

Étapes à la main, si besoin :

```bash
python src/ingestion.py                          # CSV → DuckDB (raw)
cd dbt_project && dbt build --profiles-dir .     # staging + marts + tests
cd ..
python src/train_models.py                       # 4 détecteurs → reports/model_comparison.csv
python src/log_comparison.py                     # → MLflow
python src/train_stacking.py                     # 2 variantes du modèle composé
python src/train_final_stacking.py               # modèle final → registre MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Scripts d'exploration : `src/explore_raw.py` (contrôles de la couche brute), `src/check_features.py` (cohérence Python/dbt), `src/check_ecart.py` (écart de température par qualité).

---

## 10. Structure du dépôt

```
.
├── src/
│   ├── ingestion.py              # dlt : CSV → DuckDB
│   ├── features.py               # variables du modèle (miroir de dbt)
│   ├── train_models.py           # 4 détecteurs non supervisés
│   ├── train_stacking.py         # variantes du modèle composé LR → XGBoost
│   ├── train_final_stacking.py   # modèle final + registre MLflow
│   ├── log_comparison.py         # essais de comparaison → MLflow
│   ├── api.py                    # API FastAPI
│   ├── interface.html            # interface du technicien
│   ├── journal.py                # journal des prédictions (DuckDB)
│   └── exemple_csv.py, tester_csv.py, check_*.py, explore_raw.py
├── dbt_project/
│   ├── models/staging/           # stg_sensor_readings
│   ├── models/marts/             # fct_machine_features
│   ├── tests/                    # tests SQL personnalisés
│   └── models/schema.yml         # tests génériques
├── orchestration/definitions.py  # Dagster
├── tests/                        # 39 tests pytest
├── reports/                      # comparaisons de modèles (CSV)
├── scripts/                      # bootstrap.py, fetch_data.py
├── docker/entrypoint.sh
├── Dockerfile · docker-compose.yml · requirements.txt
├── .github/workflows/ci.yml
└── data/raw/ai4i2020.csv
```

---

## 11. Choix de conception

- **Une définition des variables, deux implémentations vérifiées.** Le SQL (dbt) alimente l'entraînement, le Python (`features.py`) sert l'API, et un test garantit leur égalité. C'est la parade classique au décalage entre entraînement et production (*training-serving skew*).
- **Pas de fuite de données.** La régression logistique du modèle composé est entraînée en validation croisée interne, et le seuil est choisi sur des prédictions hors échantillon.
- **Qualité avant modèle.** Les données sont contrôlées par dbt avant tout entraînement, et un contrôle Dagster bloque la chaîne si le volume est anormal.
- **Étiquettes douteuses écartées.** Les 27 lignes contradictoires sont exclues plutôt que corrigées arbitrairement.
- **Promotion de modèle manuelle.** Un ré-entraînement ne remplace jamais le modèle servi sans vérification humaine.
- **Seuil porté par le modèle.** Le seuil d'anomalie est un paramètre de l'essai MLflow qui a produit le modèle : l'API charge toujours un modèle et son seuil cohérents.
- **Tout est reproductible.** Versions figées, graine fixe, une commande Docker, et la CI rejoue les vérifications à chaque modification.

---

## 12. Limites connues

- **Données synthétiques** : les performances ne se transposent pas telles quelles à de vraies machines. Les pannes suivent des règles de génération, ce qui avantage les modèles capables de les retrouver.
- **Très peu de pannes** (339) : les écarts-types entre plis restent sensibles (±0,035 de PR-AUC pour le meilleur modèle).
- **Pas de dimension temporelle** : chaque machine est une observation indépendante, sans historique. Ce n'est pas une prévision de durée de vie restante.
- **Pas d'authentification** sur l'API : à protéger avant toute exposition hors d'un réseau local.
- **Modèle sérialisé avec cloudpickle** (MLflow le signale) : ne charger que des modèles dont on maîtrise la provenance.
- **Mono-machine** : DuckDB et SQLite sont des fichiers locaux, adaptés à la démonstration, pas à plusieurs instances en parallèle.


---

## Données et crédits

AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository (jeu synthétique public). Consulter la page du jeu de données pour sa licence et sa citation.
