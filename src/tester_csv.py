"""Envoie data/exemple_machines.csv à l'API et compare les réponses à la vraie colonne machine_failure.

L'API doit tourner (python src/api.py), dans un autre terminal :
    python src/tester_csv.py
"""
import json
import urllib.request
import uuid
from pathlib import Path

import pandas as pd

CSV = Path("data/exemple_machines.csv")
URL = "http://127.0.0.1:8000/predict_csv"

# Envoi du fichier (multipart/form-data), sans dépendance supplémentaire.
frontiere = uuid.uuid4().hex
corps = (f"--{frontiere}\r\nContent-Disposition: form-data; name=\"fichier\"; filename=\"{CSV.name}\"\r\n"
         f"Content-Type: text/csv\r\n\r\n").encode() + CSV.read_bytes() + f"\r\n--{frontiere}--\r\n".encode()
requete = urllib.request.Request(URL, data=corps, headers={
    "Content-Type": f"multipart/form-data; boundary={frontiere}"})
try:
    reponse = json.load(urllib.request.urlopen(requete))
except urllib.error.HTTPError as e:
    raise SystemExit(f"Erreur {e.code} : {e.read().decode()}")

reel = pd.read_csv(CSV)
res = pd.DataFrame(reponse["resultats"])
res["panne_reelle"] = reel["machine_failure"].values
res["cause"] = ["bien détecté" if (a == "anormal") == (p == 1) else
                ("PANNE MANQUÉE" if p == 1 else "fausse alerte")
                for a, p in zip(res["resultat"], res["panne_reelle"])]
print(res[["identifiant", "panne_reelle", "resultat", "score", "niveau_alerte", "cause"]].to_string(index=False))

vp = ((res["resultat"] == "anormal") & (res["panne_reelle"] == 1)).sum()
print(f"\nPannes détectées : {vp} sur {int(res['panne_reelle'].sum())}")
print(f"Fausses alertes  : {int(((res['resultat'] == 'anormal') & (res['panne_reelle'] == 0)).sum())} "
      f"sur {int((res['panne_reelle'] == 0).sum())} machines normales")
