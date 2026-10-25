"""Tests de l'API (modèle factice : voir conftest.py)."""
from conftest import MACHINE_NORMALE, SURCHARGE, SURCHARGE_FORTE

ENTETE = "quality_level,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_nm,tool_wear_min"


def csv(*lignes, entete=ENTETE) -> dict:
    return {"fichier": ("machines.csv", "\n".join([entete, *lignes]), "text/csv")}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"statut": "ok", "modele": "ai4i_stacking_10var", "version": 99,
                        "seuil_anomalie": 0.575, "seuil_alerte_elevee": 0.8}


def test_page_d_accueil(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "<html" in r.text


# ------------------------------------------------------------------ /predict

def test_machine_normale(client):
    r = client.post("/predict", json=MACHINE_NORMALE)
    assert r.status_code == 200
    j = r.json()
    assert set(j) == {"resultat", "score", "niveau_alerte", "seuil"}
    assert j["resultat"] == "normal" and j["niveau_alerte"] == "aucune"
    assert 0 <= j["score"] < j["seuil"] == 0.575


def test_surcharge_en_qualite_L_est_anormale_alerte_moderee(client):
    j = client.post("/predict", json=SURCHARGE).json()
    assert j["resultat"] == "anormal" and j["niveau_alerte"] == "modérée"


def test_meme_valeur_en_qualite_H_est_normale(client):
    assert client.post("/predict", json={**SURCHARGE, "quality_level": "H"}).json()["resultat"] == "normal"


def test_alerte_elevee(client):
    j = client.post("/predict", json=SURCHARGE_FORTE).json()
    assert j["resultat"] == "anormal" and j["niveau_alerte"] == "élevée" and j["score"] >= 0.8


def test_qualite_en_minuscules_acceptee(client):
    assert client.post("/predict", json={**MACHINE_NORMALE, "quality_level": " l "}).status_code == 200


def test_qualite_invalide_refusee(client):
    assert client.post("/predict", json={**MACHINE_NORMALE, "quality_level": "X"}).status_code == 422


def test_valeurs_hors_bornes_refusees(client):
    for champ, valeur in [("torque_nm", -3), ("tool_wear_min", 5000), ("rotational_speed_rpm", 0),
                          ("air_temperature_k", 10)]:
        assert client.post("/predict", json={**MACHINE_NORMALE, champ: valeur}).status_code == 422, champ


def test_champ_manquant_refuse(client):
    machine = dict(MACHINE_NORMALE)
    del machine["torque_nm"]
    assert client.post("/predict", json=machine).status_code == 422


# ------------------------------------------------------------------ /predict_csv

def test_csv_plusieurs_machines(client):
    r = client.post("/predict_csv", files=csv("L,298.1,308.6,1551,42.8,0", "L,300,310,1400,57,200",
                                              "L,300,310,1400,60,250"))
    j = r.json()
    assert r.status_code == 200
    assert (j["n_machines"], j["n_anormales"], j["n_alerte_elevee"]) == (3, 2, 1)
    assert [x["ligne"] for x in j["resultats"]] == [1, 2, 3]
    assert [x["niveau_alerte"] for x in j["resultats"]] == ["aucune", "modérée", "élevée"]


def test_csv_avec_identifiants(client):
    r = client.post("/predict_csv", files=csv("7,L,298.1,308.6,1551,42.8,0", entete="udi," + ENTETE))
    assert r.json()["resultats"][0]["identifiant"] == "7"


def test_csv_au_format_du_fichier_ai4i(client):
    # En-têtes d'origine, séparateur « ; » et virgule décimale (Excel en français).
    entete = ("UDI;Product ID;Type;Air temperature [K];Process temperature [K];"
              "Rotational speed [rpm];Torque [Nm];Tool wear [min]")
    r = client.post("/predict_csv", files=csv("1;M14860;M;298,1;308,6;1551;42,8;0", entete=entete))
    assert r.status_code == 200
    assert r.json()["resultats"][0]["identifiant"] == "1" and r.json()["n_anormales"] == 0


def test_csv_colonnes_manquantes(client):
    r = client.post("/predict_csv", files=csv("1,2", entete="a,b"))
    assert r.status_code == 422 and "Colonnes manquantes" in r.json()["detail"]


def test_csv_erreurs_avec_numero_de_ligne(client):
    r = client.post("/predict_csv", files=csv("L,298.1,308.6,1551,42.8,0", "Z,298,308,1500,40,0",
                                              "L,298,308,1500,-4,0"))
    exemples = r.json()["detail"]["exemples"]
    assert r.status_code == 422
    assert any("ligne 3" in e and "qualité" in e for e in exemples)
    assert any("ligne 4" in e and "torque_nm" in e for e in exemples)


def test_csv_vide_ou_sans_machine(client):
    assert client.post("/predict_csv", files={"fichier": ("v.csv", "", "text/csv")}).status_code == 422
    assert client.post("/predict_csv", files=csv()).status_code == 422


# ------------------------------------------------------------------ journal

def test_les_predictions_sont_journalisees(client):
    client.post("/predict", json={**MACHINE_NORMALE, "quality_level": "l"})
    client.post("/predict_csv", files=csv("7,L,300,310,1400,57,200", "8,L,300,310,1400,60,250",
                                          entete="udi," + ENTETE))
    j = client.get("/historique?n=10").json()
    assert j["resume"] == {"total": 3, "anormales": 2, "alertes_elevees": 1}
    sources = sorted(d["source"] for d in j["dernieres"])
    assert sources == ["csv", "csv", "formulaire"]
    assert all(d["modele_version"] == 99 for d in j["dernieres"])
    assert {d["quality_level"] for d in j["dernieres"]} == {"L"}


def test_une_requete_refusee_n_est_pas_journalisee(client):
    client.post("/predict", json={**MACHINE_NORMALE, "quality_level": "X"})
    assert client.get("/historique").json()["resume"]["total"] == 0


def test_une_panne_du_journal_ne_bloque_pas_la_prediction(client, monkeypatch):
    import journal
    monkeypatch.setattr(journal, "enregistrer", lambda lot: (_ for _ in ()).throw(OSError("disque plein")))
    r = client.post("/predict", json=SURCHARGE)
    assert r.status_code == 200 and r.json()["resultat"] == "anormal"
