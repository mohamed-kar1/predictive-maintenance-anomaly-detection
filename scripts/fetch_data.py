"""Place data/raw/ai4i2020.csv s'il est absent (téléchargement depuis UCI, jeu public AI4I 2020)."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

CSV = Path("data/raw/ai4i2020.csv")
URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"


def main() -> int:
    if CSV.exists():
        print(f"[data] {CSV} déjà présent.")
        return 0
    print(f"[data] {CSV} absent : téléchargement depuis UCI...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as r:
            archive = zipfile.ZipFile(io.BytesIO(r.read()))
        nom = next(n for n in archive.namelist() if n.lower().endswith("ai4i2020.csv"))
        CSV.parent.mkdir(parents=True, exist_ok=True)
        CSV.write_bytes(archive.read(nom))
    except Exception as e:  # réseau, URL changée, archive inattendue
        print(f"[data] Échec ({e}).\n       Télécharge ai4i2020.csv à la main et place-le dans {CSV}.", file=sys.stderr)
        return 1
    print(f"[data] {CSV} téléchargé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
