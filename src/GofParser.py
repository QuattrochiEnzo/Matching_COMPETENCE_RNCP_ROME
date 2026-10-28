import requests
import csv
from time import sleep

OUTPUT_FILE = "formations_gof_complet.csv"
MAX_ID = 100000  

def get_formation(objet_id):
    url = f"https://referentiel-odf-backend.u-bordeaux.fr/formations/{objet_id}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "objetId" in data:
                return {
                    "objetId": data["objetId"],
                    "lib": data.get("lib", ""),
                    "annee": data.get("annee", "")
                }
    except requests.exceptions.RequestException:
        pass
    return None

def main():
    formations = []
    print("🚀 Début de l'exploration des objetId...\n")

    for i in range(1, MAX_ID + 1):
        formation = get_formation(i)
        if formation:
            formations.append(formation)
            print(f"✅ {formation['lib']} — objetId: {formation['objetId']}")

        if i % 500 == 0:
            print(f"🔄 Progression : {i}/{MAX_ID} (Formations trouvées : {len(formations)})")
            sleep(1)  # Pause légère pour éviter surcharge

    print(f"\n✅ Terminé ! {len(formations)} formations trouvées.")

    # Écriture CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["objetId", "lib", "annee"])
        writer.writeheader()
        writer.writerows(formations)

    print(f"📄 Résultats sauvegardés dans : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
