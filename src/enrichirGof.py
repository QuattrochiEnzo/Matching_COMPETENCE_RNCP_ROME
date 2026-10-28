import requests
import csv
import time

with open("formations_gof_complet.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    formations = list(reader)

competences_directes = []
formations_sans_competence = []

url_direct = "https://referentiel-odf-backend.u-bordeaux.fr/formations/{}/competences"

print(f"🔍 Extraction des compétences directes pour {len(formations)} formations...")

for formation in formations:
    objetId = formation["objetId"]
    nom = formation["lib"]

    print(f"\n📘 Formation : {nom} — objetId : {objetId}")
    nb_comp = 0

    try:
        response = requests.get(url_direct.format(objetId), timeout=10)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            for comp in data:
                competence = comp.get("competence")
                if not isinstance(competence, dict):
                    continue

                lib = competence.get("lib", "").strip()
                type_ = competence.get("type", {}).get("lib", "")
                discipline = competence.get("discipline", {})
                acquisition = comp.get("acquisition", {})

                discipline_lib = discipline.get("lib", "") if isinstance(discipline, dict) else ""
                acquisition_lib = acquisition.get("lib", "") if isinstance(acquisition, dict) else ""

                print(f" - {lib}")
                competences_directes.append({
                    "objetId": objetId,
                    "formation": nom,
                    "type": type_,
                    "lib": lib,
                    "discipline": discipline_lib,
                    "acquisition": acquisition_lib
                })
                nb_comp += 1

            print(f"✅ Total compétences récupérées : {nb_comp}")
        else:
            print("⚠️  Donnée inattendue (pas une liste)")

        if nb_comp == 0:
            formations_sans_competence.append({"objetId": objetId, "formation": nom})

    except Exception as e:
        print(f"❌ Exception : {e}")
        formations_sans_competence.append({"objetId": objetId, "formation": nom})

    time.sleep(0.2)

# Sauvegarde des résultats
if competences_directes:
    with open("competences_directes.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=competences_directes[0].keys())
        writer.writeheader()
        writer.writerows(competences_directes)
    print(f"\n✅ Compétences extraites : {len(competences_directes)}")

if formations_sans_competence:
    with open("formations_sans_competence.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["objetId", "formation"])
        writer.writeheader()
        writer.writerows(formations_sans_competence)
    print(f"⚠️ Formations sans compétences trouvées : {len(formations_sans_competence)}")

print("\n✅ Extraction terminée.")
