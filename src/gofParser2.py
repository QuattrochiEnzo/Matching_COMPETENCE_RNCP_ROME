import requests
import time
import csv

search_terms = [
    "Agrosciences","Mathématiques", "Biologie","Biologie médicale", "Chimie","Chirurgie","Comptabilité" ,"Physique", "Psychologie",
    "Informatique", "Santé","Santé publique", "STAPS", "Sociologie", "Droit","Energie électrique","Energetique","Formation paramedical",
    "Économie", "Gestion", "Médecine", "Enseignement", "Commerce","Génétique médicale",
    "Aéronautique", "Électronique", "Pharmacie", "Pharmacologie","Mécanique",
    "Sciences cognitives", "Sciences politiques", "Environnement",
    "Anthropologie", "Éducation", "Épistémologie", "Bibliothéconomie",
    "Automatique", "Génie Civil", "Ingénierie", "Sciences du vivant","Science de la terre",
    "Sciences de la terre", "Science de la vigne et du vin","Science de l'éducation","Science du vivant",
    "Spécialités médicales", "Management", "Maïeutique", "Odontologie",
    "Oncologie", "Médecine d'urgence", "Maladies infectieuses","Maladies infectieuses et tropicales","math", "maths", "mathématiques", "algèbre", "probabilité", "statistique", "analyse",
    "biologie", "chimie", "physique", "psychologie", "sociologie", "sciences cognitives",
    "sciences politiques", "informatique", "santé", "médecine", "médecine d'urgence",
    "staps", "droit", "économie", "gestion", "commerce", "management", "marketing",
    "pharmacie", "pharmacologie", "odontologie", "oncologie", "gynécologie", "maïeutique",
    "aéronautique", "automatique", "électronique", "énergie", "électrique", "mécanique",
    "génie", "ingénierie", "civil", "environnement", "écologie", "agronomie", "agro",
    "épistémologie", "sciences de la terre", "sciences du vivant", "sciences de la vigne",
    "anthropologie", "archéologie", "sciences de l'éducation", "enseignement", "bibliothéconomie",
    "paramédical", "santé publique", "maladies infectieuses", "spécialités médicales"
]

formations_dict = {}

def fetch_formations_by_term(term):
    page = 0
    all_results = []
    size = 100
    while True:
        url = f"https://referentiel-odf-backend.u-bordeaux.fr/formations/recherche?search={term}&size={size}&page={page}"
        r = requests.get(url)
        if r.status_code != 200:
            break
        results = r.json()
        if not results or len(results) < size:
            all_results.extend(results)
            break
        all_results.extend(results)
        page += 1
        time.sleep(0.1)
    return all_results

print("🔍 Étape 1 : Récupération des formations avec objetId...")

for term in search_terms:
    print(f"Recherche : '{term}'")
    try:
        formations = fetch_formations_by_term(term)
        for f in formations:
            objetId = f.get("objetId")
            if objetId and objetId not in formations_dict:
                formations_dict[objetId] = f
                print(f"✅ {f.get('lib')} — objetId: {objetId}")
    except Exception as e:
        print(f"❌ Erreur pour '{term}': {e}")

print(f"\n✅ Total formations récupérées : {len(formations_dict)}")

with open("formations_gof_keyword.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["objetId", "lib", "annee"])
    writer.writeheader()
    for obj in formations_dict.values():
        writer.writerow({
            "objetId": obj.get("objetId"),
            "lib": obj.get("lib"),
            "annee": obj.get("annee")
        })

print("\n📄 Fichier généré : formations_gof.csv")
