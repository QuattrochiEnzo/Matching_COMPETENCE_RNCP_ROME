import csv

def lire_formation_csv(fichier):
    with open(fichier, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return {row['objetId']: row for row in reader}

# Charge les deux fichiers
complet = lire_formation_csv("formations_gof_complet.csv")
keywords = lire_formation_csv("formations_gof_keyword.csv")  # change le nom si nécessaire

# Formations qui sont dans keywords mais pas dans complet
manquantes = [v for k, v in keywords.items() if k not in complet]

# Affiche les formations manquantes
print(f"\n📌 Formations présentes uniquement dans la méthode mots-clés ({len(manquantes)}):\n")
for formation in manquantes:
    print(f"🔹 {formation['lib']} — objetId: {formation['objetId']}")

# Enregistre si besoin
with open("formations_manquantes.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["objetId", "lib", "annee"])
    writer.writeheader()
    writer.writerows(manquantes)
