import pandas as pd
import json
import re
import os
from sentence_transformers import SentenceTransformer, util
from rncpParser import get_details_bloc_competence
# === Liste de verbes vides à neutraliser ===
verbes_vides = [
    "préparer", "organiser", "assurer", "réaliser", "effectuer", "mettre en œuvre",
    "appliquer", "contribuer", "participer", "exécuter", "intervenir", "traiter",
    "gérer", "suivre", "contrôler", "analyser", "utiliser", "exploiter",
    "concevoir", "définir", "animer", "piloter", "adapter", "élaborer", "entretenir", "régler"
]

def nettoyer_phrase(texte):
    """
    Supprime les verbes vides d’une phrase pour ne garder que les éléments porteurs de sens.
    """
    texte_min = texte.lower()
    for verbe in verbes_vides:
        texte_min = re.sub(rf"\b{verbe}\b", "", texte_min)
    return texte_min.strip()

# === Chargement des données ===
df_rncp = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_05_20.csv", sep=';')
rncp_competences = (
    df_rncp.groupby("Numero_Fiche")
    .agg({"Bloc_Competences_Libelle": lambda x: [c for c in x.dropna()]})
    .to_dict()["Bloc_Competences_Libelle"]
)

with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
    fiches_rome = json.load(f)

def get_rome_savoir_faire(code):
    for fiche in fiches_rome:
        if fiche.get("rome", {}).get("code_rome") == code:
            return [
                item["libelle"]
                for bloc in fiche.get("competences", {}).get("savoir_faire", {}).get("enjeux", [])
                for item in bloc.get("items", [])
                if "libelle" in item
            ]
    return []

if __name__ == "__main__":
    rncp_codes = ["RNCP36784", "RNCP37928", "RNCP34067"]
    rome_codes = ["H2502", "H1502", "H1401", "H1402"]
    seuil = 0.35
    logs = []

    # Chargement des compétences ROME
    with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
        fiches_rome = json.load(f)

    # Chargement du CSV RNCP pour récupérer les blocs
    df_rncp = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_05_20.csv", sep=';')

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    for code_rncp in rncp_codes:
        # 1. Trouver tous les blocs liés à la fiche RNCP
        codes_blocs = df_rncp[df_rncp["Numero_Fiche"] == code_rncp]["Bloc_Competences_Code"].dropna().unique()
        if not len(codes_blocs):
            for code_rome in rome_codes:
                logs.append({
                    "RNCP": code_rncp,
                    "ROME": code_rome,
                    "Taux_couverture": "N/A",
                    "Détail": "Fiche RNCP introuvable"
                })
            continue

        # 2. Extraire les compétences enrichies
        competences_rncp = []
        for code_bc in codes_blocs:
            try:
                details = get_details_bloc_competence(code_bc)
                competences_rncp.extend(details["competences"])
            except Exception as e:
                print(f"❌ Bloc {code_bc} : {e}")

        if not competences_rncp:
            for code_rome in rome_codes:
                logs.append({
                    "RNCP": code_rncp,
                    "ROME": code_rome,
                    "Taux_couverture": "N/A",
                    "Détail": "Aucune compétence extraite"
                })
            continue

        rncp_nettoye = [nettoyer_phrase(c) for c in competences_rncp]
        rncp_embs = model.encode(rncp_nettoye, convert_to_tensor=True)

        for code_rome in rome_codes:
            competences_rome = get_rome_savoir_faire(code_rome)
            if not competences_rome:
                logs.append({
                    "RNCP": code_rncp,
                    "ROME": code_rome,
                    "Taux_couverture": "N/A",
                    "Détail": "Fiche ROME vide"
                })
                continue

            rome_nettoye = [nettoyer_phrase(c) for c in competences_rome]
            rome_embs = model.encode(rome_nettoye, convert_to_tensor=True)

            similarities = util.cos_sim(rome_embs, rncp_embs)
            max_scores = similarities.max(dim=1).values
            nb_rome = len(competences_rome)
            nb_couvertes = (max_scores > seuil).sum().item()
            taux = nb_couvertes / nb_rome * 100

            logs.append({
                "RNCP": code_rncp,
                "ROME": code_rome,
                "Taux_couverture": round(taux, 2),
                "Détail": f"{nb_couvertes} / {nb_rome} compétences ROME couvertes"
            })

    # Sauvegarde des résultats
    os.makedirs("../logs", exist_ok=True)
    log_df = pd.DataFrame(logs)
    log_path = "../logs/logs_embeddings_rncp_rome.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n✅ Résultats enregistrés dans : {log_path}")