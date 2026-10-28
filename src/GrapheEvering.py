import pandas as pd
import json
import re
import os
from sentence_transformers import SentenceTransformer, util
from rncpParser import get_details_bloc_competence

verbes_vides = [
    "préparer", "organiser", "assurer", "réaliser", "effectuer", "mettre en œuvre",
    "appliquer", "contribuer", "participer", "exécuter", "intervenir", "traiter",
    "gérer", "suivre", "contrôler", "analyser", "utiliser", "exploiter",
    "concevoir", "définir", "animer", "piloter", "adapter", "élaborer", "entretenir", "régler"
]

def nettoyer_phrase(texte):
    texte_min = texte.lower()
    for verbe in verbes_vides:
        texte_min = re.sub(rf"\\b{verbe}\\b", "", texte_min)
    return texte_min.strip()

def get_rome_savoir_faire(code, fiches_rome):
    for fiche in fiches_rome:
        if fiche.get("rome", {}).get("code_rome") == code:
            return [
                item["libelle"]
                for bloc in fiche.get("competences", {}).get("savoir_faire", {}).get("enjeux", [])
                for item in bloc.get("items", [])
                if "libelle" in item
            ]
    return []

def cosine_similarity_matrix(v1, v2):
    return util.cos_sim(v1, v2).cpu().numpy()

def charger_competences_rncp(fiches_rncp):
    df_rncp = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_06_30.csv", sep=';')
    cache = {}
    rncp_competences = {}
    for code in fiches_rncp:
        codes_blocs = df_rncp[df_rncp["Numero_Fiche"] == code]["Bloc_Competences_Code"].dropna().unique()
        competences = []
        for code_bc in codes_blocs:
            try:
                details = get_details_bloc_competence(code_bc, cache)
                competences.extend(details["competences"])
            except Exception as e:
                print(f"❌ Erreur sur {code_bc} : {e}")
        rncp_competences[code] = competences
    return rncp_competences

def charger_competences_rome(fiches_rome_codes, fiches_rome_data):
    return {
        code: get_rome_savoir_faire(code, fiches_rome_data)
        for code in fiches_rome_codes
    }

def calculer_couverture(rome_competences_dict, rncp_competences_dict, seuil=0.35):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    resume = []
    for rome_code, rome_competences in rome_competences_dict.items():
        if not rome_competences:
            continue
        rome_clean = [nettoyer_phrase(c) for c in rome_competences]
        rome_vecs = model.encode(rome_clean, convert_to_tensor=True)
        for rncp_code, rncp_competences in rncp_competences_dict.items():
            if not rncp_competences:
                continue
            rncp_clean = [nettoyer_phrase(c) for c in rncp_competences]
            rncp_vecs = model.encode(rncp_clean, convert_to_tensor=True)
            sim_matrix = cosine_similarity_matrix(rome_vecs, rncp_vecs)
            nb_couvertes = sum(1 for i in range(len(rome_competences)) if sim_matrix[i].max() >= seuil)
            total_rome = len(rome_competences)
            pourcentage = round((nb_couvertes / total_rome) * 100, 1) if total_rome > 0 else 0
            resume.append({
                "rncp_code": rncp_code,
                "rome_code": rome_code,
                "nb_competences_rome": total_rome,
                "nb_couvertes": nb_couvertes,
                "pourcentage_couverture": pourcentage
            })
    return resume

def sauvegarder_resume(resume, filename="resume_couverture.csv"):
    os.makedirs("../logs", exist_ok=True)
    df_resume = pd.DataFrame(resume)
    df_resume.to_csv(f"../logs/{filename}", index=False)

if __name__ == "__main__":
    import json

    # Charger les données ROME
    with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
        fiches_rome_data = json.load(f)

    # Liste complète des codes ROME (18)
    fiches_rome_codes = [
        "I1102", "I1203", "I1204", "H1402", "H1404", "H1502", "H2502", "H2503",
        "H2505", "H2506", "H3102", "I1304", "H1602", "H2501", "I1101", "I1202",
        "I1305", "H2504"
    ]

    # Liste complète des codes RNCP (18)
    fiches_rncp_codes = [
        "RNCP40231", "RNCP36808", "RNCP36816", "RNCP34067", "RNCP34870", "RNCP36809",
        "RNCP36894", "RNCP36784", "RNCP35307", "RNCP37540", "RNCP38680", "RNCP38677",
        "RNCP35831", "RNCP39130", "RNCP30180", "RNCP24186", "RNCP11790", "RNCP40956"
    ]

    # Charger compétences
    rome_comp_dict = charger_competences_rome(fiches_rome_codes, fiches_rome_data)
    rncp_comp_dict = charger_competences_rncp(fiches_rncp_codes)

    # Calcul des couvertures + sauvegarde CSV
    resume_result = calculer_couverture(rome_comp_dict, rncp_comp_dict, seuil=0.5)
    sauvegarder_resume(resume_result)
    print("✅ Calcul résumé terminé et sauvegardé dans ../logs/resume_couverture.csv")

