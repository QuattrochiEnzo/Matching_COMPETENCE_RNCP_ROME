import pandas as pd
import json
import re
import argparse
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
        texte_min = re.sub(rf"\b{verbe}\b", "", texte_min)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rncp", required=True, help="Code de la fiche RNCP (ex: RNCP34067)")
    parser.add_argument("--rome", required=True, help="Code ROME (ex: H1402)")
    parser.add_argument("--seuil", type=float, default=0.35, help="Seuil de similarité")
    args = parser.parse_args()

    fiche_rncp = args.rncp.upper()
    fiche_rome = args.rome.upper()
    seuil = args.seuil

    # === 1. Charger les blocs de la fiche RNCP
    df_rncp = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_05_20.csv", sep=';')
    codes_blocs = df_rncp[df_rncp["Numero_Fiche"] == fiche_rncp]["Bloc_Competences_Code"].dropna().unique()

    if not len(codes_blocs):
        raise ValueError(f"Aucun bloc trouvé pour la fiche RNCP {fiche_rncp}")

    # === 2. Scraper les compétences RNCP
    toutes_competences_rncp = []
    cache = {}
    for code_bc in codes_blocs:
        try:
            details = get_details_bloc_competence(code_bc, cache)
            toutes_competences_rncp.extend(details["competences"])
        except Exception as e:
            print(f"❌ Erreur sur {code_bc} : {e}")

    if not toutes_competences_rncp:
        raise ValueError(f"Aucune compétence enrichie récupérée pour la fiche RNCP {fiche_rncp}")

    # === 3. Charger les compétences ROME
    with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
        fiches_rome = json.load(f)
    competences_rome = get_rome_savoir_faire(fiche_rome, fiches_rome)

    if not competences_rome:
        raise ValueError(f"Aucune compétence trouvée pour le code ROME {fiche_rome}")

    # === 4. Nettoyage
    competences_rncp_clean = [nettoyer_phrase(c) for c in toutes_competences_rncp]
    competences_rome_clean = [nettoyer_phrase(c) for c in competences_rome]

    # === 5. Embedding + similarité
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    rncp_vecs = model.encode(competences_rncp_clean, convert_to_tensor=True)
    rome_vecs = model.encode(competences_rome_clean, convert_to_tensor=True)
    sim_matrix = cosine_similarity_matrix(rncp_vecs, rome_vecs)  # reverse !

    logs = []
    for i, original in enumerate(toutes_competences_rncp):
        scores_rome = sim_matrix[i]
        score_max = scores_rome.max()
        indice_max = scores_rome.argmax()
        competence_associee = competences_rome[indice_max]
        emoji = "✅" if score_max >= seuil else "❌"

        # Top 3
        top_indices = scores_rome.argsort()[::-1]
        top_seen = set()
        top_texts = []
        for idx in top_indices:
            texte = competences_rome[idx].strip()
            if texte not in top_seen:
                score = scores_rome[idx]
                top_texts.append(f"{texte} (score={score:.4f})")
                top_seen.add(texte)
            if len(top_texts) == 3:
                break
        top3_associees_str = "\n".join(top_texts)

        # Full associations
        full_associees_str = "\n".join(
            f"{competences_rome[j]} (score={scores_rome[j]:.4f})"
            for j in top_indices
        )

        logs.append({
            "comp_rncp": original,
            "score": round(score_max, 4),
            "couvert": emoji,
            "competence_rome_associee": competence_associee,
            "top_3_associees": top3_associees_str,
            "full_associees": full_associees_str
        })

        print(f"{emoji} {original[:70]:<70} → score = {score_max:.4f}")
        print(f"   ↪️ {competence_associee}")
        print(f"   🥉 Top 3 :\n{top3_associees_str}\n")

    # === Résumé
    nb_rncp = len(toutes_competences_rncp)
    nb_couvertes = sum(1 for l in logs if l["couvert"] == "✅")
    print(f"→ {nb_couvertes} / {nb_rncp} compétences RNCP couvertes")
    print(f"→ Taux de couverture : {nb_couvertes / nb_rncp * 100:.1f}%")

    # === 6. Sauvegarde
    os.makedirs("../logs", exist_ok=True)
    log_path = f"../logs/logs_embedding_reverse__{fiche_rncp}__{fiche_rome}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"\n✅ Résultats enregistrés dans : {log_path}")

