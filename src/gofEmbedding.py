import pandas as pd
import json
import re
import argparse
import os
from sentence_transformers import SentenceTransformer, util

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
    parser.add_argument("--gof", required=True, help="Intitulé exact de la formation GOF (ex: Licence : Droit)")
    parser.add_argument("--rome", required=True, help="Code ROME (ex: H1402)")
    parser.add_argument("--seuil", type=float, default=0.35, help="Seuil de similarité")
    args = parser.parse_args()

    formation_gof = args.gof
    fiche_rome = args.rome.upper()
    seuil = args.seuil

    # === 1. Charger les compétences GOF
    df_gof = pd.read_csv("../Data/competences_directes.csv")
    df_gof.columns = df_gof.columns.str.strip().str.lower()

    competences_gof = df_gof[df_gof["objetid"] == int(formation_gof)]["lib"].dropna().unique()

    if not len(competences_gof):
        raise ValueError(f"Aucune compétence trouvée pour l'ID de formation GOF '{formation_gof}'")


    # === 2. Charger les compétences ROME
    with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
        fiches_rome = json.load(f)
    competences_rome = get_rome_savoir_faire(fiche_rome, fiches_rome)

    if not competences_rome:
        raise ValueError(f"Aucune compétence trouvée pour le code ROME {fiche_rome}")

    # === 3. Nettoyage
    competences_gof_clean = [nettoyer_phrase(c) for c in competences_gof]
    competences_rome_clean = [nettoyer_phrase(c) for c in competences_rome]

    # === 4. Embedding + similarité
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    gof_vecs = model.encode(competences_gof_clean, convert_to_tensor=True)
    rome_vecs = model.encode(competences_rome_clean, convert_to_tensor=True)
    sim_matrix = cosine_similarity_matrix(rome_vecs, gof_vecs)

    logs = []
    for i, original in enumerate(competences_rome):
        scores_gof = sim_matrix[i]
        score_max = scores_gof.max()
        indice_max = scores_gof.argmax()
        competence_associee = competences_gof[indice_max]
        emoji = "✅" if score_max >= seuil else "❌"

        # Top 3
        top_indices = scores_gof.argsort()[::-1]
        top_seen = set()
        top_texts = []
        for idx in top_indices:
            texte = competences_gof[idx].strip()
            if texte not in top_seen:
                score = scores_gof[idx]
                top_texts.append(f"{texte} (score={score:.4f})")
                top_seen.add(texte)
            if len(top_texts) == 3:
                break
        top3_associees_str = "\n".join(top_texts)

        # Full associations
        full_associees_str = "\n".join(
            f"{competences_gof[j]} (score={scores_gof[j]:.4f})"
            for j in top_indices
        )

        logs.append({
            "comp_rome": original,
            "score": round(score_max, 4),
            "couvert": emoji,
            "competence_gof_associee": competence_associee,
            "top_3_associees": top3_associees_str,
            "full_associees": full_associees_str
        })

        print(f"{emoji} {original[:70]:<70} → score = {score_max:.4f}")
        print(f"   ↪️ {competence_associee}")
        print(f"   🥉 Top 3 :\n{top3_associees_str}\n")

    # === Résumé
    nb_rome = len(competences_rome)
    nb_couvertes = sum(1 for l in logs if l["couvert"] == "✅")
    print(f"→ {nb_couvertes} / {nb_rome} compétences ROME couvertes")
    print(f"→ Taux de couverture : {nb_couvertes / nb_rome * 100:.1f}%")

    # === 5. Sauvegarde
    os.makedirs("../logs", exist_ok=True)
    log_path = f"../logs/logs_embedding__GOF_{formation_gof.replace(' ', '_')}__{fiche_rome}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"\n✅ Résultats enregistrés dans : {log_path}")
