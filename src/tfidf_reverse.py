import pandas as pd
import json
import stanza
import argparse
import os
from rncpParser import get_details_bloc_competence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialisation Stanza
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,lemma', use_gpu=False)

def lemmatiser_texte(texte):
    doc = nlp(texte)
    lemmes = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.lemma not in [",", ".", ";", ":", "-", "'", "\""]:
                lemmes.append(word.lemma.lower())
    return ' '.join(lemmes)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rncp", required=True, help="Code fiche RNCP (ex: RNCP34067)")
    parser.add_argument("--rome", required=True, help="Code fiche ROME (ex: H1402)")
    parser.add_argument("--seuil", type=float, default=0.15, help="Seuil de similarité")
    args = parser.parse_args()

    fiche_rncp = args.rncp.upper()
    fiche_rome = args.rome.upper()
    seuil = args.seuil

    # === 1. Chargement des blocs
    df_rncp = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_05_20.csv", sep=';')
    codes_blocs = df_rncp[df_rncp["Numero_Fiche"] == fiche_rncp]["Bloc_Competences_Code"].dropna().unique()

    if not len(codes_blocs):
        raise ValueError(f"Aucun bloc trouvé pour la fiche RNCP {fiche_rncp}")

    # === 2. Extraction des compétences RNCP
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

    # === 3. Chargement des compétences ROME
    with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
        fiches_rome = json.load(f)
    competences_rome = get_rome_savoir_faire(fiche_rome, fiches_rome)

    if not competences_rome:
        raise ValueError(f"Aucune compétence trouvée pour le code ROME {fiche_rome}")

    # === 4. Lemmatisation
    rncp_lems = [lemmatiser_texte(c) for c in toutes_competences_rncp]
    rome_lems = [lemmatiser_texte(c) for c in competences_rome]

    # === 5. TF-IDF + Similarité
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(rncp_lems + rome_lems)
    n_rncp = len(rncp_lems)
    rncp_vectors = tfidf[:n_rncp]
    rome_vectors = tfidf[n_rncp:]
    sim_matrix = cosine_similarity(rncp_vectors, rome_vectors)

    # === 6. Logs inversés (pour chaque compétence RNCP → meilleure ROME)
    nb_rncp = len(toutes_competences_rncp)
    nb_couvertes = 0
    logs = []

    print(f"\n🧠 ROME {fiche_rome} vs RNCP {fiche_rncp} (vue inversée)")

    for i, original in enumerate(toutes_competences_rncp):
        scores_i = list(sim_matrix[i])
        index_tries = sorted(range(len(scores_i)), key=lambda j: scores_i[j], reverse=True)

        # === Top 3 sans doublons
        top_texts = []
        top_seen = set()
        for idx in index_tries:
            texte = competences_rome[idx].strip()
            if texte not in top_seen:
                score = scores_i[idx]
                top_texts.append(f"{texte} (score={score:.4f})")
                top_seen.add(texte)
            if len(top_texts) == 3:
                break

        all_assoc = "\n".join([f"{competences_rome[j]} (score={scores_i[j]:.4f})" for j in index_tries])
        score_max = scores_i[index_tries[0]]
        best_competence = competences_rome[index_tries[0]]
        emoji = "✅" if score_max > seuil else "❌"

        if score_max > seuil:
            nb_couvertes += 1

        print(f"{emoji} {original[:70]:<70} → score = {score_max:.4f}")
        print(f"   ↪️ {best_competence}\n")

        logs.append({
            "comp_rncp": original,
            "score": round(score_max, 4),
            "couvert": emoji,
            "competence_rome_associee": best_competence,
            "top_3_associees": "\n".join(top_texts),
            "full_associees": all_assoc
        })

    print(f"→ {nb_couvertes} / {nb_rncp} compétences RNCP couvertes")
    print(f"→ Taux de couverture : {nb_couvertes / nb_rncp * 100:.1f}%")

    # === 7. Sauvegarde
    os.makedirs("../logs", exist_ok=True)
    log_path = f"../logs/logs_tfidf_reverse__{fiche_rncp}__{fiche_rome}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"\n✅ Résultats enregistrés dans : {log_path}")
