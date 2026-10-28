import pandas as pd
import json
import stanza
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rncpParser import get_details_bloc_competence

# Préparation de stanza
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,lemma', use_gpu=False)

# Fonction de lemmatisation
def lemmatiser_texte(texte):
    doc = nlp(texte.lower())
    return ' '.join(
        word.lemma.lower()
        for sent in doc.sentences
        for word in sent.words
        if word.lemma not in {",", ".", ";", ":", "-", "'", "\""}
    )

# Chargement des données
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

    logs = []

    # Chargement fichiers
    fiches_rome = json.load(open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1"))
    df_rncp = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_05_20.csv", sep=";")

    for code_rncp in rncp_codes:
        # Obtenir les blocs de la fiche
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

        # Récupérer les compétences détaillées
        competences_rncp = []
        for code_bc in codes_blocs:
            try:
                details = get_details_bloc_competence(code_bc)
                competences_rncp.extend(details["competences"])
            except Exception as e:
                print(f"❌ Erreur sur {code_bc} : {e}")

        if not competences_rncp:
            for code_rome in rome_codes:
                logs.append({
                    "RNCP": code_rncp,
                    "ROME": code_rome,
                    "Taux_couverture": "N/A",
                    "Détail": "Aucune compétence extraite"
                })
            continue

        rncp_lems = [lemmatiser_texte(c) for c in competences_rncp]

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

            rome_lems = [lemmatiser_texte(c) for c in competences_rome]

            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            tfidf = vectorizer.fit_transform(rncp_lems + rome_lems)
            n_rncp = len(rncp_lems)
            rome_vectors = tfidf[n_rncp:]
            rncp_vectors = tfidf[:n_rncp]
            similarities = cosine_similarity(rome_vectors, rncp_vectors).max(axis=1)

            nb_rome = len(competences_rome)
            nb_couvertes = (similarities > 0.15).sum()
            taux = nb_couvertes / nb_rome * 100

            logs.append({
                "RNCP": code_rncp,
                "ROME": code_rome,
                "Taux_couverture": round(taux, 2),
                "Détail": f"{nb_couvertes} / {nb_rome} compétences ROME couvertes"
            })

    os.makedirs("../logs", exist_ok=True)
    log_df = pd.DataFrame(logs)
    log_path = "../logs/logs_tfidf_rncp_rome.csv"
    log_df.to_csv(log_path, index=False)
    print(f"✅ Fichier de logs généré : {log_path}")