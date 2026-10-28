import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pdfplumber
import tempfile
import os
import base64
import pickle
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

dash.register_page(__name__, path="/cv")

# === 📦 Cache global paresseux
cache = {
    "df_rome": None,
    "model": None,
    "embeddings": None
}

# === 🧠 Chargement différé du modèle + embeddings
def get_model_and_rome_embeddings():
    cache_path = "../Data/ROME/rome_embeddings_cache.pkl"

    # Si tout est déjà chargé en mémoire
    if cache["df_rome"] is not None and cache["model"] is not None and cache["embeddings"] is not None:
        return cache["df_rome"], cache["model"], cache["embeddings"]

    # Si un fichier pickle existe, on le charge directement
    if os.path.exists(cache_path):
        print("📦 Chargement du cache .pkl...")
        with open(cache_path, "rb") as f:
            cache["df_rome"], cache["embeddings"] = pickle.load(f)
        print(f"✅ {len(cache['df_rome'])} compétences ROME chargées (depuis cache)")
    else:
        print("📂 Chargement du JSON ROME...")
        with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
            data_rome = json.load(f)

        competences, codes, metiers = [], [], []
        for fiche in data_rome:
            code = fiche.get("rome", {}).get("code_rome")
            intitule = fiche.get("rome", {}).get("intitule")
            savoir_faire = fiche.get("competences", {}).get("savoir_faire", {}).get("enjeux", [])
            for bloc in savoir_faire:
                for item in bloc.get("items", []):
                    libelle = item.get("libelle", "").strip()
                    if libelle:
                        competences.append(libelle)
                        codes.append(code)
                        metiers.append(intitule)

        df = pd.DataFrame({
            "code_rome": codes,
            "intitule_metier": metiers,
            "libelle_competence": competences
        }).dropna()

        df["libelle_competence"] = df["libelle_competence"].astype(str)
        df = df[df["libelle_competence"].str.strip() != ""]
        cache["df_rome"] = df
        print(f"✅ {len(df)} compétences ROME chargées.")

        # Chargement du modèle
        if cache["model"] is None:
            print("⚙️ Chargement du modèle all-MiniLM-L6-v2...")
            cache["model"] = SentenceTransformer("all-MiniLM-L6-v2")

        # Calcul des embeddings
        print("🧠 Encodage des compétences ROME...")
        cache["embeddings"] = cache["model"].encode(
            df["libelle_competence"].tolist(),
            convert_to_tensor=True
        )
        print("✅ Embeddings générés.")

        # Sauvegarde dans un .pkl
        with open(cache_path, "wb") as f:
            pickle.dump((df, cache["embeddings"]), f)
        print("📁 Embeddings et DataFrame sauvegardés.")

    # Chargement du modèle si ce n'était pas encore fait
    if cache["model"] is None:
        print("⚙️ Chargement du modèle all-MiniLM-L6-v2...")
        cache["model"] = SentenceTransformer("all-MiniLM-L6-v2")

    return cache["df_rome"], cache["model"], cache["embeddings"]

# === 📰 Layout
layout = dbc.Container([
    html.H2("📄 Analyse de CV – Suggestions de métiers ROME"),
    dcc.Upload(
        id='upload-cv',
        children=html.Div(["Glissez un fichier PDF ici ou cliquez pour uploader."]),
        style={
            'width': '100%', 'padding': '2em', 'borderWidth': '2px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center'
        },
        multiple=False
    ),
    dcc.Loading(
        type="circle",
        color="#2a4d69",
        children=[
            html.Div(id='cv-status', style={'marginTop': '2em'}),
            html.Div(id='cv-suggestions', style={'marginTop': '2em'})
        ]
    )
], fluid=True)

# === 📄 Extraction texte PDF
def extract_text_from_pdf(contents, filename):
    content_type, content_string = contents.split(',')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(base64.b64decode(content_string))
        tmp_path = tmp.name
    with pdfplumber.open(tmp_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    os.remove(tmp_path)
    return text


def extraire_sections_structurees(texte):
    lignes = texte.split("\n")

    blocs_valides = {"FORMATION", "EXPERIENCE", "EXPERIENCES", "PROJETS", "COMPETENCES"}
    blocs_inutiles = {"CONTACT", "CENTRES D’INTÉRÊT", "CENTRES D INTERET", "INTERET", "INTERÊTS", "LOISIRS", "LANGUES"}

    sections = []
    current_section = None
    contenu = []

    for ligne in lignes:
        ligne_clean = ligne.strip()
        upper = ligne_clean.upper()
        condensed = upper.replace(" ", "")

        if condensed in blocs_valides:
            if current_section and contenu:
                sections.append({"section": current_section, "texte": ' '.join(contenu)})
            current_section = condensed
            contenu = []
        elif condensed in blocs_inutiles:
            if current_section and contenu:
                sections.append({"section": current_section, "texte": ' '.join(contenu)})
            current_section = None
            contenu = []
        elif ligne_clean and current_section:
            contenu.append(ligne_clean)

    if current_section and contenu:
        sections.append({"section": current_section, "texte": ' '.join(contenu)})

    return sections





# === Callback principal
@callback(
    Output('cv-status', 'children'),
    Output('cv-suggestions', 'children'),
    Input('upload-cv', 'contents'),
    State('upload-cv', 'filename')
)
def analyser_cv(contents, filename):
    print("📥 Upload reçu !", contents is not None, filename)

    try:
        if contents is None:
            return dbc.Alert("Aucun fichier uploadé.", color="warning"), ""

        # 1. Extraction brute
        text = extract_text_from_pdf(contents, filename)
        sections = extraire_sections_structurees(text)
        if not sections:
            return dbc.Alert("Aucune section exploitable détectée dans le CV.", color="danger"), ""

        texts = [s["texte"] for s in sections]

        # 2. Chargement paresseux
        df_rome, model, rome_embeddings = get_model_and_rome_embeddings()

        # 3. Embedding des blocs de section du CV
        phrases_cv = model.encode(texts, convert_to_tensor=True)

        # 4. Similarité
        sim_matrix = util.pytorch_cos_sim(phrases_cv, rome_embeddings)
        best_scores = sim_matrix.max(dim=0).values
        best_cv_index = sim_matrix.argmax(dim=0).tolist()

        df_rome["score"] = best_scores.cpu().numpy()
        df_rome["section_associee"] = [sections[i]["section"] for i in best_cv_index]
        df_rome["phrase_cv_associee"] = [sections[i]["texte"] for i in best_cv_index]

        # 5. Agrégation
        def get_best_entry(column, full_scores):
            if column.empty:
                return ""
            score_subset = full_scores.reindex(column.index)
            best_idx = score_subset.idxmax()
            return column.loc[best_idx]

        grouped = df_rome.groupby(["code_rome", "intitule_metier"]).agg({
            "score": "mean",
            "libelle_competence": lambda x: get_best_entry(x, df_rome["score"]),
            "phrase_cv_associee": lambda x: get_best_entry(x, df_rome["score"]),
            "section_associee": lambda x: get_best_entry(x, df_rome["score"])
        }).reset_index()

        top_metiers = grouped.sort_values("score", ascending=False).head(5)

        # 6. Affichage
        suggestions = []
        for _, row in top_metiers.iterrows():
            suggestions.append(html.Li([
                html.B(f"{row['code_rome']} – {row['intitule_metier']}"),
                html.Br(),
                html.Span(f"📁 Section du CV : {row['section_associee']}"),
                html.Br(),
                html.Span(f"🔥 Compétence matchée : {row['libelle_competence']}"),
                html.Br(),
                html.Span(f"💬 Contenu analysé : {row['phrase_cv_associee'][:300]}..."),
                html.Br(),
                html.I(f"Score moyen global : {row['score']:.2f}"),
                html.Hr()
            ]))

        return dbc.Alert(f"✅ Fichier '{filename}' analysé avec succès.", color="success"), html.Ul(suggestions)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert("❌ Erreur : " + str(e), color="danger"), ""