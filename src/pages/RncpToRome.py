import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import os
import pickle
import json
import re
import torch
from sentence_transformers import SentenceTransformer, util
from rncpParser import get_details_bloc_competence, load_cache

dash.register_page(__name__, path="/rncp-to-rome", name="RNCP → ROME")

# Chargement des données
rncp_df = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_06_30.csv", sep=";")
codes_rncp = sorted(rncp_df["Numero_Fiche"].dropna().unique().tolist())
df_metiers = pd.read_csv("../Data/RNCP/export_fiches_CSV_Standard_2025_06_30.csv", sep=";")
mapping_metiers = dict(zip(df_metiers["Numero_Fiche"], df_metiers["Intitule"]))

# Cache modèle + embeddings
cache = {"rome_df": None, "model": None, "embeddings": None}
def get_model_and_embeddings_rome_excel():
    if all(v is not None for v in cache.values()):
        return cache["rome_df"], cache["model"], cache["embeddings"]

    cache_path = "../Data/ROME/rome_embeddings_from_excel.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache["rome_df"], cache["embeddings"] = pickle.load(f)
    else:
        rome_df = pd.read_excel("../Data/ROME/ROME Arborescence des Competences 24M11.xlsx", sheet_name="Données brutes Comp.",
                                usecols=["4 - Macro-compétence", "5 - Compétence"]).dropna().drop_duplicates()
        rome_texts = pd.concat([rome_df["4 - Macro-compétence"], rome_df["5 - Compétence"]]).dropna().unique().tolist()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(rome_texts, convert_to_tensor=True)
        with open(cache_path, "wb") as f:
            pickle.dump((rome_texts, embeddings), f)
        cache["rome_df"] = rome_texts
        cache["embeddings"] = embeddings
        cache["model"] = model
        return rome_texts, model, embeddings

    if cache["model"] is None:
        cache["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    return cache["rome_df"], cache["model"], cache["embeddings"]

# Extraction compétences RNCP
def get_competences_rncp(code_rncp):
    df_blocs = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_06_30.csv", sep=';')
    codes_blocs = df_blocs[df_blocs["Numero_Fiche"] == code_rncp]["Bloc_Competences_Code"].dropna().unique()
    if not len(codes_blocs):
        raise ValueError(f"Aucun bloc trouvé pour la fiche RNCP {code_rncp}")
    toutes = []
    cache_local = load_cache()
    for code_bc in codes_blocs:
        try:
            details = get_details_bloc_competence(code_bc, cache_local)
            toutes.extend(details["competences"])
        except Exception as e:
            print(f"❌ Erreur sur {code_bc} : {e}")
    if not toutes:
        raise ValueError(f"Aucune compétence enrichie récupérée pour la fiche RNCP {code_rncp}")
    return toutes

# Layout
layout = dbc.Container(
    [
        dcc.Markdown("""
            <style>
            .copy-btn {
                background-color: transparent;
                border: none;
                cursor: pointer;
                font-size: 14px;
                margin-left: 8px;
            }

            /* Empêche de sélectionner du HTML dans la cellule */
            .dash-table-container td div {
                user-select: none;
            }
            </style>
            """, dangerously_allow_html=True),

        html.H2("🧠 Correspondance RNCP → ROME", className="text-center mb-4", style={"color": "#997b00"}),

        dbc.Card(
            [
                dbc.CardHeader("Sélectionnez une fiche RNCP", style={"backgroundColor": "#ffe066"}),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="code_bc_input",
                        options=[{"label": f"{code} — {mapping_metiers.get(code, '')}", "value": code}
                                 for code in codes_rncp],
                        placeholder="Choisir un code RNCP (ex: RNCP34067)",
                        searchable=True,
                        style={"marginBottom": "20px"}
                    ),
                    dcc.Loading(
                        id="loading-spinner",
                        type="circle",
                        color="#2a4d69",
                        children=html.Div(id="rncp_result", style={"minHeight": "200px"})
                    ),
                    html.Div(id="error_msg", className="text-danger mt-3", style={"fontWeight": "bold"})
                ])
            ],
            style={"backgroundColor": "#fffbe6", "boxShadow": "0 0 10px rgba(0,0,0,0.05)", "padding": "20px"}
        ),

        html.Div(className="mt-5"),
        html.P("Outil développé au LaBRI — Enzo Quattrochi", className="text-center",
               style={"color": "#adb5bd", "fontSize": "0.9em"}),
    ],
    className="p-5",
    fluid=True,
)




# Callback
@callback(
    Output("rncp_result", "children"),
    Output("error_msg", "children"),
    Input("code_bc_input", "value"),
    prevent_initial_call=True,
    allow_duplicate=True
)
def update_matching(code_input):
    if not code_input:
        return "", "⚠️ Veuillez sélectionner une fiche RNCP."

    try:
        comp_rncp = get_competences_rncp(code_input)
    except Exception as e:
        return "", str(e)

    rome_texts, model, rome_embeddings = get_model_and_embeddings_rome_excel()

    lignes = []
    for comp in comp_rncp:
        emb = model.encode(comp, convert_to_tensor=True)
        scores = util.cos_sim(emb, rome_embeddings)[0]
        top_scores = torch.topk(scores, k=3)
        top_indices = top_scores.indices.tolist()
        top_values = top_scores.values.tolist()

        top_matches_html = ""
        for i, val in zip(top_indices, top_values):
            rome_text = rome_texts[i]
            rome_score = round(val, 3)
            top_matches_html += f"- {rome_text} (score={rome_score})\n"

        lignes.append({
            "Compétence RNCP": comp,
            "Top 3 compétences ROME les plus proches": top_matches_html
        })

    df_result = pd.DataFrame(lignes)
    table = dash_table.DataTable(
        columns=[{"name": col, "id": col, "presentation": "markdown"} for col in df_result.columns],
        data=df_result.to_dict("records"),
        markdown_options={"html": False},
        cell_selectable=False,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "whiteSpace": "pre-line",
            "height": "auto",
            "fontSize": "14px",
            "padding": "8px"
        },
        style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
    )

    intitule_rncp = mapping_metiers.get(code_input, "Intitulé inconnu")
    return html.Div([
        html.H4(f"🧠 Fiche RNCP {code_input} : {intitule_rncp}"),
        table
    ]), ""
