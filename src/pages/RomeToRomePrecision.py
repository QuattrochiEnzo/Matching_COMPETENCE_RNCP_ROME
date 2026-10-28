import dash
from dash import html, dcc, callback, Input, Output, dash_table
import pandas as pd
import json
import os
import pickle
import torch
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format
from sentence_transformers import SentenceTransformer, util

dash.register_page(__name__, path="/rome-to-rome-precision", name="ROME ↔ ROME (détail)")

# === Données et modèle
cache = {"df_rome": None, "model": None, "embeddings": None}

def get_model_and_rome_embeddings():
    cache_path = "../Data/ROME/rome_embeddings_cache.pkl"

    if cache["df_rome"] is not None and cache["model"] is not None and cache["embeddings"] is not None:
        return cache["df_rome"], cache["model"], cache["embeddings"]

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache["df_rome"], cache["embeddings"] = pickle.load(f)
    else:
        raise FileNotFoundError("rome_embeddings_cache.pkl introuvable")

    cache["model"] = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return cache["df_rome"], cache["model"], cache["embeddings"]

with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
    data_rome = json.load(f)

codes_rome = sorted(set(f["rome"]["code_rome"] for f in data_rome))
mapping_rome = {
    f["rome"]["code_rome"]: f["rome"]["intitule"] for f in data_rome if "rome" in f
}

# === Layout
layout = html.Div(
    style={"padding": "30px"},
    children=[
        dbc.NavbarSimple(
            brand="🔍 Comparaison ROME ↔ ROME (détail)",
            brand_href="/",
            color="#ffd6e0",
            dark=False,
            fluid=True
        ),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Code ROME 1 :"),
                    dcc.Dropdown(
                        id="input-rome-a",
                        options=[{"label": f"{code} — {mapping_rome[code]}", "value": code} for code in codes_rome],
                        placeholder="Sélectionner un code ROME (source)",
                        searchable=True
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Code ROME 2 :"),
                    dcc.Dropdown(
                        id="input-rome-b",
                        options=[{"label": f"{code} — {mapping_rome[code]}", "value": code} for code in codes_rome],
                        placeholder="Sélectionner un code ROME (cible)",
                        searchable=True
                    ),
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Label("Seuil :"),
                    dcc.Input(
                        id="input-seuil-rome2rome",
                        type="number",
                        step=0.01,
                        value=0.50,
                        style={"width": "100%"}
                    )
                ], width=3)
            ]),

            html.Div("💡 Comparaison compétence par compétence. Le top 3 de chaque correspondance est affiché si le score est supérieur au seuil.", style={
                "marginBottom": "15px",
                "fontSize": "14px",
                "fontStyle": "italic"
            }),

            dcc.Loading(
                type="circle",
                color="#c2255c",
                children=[
                    html.Div(id="summary-rome2rome", style={"fontWeight": "bold", "marginBottom": "15px", "fontSize": "18px"}),
                    html.Div(id="output-rome2rome")
                ]
            )
        ], style={
            "backgroundColor": "#fff0f6",
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"
        })
    ]
)


# === Callback
@callback(
    Output("summary-rome2rome", "children"),
    Output("output-rome2rome", "children"),
    Input("input-rome-a", "value"),
    Input("input-rome-b", "value"),
    Input("input-seuil-rome2rome", "value")
)
def compare_rome_rome(code_a, code_b, seuil):
    if not code_a or not code_b:
        return "", html.Div("❌ Veuillez sélectionner deux codes ROME.", style={"color": "red"})

    df_rome, model, rome_embeddings = get_model_and_rome_embeddings()

    df_a = df_rome[df_rome["code_rome"] == code_a]
    df_b = df_rome[df_rome["code_rome"] == code_b]
    if df_a.empty or df_b.empty:
        return "", html.Div("❌ Compétences introuvables pour un des deux codes.", style={"color": "red"})

    emb_b = model.encode(df_b["libelle_competence"].tolist(), convert_to_tensor=True)
    comp_b = df_b["libelle_competence"].tolist()

    lignes = []
    for i, row in df_a.iterrows():
        comp_a = row["libelle_competence"]
        emb_a = model.encode(comp_a, convert_to_tensor=True)

        scores = util.cos_sim(emb_a, emb_b)[0]
        top_k = torch.topk(scores, k=3)
        top_indices = top_k.indices.tolist()
        top_values = top_k.values.tolist()

        top_matches = []
        for idx, score in zip(top_indices, top_values):
            val = float(score)
            if val >= seuil:
                top_matches.append(f"{comp_b[idx]} (score={val:.3f})")

        lignes.append({
            "Compétence ROME A": comp_a,
            "Score max": round(float(top_values[0]), 3),
            "Compétence ROME B la plus proche": comp_b[top_indices[0]],
            "Top 3 compétences B": "<br>".join(top_matches)
        })

    df_result = pd.DataFrame(lignes)

    table = dash_table.DataTable(
        columns=[
            {"name": "Compétence ROME A", "id": "Compétence ROME A"},
            {"name": "Score max", "id": "Score max", "type": "numeric", "format": Format(precision=3)},
            {"name": "Compétence ROME B la plus proche", "id": "Compétence ROME B la plus proche"},
            {"name": "Top 3 compétences B", "id": "Top 3 compétences B", "presentation": "markdown"}
        ],
        cell_selectable=False,
        data=df_result.to_dict("records"),
        markdown_options={"html": True},
        style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "auto"},
        style_cell={
            "textAlign": "left", "padding": "8px", "fontSize": "14px", "whiteSpace": "pre-line"
        },
        style_header={"backgroundColor": "#dee2ff", "fontWeight": "bold"},
        style_data_conditional=[
            {
                "if": {"column_id": "Score max", "filter_query": "{Score max} >= " + str(seuil)},
                "backgroundColor": "#D3F9D8"
            },
            {
                "if": {"column_id": "Score max", "filter_query": "{Score max} < " + str(seuil)},
                "backgroundColor": "#FFE3E3"
            }
        ],
        style_as_list_view=True
    )

    summary = html.Div([
        html.Div(f"🔹 Fiche ROME A : {code_a} — {mapping_rome.get(code_a, '')}", style={"marginBottom": "5px"}),
        html.Div(f"🔸 Fiche ROME B : {code_b} — {mapping_rome.get(code_b, '')}", style={"marginBottom": "15px"}),
        html.Div(f"💡 Total : {len(df_result)} correspondances analysées")
    ])

    return summary, table
