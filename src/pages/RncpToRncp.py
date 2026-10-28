import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
from dash.dash_table.Format import Format
from sentence_transformers import SentenceTransformer, util
import pickle
from rncpParser import get_details_bloc_competence

dash.register_page(__name__, path="/rncp-to-rncp", name="RNCP → RNCP (similaires)")



cache = {"df_rncp": None, "model": None, "embeddings": None}

def get_model_and_rncp_embeddings():
    df_blocs = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_06_30.csv", sep=';')
    print(df_blocs.columns.tolist())
    all_rows = []
    cache = {}

    for fiche_rncp, group in df_blocs.groupby("Numero_Fiche"):
        intitule = "Intitulé inconnu"
        codes_blocs = group["Bloc_Competences_Code"].dropna().unique()

        for code_bc in codes_blocs:
            try:
                details = get_details_bloc_competence(code_bc, cache)
                for competence in details.get("competences", []):
                    all_rows.append({
                        "code_rncp": fiche_rncp,
                        "intitule_formation": intitule,
                        "libelle_competence": competence.strip()
                    })
            except Exception as e:
                print(f"❌ Erreur sur bloc {code_bc} : {e}")

    df_comp = pd.DataFrame(all_rows)
    df_comp = df_comp[df_comp["libelle_competence"].str.strip() != ""]

    os.makedirs("../Data/RNCP", exist_ok=True)
    output_path = "../Data/RNCP/competences_rncp.csv"
    df_comp.to_csv(output_path, index=False)
    print(f"✅ {len(df_comp)} compétences RNCP enregistrées dans {output_path}")

# === Chargement modèle + embeddings
df_rncp, model, embeddings = get_model_and_rncp_embeddings()

if df_rncp is None:
    print("⚠️ Aucune donnée RNCP chargée — la page sera vide.")
    df_rncp = pd.DataFrame(columns=["code_rncp", "intitule_formation", "libelle_competence"])
    embeddings = None

codes_rncp = sorted(df_rncp["code_rncp"].unique())


# === Interface
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
        .dash-table-container td div {
            user-select: text !important;
        }
        </style>
        """, dangerously_allow_html=True),

        html.H2("🔁 Suggestions RNCP → RNCP", className="text-center mb-4", style={"color": "#1864ab"}),

        dbc.Card(
            [
                dbc.CardHeader("Sélectionnez une fiche RNCP", style={"backgroundColor": "#d0ebff"}),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="input-code-rncp",
                        options=[{"label": code, "value": code} for code in codes_rncp],
                        placeholder="Choisir un code RNCP",
                        searchable=True,
                        style={"marginBottom": "20px"}
                    ),
                    dcc.Input(
                        id="seuil-couverture-rncp",
                        type="number",
                        step=0.01,
                        value=0.50,
                        placeholder="Seuil de couverture",
                        style={"width": "100%", "marginBottom": "20px"}
                    ),
                    dcc.Loading(
                        type="circle",
                        color="#2a4d69",
                        children=html.Div(id="table-output-rncp", style={"minHeight": "200px"})
                    ),
                ])
            ],
            style={"backgroundColor": "#e3f2fd", "boxShadow": "0 0 10px rgba(0,0,0,0.05)", "padding": "20px"}
        ),

        html.Div(className="mt-5"),
        html.P("Outil développé au LaBRI — Enzo Quattrochi", className="text-center",
               style={"color": "#adb5bd", "fontSize": "0.9em"}),
    ],
    className="p-5",
    fluid=True,
)

@callback(
    Output("table-output-rncp", "children"),
    Input("input-code-rncp", "value"),
    Input("seuil-couverture-rncp", "value"),
    prevent_initial_call=True
)
def update_comparaison_rncp(code_rncp_source, seuil):
    if not code_rncp_source:
        return html.Div("❌ Veuillez sélectionner un code RNCP.", style={"color": "red"})

    comp_source = df_rncp[df_rncp["code_rncp"] == code_rncp_source].copy()
    if comp_source.empty:
        return html.Div("❌ Code RNCP source introuvable.", style={"color": "red"})

    intitule_rncp = comp_source["intitule_formation"].iloc[0]
    summary = html.Div([
        html.Div([
            html.B("📌 Fiche RNCP ", style={"marginRight": "5px"}),
            html.B(f"{code_rncp_source} : {intitule_rncp}")
        ], style={"marginBottom": "15px"})
    ])

    seuil_str = str(seuil).replace(".", "_")
    log_path = f"../logs/logs_rncp_to_rncp__{code_rncp_source}__seuil_{seuil_str}.csv"
    if os.path.exists(log_path):
        df_resultats = pd.read_csv(log_path)
        df_resultats = df_resultats.sort_values(by="couverture", ascending=False).head(5)
        print(f"📁 Résultats chargés depuis le cache : {log_path}")
        table = dash_table.DataTable(
            columns=[
                {"name": "Code RNCP", "id": "code_rncp"},
                {"name": "Intitulé", "id": "intitule"},
                {"name": "Couverture (%)", "id": "couverture", "type": "numeric", "format": Format(precision=1)},
                {"name": "Score moyen", "id": "score_moyen", "type": "numeric", "format": Format(precision=4)},
            ],
            data=df_resultats.to_dict("records"),
            cell_selectable=False,
            style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "auto"},
            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "14px", "whiteSpace": "normal"},
            style_header={"backgroundColor": "#d0ebff", "fontWeight": "bold"},
            sort_action="native"
        )
        return html.Div([summary, table])

    # Calcul si pas en cache
    vec_source = model.encode(comp_source["libelle_competence"].tolist(), convert_to_tensor=True)
    rows = []
    for code in codes_rncp:
        if code == code_rncp_source:
            continue
        comp_target = df_rncp[df_rncp["code_rncp"] == code]
        if comp_target.empty:
            continue
        vec_target = model.encode(comp_target["libelle_competence"].tolist(), convert_to_tensor=True)
        sim_matrix = util.cos_sim(vec_source, vec_target).cpu().numpy()
        couverture = np.mean(sim_matrix.max(axis=1) >= seuil)
        score_moyen = sim_matrix.mean()
        rows.append({
            "code_rncp": code,
            "intitule": comp_target["intitule_formation"].iloc[0],
            "couverture": float(round(couverture, 1)),
            "score_moyen": round(score_moyen, 4)
        })

    df_resultats = pd.DataFrame(rows).sort_values(by="couverture", ascending=False)
    os.makedirs("../logs", exist_ok=True)
    df_resultats.to_csv(log_path, index=False)
    print(f"✅ Résultats enregistrés dans : {log_path}")

    table = dash_table.DataTable(
        columns=[
            {"name": "Code RNCP", "id": "code_rncp"},
            {"name": "Intitulé", "id": "intitule"},
            {"name": "Couverture (%)", "id": "couverture", "type": "numeric", "format": Format(precision=1)},
            {"name": "Score moyen", "id": "score_moyen", "type": "numeric", "format": Format(precision=4)},
        ],
        data=df_resultats.to_dict("records"),
        cell_selectable=False,
        style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "14px", "whiteSpace": "normal"},
        style_header={"backgroundColor": "#d0ebff", "fontWeight": "bold"},
        sort_action="native"
    )

    return html.Div([summary, table])
