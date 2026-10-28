import dash
from dash import html, dcc, callback, Output, Input, State, dash_table
import pandas as pd
import os
import subprocess
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import json
dash.register_page(__name__, path="/comparaison")


df_metiers = pd.read_csv("../Data/RNCP/export_fiches_CSV_Standard_2025_05_20.csv", sep=";")
mapping_metiers = dict(zip(df_metiers["Numero_Fiche"], df_metiers["Intitule"]))
df_rncp = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_05_20.csv", sep=";")
codes_rncp = sorted(df_rncp["Numero_Fiche"].dropna().unique().tolist())

with open("../Data/ROME/unix_fiche_emploi_metier_v458.json", encoding="ISO-8859-1") as f:
    data_rome = json.load(f)
codes_rome = sorted(set(f["rome"]["code_rome"] for f in data_rome))
mapping_rome = {
    fiche["rome"]["code_rome"]: fiche["rome"]["intitule"]
    for fiche in data_rome
    }


layout = html.Div(
    style={"padding": "30px"},
    children=[
        dbc.NavbarSimple(
            brand="📊 Comparaison Embedding vs TFIDF",
            brand_href="/",
            color="#f3af5a",
            dark=True,
            fluid=True
        ),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Code RNCP :"),
                    dcc.Dropdown(
                        id="input-rncp-comp",
                        options=[{"label": code, "value": code} for code in codes_rncp],
                        placeholder="Sélectionner un code RNCP",
                        searchable=True
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Code ROME :"),
                    dcc.Dropdown(
                        id="input-rome-comp",
                        options=[{"label": code, "value": code} for code in codes_rome],
                        placeholder="Sélectionner un code ROME",
                        searchable=True
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Seuil Embedding :"),
                    dcc.Input(
                        id="input-seuil-embed-comp",
                        type="number",
                        step=0.01,
                        value=0.35,
                        style={"width": "100%"}
                    )
                ], width=2),
                dbc.Col([
                    dbc.Label("Seuil TF-IDF :"),
                    dcc.Input(
                        id="input-seuil-tfidf-comp",
                        type="number",
                        step=0.01,
                        value=0.15,
                        style={"width": "100%"}
                    )
                ], width=2),
                dbc.Col([
                    dbc.Label("Options :"),
                    dcc.Checklist(
                        id="reverse-toggle-comp",
                        options=[{"label": "Inverser RNCP/ROME", "value": "reverse"}],
                        value=[],
                        inputStyle={"marginRight": "8px"},
                        labelStyle={"marginTop": "8px"}
                    )
                ], width=4),
            ], className="mb-3"),

            html.Div("💡 La couverture est recalculée dynamiquement sans relancer les scripts.", style={
                "marginBottom": "15px",
                "fontSize": "14px",
                "fontStyle": "italic"
            }),

            dcc.Loading(
                type="circle",
                color="#2a4d69",
                children=[
                    html.Div(id="summary-comp", style={"fontWeight": "bold", "marginBottom": "15px", "fontSize": "18px"}),
                    html.Div(id="output-resultats-comp")
                ]
            )
        ], style={
            "backgroundColor": "#fff9db",
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"
        })
    ]
)

@callback(
    Output("summary-comp", "children"),
    Output("output-resultats-comp", "children"),
    Input("input-rncp-comp", "value"),
    Input("input-rome-comp", "value"),
    Input("input-seuil-embed-comp", "value"),
    Input("input-seuil-tfidf-comp", "value"),
    Input("reverse-toggle-comp", "value")
)

def update_comparaison(code_rncp, code_rome, seuil_embed, seuil_tfidf, reverse_toggle):
    if not code_rncp or not code_rome or seuil_embed is None or seuil_tfidf is None:
        return "", ""

    reverse = "reverse" in reverse_toggle

    script_embed = "embedding_reverse.py" if reverse else "poke_embedding_detail.py"
    script_tfidf = "tfidf_reverse.py" if reverse else "poke_tfidf_detail.py"

    log_embed = f"../logs/logs_embedding{'_reverse' if reverse else ''}__{code_rncp.upper()}__{code_rome.upper()}.csv"
    log_tfidf = f"../logs/logs_tfidf{'_reverse' if reverse else ''}__{code_rncp.upper()}__{code_rome.upper()}.csv"

    if not os.path.exists(log_embed):
        subprocess.run(["python3", script_embed, "--rncp", code_rncp, "--rome", code_rome])
    if not os.path.exists(log_tfidf):
        subprocess.run(["python3", script_tfidf, "--rncp", code_rncp, "--rome", code_rome])

    if not os.path.exists(log_embed) or not os.path.exists(log_tfidf):
        return "", html.Div("❌ Résultats manquants.", style={"color": "red"})

    df_embed = pd.read_csv(log_embed)
    df_tfidf = pd.read_csv(log_tfidf)

    if reverse:
        df_embed = df_embed.set_index("comp_rncp")
        df_tfidf = df_tfidf.set_index("comp_rncp")
    else:
        df_embed = df_embed.set_index("comp_rome")
        df_tfidf = df_tfidf.set_index("comp_rome")

    common = df_embed.join(df_tfidf, lsuffix="_embed", rsuffix="_tfidf", how="inner").reset_index()
    col_key = "comp_rncp" if reverse else "comp_rome"

    common["Couvert par Embedding"] = common["score_embed"].apply(lambda s: "✅" if s >= seuil_embed else "❌")
    common["Couvert par TF-IDF"] = common["score_tfidf"].apply(lambda s: "✅" if s >= seuil_tfidf else "❌")

    nb_total = len(common)
    nb_embed = sum(common["Couvert par Embedding"] == "✅")
    nb_tfcomp = sum(common["Couvert par TF-IDF"] == "✅")
    taux_embed = round(nb_embed / nb_total * 100, 1)
    taux_tfcomp = round(nb_tfcomp / nb_total * 100, 1)

    intitule_rncp = mapping_metiers.get(code_rncp, "Métier RNCP inconnu")
    intitule_rome = mapping_rome.get(code_rome, "Métier ROME inconnu")

    summary = html.Div([
        html.Div(f"🧠 Fiche RNCP {code_rncp} : {intitule_rncp}", style={"marginBottom": "5px"}),
        html.Div(f"📌 Fiche ROME {code_rome} : {intitule_rome}", style={"marginBottom": "15px"}),
        html.Div(f"✅ Couverture Embedding : {nb_embed}/{nb_total} ({taux_embed}%) • TF-IDF : {nb_tfcomp}/{nb_total} ({taux_tfcomp}%)")
    ])

    table = dash_table.DataTable(
        columns=[
            {"name": f"Compétence {'RNCP' if reverse else 'ROME'}", "id": col_key},
            {"name": "Score Embedding", "id": "score_embed", "type": "numeric"},
            {"name": "Score TF-IDF", "id": "score_tfidf", "type": "numeric"},
            {"name": "Couvert par TF-IDF", "id": "Couvert par TF-IDF"},
            {"name": "Couvert par Embedding", "id": "Couvert par Embedding"}
        ],
        cell_selectable=False,
        data=common.to_dict("records"),
        sort_action="native",
        style_table={
            "overflowX": "auto",
            "maxHeight": "600px",
            "overflowY": "auto",
            "border": "1px solid #ccc"
        },
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "fontSize": "14px",
            "minWidth": "150px",
            "maxWidth": "500px",
            "whiteSpace": "normal"
        },
        style_header={
            "backgroundColor": "#F9E79F",
            "fontWeight": "bold"
        },
        style_data_conditional=[
            {
                "if": {"column_id": "Couvert par Embedding", "filter_query": "{Couvert par Embedding} = '✅'"},
                "backgroundColor": "#D5F5E3"
            },
            {
                "if": {"column_id": "Couvert par TF-IDF", "filter_query": "{Couvert par TF-IDF} = '✅'"},
                "backgroundColor": "#D6EAF8"
            },
            {
                "if": {"column_id": "Couvert par Embedding", "filter_query": "{Couvert par Embedding} = '❌'"},
                "backgroundColor": "#FADBD8"
            },
            {
                "if": {"column_id": "Couvert par TF-IDF", "filter_query": "{Couvert par TF-IDF} = '❌'"},
                "backgroundColor": "#FADBD8"
            }
        ],
        style_as_list_view=True
    )

    return summary, table