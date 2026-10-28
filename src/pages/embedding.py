import dash
from dash import html, dcc, callback, Input, Output, dash_table
import pandas as pd
import os
import subprocess
import json
import dash_bootstrap_components as dbc

from dash.dash_table.Format import Format

dash.register_page(__name__, path="/embedding")

df_metiers = pd.read_csv("../Data/RNCP/export_fiches_CSV_Standard_2025_06_30.csv", sep=";")
mapping_metiers = dict(zip(df_metiers["Numero_Fiche"], df_metiers["Intitule"]))
df_rncp = pd.read_csv("../Data/RNCP/export_fiches_CSV_Blocs_De_Compétences_2025_06_30.csv", sep=";")
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
            brand="📊 Comparaison Embedding",
            brand_href="/",
            color="#d1a0fa",
            dark=True,
            fluid=True
        ),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Code RNCP :"),
                    dcc.Dropdown(
                        id="input-rncp-emb",
                        options=[{"label": code, "value": code} for code in codes_rncp],
                        placeholder="Sélectionner un code RNCP",
                        searchable=True
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Code ROME :"),
                    dcc.Dropdown(
                        id="input-rome-emb",
                        options=[{"label": code, "value": code} for code in codes_rome],
                        placeholder="Sélectionner un code ROME",
                        searchable=True
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Seuil :"),
                    dcc.Input(
                        id="input-seuil-emb",
                        type="number",
                        step=0.01,
                        value=0.35,
                        style={"width": "100%"}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Label("Options :"),
                    dcc.Checklist(
                        id="reverse-toggle-emb",
                        options=[{"label": "Inverser RNCP/ROME", "value": "reverse"}],
                        value=[],
                        inputStyle={"marginRight": "8px"},
                        labelStyle={"display": "block", "marginTop": "8px"}
                    )
                ], width=4)
            ], className="mb-3"),

            html.Div("💡 Le tableau se met à jour dynamiquement. Cliquez sur une ligne pour voir toutes les correspondances RNCP.", style={
                "marginBottom": "15px",
                "fontSize": "14px",
                "fontStyle": "italic"
            }),

            dcc.Loading(
                type="circle",
                color="#2a4d69",
                children=[
                    html.Div(id="summary-emb", style={"fontWeight": "bold", "marginBottom": "15px", "fontSize": "18px"}),
                    html.Div(id="output-emb")
                ]
            )
        ], style={
            "backgroundColor": "#f8f0fc",
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"
        })
    ]
)

@callback(
    Output("summary-emb", "children"),
    Output("output-emb", "children"),
    Input("input-rncp-emb", "value"),
    Input("input-rome-emb", "value"),
    Input("input-seuil-emb", "value"),
    Input("reverse-toggle-emb", "value")
)

def update_table(code_rncp, code_rome, seuil, reverse_toggle):
    if not code_rncp or not code_rome:
        return "", html.Div("❌ Veuillez entrer un code RNCP et un code ROME.", style={"color": "red"})

    reverse = "reverse" in reverse_toggle
    script = "embedding_reverse.py" if reverse else "poke_embedding_detail.py"
    log_embed = f"../logs/logs_embedding{'_reverse' if reverse else ''}__{code_rncp.upper()}__{code_rome.upper()}.csv"

    if not os.path.exists(log_embed):
        subprocess.run(["python3", script, "--rncp", code_rncp, "--rome", code_rome])

    if not os.path.exists(log_embed):
        return "", html.Div("❌ Résultats manquants.", style={"color": "red"})

    df = pd.read_csv(log_embed)
    df["couvert"] = df["score"].apply(lambda s: "✅" if s >= seuil else "❌")

    if "top_3_associees" not in df.columns:
        df["top_3_associees"] = ""

    def details_cell(row):
        try:
            top3 = row["top_3_associees"]
            if isinstance(top3, float):
                return ""
            lignes = top3.strip().split("<br>" if "<br>" in top3 else "\n")
            lignes_filtrees = []
            for ligne in lignes:
                if "score" in ligne:
                    try:
                        s = float(ligne.split("score")[-1].replace("=", "").replace(")", "").strip())
                        if s >= seuil:
                            lignes_filtrees.append(ligne)
                    except:
                        lignes_filtrees.append(ligne)
                else:
                    lignes_filtrees.append(ligne)
            if not lignes_filtrees:
                return ""
            top3_html = "<br>".join(lignes_filtrees)
            return f"<details><summary>Voir +</summary><div>{top3_html}</div></details>"
        except Exception:
            return ""

    df["comp_rncp_details"] = df.apply(details_cell, axis=1)

    nb_total = len(df)
    nb_couverts = sum(df["couvert"] == "✅")
    taux = round(nb_couverts / nb_total * 100, 1)
    intitule_rncp = mapping_metiers.get(code_rncp, "Métier RNCP inconnu")
    intitule_rome = mapping_rome.get(code_rome, "Métier ROME inconnu")

    summary = html.Div([
        html.Div(f"🧠 Fiche RNCP {code_rncp} : {intitule_rncp}", style={"marginBottom": "5px"}),
        html.Div(f"📌 Fiche ROME {code_rome} : {intitule_rome}", style={"marginBottom": "15px"}),
        html.Div(f"✅ {nb_couverts} / {nb_total} compétences couvertes ({taux}%)")
    ])

    if reverse:
        colonnes = [
            {"name": "Compétence RNCP", "id": "comp_rncp"},
            {"name": "Score", "id": "score", "type": "numeric", "format": Format(precision=4)},
            {"name": "Couvert", "id": "couvert"},
            {"name": "Compétence ROME associée", "id": "competence_rome_associee"},
            {"name": "Voir +", "id": "comp_rncp_details", "presentation": "markdown"},
        ]
    else:
        colonnes = [
            {"name": "Compétence ROME", "id": "comp_rome"},
            {"name": "Score", "id": "score", "type": "numeric", "format": Format(precision=4)},
            {"name": "Couvert", "id": "couvert"},
            {"name": "Compétence RNCP associée", "id": "competence_rncp_associee"},
            {"name": "Voir +", "id": "comp_rncp_details", "presentation": "markdown"},
        ]

    table = dash_table.DataTable(
        columns=colonnes,
        data=df.to_dict("records"),
        cell_selectable=False,
        sort_action="native",
        style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "auto", "border": "1px solid #ccc"},
        style_cell={
            "textAlign": "left", "padding": "8px", "fontSize": "14px",
            "minWidth": "150px", "maxWidth": "500px", "whiteSpace": "normal"
        },
        style_header={"backgroundColor": "#F9E79F", "fontWeight": "bold"},
        style_data_conditional=[
            {
                "if": {"column_id": "couvert", "filter_query": "{couvert} = '✅'"},
                "backgroundColor": "#D5F5E3"
            },
            {
                "if": {"column_id": "couvert", "filter_query": "{couvert} = '❌'"},
                "backgroundColor": "#FADBD8"
            }
        ],
        style_as_list_view=True,
        markdown_options={"html": True}
    )

    return summary, table
