import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
from dash.dash_table.Format import Format
from sentence_transformers import util
from .cvAnalyse import get_model_and_rome_embeddings

dash.register_page(__name__, path="/rome-to-rome", name="ROME → ROME (similaires)")


# === Chargement du modèle + embeddings depuis le cache
df_rome, model, embeddings = get_model_and_rome_embeddings()
codes_rome = sorted(df_rome["code_rome"].unique())

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

    /* ✅ Permet de sélectionner le texte dans les cellules */
    .dash-table-container td div {
        user-select: text !important;
    }
    </style>
""", dangerously_allow_html=True),


        html.H2("🔁 Suggestions ROME → ROME", className="text-center mb-4", style={"color": "#1864ab"}),

        dbc.Card(
            [
                dbc.CardHeader("Sélectionnez une fiche ROME", style={"backgroundColor": "#d0ebff"}),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="input-code-rome",
                        options=[{"label": code, "value": code} for code in codes_rome],
                        placeholder="Choisir un code ROME",
                        searchable=True,
                        style={"marginBottom": "20px"}
                    ),
                    dcc.Input(
                        id="seuil-couverture",
                        type="number",
                        step=0.01,
                        value=0.50,
                        placeholder="Seuil de couverture",
                        style={"width": "100%", "marginBottom": "20px"}
                    ),
                    dcc.Loading(
                        type="circle",
                        color="#2a4d69",
                        children=html.Div(id="table-output", style={"minHeight": "200px"})
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
    Output("table-output", "children"),
    Input("input-code-rome", "value"),
    Input("seuil-couverture", "value"),
    allow_duplicate=True
)
def update_comparaison(code_rome_source, seuil):
    if not code_rome_source:
        return html.Div("❌ Veuillez sélectionner un code ROME.", style={"color": "red"})

    comp_source = df_rome[df_rome["code_rome"] == code_rome_source].copy()
    if comp_source.empty:
        return html.Div("❌ Code ROME source introuvable.", style={"color": "red"})

    # === Intitulé de la fiche ROME source
    intitule_rome = comp_source["intitule_metier"].iloc[0]
    summary = html.Div([
        html.Div([
            html.B("📌 Fiche ROME ", style={"marginRight": "5px"}),
            html.B(f"{code_rome_source} : {intitule_rome}")
        ], style={"marginBottom": "15px"})
    ])

    seuil_str = str(seuil).replace(".", "_")
    log_path = f"../logs/logs_rome_to_rome__{code_rome_source}__seuil_{seuil_str}.csv"
    if os.path.exists(log_path):
        df_resultats = pd.read_csv(log_path)
        df_resultats = df_resultats.sort_values(by="couverture", ascending=False).head(5)  # ✅ nouveau
        print(f"📁 Résultats chargés depuis le cache : {log_path}")
        table = dash_table.DataTable(
            columns=[
                {"name": "Code ROME", "id": "code_rome", "presentation": "markdown"},
                {"name": "Intitulé", "id": "intitule", "presentation": "markdown"},
                {"name": "Couverture (%)", "id": "couverture", "type": "numeric", "format": Format(precision=1)},
                {"name": "Score moyen", "id": "score_moyen", "type": "numeric", "format": Format(precision=4)},
            ],
            cell_selectable=False,
            data=df_resultats.to_dict("records"),
            style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "auto"},
            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "14px", "whiteSpace": "normal"},
            style_header={"backgroundColor": "#d0ebff", "fontWeight": "bold"},
            sort_action="native"
        )
        return html.Div([summary, table])


    # Calcul si pas dans cache
    vec_source = model.encode(comp_source["libelle_competence"].tolist(), convert_to_tensor=True)
    rows = []
    for code in codes_rome:
        if code == code_rome_source:
            continue
        comp_target = df_rome[df_rome["code_rome"] == code]
        if comp_target.empty:
            continue
        vec_target = model.encode(comp_target["libelle_competence"].tolist(), convert_to_tensor=True)
        sim_matrix = util.cos_sim(vec_source, vec_target).cpu().numpy()
        couverture = np.mean(sim_matrix.max(axis=1) >= seuil)
        score_moyen = sim_matrix.mean()
        rows.append({
            "code_rome": code,
            "intitule": comp_target["intitule_metier"].iloc[0],
            "couverture": float(round(couverture, 1)),
            "score_moyen": round(score_moyen, 4)
        })

    df_resultats = pd.DataFrame(rows).sort_values(by="couverture", ascending=False)
    os.makedirs("../logs", exist_ok=True)
    seuil_str = str(seuil).replace(".", "_")
    chemin_csv = f"../logs/logs_rome_to_rome__{code_rome_source}__seuil_{seuil_str}.csv"
    df_resultats.to_csv(chemin_csv, index=False)
    print(f"✅ Résultats enregistrés dans : {chemin_csv}")

    table = dash_table.DataTable(
        columns=[
            {"name": "Code ROME", "id": "code_rome"},
            {"name": "Intitulé", "id": "intitule"},
            {"name": "Couverture (%)", "id": "couverture", "type": "numeric", "format": Format(precision=1)},
            {"name": "Score moyen", "id": "score_moyen", "type": "numeric", "format": Format(precision=4)}
        ],
        data=df_resultats.to_dict("records"),
        cell_selectable=False,
        style_table={"overflowX": "auto", "maxHeight": "600px", "overflowY": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "14px", "whiteSpace": "normal"},
        style_header={"backgroundColor": "#d0ebff", "fontWeight": "bold"},
        sort_action="native"
    )
    return html.Div([summary, table])
