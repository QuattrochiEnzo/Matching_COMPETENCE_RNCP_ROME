import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/", name="Accueil")

layout = dbc.Container(
    [
        html.H1("Comparateur RNCP ↔ ROME", className="text-center mb-4", style={"color": "#4a4e69"}),

        html.P(
            "Bienvenue sur l'outil de comparaison entre les compétences des fiches RNCP et les métiers ROME.",
            className="lead text-center",
            style={"color": "#6c757d"},
        ),

        html.Hr(),

        # Ligne 1
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("🔎 Méthode Embedding", style={"backgroundColor": "#e0bbff"}),
                        dbc.CardBody(
                            [
                                html.P("Utilise des embeddings sémantiques pour mesurer la similarité entre les compétences."),
                                dbc.Button("Aller à Embedding", color="primary", href="/embedding"),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#f8f0fc"}
                ),
                md=4,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("🧮 Méthode TF-IDF", style={"backgroundColor": "#b2f2bb"}),
                        dbc.CardBody(
                            [
                                html.P("Utilise des représentations TF-IDF classiques sur les mots."),
                                dbc.Button("Aller à TF-IDF", color="success", href="/tfidf"),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#ebfbee"}
                ),
                md=4,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("⚖️ Comparaison croisée", style={"backgroundColor": "#ffd8a8"}),
                        dbc.CardBody(
                            [
                                html.P("Compare les résultats des deux méthodes pour une même paire RNCP–ROME."),
                                dbc.Button("Aller à Comparaison", color="warning", href="/comparaison"),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#fff9db"}
                ),
                md=4,
            ),
        ], className="mt-4"),

        # Ligne 2
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("📄 Analyse CV", style={"backgroundColor": "#c5f6fa"}),
                        dbc.CardBody(
                            [
                                html.P("Propose des métiers ROME en analysant un CV PDF."),
                                dbc.Button("Aller à Analyse CV", color="info", href="/cv"),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#e3fafc"}
                ),
                md=6,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("🧠 Correspondance RNCP → ROME", style={"backgroundColor": "#ffe066"}),
                        dbc.CardBody(
                            [
                                html.P("Associe chaque compétence d’une fiche RNCP à la compétence ROME la plus proche."),
                                dbc.Button("Aller à RNCP → ROME", color="warning", href="/rncp-to-rome"),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#fffbe6"}
                ),
                md=6,
            )
        ], className="mt-4"),

        # Ligne 3
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("🔁 Suggestions ROME → ROME", style={"backgroundColor": "#d0ebff"}),
                        dbc.CardBody(
                            [
                                html.P("Propose des métiers ROME proches d’un code ROME donné."),
                                dbc.Button("Aller à ROME → ROME", color="primary", href="/rome-to-rome"),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#e3f2fd"}
                ),
                md=6,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("📋 Détail ROME ↔ ROME Détails", style={"backgroundColor": "#ffd6e0"}),
                        dbc.CardBody(
                            [
                                html.P("Compare compétence par compétence deux fiches ROME avec scores et top 3."),
                                dbc.Button("Aller à Détail ROME ↔ ROME", color="danger", href="/rome-to-rome-precision"),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#fff0f6"}
                ),
                md=6,
            ),

        ], className="mt-4"),

        html.Div(className="mt-5"),
        html.P("Projet réalisé dans le cadre d’un stage au LaBRI — Enzo Quattrochi",
               className="text-center", style={"color": "#adb5bd", "fontSize": "0.9em"}),
    ],
    fluid=True,
    className="p-5",
)
