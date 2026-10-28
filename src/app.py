import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1(
        "🧐 Analyse TF-IDF et Embedding",
        style={
            "textAlign": "center",
            "color": "#4a4e69",
            "backgroundColor": "#f0f0f5",
            "padding": "20px",
            "borderRadius": "10px",
            "marginBottom": "30px",
            "fontSize": "36px",
            "fontFamily": "Arial, sans-serif",
            "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.1)",
        }
    ),

    dbc.Nav([
        dbc.NavLink("🔍 Embedding", href="/embedding", active="exact"),
        dbc.NavLink("🧮 TF-IDF", href="/tfidf", active="exact"),
        dbc.NavLink("📊 Comparaison", href="/comparaison", active="exact"),
        dbc.NavLink("📄 Analyse CV", href="/cv", active="exact"),
        dbc.NavLink("🧠 RNCP → ROME", href="/rncp-to-rome", active="exact"),
        dbc.NavLink("🤝 ROME → ROME", href="/rome-to-rome", active="exact"),
        dbc.NavLink("🔍🤝 ROME → ROME (détails)", href="/rome-to-rome-precision", active="exact")
    ], pills=True, className="mb-4"),

    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=True)
