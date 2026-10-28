import argparse
import os
import pandas as pd
from tulip import tlp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération de graphe Tulip depuis les logs ROME/RNCP")
    parser.add_argument("--rncp", type=str, required=True, help="Ex: RNCP34067")
    parser.add_argument("--rome", type=str, required=True, help="Ex: H1401")
    parser.add_argument("--method", type=str, choices=["embedding", "tfidf"], required=True)
    parser.add_argument("--reverse", type=lambda x: x.lower() == "true", default=False, help="True pour inverser RNCP/ROME")
    parser.add_argument("--seuil", type=float, default=0.15, help="Seuil minimal pour filtrer les scores")
    args = parser.parse_args()

    fiche_rncp = args.rncp.upper()
    fiche_rome = args.rome.upper()
    seuil = args.seuil

    # === 📁 Déduction du nom de fichier CSV
    prefix = {
        ("embedding", False): "logs_embedding",
        ("embedding", True): "logs_embedding_reverse",
        ("tfidf", False): "logs_tfidf",
        ("tfidf", True): "logs_tfidf_reverse"
    }.get((args.method, args.reverse))

    if prefix is None:
        raise ValueError("Méthode ou sens non reconnu")

    CSV_PATH = f"../logs/{prefix}__{fiche_rncp}__{fiche_rome}.csv"
    TLP_OUTPUT = f"../logs/graphe_{prefix}__{fiche_rncp}__{fiche_rome}.tlp"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Fichier introuvable : {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # === 📌 Colonnes selon sens
    if args.reverse:
        rome_text_col = "competence_rome_associee"
        rncp_text_col = "comp_rncp"
    else:
        rome_text_col = "comp_rome"
        rncp_text_col = "competence_rncp_associee"

    print("📂 Colonnes détectées :", df.columns.tolist())
    print(f"📄 Lecture : ROME ← {rome_text_col} | RNCP ← {rncp_text_col} | Seuil = {seuil}")

    # === 🧠 Création du graphe Tulip
    graph = tlp.newGraph()
    label_prop = graph.getStringProperty("label")
    type_prop = graph.getStringProperty("type")
    score_prop = graph.getDoubleProperty("score")
    details_prop = graph.getStringProperty("details")  # 💬 nouvelle propriété pour full_associees

    node_map_rome = {}
    node_map_rncp = {}
    edge_count = 0

    for _, row in df.iterrows():
        if row["score"] < seuil:
            continue

        rome_text = str(row[rome_text_col])
        rncp_text = str(row[rncp_text_col])
        full_associees = str(row.get("full_associees", "")).strip()

        # Création ou récupération des nœuds
        if rome_text not in node_map_rome:
            n_rome = graph.addNode()
            label_prop[n_rome] = rome_text
            type_prop[n_rome] = "ROME"
            node_map_rome[rome_text] = n_rome
        else:
            n_rome = node_map_rome[rome_text]

        rncp_id = f"{fiche_rncp} - {rncp_text}"
        if rncp_id not in node_map_rncp:
            n_rncp = graph.addNode()
            label_prop[n_rncp] = rncp_id
            type_prop[n_rncp] = "RNCP"
            node_map_rncp[rncp_id] = n_rncp
        else:
            n_rncp = node_map_rncp[rncp_id]

        edge = graph.addEdge(n_rome, n_rncp)
        score_prop[edge] = row["score"]
        details_prop[edge] = full_associees
        edge_count += 1

    # 💾 Sauvegarde du graphe
    tlp.saveGraph(graph, TLP_OUTPUT)
    print(f"\n✅ Graphe généré : {TLP_OUTPUT}")
    print(f"📌 Nœuds ROME : {len(node_map_rome)}")
    print(f"📌 Nœuds RNCP : {len(node_map_rncp)}")
    print(f"🔗 Arêtes totales : {edge_count}")