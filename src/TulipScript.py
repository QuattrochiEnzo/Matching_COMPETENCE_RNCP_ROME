from tulip import tlp
import csv

# === 1. Spécifie ici le chemin du fichier CSV ===
csv_path = "/autofs/unityaccount/cremi/equattrochi/Bureau/STAGE/labri-stage-modelisation_rncp_rome/logs/graphe_metiers_similarites_texte.csv"

# === 2. Création du graphe ===
graph = tlp.newGraph()
graph.setName("Graphe des similarités entre métiers (texte)")

viewLabel = graph.getStringProperty("viewLabel")
weightProp = graph.getDoubleProperty("weight")

# === 3. Lecture et ajout des nœuds et arêtes ===
node_map = {}

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        src_name = row["source"]
        tgt_name = row["target"]
        weight = float(row["weight"])

        # Création des nœuds s’ils n’existent pas
        if src_name not in node_map:
            n_src = graph.addNode()
            viewLabel[n_src] = src_name
            node_map[src_name] = n_src
        else:
            n_src = node_map[src_name]

        if tgt_name not in node_map:
            n_tgt = graph.addNode()
            viewLabel[n_tgt] = tgt_name
            node_map[tgt_name] = n_tgt
        else:
            n_tgt = node_map[tgt_name]

        # Création de l’arête avec poids
        edge = graph.addEdge(n_src, n_tgt)
        weightProp[edge] = weight

tlp.saveGraph(graph, "/autofs/unityaccount/cremi/equattrochi/Bureau/STAGE/labri-stage-modelisation_rncp_rome/logs/graphe_metiers_similarites_texte.tlp")
print("📁 Graphe sauvegardé dans logs/graphe_metiers_similarites_texte.tlp")

