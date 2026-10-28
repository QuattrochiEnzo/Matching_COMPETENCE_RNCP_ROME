from tulip import tlp
import pandas as pd

graph = tlp.newGraph()
graph.setName("Graphe ROME–RNCP simple")

viewLabel = graph.getStringProperty("viewLabel")
nodeType = graph.getStringProperty("type")
edgeCouverture = graph.getDoubleProperty("couverture")

df = pd.read_csv("/net/cremi/equattrochi/Bureau/STAGE/labri-stage-modelisation_rncp_rome/logs/resume_couverture.csv")

node_map = {}

# Créer les nœuds ROME
for code in df["rome_code"].unique():
    n = graph.addNode()
    viewLabel[n] = code  # Label = code ROME
    nodeType[n] = "ROME"
    node_map[code] = n

# Créer les nœuds RNCP
for code in df["rncp_code"].unique():
    n = graph.addNode()
    viewLabel[n] = code  # Label = code RNCP
    nodeType[n] = "RNCP"
    node_map[code] = n

# Créer les arêtes si % couverture ≥ 30
SEUIL = 30
for _, row in df.iterrows():
    if row["pourcentage_couverture"] >= SEUIL:
        e = graph.addEdge(node_map[row["rome_code"]], node_map[row["rncp_code"]])
        edgeCouverture[e] = row["pourcentage_couverture"]

tlp.saveGraph(graph, "/net/cremi/equattrochi/Bureau/STAGE/labri-stage-modelisation_rncp_rome/logs/graph_rome_rncp_simple.tlp")
print("✅ Graphe Tulip sauvegardé avec code des nœuds affiché")
