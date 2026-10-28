import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

# === 1. Chargement des données métiers ===
df = pd.read_excel("../Data/dataset_métiers_datagouvernance.xlsx", sheet_name="Missions")
df = df[['Titre du poste', 'Tâches']].dropna(subset=['Tâches'])

# === 2. Chargement et encodage des compétences ROME depuis l'Excel ===
rome_excel_path = "../Data/ROME/ROME Arborescence des Competences 24M11.xlsx"
rome_cache_path = "../Data/ROME/rome_embeddings_from_excel.pkl"

if os.path.exists(rome_cache_path):
    with open(rome_cache_path, "rb") as f:
        rome_texts, rome_embeddings = pickle.load(f)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # même modèle que pour l'encodage initial
else:
    rome_df = pd.read_excel(rome_excel_path, sheet_name="Données brutes Comp.",
                            usecols=["4 - Macro-compétence", "5 - Compétence"]).dropna().drop_duplicates()
    rome_texts = pd.concat([
        rome_df["4 - Macro-compétence"],
        rome_df["5 - Compétence"]
    ]).dropna().unique().tolist()

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    rome_embeddings = model.encode(rome_texts, convert_to_tensor=True)

    with open(rome_cache_path, "wb") as f:
        pickle.dump((rome_texts, rome_embeddings), f)

# === 3. Fonction pour associer 1 à 2 compétences ROME à chaque tâche ===
def associe_competences_rome(tache, seuil=0.5):
    emb = model.encode(tache, convert_to_tensor=True)
    scores = util.cos_sim(emb, rome_embeddings)[0]
    top_scores = torch.topk(scores, k=3)
    selected = []

    for idx, val in zip(top_scores.indices.tolist(), top_scores.values.tolist()):
        if val >= seuil or len(selected) == 0:
            selected.append((rome_texts[idx], val))
        if len(selected) == 2:
            break
    return selected

# === 4. Association métier → compétences ROME via les tâches ===
metier_competences = defaultdict(list)

for _, row in df.iterrows():
    poste = row['Titre du poste']
    taches = row['Tâches'].split("  ")  # séparation par double espace
    for tache in taches:
        rome_matches = associe_competences_rome(tache)
        metier_competences[poste].extend(rome_matches)

# === 5. Construction de la matrice métier × compétences ROME ===
all_romes = sorted(set(txt for lst in metier_competences.values() for txt, _ in lst))
rome_to_idx = {r: i for i, r in enumerate(all_romes)}

X = np.zeros((len(metier_competences), len(all_romes)))
metiers = list(metier_competences.keys())

for i, poste in enumerate(metiers):
    for rome_txt, _ in metier_competences[poste]:
        X[i, rome_to_idx[rome_txt]] = 1

# === 6. Clustering KMeans avec test de différents k ===
k_range = range(2, 15)
scores = []
models = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)
    models.append((kmeans, labels))
    print(f"k = {k} → silhouette score = {round(score, 3)}")

# === 7. Affichage du Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, marker='o')
plt.title("Silhouette Score (clustering sur les compétences ROME)")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.xticks(k_range)
plt.show()

# === 8. Choix du meilleur k dans une plage raisonnable
best_k = None
best_score = -1
for i, k in enumerate(k_range):
    if 4 <= k <= 10 and scores[i] > best_score:
        best_k = k
        best_score = scores[i]
        best_labels = models[i][1]

print(f"\n✅ Meilleur k retenu : {best_k} (silhouette = {round(best_score, 3)})")

# === 9. PCA + Visualisation des clusters
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=best_labels, cmap="tab10", s=60)
plt.title(f"Clustering via projection ROME (k = {best_k})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()
# === 9bis. UMAP + Visualisation des clusters
try:
    import umap
except ImportError:
    raise ImportError("UMAP n'est pas installé. Fais : pip install umap-learn")

umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

plt.figure(figsize=(10, 6))
scatter_umap = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=best_labels, cmap="tab10", s=60)
plt.title(f"Clustering via projection UMAP (k = {best_k})")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter_umap, label="Cluster")
plt.grid(True)
plt.show()

# === 10. Export des résultats
df_export = pd.DataFrame({
    "Titre du poste": metiers,
    "Cluster_ROME": best_labels
})
df_export.to_csv("../logs/clusters_rome.csv", index=False)
print("📁 Fichier 'clusters_rome.csv' généré.")

# === Sauvegarde pour analyse ultérieure ===
np.save("../logs/features_matrix_rome.npy", X)
with open("../logs/features_metiers.txt", "w") as f:
    for m in metiers:
        if pd.notnull(m):
            f.write(str(m).strip() + "\n")


print("✅ Matrice X et liste de métiers sauvegardées.")

# === 11. Graphe de similarité entre métiers (pour Tulip)
similarities = cosine_similarity(X)

edges = []
for i in range(len(metiers)):
    for j in range(i + 1, len(metiers)):
        sim = similarities[i, j]
        if sim > 0.3:  # tu peux ajuster le seuil si le graphe est trop dense
            edges.append({
                "source": metiers[i],
                "target": metiers[j],
                "weight": round(sim, 3)
            })

df_sim = pd.DataFrame(edges)
df_sim.to_csv("../logs/graphe_metiers_similarites.csv", index=False)
print("📁 Fichier 'graphe_metiers_similarites.csv' généré (pour Tulip)")