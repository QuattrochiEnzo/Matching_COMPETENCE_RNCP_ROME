import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Charger et grouper les tâches par métier ===
file_path = "../Data/dataset_métiers_datagouvernance.xlsx"
df = pd.read_excel(file_path, sheet_name="Missions")
df = df[['Titre du poste', 'Tâches']].dropna(subset=['Tâches'])

grouped = df.groupby("Titre du poste")["Tâches"].apply(lambda x: " ".join(x)).reset_index()

# === 2. Embedding ===
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(grouped["Tâches"].tolist(), show_progress_bar=True)

# === 3. Test des valeurs de k ===
k_range = range(2, 15)
scores = []
models = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    scores.append(score)
    models.append((kmeans, labels))
    print(f"k = {k} → silhouette score = {round(score, 3)}")

# === 4. Affichage du silhouette score pour tous les k ===
plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, marker='o')
plt.title("Silhouette Score selon k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.xticks(k_range)
plt.show()

# === 5. Choisir le meilleur k dans une plage "raisonnable" ===
best_k = None
best_score = -1
for i, k in enumerate(k_range):
    if 4 <= k <= 10 and scores[i] > best_score:
        best_k = k
        best_score = scores[i]
        best_labels = models[i][1]

print(f"\n✅ Meilleur k retenu : {best_k} (silhouette = {round(best_score, 3)})")

# === 6. PCA + Visualisation des clusters choisis ===
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=best_labels, cmap="tab10", s=60)
plt.title(f"Visualisation des clusters (k = {best_k})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()
# === 6 bis. Visualisation additionnelle pour k = 9 ===
k_target = 9
kmeans_9 = KMeans(n_clusters=k_target, random_state=42)
labels_9 = kmeans_9.fit_predict(embeddings)


reduced_9 = pca.transform(embeddings)  

plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_9[:, 0], reduced_9[:, 1], c=labels_9, cmap="tab10", s=60)
plt.title(f"Visualisation des clusters (k = {k_target})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()


# === 7. Export des résultats ===
grouped["Cluster"] = best_labels
grouped.to_csv("../logs/clusters_metiers_final.csv", index=False)
print("📁 Fichier 'clusters_metiers_final.csv' généré.")

# === 8. Graphe de similarité entre métiers (pour Tulip)
similarities = cosine_similarity(embeddings)
metiers = grouped["Titre du poste"].tolist()

edges = []
for i in range(len(metiers)):
    for j in range(i + 1, len(metiers)):
        sim = similarities[i, j]
        if sim > 0.3:  # seuil à ajuster selon la densité souhaitée
            edges.append({
                "source": metiers[i],
                "target": metiers[j],
                "weight": round(sim, 3)
            })

df_sim = pd.DataFrame(edges)
df_sim.to_csv("../logs/graphe_metiers_similarites_texte.csv", index=False)
print("📁 Fichier 'graphe_metiers_similarites_texte.csv' généré (pour Tulip)")