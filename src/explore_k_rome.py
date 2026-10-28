import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Chargement des données ===
df_text = pd.read_csv("../logs/clusters_metiers_final.csv")
df_rome = pd.read_csv("../logs/clusters_rome.csv")
merged = pd.merge(df_text, df_rome, on="Titre du poste", how="inner")

# === 2. Chargement de la matrice métier × compétences ROME ===
# Recalculée à partir du fichier de base si pas stockée. Ici, on suppose que tu l’as enregistrée :
features_path = "../logs/features_matrix_rome.npy"
metiers_path = "../logs/features_metiers.txt"

if not os.path.exists(features_path) or not os.path.exists(metiers_path):
    raise FileNotFoundError("❌ Fichier features_matrix_rome.npy ou features_metiers.txt manquant. "
                            "Pense à sauvegarder la matrice X et les noms de métiers après projection.")

X = np.load(features_path)
with open(metiers_path, "r") as f:
    metiers = [line.strip() for line in f]

# On garde les lignes correspondant aux métiers présents dans le merge
mask = [i for i, m in enumerate(metiers) if m in merged["Titre du poste"].values]
X = X[mask]
y_true = merged.set_index("Titre du poste").loc[[metiers[i] for i in mask]]["Cluster"].values

# === 3. Test de plusieurs k
k_values = range(4, 11)
ari_scores = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    sil_score = silhouette_score(X, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    silhouette_scores.append(sil_score)
    ari_scores.append(ari)

    print(f"\n🔹 k = {k}")
    print(f"- Silhouette Score : {sil_score:.3f}")
    print(f"- Adjusted Rand Index (vs texte) : {ari:.3f}")

    # matrice de confusion
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Matrice de confusion : texte vs ROME (k={k})")
    plt.xlabel("Cluster ROME (k=" + str(k) + ")")
    plt.ylabel("Cluster texte")
    plt.tight_layout()
    plt.show()

# === 4. Graphe récapitulatif des scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, label="Silhouette", marker='o')
plt.plot(k_values, ari_scores, label="Adjusted Rand Index (vs texte)", marker='s')
plt.title("Scores selon k (clustering ROME)")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
