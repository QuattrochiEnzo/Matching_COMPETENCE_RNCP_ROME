import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Chargement des clusters ===
df_text = pd.read_csv("../logs/clusters_metiers_final.csv")
df_rome = pd.read_csv("../logs/clusters_rome.csv")

# === 2. Fusion des deux sources sur le titre du poste ===
merged = pd.merge(df_text, df_rome, on="Titre du poste", how="inner")
print(f"🔢 Nombre de métiers communs aux deux clusters : {len(merged)}")

# === 3. Métriques de comparaison ===
ari = adjusted_rand_score(merged["Cluster"], merged["Cluster_ROME"])
homogeneity = homogeneity_score(merged["Cluster"], merged["Cluster_ROME"])
completeness = completeness_score(merged["Cluster"], merged["Cluster_ROME"])

print("\n🎯 Comparaison des clusters (texte vs ROME) :")
print(f"- Adjusted Rand Index (ARI)      : {ari:.3f}")
print(f"- Homogeneity Score               : {homogeneity:.3f}")
print(f"- Completeness Score              : {completeness:.3f}")

# === 4. Matrice de confusion (optionnel mais visuel) ===
cm = confusion_matrix(merged["Cluster"], merged["Cluster_ROME"])
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Matrice de confusion : Clusters texte vs ROME")
plt.xlabel("Cluster ROME")
plt.ylabel("Cluster texte")
plt.tight_layout()
plt.show()
