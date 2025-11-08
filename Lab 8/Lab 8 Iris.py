import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Wczytanie zbioru danych Iris
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_target = iris.target

# Wyświetlenie przykładowych danych
print("\nPrzykładowe dane zbioru Iris:\n", iris_data.head())
print("\nEtykiety rzeczywiste (przykładowe):\n", iris_target[:5])

# Standaryzacja danych Iris
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_data)

# Zastosowanie algorytmu Aglomeracyjnego klasteryzacji
n_clusters_agglomerative = 3  # Zakładamy 3 klastry (powiązane z typami irysów)
agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters_agglomerative, linkage='ward')
clusters = agglomerative_clustering.fit_predict(iris_scaled)

# Dodanie wyników klasteryzacji do ramki wynikowej
iris_clustered = iris_data.copy()
iris_clustered['Agglomerative_Cluster'] = clusters

# Analiza klastrów - średnie wartości w każdym klastrze
cluster_summary_iris = iris_clustered.groupby('Agglomerative_Cluster').mean()
print("\nŚrednie wartości cech w klastrach (Agglomerative Clustering):\n", cluster_summary_iris)

# Wizualizacja klastrów w 2D za pomocą PCA
iris_pca = PCA(n_components=2)
iris_pca_data = pd.DataFrame(iris_pca.fit_transform(iris_scaled), columns=['PC1', 'PC2'])

plt.figure(figsize=(12, 6))

# Wykres klastrów
plt.subplot(1, 2, 1)
sns.scatterplot(
    x=iris_pca_data['PC1'], 
    y=iris_pca_data['PC2'], 
    hue=clusters, 
    palette='Set2', 
    s=50
)
plt.title("Wizualizacja klastrów (Agglomerative Clustering)")
plt.xlabel("Główna składowa 1 (PC1)")
plt.ylabel("Główna składowa 2 (PC2)")
plt.legend(title="Klastry")

# Porównanie z prawdziwymi etykietami
plt.subplot(1, 2, 2)
sns.scatterplot(
    x=iris_pca_data['PC1'], 
    y=iris_pca_data['PC2'], 
    hue=iris_target, 
    palette='tab10', 
    s=50
)
plt.title("Wizualizacja rzeczywistych etykiet Iris")
plt.xlabel("Główna składowa 1 (PC1)")
plt.ylabel("Główna składowa 2 (PC2)")
plt.legend(title="Prawdziwe etykiety")

plt.tight_layout()
plt.show()

# Porównanie wyników: etykiety rzeczywiste a klasy klastrowe
print("\n=== Porównanie wyników ===")

# Miara zgodności: Indeks Rand
adjusted_rand = adjusted_rand_score(iris_target, clusters)
print(
    f"Indeks zgodności (Adjusted Rand Index) między klastrami a etykietami: {adjusted_rand:.2f}"
)

# Wyświetlenie rzeczywistych i przypisanych etykiet
print("\nEtykiety rzeczywiste (przykładowe):", iris_target[:10])
print("Etykiety klastrów (przykładowe):", clusters[:10])