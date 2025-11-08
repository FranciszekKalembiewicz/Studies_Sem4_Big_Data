import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytanie zbioru danych Wine
wine = load_wine()
wine_data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_target = wine.target

# Wyświetlenie pierwszych kilku wierszy danych
print("\nPrzykładowe dane zbioru Wine:\n", wine_data.head())
print("\nPrzykładowe etykiety rzeczywiste:\n", wine_target[:5])

# Standaryzacja danych
scaler = StandardScaler()
wine_scaled = scaler.fit_transform(wine_data)

# Zastosowanie modelu mieszanych rozkładów Gaussa (GMM)
n_components_gmm = 3  # Zakładamy 3 grupy, ponieważ dane Wine mają 3 klasy etykiet
gmm = GaussianMixture(n_components=n_components_gmm, random_state=42)
gmm_labels = gmm.fit_predict(wine_scaled)

# Dodanie wyników klasteryzacji do ramki danych
wine_clustered = wine_data.copy()
wine_clustered['GMM_Cluster'] = gmm_labels

# Analiza wyników klasteryzacji
cluster_summary_wine = wine_clustered.groupby('GMM_Cluster').mean()
print("\nŚrednie wartości w każdej grupie (GMM):\n", cluster_summary_wine)

# Wizualizacja klastrów w 2D za pomocą PCA
pca = PCA(n_components=2)
wine_pca = pd.DataFrame(pca.fit_transform(wine_scaled), columns=['PC1', 'PC2'])

plt.figure(figsize=(12, 6))

# Wykres klastrów
plt.subplot(1, 2, 1)
sns.scatterplot(
    x=wine_pca['PC1'], 
    y=wine_pca['PC2'], 
    hue=gmm_labels, 
    palette='Set2', 
    s=50
)
plt.title("Wizualizacja klastrów (GMM)")
plt.xlabel("Główna składowa 1 (PC1)")
plt.ylabel("Główna składowa 2 (PC2)")
plt.legend(title="Klastry")

# Porównanie z rzeczywistymi etykietami
plt.subplot(1, 2, 2)
sns.scatterplot(
    x=wine_pca['PC1'], 
    y=wine_pca['PC2'], 
    hue=wine_target, 
    palette='tab10', 
    s=50
)
plt.title("Wizualizacja rzeczywistych etykiet Wine")
plt.xlabel("Główna składowa 1 (PC1)")
plt.ylabel("Główna składowa 2 (PC2)")
plt.legend(title="Prawdziwe etykiety")

plt.tight_layout()
plt.show()

# Porównanie wyników: etykiety rzeczywiste vs. etykiety z GMM
print("\n=== Ocena skuteczności modelu ===")

# Wskaźnik Rand (Adjusted Rand Index)
adjusted_rand = adjusted_rand_score(wine_target, gmm_labels)
print(f"Indeks zgodności (Adjusted Rand Index): {adjusted_rand:.2f}")

# Wskaźnik sylwetki (Silhouette Score)
silhouette_avg = silhouette_score(wine_scaled, gmm_labels)
print(f"Średni wynik sylwetki (Silhouette Score): {silhouette_avg:.2f}")

# Wyświetlenie rzeczywistych i przypisanych etykiet dla kilku próbek
print("\nEtykiety rzeczywiste (przykładowe):", wine_target[:10])
print("Etykiety klastrów (GMM, przykładowe):", gmm_labels[:10])