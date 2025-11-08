import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# Wczytanie danych Diabetes
diabetes = load_diabetes()
diabetes_data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
diabetes_target = diabetes.target  # Cukrzyca - wynik regresji (np. poziom glukozy)

# Wyświetlenie pierwszych kilku wierszy danych
print("\nPrzykładowe dane zbioru Diabetes:\n", diabetes_data.head())
print("\nPrzykładowe wartości docelowe:\n", diabetes_target[:5])

# Standaryzacja danych
scaler = StandardScaler()
diabetes_scaled = scaler.fit_transform(diabetes_data)

# Hierarchiczne klastrowanie
# Użycie linkage do wizualizacji dendrogramu
linked = linkage(diabetes_scaled, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=5, labels=None)
plt.title("Dendrogram dla danych Diabetes")
plt.xlabel("Próbki pacjentów")
plt.ylabel("Odległość (Ward's method)")
plt.tight_layout()
plt.show()

# Wybór liczby klastrów i klastrowanie
n_clusters = 3  # Liczba klastrów do wyodrębnienia
hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
clusters = hierarchical_clustering.fit_predict(diabetes_scaled)

# Dodanie wyników klastrowania do danych
diabetes_clustered = diabetes_data.copy()
diabetes_clustered['Cluster'] = clusters

# Analiza średnich wartości w klastrach
cluster_summary = diabetes_clustered.groupby('Cluster').mean()
print("\nŚrednie wartości cech w klastrach:\n", cluster_summary)

# Wizualizacja klastrów
# Redukcja wymiarów do 2D za pomocą PCA
pca = PCA(n_components=2)
diabetes_pca = pca.fit_transform(diabetes_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=diabetes_pca[:, 0],
    y=diabetes_pca[:, 1],
    hue=clusters,
    palette="Set2",
    s=50
)
plt.title("Wizualizacja klastrów (Agregacyjne klastrowanie)")
plt.xlabel("Główna składowa 1 (PC1)")
plt.ylabel("Główna składowa 2 (PC2)")
plt.legend(title="Klastry")
plt.show()

# Analiza potencjalnych implikacji klastrów
print("\n=== Interpretacja wyników ===")
print("- Klaster 0: Może odpowiadać pacjentom o średnich wartościach cech wynikających z danych.")
print("- Klaster 1: Typowe cechy pacjentów z poziomem glukozy/materiałem genetycznym różniącym się znacznie od mediany.")
print("- Klaster 2: Pacjenci wyróżniającymi się cechami (np. BMI, poziom glukozy, wiek itp.), co potencjalnie może wskazywać na skomplikowane przypadki.")
print("Uzyskana segmentacja może pomóc w lepszym dopasowaniu strategii leczenia, biorąc pod uwagę indywidualne różnice pacjentów.")