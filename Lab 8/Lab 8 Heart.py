import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Generowanie danych symulacyjnych dla chorób serca
data, _ = make_classification(
    n_samples=500, n_features=6, n_informative=5, n_redundant=1, random_state=42
)
columns = ['Cholesterol', 'Max_Heart_Rate', 'Resting_BP', 'BMI', 'Age', 'Oldpeak']
heart_data = pd.DataFrame(data, columns=columns)

# Standaryzacja danych
scaler = StandardScaler()
scaled_heart_data = scaler.fit_transform(heart_data)

# Redukcja wymiaru do 2D dla wizualizacji
pca = PCA(n_components=2)
heart_pca = pd.DataFrame(pca.fit_transform(scaled_heart_data), columns=["PC1", "PC2"])

# Inicjalizacja modeli
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
gmm = GaussianMixture(n_components=3, random_state=42)
dbscan = DBSCAN(eps=0.5, min_samples=5)
meanshift = MeanShift()

# Dopasowanie modeli do danych i przewidywanie klastrów
models = {
    "K-means": kmeans,
    "GMM": gmm,
    "DBSCAN": dbscan,
    "Mean Shift": meanshift
}

results = {}
for name, model in models.items():
    # Dopasowanie i prognozowanie klastrów
    clusters = model.fit_predict(scaled_heart_data)

    # Obliczanie metryk tylko dla przypadków, gdzie zidentyfikowano więcej niż 1 klaster
    if len(set(clusters)) > 1:  # Sprawdzenie, czy istnieje więcej niż jeden klaster
        silhouette = silhouette_score(scaled_heart_data, clusters)
        calinski = calinski_harabasz_score(scaled_heart_data, clusters)
        davies = davies_bouldin_score(scaled_heart_data, clusters)
    else:
        silhouette = None
        calinski = None
        davies = None
        print(f"\nUwaga: Algorytm {name} zidentyfikował tylko jeden klaster lub wszystkie punkty jako szum.")

    # Przechowaj wyniki
    results[name] = {
        "Clusters": clusters,
        "Silhouette Score": silhouette,
        "Calinski-Harabasz Score": calinski,
        "Davies-Bouldin Score": davies
    }

# Wyświetlanie wyników metryk
for method, metrics in results.items():
    print(f"\n=== Metoda: {method} ===")
    print(f"Silhouette Score: {metrics['Silhouette Score']}")
    print(f"Calinski-Harabasz Score: {metrics['Calinski-Harabasz Score']}")
    print(f"Davies-Bouldin Score: {metrics['Davies-Bouldin Score']}")

# Wizualizacja klastrów dla każdej metody
plt.figure(figsize=(15, 12))

for i, (name, metrics) in enumerate(results.items(), start=1):
    plt.subplot(2, 2, i)
    sns.scatterplot(
        x=heart_pca["PC1"],
        y=heart_pca["PC2"],
        hue=metrics["Clusters"],
        palette="Set2",
        s=50,
        legend=None
    )
    plt.title(f"Klastrowanie: {name}")
    plt.xlabel("Główna składowa 1")
    plt.ylabel("Główna składowa 2")

plt.tight_layout()
plt.show()

# Podsumowanie wyników
print("\n=== Interpretacja wyników ===")
print("Silhouette Score: Wyższa wartość oznacza lepsze dopasowanie i separację klastrów.")
print("Calinski-Harabasz Score: Wyższa wartość oznacza lepszą strukturę klastrów.")
print("Davies-Bouldin Score: Niższa wartość oznacza bardziej kompaktowe i dobrze rozdzielone klastry.")
print("\nPorównanie powyższych metryk pozwala zidentyfikować metodę najlepiej dopasowaną do danych.")