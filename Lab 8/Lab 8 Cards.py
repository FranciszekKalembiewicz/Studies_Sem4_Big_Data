import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych z pliku CSV
file_name = "UCI_Credit_Card.csv"
data = pd.read_csv(file_name)

# Wyświetlenie pierwszych wierszy dla wglądu
print("Przykładowe dane:\n", data.head())

# Sprawdzanie brakujących wartości w danych
print("\nLiczba brakujących wartości przed czyszczeniem:")
print(data.isnull().sum())

# Usuwanie wierszy z brakującymi wartościami
data_cleaned = data.dropna()

# Sprawdzanie brakujących wartości po czyszczeniu
print("\nLiczba brakujących wartości po czyszczeniu:")
print(data_cleaned.isnull().sum())

# Kodowanie zmiennych kategorycznych
# Przykład: Zakodowanie kolumny 'SEX', 'EDUCATION', 'MARRIAGE' jako kategorie liczbowe
categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
data_cleaned[categorical_columns] = data_cleaned[categorical_columns].apply(pd.Categorical)

# One-hot encoding (kodowanie zerami i jedynkami dla zmiennych kategorycznych)
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns)

# Wyświetlenie przetworzonych danych
print("\nDane po zakodowaniu zmiennych kategorycznych:\n", data_encoded.head())

# Zapis przetworzonych danych do nowego pliku (opcjonalne)
data_encoded.to_csv("credit_card_default_cleaned.csv", index=False)
print("\nDane zostały oczyszczone i zapisane do pliku 'credit_card_default_cleaned.csv'.")

# Zastosowanie algorytmu K-means
# Wybór cech (zachowania finansowe klientów)
financial_features = [
    'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
    'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Standaryzacja danych (skalowanie do średniej 0 i odchylenia standardowego 1)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded[financial_features])

# Określenie optymalnej liczby klastrów (metoda "łokcia")
inertia = []
range_clusters = range(1, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Wizualizacja metody "łokcia"
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.title("Metoda łokcia - optymalna liczba klastrów")
plt.xlabel("Liczba klastrów")
plt.ylabel("Inertia")
plt.show()

# Na podstawie wykresu wybierz optymalną liczbę klastrów, np. k=4
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data_scaled)

# Dodanie wyników klastrowania do zbioru danych
data_encoded['Cluster'] = kmeans.labels_

# Analiza klastrów - obliczenie średnich wartości cech w każdym klastrze
cluster_summary = data_encoded.groupby('Cluster')[financial_features].mean()
print("\nŚrednie wartości cech w każdym klastrze (K-means):\n")
print(cluster_summary)

# Wizualizacja: Rozkład klastrów w dwóch wymiarach (K-means)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data_pca[:, 0], y=data_pca[:, 1], hue=data_encoded['Cluster'], palette='Set2', s=50
)
plt.title("Wizualizacja klastrów (K-means)")
plt.xlabel("Główna składowa 1")
plt.ylabel("Główna składowa 2")
plt.legend(title="Klastry")
plt.show()

# Zastosowanie metody Mean Shift
print("\n=== Mean Shift Clustering ===")

# Mean Shift identifikuje naturalne grupy w danych
mean_shift = MeanShift()
mean_shift.fit(data_scaled)

# Dodanie wyników klastrowania do zbioru danych
data_encoded['MeanShift_Cluster'] = mean_shift.labels_

# Analiza klastrów - obliczenie średnich wartości cech w klastrach (Mean Shift)
mean_shift_summary = data_encoded.groupby('MeanShift_Cluster')[financial_features].mean()
print("\nŚrednie wartości cech w klastrach (Mean Shift):\n")
print(mean_shift_summary)

# Porównanie liczby klastrów z K-means
print("\nLiczba klastrów z K-means:", len(data_encoded['Cluster'].unique()))
print("Liczba klastrów z Mean Shift:", len(data_encoded['MeanShift_Cluster'].unique()))

# Wizualizacja: Rozkład klastrów w dwóch wymiarach (Mean Shift)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data_pca[:, 0], y=data_pca[:, 1], hue=data_encoded['MeanShift_Cluster'], palette='tab10', s=50
)
plt.title("Wizualizacja klastrów (Mean Shift)")
plt.xlabel("Główna składowa 1")
plt.ylabel("Główna składowa 2")
plt.legend(title="Klastry")
plt.show()

# Porównanie ogólne wyników
print("\n=== Porównanie wyników ===")
print("Mean Shift wykrywa klastry na podstawie struktury danych, natomiast K-means wymaga założenia liczby klastrów a priori.")
print("Porównaj liczby klastrów i analizę średnich wartości cech w obu metodach, aby zrozumieć różnicę.")