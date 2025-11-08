import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import seaborn as sns

# 1. Wczytanie zbioru danych Olivetti Faces
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data  # obrazy spłaszczone
images = faces.images  # obrazy 64x64
n_samples, n_features = X.shape

print(f'Liczba obrazów: {n_samples}, liczba pikseli: {n_features}')

# 2. Redukcja wymiarowości (PCA) do 2D w celu wizualizacji
pca = PCA(n_components=2, whiten=True, random_state=42)
X_pca = pca.fit_transform(X)

# 3. Klastrowanie DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_pca)

# 4. Liczba klastrów i odosobnionych punktów
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'Liczba klastrów: {n_clusters}')
print(f'Liczba odosobnionych punktów (noise): {n_noise}')

# 5. Wizualizacja 2D klastrów
plt.figure(figsize=(10, 6))
palette = sns.color_palette('hsv', n_colors=n_clusters)

for cluster_id in set(labels):
    mask = labels == cluster_id
    color = 'k' if cluster_id == -1 else palette[cluster_id % len(palette)]
    label = 'Odosobnione' if cluster_id == -1 else f'Klaster {cluster_id}'
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=30, color=color, label=label, alpha=0.6)

plt.title('Wizualizacja klastrów DBSCAN (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Przykładowe twarze z każdego klastra
def plot_faces(images, labels, cluster_id, max_faces=10):
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        print(f"Brak twarzy w klastrze {cluster_id}")
        return
    plt.figure(figsize=(10, 2))
    for i, image_idx in enumerate(idx[:max_faces]):
        plt.subplot(1, max_faces, i + 1)
        plt.imshow(images[image_idx], cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Twarze z klastra {cluster_id}' if cluster_id != -1 else 'Twarze odosobnione (noise)')
    plt.show()

# 7. Wyświetl twarze z kilku klastrów (w tym outlierów)
for cluster_id in sorted(set(labels)):
    plot_faces(images, labels, cluster_id)
