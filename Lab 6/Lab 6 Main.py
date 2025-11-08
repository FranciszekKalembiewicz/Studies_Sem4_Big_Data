import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

#Zadanie 1
#Wczytuje dane proponowane przez Pana
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names

#Standaryzacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#Wizualizacja
plt.figure(figsize=(8,6))
colors = ['navy', 'darkorange']
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, lw=2, label=target_name)

plt.title('PCA Breast Cancer Dataset')
plt.xlabel('Pierwsza główna składowa')
plt.ylabel('Druga główna składowa')
plt.legend()
plt.grid(True)
plt.show()

#Zadanie 2
#Wczytanie danych
digits = load_digits()
X = digits.data
y = digits.target

#Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Redukcja wymiarowości za pomocą t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

#Wizualizacja wyników
plt.figure(figsize=(10,8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Cyfry", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE na zbiorze Digits')
plt.xlabel('Wymiar 1')
plt.ylabel('Wymiar 2')
plt.grid(True)
plt.tight_layout()
plt.show()

#Zadania 3
#Wczytanie zbioru danych polecanych przez Pana z minimum 70 twarzami
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

#Obrazy jako płaskie wektory
X = lfw_people.data
n_samples, n_features = X.shape
#Wysokość i szerokość zdjęć
h, w = lfw_people.images.shape[1:]
print(f'Rozmiar danych: {n_samples} próbek, {n_features} cech')

#Normalizacja danych
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#NMF
n_components = 15  # liczba cech (twarzy-bazowych), które chcemy znaleźć
nmf = NMF(n_components=n_components, init='nndsvda', random_state=42, max_iter=2000)
W = nmf.fit_transform(X_scaled)
H = nmf.components_

#Wizualizacja
fig, axes = plt.subplots(3, 5, figsize=(12, 8), subplot_kw={'xticks':[], 'yticks':[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(H[i].reshape(h, w), cmap='gray')
    ax.set_title(f'Cech {i+1}')

plt.suptitle('Wykryte cechy twarzy za pomocą NMF', fontsize=16)
plt.tight_layout()
plt.show()

#Zadania 4
#Wczytuje zbiór danych o winach
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

#Standaryzacja danych wymagane przed PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

#Analiza wariancji wyjaśnianej przez każdy komponent
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

#Wizualizacjia: Wariancja wyjaśniona przez kolejne komponenty
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.title('Skumulowana wariancja wyjaśniona przez PCA')
plt.xlabel('Liczba komponentów')
plt.ylabel('Skumulowana wariancja')
plt.grid()
plt.show()

#Wybór liczby komponentów
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f'Wybrano {n_components_95} komponentów, które wyjaśniają co najmniej 95% wariancji.')

#Przekształcenie danych do nowej przestrzeni
pca_final = PCA(n_components=n_components_95)
X_reduced = pca_final.fit_transform(X_scaled)

#Wizualizacja danych dla 2 komponentów
plt.figure(figsize=(8,6))

for target, color, label in zip([0, 1, 2], ['red', 'green', 'blue'], target_names):
    plt.scatter(X_reduced[y == target, 0], X_reduced[y == target, 1], alpha=0.7, color=color, label=label)

plt.title('Wina w przestrzeni dwóch pierwszych komponentów PCA')
plt.xlabel('Pierwszy komponent')
plt.ylabel('Drugi komponent')
plt.legend()
plt.grid()
plt.show()

#Zadanie 5
#Wczytaj dane
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data

#Proste czyszczenie tekstu
def preprocess_text(text):
    text = text.lower()  # małe litery
    text = re.sub(r'\W+', ' ', text)  # usuń wszystko oprócz liter
    return text

processed_docs = [preprocess_text(doc) for doc in documents]

#Wektor cech
vectorizer = CountVectorizer(
    stop_words='english',  # automatyczne usunięcie angielskich stop words
    max_df=0.95,
    min_df=2,
    max_features=2000
)
X = vectorizer.fit_transform(processed_docs)

#LDA
n_topics = 20
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
X_topics = lda_model.fit_transform(X)

#Przypisanie dokumentów do tematów
document_topics = np.argmax(X_topics, axis=1)

#Wykres
plt.figure(figsize=(12,6))
sns.countplot(x=document_topics, hue=document_topics, palette='tab20', legend=False)
plt.xlabel('Temat (Topic)')
plt.ylabel('Liczba dokumentów')
plt.title('Rozkład dokumentów na tematy (LDA)')
plt.xticks(range(n_topics))
plt.grid(axis='y')
plt.show()

#Wypisanie słów dla każdego tematu
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Temat {topic_idx}: ", end='')
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

print("\nNajważniejsze słowa dla każdego tematu:\n")
display_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words=10)