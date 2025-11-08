# Standardowe biblioteki
from io import StringIO

# Biblioteki do analizy danych i wizualizacji
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn: modele klasyfikacji
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Sklearn: dane przykładowe
from sklearn.datasets import load_iris, load_breast_cancer, load_digits

# Sklearn: przetwarzanie danych
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sklearn: metryki
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Zadanie 1
# 1. Wczytanie zbioru danych
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 2. Eksploracja danych
print("Pierwsze 5 wierszy:")
print(df.head())

print("\nInformacje o kolumnach:")
print(df.info())

print("\nStatystyki opisowe:")
print(df.describe())

print("\nLiczba przykładów w każdej klasie:")
print(df['target'].value_counts())

# 3. Przygotowanie danych
X = iris.data
y = iris.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standaryzacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Wybór najlepszego k
accuracies = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)
    print(f"k={k}, dokładność: {acc:.4f}")

# Wykres dokładności
plt.figure(figsize=(8, 5))
plt.plot(k_range, accuracies, marker='o')
plt.title("Dokładność klasyfikatora KNN dla różnych wartości k")
plt.xlabel("Liczba sąsiadów (k)")
plt.ylabel("Dokładność")
plt.grid(True)
plt.show()

# Najlepsze k
best_k = k_range[np.argmax(accuracies)]
print(f"\nNajlepsze k: {best_k}")

# 5. Klasyfikator KNN z najlepszym k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# 6. Ocena jakości klasyfikatora
print("\nMacierz pomyłek:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Dokładność
accuracy = accuracy_score(y_test, y_pred)
# Precyzja i czułość (średnia dla wielu klas)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Specyficzność (dla każdej klasy osobno)
def specificity(cm):
    spec_list = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[:, i], i))
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0
        spec_list.append(spec)
    return np.mean(spec_list)

spec = specificity(cm)

# Wyświetlenie metryk
print(f"\nDokładność: {accuracy:.4f}")
print(f"Precyzja (średnia): {precision:.4f}")
print(f"Czułość (średnia): {recall:.4f}")
print(f"Specyficzność (średnia): {spec:.4f}")

# Zadanie 2
# 1. Wczytanie danych
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Tworzymy DataFrame do eksploracji
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 2. Eksploracja i czyszczenie danych
print("Informacje o danych:")
print(df.info())

print("\nBrakujące wartości:")
print(df.isnull().sum())  # nie ma braków, ale sprawdzamy

# 3. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model regresji logistycznej
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 6. Ocena jakości modelu
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)

# Specyficzność (TN / (TN + FP))
def specificity(cm):
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp) if (tn + fp) != 0 else 0

spec = specificity(cm)

# 7. Wyniki
print("\nMacierz pomyłek:")
print(cm)

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

print(f"Dokładność: {accuracy:.4f}")
print(f"Precyzja: {precision:.4f}")
print(f"Czułość (Recall): {recall:.4f}")
print(f"Specyficzność: {spec:.4f}")

# Zadanie 3
# 1. Wczytanie danych
digits = load_digits()
X = digits.data  # obrazy są już spłaszczone do 64 cech (8x8 pikseli)
y = digits.target

print("Kształt danych:", X.shape)

# 2. Eksploracja – przykładowe obrazy
plt.figure(figsize=(8, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Cyfra: {digits.target[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3. Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Standaryzacja cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model SVM
svm = SVC(kernel='rbf', gamma='scale')  # domyślny kernel RBF
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

# 6. Ocena klasyfikatora
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nDokładność: {accuracy:.4f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Wizualizacja macierzy pomyłek
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel("Przewidziane")
plt.ylabel("Rzeczywiste")
plt.title("Macierz pomyłek – SVM na zbiorze Digits")
plt.show()

# 8. Analiza błędów
errors = np.where(y_test != y_pred)[0]
print(f"\nLiczba błędnie sklasyfikowanych próbek: {len(errors)}")

# Pokaż kilka przykładów błędów
plt.figure(figsize=(10, 4))
for i in range(min(8, len(errors))):
    idx = errors[i]
    plt.subplot(2, 4, i + 1)
    plt.imshow(digits.images[idx], cmap='gray')
    plt.title(f"R: {y_test[idx]}, P: {y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Przykłady błędnie sklasyfikowanych cyfr")
plt.tight_layout()
plt.show()

# Zadanie 4
# 1. Wczytanie danych
try:
    df = pd.read_csv("Titanic-Dataset.csv")
except FileNotFoundError:
    print("Plik 'Titanic-Dataset.csv' nie został znaleziony. Pomiń zadanie 4.")
else:
    # 2. Eksploracyjna analiza danych
    print("Informacje o danych:")
    print(df.info())

    print("\nBraki danych:")
    print(df.isnull().sum())

    print("\nStatystyki opisowe:")
    print(df.describe(include='all'))

    # Wizualizacja: przeżywalność
    sns.countplot(x='Survived', data=df)
    plt.title("Przeżywalność pasażerów")
    plt.show()

    # Przeżywalność wg płci i klasy
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title("Przeżywalność wg płci")
    plt.show()

    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title("Przeżywalność wg klasy")
    plt.show()

    # 3. Przygotowanie danych
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])  # usuwamy mało przydatne lub trudne do obróbki kolumny

    # Uzupełnianie braków - poprawione aby uniknąć warningów
    df = df.assign(Age=df['Age'].fillna(df['Age'].median()))
    df = df.assign(Embarked=df['Embarked'].fillna(df['Embarked'].mode()[0]))

    # Kodowanie zmiennych kategorycznych
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # 4. Podział danych na cechy i etykiety
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Budowa modeli klasyfikacji

    # Regresja logistyczna
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)
    log_proba = log_model.predict_proba(X_test)[:,1]

    # Drzewo decyzyjne
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)
    tree_proba = tree_model.predict_proba(X_test)[:,1]

    # 6. Ocena modeli

    print("=== REGRESJA LOGISTYCZNA ===")
    print("Dokładność:", accuracy_score(y_test, log_pred))
    print("ROC-AUC:", roc_auc_score(y_test, log_proba))
    print("\nRaport klasyfikacji:\n", classification_report(y_test, log_pred, zero_division=0))

    print("\n=== DRZEWO DECYZYJNE ===")
    print("Dokładność:", accuracy_score(y_test, tree_pred))
    print("ROC-AUC:", roc_auc_score(y_test, tree_proba))
    print("\nRaport klasyfikacji:\n", classification_report(y_test, tree_pred, zero_division=0))

    # Krzywe ROC dla obu modeli
    fpr_log, tpr_log, _ = roc_curve(y_test, log_proba)
    fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_proba)

    plt.plot(fpr_log, tpr_log, label='Logistic Regression')
    plt.plot(fpr_tree, tpr_tree, label='Decision Tree')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend()
    plt.show()

    # 7. Interpretacja cech (dla regresji logistycznej)
    importance = pd.Series(log_model.coef_[0], index=X.columns)
    importance.sort_values(ascending=False).plot(kind='barh', title='Wpływ cech na przeżycie (Regresja logistyczna)')
    plt.tight_layout()
    plt.show()

    print("\nWpływ cech (regresja logistyczna):")
    print(importance.sort_values(ascending=False))

# Zadanie 5
# 1. Wczytaj dane
try:
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
except FileNotFoundError:
    print("Plik 'heart_failure_clinical_records_dataset.csv' nie został znaleziony. Pomiń zadanie 5.")
else:
    # 2. Eksploracyjna analiza danych
    print("Podstawowe informacje:")
    print(df.info())
    print("\nStatystyki opisowe:")
    print(df.describe())

    # Korelacja cech
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Macierz korelacji cech")
    plt.show()

    # Histogram zmiennych wg DEATH_EVENT
    features_to_plot = ['age', 'ejection_fraction', 'serum_creatinine', 'platelets']
    for feature in features_to_plot:
        plt.figure()
        sns.histplot(data=df, x=feature, hue='DEATH_EVENT', kde=True, bins=10)
        plt.title(f"{feature} a ryzyko śmierci")
        plt.show()

    # 3. Przygotowanie danych
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Klasyfikator SVM
    svm_model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    svm_model.fit(X_train, y_train)

    # 5. Ocena klasyfikatora
    y_pred = svm_model.predict(X_test)
    y_proba = svm_model.predict_proba(X_test)[:, 1]

    print("\n=== Raport klasyfikacji SVM ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    conf = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf.ravel()

    # Metryki: precyzja, czułość, specyficzność
    precision = tp / (tp + fp) if (tp + fp) else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    print(f"Precyzja: {precision:.2f}")
    print(f"Czułość (Recall): {sensitivity:.2f}")
    print(f"Specyficzność: {specificity:.2f}")
    print(f"Dokładność: {accuracy_score(y_test, y_pred):.2f}")

    # 6. Wpływ cech
    correlations = df.corr()['DEATH_EVENT'].sort_values(ascending=False)
    print("\nKorelacja cech z DEATH_EVENT:")
    print(correlations)

    correlations.drop('DEATH_EVENT').plot(kind='barh', title='Korelacja cech z ryzykiem zgonu')
    plt.tight_layout()
    plt.show()