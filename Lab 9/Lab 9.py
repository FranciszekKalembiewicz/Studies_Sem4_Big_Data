from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Wczytanie danych Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(f"Liczba obrazów treningowych: {x_train.shape[0]}")
print(f"Liczba obrazów testowych: {x_test.shape[0]}")
print(f"Rozmiar pojedynczego obrazu: {x_train.shape[1]}x{x_train.shape[2]}")

# Normalizacja wartości pikseli do zakresu od 0 do 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Rozszerzenie wymiarów dla warstw konwolucyjnych (28, 28) -> (28, 28, 1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# Przekształcenie etykiet kategorii na postać one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Budowa modelu sieci neuronowej
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

# Kompilacja modelu
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Trenowanie modelu z danymi walidacyjnymi
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test)
)

# Ewaluacja modelu na zbiorze testowym
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Przewidywanie na nowych danych (zbiór testowy)
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Macierz pomyłek
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Macierz pomyłek')
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywista')
plt.show()

# Raport klasyfikacji
print("Raport klasyfikacji:")
print(classification_report(y_true, y_pred))

wnioski = """
Wnioski dotyczące wydajności modelu

Model osiągnął wysoką dokładność. Z macierzy pomyłek wynika jednak, że skuteczność rozpoznawania poszczególnych klas różni się:

- Najlepsza skuteczność została osiągnięta dla klas dobrze od siebie odróżnialnych, np. „Sneaker” czy „Ankle boot”, gdzie przez charakterystyczne cechy ubrania łatwiej na obrazkach rozróżnić.
- Niższa skuteczność obserwowana jest dla klas o podobnych wzorach, np. „Shirt”, „Pullover” i „Coat”. Klasy te są często często mylone przez model zapewne przez podobny krztałt i brak cech mocno wyróżniających się na fotografiach.

Podsumowanie
Model radzi sobie dobrze, lecz niektóre typy ubrań są dla niego trudniejsze do rozpoznania. 
"""
print(wnioski)