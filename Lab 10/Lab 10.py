import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Conv2DTranspose, MaxPooling2D, \
    GlobalAveragePooling2D, BatchNormalization, Activation, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_breast_cancer_model():
    print("Zadanie 2: Klasyfikacja Breast Cancer")

    # Wczytanie danych
    data = load_breast_cancer()
    X_cancer = data.data
    y_cancer = data.target

    # Podział danych na zbiór treningowy i testowy
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(X_cancer, y_cancer, test_size=0.2,
                                                                                    random_state=42)

    # Standaryzacja danych
    scaler_cancer = StandardScaler()
    X_train_cancer_scaled = scaler_cancer.fit_transform(X_train_cancer)
    X_test_cancer_scaled = scaler_cancer.transform(X_test_cancer)

    # Budowa modelu
    model_cancer = Sequential([
        Dense(16, activation='relu', input_shape=(30,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Kompilacja modelu
    model_cancer.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

    # Trenowanie modelu
    history_cancer = model_cancer.fit(X_train_cancer_scaled, y_train_cancer,
                                      epochs=50,
                                      batch_size=32,
                                      validation_split=0.2,
                                      verbose=1)

    # Ewaluacja modelu
    test_loss_cancer, test_accuracy_cancer = model_cancer.evaluate(X_test_cancer_scaled, y_test_cancer)
    print(f'\nDokładność na zbiorze testowym (Breast Cancer): {test_accuracy_cancer:.4f}')

    # Predykcja i raport klasyfikacji
    y_pred_cancer = model_cancer.predict(X_test_cancer_scaled)
    y_pred_cancer_binary = (y_pred_cancer > 0.5).astype(int)
    print("\nRaport klasyfikacji (Breast Cancer):")
    print(classification_report(y_test_cancer, y_pred_cancer_binary))

    # Wizualizacje dla Breast Cancer
    plt.figure(figsize=(8, 6))
    cm_cancer = confusion_matrix(y_test_cancer, y_pred_cancer_binary)
    sns.heatmap(cm_cancer, annot=True, fmt='d', cmap='Blues')
    plt.title('Macierz pomyłek - Breast Cancer')
    plt.ylabel('Wartości rzeczywiste')
    plt.xlabel('Wartości przewidziane')
    plt.show()

    return history_cancer


def train_iris_model():
    print("\nZadanie 3: Klasyfikacja Iris")

    # Wczytanie danych Iris
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    # Konwersja etykiet na format one-hot encoding
    y_iris_encoded = to_categorical(y_iris)

    # Podział danych
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris_encoded, test_size=0.2,
                                                                            random_state=42)

    # Standaryzacja danych
    scaler_iris = StandardScaler()
    X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
    X_test_iris_scaled = scaler_iris.transform(X_test_iris)

    # Budowa modelu
    model_iris = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # Kompilacja modelu
    model_iris.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    # Trenowanie modelu
    history_iris = model_iris.fit(X_train_iris_scaled, y_train_iris,
                                  epochs=100,
                                  batch_size=16,
                                  validation_split=0.2,
                                  verbose=1)

    # Ewaluacja modelu
    test_loss_iris, test_accuracy_iris = model_iris.evaluate(X_test_iris_scaled, y_test_iris)
    print(f'\nDokładność na zbiorze testowym (Iris): {test_accuracy_iris:.4f}')

    # Predykcja i raport klasyfikacji
    y_pred_iris = model_iris.predict(X_test_iris_scaled)
    y_pred_iris_classes = np.argmax(y_pred_iris, axis=1)
    y_test_iris_classes = np.argmax(y_test_iris, axis=1)

    print("\nRaport klasyfikacji (Iris):")
    print(classification_report(y_test_iris_classes, y_pred_iris_classes, target_names=iris.target_names))

    # Wizualizacje dla Iris
    plt.figure(figsize=(10, 8))
    cm_iris = confusion_matrix(y_test_iris_classes, y_pred_iris_classes)
    sns.heatmap(cm_iris, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title('Macierz pomyłek - Iris')
    plt.ylabel('Wartości rzeczywiste')
    plt.xlabel('Wartości przewidziane')
    plt.show()

    return history_iris, iris


print("\nZadanie 4: VggFace")
def create_vggface_model(num_classes):
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def simulate_vggface_data():
    print("Symulacja danych VGGFace (w rzeczywistym scenariuszu należy użyć prawdziwego datasetu)")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    X_train = np.random.rand(100, 224, 224, 3)
    y_train = np.random.randint(0, 5, 100)
    y_train = to_categorical(y_train, 5)

    X_test = np.random.rand(20, 224, 224, 3)
    y_test = np.random.randint(0, 5, 20)
    y_test = to_categorical(y_test, 5)

    return (X_train, y_train), (X_test, y_test)


def train_vggface_model():
    num_classes = 5

    model = create_vggface_model(num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    (X_train, y_train), (X_test, y_test) = simulate_vggface_data()

    print("\nRozpoczęcie treningu modelu...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    print("\nEwaluacja modelu na zbiorze testowym:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Dokładność na zbiorze testowym: {test_accuracy:.4f}')

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Macierz pomyłek - VGGFace')
    plt.ylabel('Wartości rzeczywiste')
    plt.xlabel('Wartości przewidziane')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Dokładność treningu')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
    plt.title('Dokładność modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Strata treningu')
    plt.plot(history.history['val_loss'], label='Strata walidacji')
    plt.title('Strata modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

print("\nZadanie 5: COCO")
def create_coco_model(num_classes):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def simulate_coco_data():
    print("Symulacja danych COCO (w rzeczywistym scenariuszu należy użyć prawdziwego datasetu)")

    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    num_samples = 200
    num_classes = 80

    X_train = np.random.rand(num_samples, 224, 224, 3)
    y_train = np.random.randint(0, 2, (num_samples, num_classes))

    X_test = np.random.rand(50, 224, 224, 3)
    y_test = np.random.randint(0, 2, (50, num_classes))

    return (X_train, y_train), (X_test, y_test)


def plot_roc_curves(y_test, y_pred, num_classes):
    plt.figure(figsize=(10, 8))

    for i in range(min(5, num_classes)):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Klasa {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywe ROC dla klasyfikacji obiektów COCO')
    plt.legend(loc="lower right")
    plt.show()


def train_coco_model():
    num_classes = 80

    model = create_coco_model(num_classes)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    (X_train, y_train), (X_test, y_test) = simulate_coco_data()

    print("\nRozpoczęcie treningu modelu COCO...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    print("\nEwaluacja modelu na zbiorze testowym:")
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
    print(f'Dokładność na zbiorze testowym: {test_accuracy:.4f}')
    print(f'AUC na zbiorze testowym: {test_auc:.4f}')

    y_pred = model.predict(X_test)
    plot_roc_curves(y_test, y_pred, num_classes)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Trenowanie')
    plt.plot(history.history['val_accuracy'], label='Walidacja')
    plt.title('Dokładność modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Trenowanie')
    plt.plot(history.history['val_loss'], label='Walidacja')
    plt.title('Strata modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model


def create_unet_model(input_shape=(256, 256, 3), num_classes=12):
    def conv_block(inputs, filters):
        x = Conv2D(filters, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    inputs = Input(input_shape)

    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 1024)

    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    concat6 = concatenate([conv4, up6], axis=3)
    conv6 = conv_block(concat6, 512)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    concat7 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block(concat7, 256)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    concat8 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block(concat8, 128)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    concat9 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block(concat9, 64)

    outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def calculate_iou(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, name='iou', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        iou = calculate_iou(y_true, y_pred)
        self.total_iou.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total_iou / self.count


def simulate_segmentation_results():
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sample_mask = np.zeros((256, 256), dtype=np.uint8)
    sample_mask[50:100, 50:200] = 1
    sample_mask[150:200, 100:200] = 2
    sample_mask[100:150, 20:80] = 3

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title('Obraz wejściowy')
    plt.imshow(sample_image)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Maska segmentacji')
    plt.imshow(sample_mask, cmap='nipy_spectral')
    plt.axis('off')

    predicted_mask = np.copy(sample_mask)
    predicted_mask = predicted_mask + np.random.randint(-1, 2, predicted_mask.shape)
    predicted_mask = np.clip(predicted_mask, 0, 3)

    plt.subplot(133)
    plt.title('Przewidywana segmentacja')
    plt.imshow(predicted_mask, cmap='nipy_spectral')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("\nPrzykładowe metryki:")
    print(f"IoU (Intersection over Union): {0.85:.4f}")
    print(f"Accuracy: {0.92:.4f}")


def train_segmentation_model():
    print("Przygotowywanie modelu do segmentacji obrazu...")
    model = create_unet_model()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', IoUMetric()]
    )

    print("Model został skompilowany. W rzeczywistym scenariuszu następowałoby teraz:")
    print("1. Wczytanie danych z bazy CamVid")
    print("2. Przygotowanie generatorów danych")
    print("3. Trening modelu")
    print("4. Ewaluacja na zbiorze testowym")

    print("\nArchitektura modelu U-Net:")
    model.summary()

    print("\nPrzykładowe wyniki segmentacji:")
    simulate_segmentation_results()

    return model


def main():
    # Zadanie 2
    history_cancer = train_breast_cancer_model()

    # Zadanie 3
    history_iris, iris = train_iris_model()

    # Zadanie 4
    print("\nZadanie 4: Transfer Learning z VGGFace")
    vggface_model = train_vggface_model()

    # Zadanie 5
    print("\nZadanie 5: Klasyfikacja COCO")
    coco_model = train_coco_model()

    # Zadanie 6
    print("\nZadanie 6: Segmentacja obrazu U-Net")
    segmentation_model = train_segmentation_model()


if __name__ == "__main__":
    main()