import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

#Funkcja używana w każdymm zadaniu więc korzystam z def
def jakosc_modelu(y_test, y_test_pred, zdanie):
    # Teraz znając y które predykuje model z tymi prawdziwymi liczymy blędy modelu
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    # Wyświetlanie wyników "metryk"
    print(f"\nJakość liczona dla", zdanie)
    print(f"Średni błąd kwadratowy (MSE): {mse:.2f}")
    print(f"Współczynnik determinacji (R²): {r2:.2f}")
    return mse, r2

#Zadanie 3 z użyciem generatora danych
#Wczytuje plik csv wygenerowany przez poprawiony generator
df = pd.read_csv("appartments.csv")

#Tworzę zmienne x - cechy i y - "wynik"
X = df[['area', 'rooms', 'floor', 'year_of_construction']].to_numpy()
y = df[['price']].to_numpy()

#By przeprowadzić testy i sprawdzić działanie naszego modelu na danych rzeczywistych
#Dziele nasze dane na dane treningowe do których dopasujemy model i testowe na których je zweryfikuje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Tworze model i uczę go na danych treningowych
model = LinearRegression()
model.fit(X_train, y_train)

#Sprawdzam jak model spredykowałby wartość y_test znając x_test
y_test_pred = model.predict(X_test)

#Jakość MSE i R^2
jakosc_modelu(y_test, y_test_pred,"Zadania 3 z danymi z generatora")

#Wyświetlanie wyników w postaci grafu
plt.scatter(y_test, y_test_pred, color='blue', label="Rzeczywiste vs Przewidywane")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Idealna linia 1:1")
plt.xlabel("Rzeczywiste ceny mieszkań")
plt.ylabel("Przewidywane ceny mieszkań")
plt.title("Porównanie rzeczywistych i przewidywanych cen")
plt.legend()
plt.tight_layout()
plt.show()

#Zadanie 3 na rzeczywistych danych
#Wczytuje plik z zinternetu z nieprzygotowanymi danymi
df = pd.read_csv("apartments_pl_2024_06.csv")
#Po zapoznaniu z plikiem, tworzę df z cechami nas interesującymi
df = df[['squareMeters', 'rooms', 'floor', 'buildYear', 'price']]

#Sprawdzam ilość obserwacji i ilość brakujących danych
#print(df.isnull().sum())
#print(df.shape)

#Usuwam obserwacje z brakującymi danymi
df = df.dropna()

#Upewniam się czy wszystkie puste dane zniknęły
#print(df.isnull().sum())
#print(df.shape)

#Tworzę zmienne x - cechy i y - "wynik"
X = df[['squareMeters', 'rooms', 'floor', 'buildYear']].to_numpy()
y = df[['price']].to_numpy()

#By przeprowadzić testy i sprawdzić działanie naszego modelu na danych rzeczywistych
#Dziele nasze dane na dane treningowe do których dopasujemy model i testowe na których je zweryfikuje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Tworze model i uczę go na danych treningowych
model = LinearRegression()
model.fit(X_train, y_train)

#Sprawdzam jak model spredykowałby wartość y_test znając x_test
y_test_pred = model.predict(X_test)
print(f"Predykowana cena za mieszkanie o podanych wymiarach to: {np.round(model.predict([[90, 4, 3, 2000]])[0][0], 2)}zł")

#Jakość MSE i R^2
jakosc_modelu(y_test, y_test_pred, "Zadanie 3 z danymi rzeczywistymi")

#Wyświetlanie wyników w postaci grafu
plt.scatter(y_test, y_test_pred, color='blue', label="Rzeczywiste vs Przewidywane")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Idealna linia 1:1")
plt.xlabel("Rzeczywiste ceny mieszkań")
plt.ylabel("Przewidywane ceny mieszkań")
plt.title("Porównanie rzeczywistych i przewidywanych cen")
plt.legend()
plt.tight_layout()
plt.show()

#Zad 4
#Wczytuje dane udostępnione przez Pana
df = pd.read_csv("temperature_and_energy_consumption.csv")
#Wizualizuje sobie dane by się z nimi zapoznać
#print(df.head())

#Konwersja cechy "time_n" na date
df['time_n'] = pd.to_datetime(df['time_n'])
#Po przekonwertowaniu zamieniamy date na miesiące by stworzyć zmienną X jako ceche
df['month'] = df['time_n'].dt.month

#Ustalam zmienne X jako cecha i y jako wyniki dla dni
X = df[['month']].to_numpy()
y = df[['temperature']].to_numpy()

#Podział danych na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Transformacja cech do wielomianowych
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

#Wybranie modelu i trening
model_lin = LinearRegression()
model_lin.fit(X_train_poly, y_train)

#Zebranie danych spredykowanych przez model
y_test_pred = model_lin.predict(X_test_poly)

#Jakość MSE i R^2
jakosc_modelu(y_test, y_test_pred, "Zadanie 4 z modelem liniowym")

#Wizualizacja
plt.scatter(X, y, color='blue', label="Rzeczywiste dane")
X_sorted = np.sort(X, axis=0)
X_sorted_poly = poly.transform(X_sorted)
y_pred_sorted = model_lin.predict(X_sorted_poly)
plt.plot(X_sorted, y_pred_sorted, color='red', label="Regresja wielomianowa")
plt.xlabel("Miesiąc")
plt.ylabel("Temperatura (°C)")
plt.title("Regresja wielomianowa temperatur w czasie")
plt.show()

#Zad 5
#Lista wartości alpha do testowania
alphas = [0.01, 0.1, 1, 10, 100]

#Regresja liniowa
lin_model = LinearRegression()
lin_model.fit(X_train_poly, y_train)
y_test_pred_lin = lin_model.predict(X_test_poly)
jakosc_modelu(y_test, y_test_pred_lin, "Zadanie 5 model liniowy (Wielomianowy)")

#Testowanie przez wszystkie alphy modeli Ridge i Lasso i wizualizacja całości
for a in alphas:
    #Regresja Ridge
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_poly, y_train)
    y_pred_ridge = ridge.predict(X_test_poly)
    jakosc_modelu(y_test, y_pred_ridge, f"Ridge alpha={a}")

    #Regresja Lasso
    lasso = Lasso(alpha=a, max_iter=500000)
    lasso.fit(X_train_poly, y_train)
    y_pred_lasso = lasso.predict(X_test_poly)
    jakosc_modelu(y_test, y_pred_lasso, f"Lasso alpha={a}")

    #Wizualizacja wyników
    plt.figure(figsize=(12, 6))

    #Regresja liniowa
    X_full_poly = poly.transform(X)
    y_pred_full_lin = lin_model.predict(X_full_poly)
    plt.plot(X, y_pred_full_lin, color='red', label="Regresja liniowa")
    #Regresja Ridge
    y_pred_ridge_full = ridge.predict(X_full_poly)
    plt.plot(X, y_pred_ridge_full, color='green', linestyle='dashed', label=f"Regresja Ridge (alpha={a})")
    #Regresja Lasso
    y_pred_lasso_full = lasso.predict(X_full_poly)
    plt.plot(X, y_pred_lasso_full, color='blue', linestyle='dotted', label=f"Regresja Lasso (alpha={a})")
    #Punkty rzeczywiste
    plt.scatter(X, y, color='blue', alpha=0.5, label="Rzeczywiste dane")

    plt.xlabel("Dni od początku roku")
    plt.ylabel("Temperatura (°C)")
    plt.title(f"Porównanie regresji: Liniowa, Ridge i Lasso (alpha={a})")
    plt.legend()
    plt.show()

print("Wychodzi na to że najlepszy wynik jest dla alphy 0.01, a wielomanów 'degree' 3/4")
print("Jeśli chodzi o wybór modelu to dla niskiej wartości alphy wyniki mocno się nie różnią lecz gdybym miał wybrać model to zostałbym przy liniowym")

#Zad 6
#Wczytuje dane, które dostaliśmy od Pana
df = pd.read_csv("dane_medyczne.csv")
#Wyświetlam dane by się z nimi zapoznać
#print(df.head())

#Sprawdzam ilość brakujących wartości i je usuwam poniważ miałem jeskiś błąd i upewniam się że nie jest to powód brakujących danych
#print("\nBraki w danych:\n", df.isnull().sum())
#df = df.dropna()

#Definiowanie cech (wszystko oprócz czasu) i zmiennej y która jest "wynikiem"
X = df.drop(columns=["czas_przezycia"])
y = df["czas_przezycia"]

#Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degree = 2
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# #Lista parametrów do przetestowania różne C i kernele
# param_grid = {
#     "C": [0.1, 1, 10, 100],
#     "kernel": ["linear", "rbf", "poly"]
# }
#
# results = []
#
# #Testowanie każdego zestawu parametrów
# for C in param_grid["C"]:
#     for kernel in param_grid["kernel"]:
#         svr = SVR(C=C, kernel=kernel)
#         svr.fit(X_train, y_train)
#         y_pred = svr.predict(X_test)
#
#         # Obliczanie metryk
#         mae = mean_absolute_error(y_test, y_pred)
#         mse = mean_squared_error(y_test, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test, y_pred)
#
#         # Zapisanie wyników
#         results.append([C, kernel, mse, r2])
#
#         print(f"Przetestowano: C={C}, kernel={kernel} -> MSE={mse:.4f}, R²={r2:.4f}")

"""
WYNIKI ŁADOWAŁO SIĘ BARDZO DŁUGO WIĘC NIE BĘDĘ LICZYŁ JESZCZE RAZ TYLKO WYCIĄGNE WNIOSKI Z TEGO TESTU
Przetestowano: C=0.1, kernel=linear -> MSE=1273.7534, R²=-46.0845
Przetestowano: C=0.1, kernel=rbf -> MSE=27.9837, R²=-0.0344
Przetestowano: C=0.1, kernel=poly -> MSE=26.4119, R²=0.0237
Przetestowano: C=1, kernel=linear -> MSE=311003.9015, R²=-11495.3197
Przetestowano: C=1, kernel=rbf -> MSE=27.3495, R²=-0.0110
Przetestowano: C=1, kernel=poly -> MSE=14.9060, R²=0.4490
Przetestowano: C=10, kernel=linear -> MSE=51011446.7969, R²=-1885647.0524
Przetestowano: C=10, kernel=rbf -> MSE=14.0102, R²=0.4821
Przetestowano: C=10, kernel=poly -> MSE=9.8247, R²=0.6368
Przetestowano: C=100, kernel=linear -> MSE=3201931922.3290, R²=-118360035.2750
Przetestowano: C=100, kernel=rbf -> MSE=14.8501, R²=0.4511
Przetestowano: C=100, kernel=poly -> MSE=14.4664, R²=0.4652

Wychodzi z tego że najlepiej poradził sobie SVR C = 10 i kernel = poly
"""

#Teraz porównuje każdy model i patrze na wyniki
#Liniowy
model_lr = LinearRegression()
model_lr.fit(X_train_poly, y_train)
y_pred_lr = model_lr.predict(X_test_poly)

mse_lr, r2_lr = jakosc_modelu(y_test, y_pred_lr, "Regresja Liniowa")

#Ridge
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train_poly, y_train)
y_pred_ridge = model_ridge.predict(X_test_poly)

mse_ridge, r2_ridge = jakosc_modelu(y_test, y_pred_ridge, "Regresja Ridge")

#Lasso
model_lasso = Lasso(alpha=0.1, max_iter=50000)
model_lasso.fit(X_train_poly, y_train)
y_pred_lasso = model_lasso.predict(X_test_poly)

mse_lasso, r2_lasso = jakosc_modelu(y_test, y_pred_lasso, "Regresja Lasso")

#SVR dla naszych najlepszych parametrów dla testu
model_svr = SVR(C=10, kernel="poly")
model_svr.fit(X_train_poly, y_train)
y_pred_svr = model_svr.predict(X_test_poly)

mse_svr, r2_svr = jakosc_modelu(y_test, y_pred_svr, "SVR")

#Wykres dla SVR
plt.scatter(y_test, y_pred_svr, alpha=0.5, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="black")
plt.xlabel("Rzeczywiste wartości")
plt.ylabel("Przewidywane wartości")
plt.title("SVR (C=10, kernel=poly) - rzeczywiste vs przewidywane")
plt.show()

#Wykres porównawczy
#Zbiór wyników
models = ["Linear", "Ridge", "Lasso", "SVR"]
mse_values = [mse_lr, mse_ridge, mse_lasso, mse_svr]

#Wykres słupkowy MSE
plt.figure(figsize=(8, 5))
plt.bar(models, mse_values, color=["blue", "green", "red", "purple"])
plt.ylabel("MSE")
plt.title("Porównanie MSE dla różnych modeli regresji (wielomianowe cechy)")
plt.show()

#Wykres R^2
r2_values = [r2_lr, r2_ridge, r2_lasso, r2_svr]
plt.figure(figsize=(8, 5))
plt.bar(models, r2_values, color=["blue", "green", "red", "purple"])
plt.ylabel("R²")
plt.title("Porównanie R² dla różnych modeli regresji")
plt.show()

#Wychodzi na to że SVR poradził sobie dużo lepiej od poprzednich modeli