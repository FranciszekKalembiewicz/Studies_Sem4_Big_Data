import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import argrelextrema
from scipy.interpolate import PchipInterpolator, interp1d, CubicSpline
from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator
from sklearn.metrics import mean_squared_error


#Wykorztuje podany przez pana generator do danych
"""
def generate_weather_data(num_stations, num_days):
    Funkcja generuje przykładowe dane meteorologiczne dla wielu stacji pomiarowych i dni
    i zapisuje je do pliku CSV.

    Parametry:
    - num_stations: liczba stacji pomiarowych
    - num_days: liczba dni pomiarowych

    Zwraca:
    - None    # Temperatury miesięczne dla stacji 1 (średnie wartości miesięczne)
    temperatures1 = np.array([-2, 0, 5, 12, 18, 23, 26, 25, 21, 15, 8, 2])

    # Generowanie dat
    np.random.seed(0)
    dates = pd.date_range(start='2023-01-01', periods=num_days)

    # Lista nazw stacji
    station_ids = ['Station_' + str(i) for i in range(1, num_stations + 1)]

    # Inicjalizacja pustych list na dane
    data = {station: [] for station in station_ids}

    # Generowanie danych dla każdego dnia
    for day in range(num_days):
        month = dates[day].month - 1  # miesiące indeksowane od 0
        base_temperature = temperatures1[month]

        for station in station_ids:
            if station == 'Station_1':
                temperature = base_temperature + np.random.uniform(low=-2, high=2)
            else:
                temperature = base_temperature + np.random.uniform(low=-4, high=4)

            # Dodanie rzadkiego skoku temperatury
            if day > 0 and np.random.rand() < 0.05:
                temperature += np.random.uniform(low=-10, high=10)

            data[station].append(temperature)

    # Tworzenie DataFrame
    df = pd.DataFrame(data)
    df['Date'] = dates
    df = df[['Date'] + station_ids]

    # Zapis do pliku CSV
    df.to_csv('weather_data.csv', index=False)
    print("Dane zapisane do pliku 'weather_data.csv'.")

# Przykład użycia:
generate_weather_data(num_stations=5, num_days=365)
"""

#Zadanie 2
#Wczytuj dane stworzone w zadaniu 1
df = pd.read_csv("weather_data.csv")
#Zmieniam kolumne Date na datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

#Usuwam brakujące dane
df = df.dropna()

#Podstawowa analiza
#Informacje o danych
print("\n>>> Informacje o danych:")
print(df.info())
#Statystyki opisowe
print("\n>>> Statystyki opisowe:")
print(df.describe())

# Zadanie 3: Interpolacja B-sklejana – wykresy oryginalnych i interpolowanych danych
stations = ['Station_1', 'Station_2', 'Station_3', 'Station_4', 'Station_5']
colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
plt.figure(figsize=(15, 10))

for i in range(len(stations)):
    station = stations[i]
    y = df[station]
    x = np.arange(len(y))

    # Interpolacja B-spline
    spline = make_interp_spline(x, y, k=3)
    y_interp = spline(x)

    # Wykres: oryginalne dane i interpolacja na jednym wykresie
    plt.subplot(3, 2, i + 1)
    plt.scatter(df.index, y, label='Oryginalne dane', color='blue', alpha=0.5)
    plt.plot(df.index, y_interp, label='Po interpolacji (B-spline)', color=colors[i], linewidth=2)
    plt.title(f"Interpolacja temperatury – {station}")
    plt.xlabel("Data")
    plt.ylabel("Temperatura [°C]")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

#Zadanie 4
#Wczytuj dane znalezione w internecie
df = pd.read_csv("AEP_hourly.csv")

#Sprawdzam dane
#print(df.head())

#Konwersja kolumny 'Datetime' do typu datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

#Usunięcie brakujących danych
df = df.dropna()

#Zmiana częstotliwości na dzienną (średnia z każdej doby)
df_daily = df.resample('D').mean()

#Podstawowa analiza
print("\n>>> Informacje o danych dziennych:")
print(df_daily.info())
print("\n>>> Statystyki opisowe:")
print(df_daily.describe())

#By lepiej zaprezentować działanie interpolacji wielomianowej robię to na mniejszej próbce bo na większej nie działa to poprawnie
sample = df_daily.iloc[:30]
x = np.arange(len(sample))
y = sample['AEP_MW'].values

#Interpolacja wielomianowa
degree = 8
coeffs = np.polyfit(x, y, deg=degree)
poly = np.poly1d(coeffs)

#Punktowa estymacja na większej liczbie punktów (do wykresu)
x_interp = np.linspace(x.min(), x.max(), 300)
y_interp = poly(x_interp)

#Wykres
plt.figure(figsize=(12, 6))
plt.plot(x_interp, y_interp, label=f'Interpolacja wielomianowa (stopień {degree})', color='blue')
plt.scatter(x, y, color='red', label='Oryginalne dane (średnie dzienne)', zorder=3)
plt.title("Interpolacja wielomianowa zużycia energii (fragment 30 dni)")
plt.xlabel("Dni (od 1. dnia w zbiorze)")
plt.ylabel("Moc [MW]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""
Mimo skorzystania z mniejszej próbki interpolacja wielomianowa w naszym przypadku i tak sobie nie poradziła
"""

#Zadanie 5
#Wczytuj dane uzykane od Pana
df = pd.read_csv("stocks_data.csv")

#Sprawdzam dane
print(df.head())

#Konwersja kolumny 'Datetime' do typu datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

#Usunięcie brakujących danych
df = df.dropna()

#Zmniejszamy plik by prezentacja metody była widoczniejsza
df_sample = df[-500:]

#Zadanie robie na całym pliku i na jego próbce
Data = [df, df_sample]

for df in Data:
    #Podstawowa analiza
    print("\n>>> Informacje o danych dziennych:")
    print(df_daily.info())
    print("\n>>> Statystyki opisowe:")
    print(df_daily.describe())


    #Przykładowa spółka
    stock = 'AAPL'
    x = df.index.values.astype('datetime64[D]').astype('float64')
    y = df[stock].values

    #Interpolator Hermite’a
    interp = PchipInterpolator(x, y)

    #Generowanie gęstszych punktów do analizy trendu
    x_dense = np.linspace(x.min(), x.max(), 5000)
    y_dense = interp(x_dense)

    #Konwersja z powrotem na daty
    x_dense_dates = pd.to_datetime(x_dense, origin='1970-01-01', unit='D')

    #Obliczenie pochodnej do identyfikacji ekstremów
    dy = np.gradient(y_dense)
    d2y = np.gradient(dy)

    #Lokalne ekstremum: miejsca, gdzie pochodna = 0 i druga pochodna ≠ 0
    extrema_indices = np.where(np.diff(np.sign(dy)))[0]
    maxima = [i for i in extrema_indices if d2y[i] < 0]
    minima = [i for i in extrema_indices if d2y[i] > 0]

    #Wizualizacja
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df[stock], label="Oryginalne dane", alpha=0.5)
    plt.plot(x_dense_dates, y_dense, label="Interpolacja Hermite’a", linewidth=2)

    #Dodanie ekstremów
    plt.scatter(x_dense_dates[maxima], y_dense[maxima], color='red', label='Maksima', zorder=5)
    plt.scatter(x_dense_dates[minima], y_dense[minima], color='green', label='Minima', zorder=5)

    plt.title(f"Analiza trendu: {stock}")
    plt.xlabel("Data")
    plt.ylabel("Cena")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Zadanie 6
#Wczytanie danych
df = pd.read_csv("road_traffic.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df = df.dropna()

#Agregacja danych godzinowych do dziennych
daily_data = df[['LGV']].resample('D').mean().dropna()

#Wykluczenie ostatnich 7 dni
train_data = daily_data.iloc[:-7]  # Wszystkie dane z wyjątkiem ostatnich 7 dni
test_data = daily_data.iloc[-7:]   # Ostatnie 7 dni jako dane testowe

#Przygotowanie danych do interpolacji (wszystkie dni treningowe)
x_train = train_data.index.values.astype('datetime64[D]').astype('float64')
y_train = train_data['LGV'].values

#Przygotowanie danych testowych (ostatnie 7 dni)
x_test = test_data.index.values.astype('datetime64[D]').astype('float64')
y_test = test_data['LGV'].values

#Dodanie 5 dni w przyszłość (prognoza)
future_days = 5
x_future = np.linspace(x_train.min(), x_train.max() + future_days, 500)

#Konwersja do dat
x_future_dates = pd.to_datetime(x_future, origin='1970-01-01', unit='D')

#Interpolatory
interpolators = {
    'Hermite (PCHIP)': PchipInterpolator(x_train, y_train),
    'Cubic Spline': CubicSpline(x_train, y_train, extrapolate=True),
    'Akima': Akima1DInterpolator(x_train, y_train)
}

#Wizualizacja prognoz interpolowanych
plt.figure(figsize=(16, 10))

for i, (name, interpolator) in enumerate(interpolators.items(), 1):
    y_future = interpolator(x_future)

    #Oblicz MSE dla danych testowych
    y_pred = interpolator(x_test)

    #Sprawdzanie NaN i usuwanie ich z y_test i y_pred
    valid_idx = ~np.isnan(y_test) & ~np.isnan(y_pred)

    if valid_idx.sum() > 0:  #Sprawdzamy, czy są jakiekolwiek elementy po usunięciu NaN
        mse = mean_squared_error(y_test[valid_idx], y_pred[valid_idx])
    else:
        mse = np.nan  #Jeśli brak danych, ustawiamy MSE na NaN

    # Wypisywanie prognozowanych wartości
    print(f"Prognozy dla metody {name}:")
    for date, pred in zip(x_test, y_pred):
        print(f"Data: {pd.to_datetime(date, unit='D')}, Prognoza: {pred:.4f}")

    # Wypisywanie MSE dla tej metody
    print(f"{name} – MSE: {mse:.6f}\n")

    plt.subplot(3, 1, i)
    plt.plot(daily_data.index, daily_data['LGV'], label='Rzeczywiste dane')
    plt.plot(x_future_dates, y_future, label='Prognoza ' + name, linewidth=2)
    plt.axvline(daily_data.index[-7], color='gray', linestyle='--', label='Granica danych')
    plt.title(f"{name} – MSE: {mse:.6f}")
    plt.xlabel("Data")
    plt.ylabel("Liczba LGV")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()