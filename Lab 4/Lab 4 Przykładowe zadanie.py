import pandas as pd
import numpy as np
import random

def generate_weather_data(num_stations, num_days):
    """
    Funkcja generuje przykładowe dane meteorologiczne dla wielu stacji pomiarowych i dni
    i zapisuje je do pliku CSV.

    Parametry:
    - num_stations: liczba stacji pomiarowych
    - num_days: liczba dni pomiarowych

    Zwraca:
    - None
    """
    # Temperatury miesięczne dla stacji 1 (średnie wartości miesięczne)
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
