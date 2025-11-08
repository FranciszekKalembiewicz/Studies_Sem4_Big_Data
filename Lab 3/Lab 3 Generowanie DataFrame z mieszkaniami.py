import pandas as pd
import random


def generate_data(N):
    data = []
    for _ in range(N):
        area = random.randint(50, 120)

        # Skorelowanie liczby pokoi z powierzchnią
        if area < 60:
            rooms = random.randint(1, 2)
        elif area < 80:
            rooms = random.randint(2, 3)
        elif area < 100:
            rooms = random.randint(3, 4)
        else:
            rooms = random.randint(4, 5)

        floor = random.randint(1, 10)
        year_of_construction = random.randint(1950, 2022)

        # Wzór na cenę mieszkania
        base_price_per_m2 = 3000  # Bazowa cena za m2
        age_factor = max(0.5, (2025 - year_of_construction) / 100)  # Starsze budynki są tańsze
        floor_factor = 1.1 if floor > 1 else 1.0  # Piętro 1 i wyżej podnosi cenę
        room_factor = 0.9 + (rooms * 0.1)  # Więcej pokoi podnosi cenę

        price = int(area * base_price_per_m2 * age_factor * floor_factor * room_factor)

        # Dodanie większej losowej wariancji +/- 10%-20%
        price = int(price * random.uniform(0.8, 1.2))

        data.append([area, rooms, floor, year_of_construction, price])

    df = pd.DataFrame(data, columns=['area', 'rooms', 'floor', 'year_of_construction', 'price'])
    df.to_csv('appartments.csv', index=False)
    print(f"Plik 'appartments.csv' został wygenerowany z {N} wierszami danych.")


generate_data(10000)


