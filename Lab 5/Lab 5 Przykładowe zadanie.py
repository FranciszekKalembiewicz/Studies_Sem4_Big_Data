import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Przykładowy szereg czasowy w formacie daty i czasu
dates = pd.date_range(start='2023-01-01', periods=100)
ts = pd.Series(np.random.randn(100), index=dates)
# Ekstrakcja cech czasowych
seasonality = ts.resample('M').mean() # Średnia wartość miesięczna
trend = ts.rolling(window=10).mean() # Średnia ruchoma
sequence_length = len(ts) # Długość sekwencji
differences = ts.diff() # Różnice między kolejnymi wartościami
variability = ts.rolling(window=10).std() # Odchylenie standardowe dla średniej ruchomej
# Wyświetlenie wyników
print("Średnia wartość miesięczna:")
print(seasonality)
print("\nŚrednia ruchoma:")
print(trend)
print("\nDługość sekwencji:", sequence_length)
print("\nRóżnice między kolejnymi wartościami:")
print(differences)
print("\nOdchylenie standardowe dla średniej ruchomej:")
print(variability)
# Rysowanie wyników
plt.figure(figsize=(12, 8))
# Wykres oryginalnego szeregu czasowego
plt.subplot(3, 2, 1)
plt.plot(ts)
plt.title('Oryginalny szereg czasowy')
plt.xlabel('Data')
plt.ylabel('Wartość')
# Wykres średniej wartości miesięcznej (sezonowość)
plt.subplot(3, 2, 2)
plt.plot(seasonality)
plt.title('Średnia wartość miesięczna')
plt.xlabel('Data')
plt.ylabel('Średnia wartość')
# Wykres średniej ruchomej (trend)
plt.subplot(3, 2, 3)
plt.plot(trend)
plt.title('Średnia ruchoma')
plt.xlabel('Data')
plt.ylabel('Średnia wartość')
# Wykres różnic między kolejnymi wartościami
plt.subplot(3, 2, 4)
plt.plot(differences)
plt.title('Różnice między kolejnymi wartościami')
plt.xlabel('Data')
plt.ylabel('Różnica')
# Wykres odchylenia standardowego dla średniej ruchomej (zmienność w czasie)
plt.subplot(3, 2, 5)
plt.plot(variability)
plt.title('Odchylenie standardowe dla średniej ruchomej')
plt.xlabel('Data')
plt.ylabel('Odchylenie standardowe')
plt.tight_layout()
plt.show()