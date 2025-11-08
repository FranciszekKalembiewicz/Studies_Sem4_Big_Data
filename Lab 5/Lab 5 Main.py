import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose

#Zadanie 2
#Wczytuje dane od Pana
df = pd.read_csv("DJIA_ClosingValues_1896-10-07-2025-03-28.csv")

#Sprawdzam pierwsze 5 linijek kodu
print("Pierwsze wiersze danych:")
print(df.head())

#Sprawdzam typy danych
print("\nTypy danych:")
print(df.dtypes)

#Zamianiam 'Date' na typ datetime, potem ustawiam ją jako indeks
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
df.set_index('Date', inplace=True)

#Sprawdzam czy brakuje jakiś danych jak tak to je usuwam, w Pana danych jest to pewnie nie potrzebne ale robie to gdybym chciał użyć własnych danych
print("\nLiczba brakujących wartości:")
print(df.isnull().sum())
df.dropna(inplace=True)

#Statystyki opisowe średnia odchylenie itd.
print("\nStatystyki opisowe:")
print(df['DJIA'].describe())

#Wykres szeregu czasowego
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['DJIA'], label="DJIA Closing Value", color='navy')
plt.title("Zadanie 1 Szereg czasowy DJIA Closing Value")
plt.xlabel("Date")
plt.ylabel("Closing Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Zadanie 3
#Statystyki opisowe
mean_val = df['DJIA'].mean()
std_val = df['DJIA'].std()
quantiles = df['DJIA'].quantile([0.25, 0.5, 0.75])

print("\n--- Statystyki opisowe ---")
print(f"Średnia: {mean_val:.2f}")
print(f"Odchylenie standardowe: {std_val:.2f}")
print("Kwantyle (25%, 50%, 75%):")
print(quantiles)

"""
--- Statystyki opisowe ---
Średnia: 3991.62
Odchylenie standardowe: 7970.76
Kwantyle (25%, 50%, 75%):
0.25     100.6800
0.50     454.9200
0.75    2730.7875
Name: DJIA, dtype: float64
"""
#Wyniki mówią o zmienności funkcji w czasie
#na początku małe wartości 100 potem zaczyna już nieregularnie rosnąć aż osiąga bardzo duże wyniki 2730

#Teraz próbuje innej metody ekstrakcji danych. Próbuje sprawdzić metode różnicy dziennej czyli jak zmieniały się dane kaźdego dnia
df['diff_1'] = df['DJIA'].diff()

#Wykres różnic
plt.figure(figsize=(14, 4))
plt.plot(df.index, df['diff_1'], color='orange')
plt.title("Pierwsza różnica (zmiana wartości DJIA)")
plt.xlabel("Data")
plt.ylabel("Zmiana")
plt.grid(True)
plt.tight_layout()
plt.show()

#Teraz patrze na średnią z każdego miesiąca jako sposób uzyskania informacji z danych
monthly = df['DJIA'].resample('ME').mean().dropna()

#Dekompozycja
decomp = seasonal_decompose(monthly, model='multiplicative', period=12)

#Wykres dekompozycji
decomp.plot()
plt.suptitle("Dekompozycja szeregu czasowego DJIA (miesięczne dane)", fontsize=14)
plt.tight_layout()
plt.show()

#Grupowanie:
#Dodanie kolumny z numerem miesiąca
df['Month_num'] = df.index.month

#Grupowanie po miesiącu: agregacja niezależnie od roku
monthly_stats = df.groupby('Month_num')['DJIA'].agg(['mean', 'std', 'min', 'max'])

#Wydrukowanie statystyk miesięcznych
print("\n--- Statystyki miesięczne (średnia bez informacji o latach) ---")
print(monthly_stats)

#Wykres sezonowości (średnia wartość DJIA w kolejnych miesiącach)
plt.figure(figsize=(10, 4))
plt.plot(monthly_stats.index, monthly_stats['mean'], marker='o', color='green', label='Średnia miesięczna')
plt.title("Średnia miesięczna DJIA (wszystkie lata)")
plt.xlabel("Miesiąc")
plt.ylabel("Średnia wartość")
plt.xticks(ticks=range(1, 13), labels=['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze', 'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru'])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""
1.Statystyki opisowe - dobre do ogólnej charakterystyki danych, ale nie pokazują zmian w czasie.
2.Różnicowanie - skuteczne w analizie zmienności i wykrywaniu gwałtownych zmian.
3.Dekompozycja - bardzo przydatna do wykrywania trendu i sezonowości (szczególnie w danych miesięcznych lub kwartalnych).
4.Grupowanie - zrobiłem też grupowanie lecz w naszym przypadku grupowanie po miesiącu nie patrząc na rok nie jest dobre, ponieważ
uważam że dane nic nam nie mówią. Dostajemy losowe informacje które mogą nas zmylić lecz też używam tej metody by pokazać że nie wszystko jest dobre
"""

#Zadanie 4
#Wczytaj dane pomijając pierwsze dwa wiersze, bo nie ma tam danych liczbowych tylko informacje
df = pd.read_csv("Hang_Seng_data_2025-03-31.csv", skiprows=2)

#Nadaje odpowiednie nazwy kolumnom, które usunąłem wyżej
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

#Konwertuje kolumnę Date na datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

#Ustawiam Date jako indeks
df.set_index('Date', inplace=True)

#Tworzę zmienną signal jako daną, którą będę analizował
signal = df['Close'].values

#Upewniam się, że nie ma danych brakujących
df.dropna(inplace=True)

"""
#Krótkie podsumowanie danych
print(df.info())
print(df.head())
"""

#Wykres ceny zamknięcia
plt.figure(figsize=(12, 5))
plt.plot(df.index, signal, label='HSI Close', color='darkgreen')
plt.title("Hang Seng Index - Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Metoda 1 Transformata Fouriera
start_fft = time.perf_counter()
fft_result = fft(signal)
fft_magnitude = np.abs(fft_result)
frequencies = fftfreq(len(signal), d=1)
end_fft = time.perf_counter()
fft_time = end_fft - start_fft

#Wykres widma częstotliwości
plt.figure(figsize=(12, 4))
plt.plot(frequencies[:len(frequencies)//2], fft_magnitude[:len(frequencies)//2], color='darkblue')
plt.title("Transformacja Fouriera - Widmo częstotliwości")
plt.xlabel("Częstotliwość")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()

#Metoda 2 Autokorelacja
def autocorrelation(x, lag):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

start_acf = time.perf_counter()
lags = 100
acf_vals = [autocorrelation(signal, lag) for lag in range(1, lags)]
end_acf = time.perf_counter()
acf_time = end_acf - start_acf

#Wykres autokorelacji
plt.figure(figsize=(12, 4))
plt.bar(range(1, lags), acf_vals, width=0.8, color='darkorange')
plt.title("Autokorelacja sygnału - Lags 1 do 100")
plt.xlabel("Lag (opóźnienie)")
plt.ylabel("Wartość autokorelacji")
plt.grid(True)
plt.tight_layout()
plt.show()

#Metoda 1 Pandas: Średnia krocząca
start_roll = time.perf_counter()
rolling_mean = df['Close'].rolling(window=30).mean()
end_roll = time.perf_counter()
roll_time = end_roll - start_roll

#Wykres średniej kroczącej
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Close'], label='Cena zamknięcia', alpha=0.5)
plt.plot(df.index, rolling_mean, label='Średnia krocząca (30)', color='red')
plt.title("Średnia krocząca (30 dni) na HSI")
plt.xlabel("Data")
plt.ylabel("Cena")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Metoda 2 Pandas: Różnicowanie
start_diff = time.perf_counter()
diff_series = df['Close'].diff()
end_diff = time.perf_counter()
diff_time = end_diff - start_diff

#Wykres różnicowania
plt.figure(figsize=(12, 4))
plt.plot(df.index, diff_series, label='Zmiana dzienna', color='purple')
plt.title("Różnicowanie szeregów czasowych (diff) na HSI")
plt.xlabel("Data")
plt.ylabel("Różnica ceny")
plt.grid(True)
plt.tight_layout()
plt.show()

#Podsumowanie
print("\n--- Porównanie czasu wykonania (w sekundach) ---")
print(f"Transformacja Fouriera (NumPy):     {fft_time:.6f} s")
print(f"Autokorelacja (NumPy):              {acf_time:.6f} s")
print(f"Rolling Mean (Pandas):              {roll_time:.6f} s")
print(f"Różnicowanie .diff() (Pandas):      {diff_time:.6f} s")