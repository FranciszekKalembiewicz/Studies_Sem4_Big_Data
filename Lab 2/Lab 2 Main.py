import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#NumPy
#Zad 1
tab = np.array([i for i in range(1,11)])
print(tab)
print(np.max(tab))
print(np.min(tab))
print(np.mean(tab))
print(np.std(tab))

#Zad 2
tab = np.array([i for i in range(12)])
tab = tab.reshape(3, 4)
print(tab)
print(tab[0][1])
sliced_tab = tab[:2, -2:]
print(sliced_tab)

#Zad 3
tab = np.array([i for i in range(10)])
print(tab)
tab = tab.reshape(2, 5)
print(tab)
tab = tab.T
print(tab)
print(tab.shape)

#Zad 4
tab1 = np.array([i for i in range(0, 16)])
tab1 = tab1.reshape(4, 4)
tab2 = np.array([i for i in range(16, 32)])
tab2 = tab2.reshape(4, 4)
print(tab1 + tab2)
print(tab1 * 3)

#Zad 5
tab = np.array([i for i in range(9)]).reshape(3,3)
tab_add = np.array([i for i in range(10,40, 10)])
print(tab_add)
tab_sum_row = tab + tab_add[:, np.newaxis]
print(tab_sum_row)
scalars = [1,2,3]
tab_product = tab * scalars
print(tab_product)

#Zad 6
tab = np.random.randint(0, 10, 100)
print(tab)
print(np.sum(tab))
print(np.mean(tab))
print(np.std(tab))
print(np.cumsum(tab))
print(np.cumprod(tab))

#Zad 7
tab = np.random.randint(0, 100, 100)
tab = np.sort(tab)
print(tab)
value_to_search = 5
index = np.searchsorted(tab, value_to_search)
print(index)

#Pandas
#Zad 8
df = pd.read_csv("IRIS.csv")
print(df.shape)
print(df.head())

#Zad 9
selected_columns = df[["sepal_length", "sepal_width"]]
print(selected_columns.head())
print(df[df["sepal_length"] > 6])
print(df[df["species"]=="Iris-setosa"])

#Zad 10
#Ilość brakujących danych
print(df.isnull().sum())
#Usunięcie danych gdzie brakuje wartości
df.dropna(inplace=True)
#Usuwanie duplikatów z df
df.drop_duplicates(inplace=True)
#Konwersja danych
#Sprawdzamy dane przed konwersją
print(df.dtypes)
#Zmieniamy z typu liczbowego na typ całkowity
print(df["petal_length"].astype(int))
print(df["sepal_width"].astype(float))

#Zad 11
print(df.groupby("species").sum())
print(df.groupby("species").mean())
print(df.groupby("species").size())
print(df.describe())
print(df.groupby("species").agg({
    "sepal_length": ["mean", "max", "min"],
    "petal_length": ["sum"],
}))

#Zad 12
df["area"] = df["sepal_length"] * df["sepal_width"]
df["area"] = df["area"].apply(lambda x: round(x, 1))
df["species_short"] = df["species"].str[5:9]
print(df)

#Zad 13
#Wykres liniowy
plt.figure(figsize=(10,5))
df["sepal_length"].plot(kind="line", color="green")
#Dwie linie na jednym wykresie
#df["sepal_width"].plot(kind="line", color="red")
#Ustawienia wykresu
plt.title("Wykres liniowy", fontsize=14)
plt.xlabel("Indeks", fontsize=12)
plt.ylabel("Długość kwiatu", fontsize=12)
#Wyświetlenie siatki na wykresie
plt.grid(True)
plt.show()

#Wykres słupkowy
plt.figure(figsize=(8,5))
df["species"].value_counts().plot(kind="bar", color=["blue", "red", "green"])
plt.title("Liczba wierszy danych dla każdego gatunku", fontsize=14)
plt.xlabel("Gatunek", fontsize=12)
plt.ylabel("Liczba próbek", fontsize=12)
#Rotacja etykiet by były czytelne
plt.xticks(rotation=90)
plt.show()

#Wykres punktowy i eksploracja relacji między zmiennymi
plt.figure(figsize=(8,5))
plt.scatter(df["sepal_length"], df["petal_length"], color="purple", alpha=0.5)
plt.title("Zależność między długościami", fontsize=14)
plt.xlabel("Sepal Length", fontsize=12)
plt.ylabel("Petal Length", fontsize=12)
plt.grid(True)
plt.show()

#Zad 14
#Tworzę nowe data frame by zagadnienie było widoczne
df1 = pd.DataFrame({"ID": [1, 2, 3], "Imię": ["Ala", "Bartek", "Celina"]})
df2 = pd.DataFrame({"ID": [2, 3, 4], "Ocena": [90, 85, 88]})
df_merged = pd.merge(df1, df2, on="ID", how="inner")
#Muszę usunąć "Imię" przez to pojawiają się błędy w kodzie
df_filtered = df_merged.drop(columns=["Imię"])
df_melted = df_filtered.melt(id_vars=["ID"], var_name="Przedmiot", value_name="Wynik")
df_pivot = df_melted.pivot_table(index="ID", columns="Przedmiot", values="Wynik", aggfunc="mean")
print(df_pivot)
df["date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
df["date"] = pd.to_datetime(df["date"])
df_filtered = df[df["date"] >= "2023-07-01"]
print(df)

#Zad 15
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label="sin(x)", color="blue", linestyle="-")
plt.xlabel("X wartości", fontsize=12)
plt.ylabel("Y wartości", fontsize=12)
plt.title("Wykres funkcji sinus", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

#Zad 16
x = np.random.rand(100) * 10
y = np.random.rand(100) * 10

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="purple", alpha=0.6, edgecolors="black", label="Losowe punkty")
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.title("Wykres punktowy - Losowe dane", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

#Zad 17
kategorie = ["A", "B", "C", "D", "E"]
wartosci = np.random.randint(10, 100, len(kategorie))

plt.figure(figsize=(8, 6))
plt.bar(kategorie, wartosci, color=["blue", "red", "green", "orange", "purple"], alpha=0.7)
plt.xlabel("Kategorie", fontsize=12)
plt.ylabel("Wartość", fontsize=12)
plt.title("Wykres słupkowy - wartości kategorii", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

#Zad 18
dane = np.random.randn(1000)

plt.figure(figsize=(8, 6))
plt.hist(dane, bins=20, color="skyblue", edgecolor="black", alpha=0.75)
plt.xlabel("Wartości", fontsize=12)
plt.ylabel("Częstotliwość", fontsize=12)
plt.title("Histogram rozkładu danych", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

#Zad 19
kategorie = ["A", "B", "C", "D"]
wartości = [25, 35, 20, 20]
kolory = ["gold", "lightcoral", "lightskyblue", "lightgreen"]

plt.figure(figsize=(7, 7))
plt.pie(wartości, labels=kategorie, colors=kolory, autopct="%1.1f%%", startangle=140, wedgeprops={"edgecolor": "black"})
plt.title("Wykres kołowy kategorii", fontsize=14)
plt.show()

#Zad 20
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x**2
y4 = np.exp(x / 5)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Wykres liniowy
axes[0, 0].plot(x, y1, color='blue', label="sin(x)")
axes[0, 0].set_title("Wykres liniowy")
axes[0, 0].legend()

# Wykres punktowy
axes[0, 1].scatter(x, y2, color='red', alpha=0.5, label="cos(x)")
axes[0, 1].set_title("Wykres punktowy")
axes[0, 1].legend()

# Wykres kwadratowy
axes[1, 0].plot(x, y3, color='green', linestyle="--", label="x^2")
axes[1, 0].set_title("Wykres funkcji kwadratowej")
axes[1, 0].legend()

# Wykres wykładniczy
axes[1, 1].plot(x, y4, color='purple', linestyle="-.", label="exp(x/5)")
axes[1, 1].set_title("Wykres wykładniczy")
axes[1, 1].legend()

fig.suptitle("Wiele wykresów", fontsize=16)
plt.tight_layout()
plt.show()

#Zad 21
#Zaimplementowanie pliku
df = pd.read_csv("updated_pollution_dataset.csv")
print(df.head())
#Wybór kolumn które użyjemy
selected_columns = ["Temperature", "PM2.5"]
df_selected = df[selected_columns].dropna()
#Mój data frame ma bardzo dużo rekordów dlatego zmniejsze go na potrzeby wizualizacji
df_selected = df_selected[0:50]

#Teraz tworzymy wykres słupkowy
df_selected.plot(kind="bar", stacked=True, figsize=(12, 6), color=["blue", "red"])
plt.title("Skumulowany wykres słupkowy")
plt.xlabel("Próbki")
plt.ylabel("Wartości")
plt.legend(["Temperature", "PM2.5"])
plt.show()

#Teraz robimy wykres 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
x = np.arange(len(df_selected))
y = df_selected["Temperature"]
z = df_selected["PM2.5"]

ax.scatter(x, y, z, c=z, cmap="coolwarm", alpha=0.7)
ax.set_xlabel("Indeks próbki")
ax.set_ylabel("Temperatura")
ax.set_zlabel("PM2.5")
plt.title("Wykres 3D")
plt.show()