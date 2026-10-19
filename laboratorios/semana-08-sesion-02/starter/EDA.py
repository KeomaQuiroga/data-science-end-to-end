import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figuras", exist_ok=True)       # crear directorio
data = pd.read_csv("viajes_taxi.csv")       # leer csv
print("Analisis exploratorio descriptivo (EDA)")

# Ver el dataset
data.info()     # informacion general
print(data.dtypes)      # tipos de datos
print(data.isna().sum())      # valores nulos
print(data.duplicated().sum())      # valores duplicados
print(data.describe())      # estadisticas descriptivas


# Visualizar data
# histograma
plt.figure()
plt.hist(data["tarifa"], bins=50)
plt.title("Distribucion de la tarifa")
plt.xlabel("USD")
plt.ylabel("Numero de viajes")
plt.savefig("./starter/figuras/histograma_tarifa.png")      # guardar figura
plt.close()

# boxplot
plt.figure()
plt.boxplot(data["tarifa"])
plt.title("Distribucion de la tarifa")
plt.savefig("./starter/figuras/boxplot_tarifa.png")      # guardar figura
plt.close()

# scatterplot
plt.figure()
plt.scatter(data["tarifa"], data["distancia_km"])
plt.title("Dispersion tarifa vs. distancia (km)")
plt.savefig("./starter/figuras/scatter_outliers-tarifa-distancia.png")      # guardar figura
plt.close()

# print(data.corr())      # matriz de correlacion, solo variables numericas


# Extraer valores atipicos
q1 = data["tarifa"].quantile(0.25)
q3 = data["tarifa"].quantile(0.75)
iqr = q3 - q1       # rango intercuartil

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

atipicos = data.query(f"tarifa < {lower_bound} or tarifa > {upper_bound}")
print("Valores atipicos:")
print(atipicos)

sin_atipicos = data.query(f"tarifa >= {lower_bound} and tarifa <= {upper_bound}")
print("Valores no atipicos:")
print(sin_atipicos)

# sin valores atipicos
plt.figure()
plt.scatter(sin_atipicos["tarifa"], sin_atipicos["distancia_km"])
plt.title("Dispersion sin outliers tarifa vs. distancia (km)")
plt.savefig("./starter/figuras/scatter_tarifa-distancia.png")      # guardar figura
plt.close()

# TODO: Filtrar para solo variables numericas y calcular la matriz de correlacion
# print(sin_atipicos.corr())      # matriz de correlacion