import os
import sys 

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from urllib.request import urlretrieve

os.makedirs("figuras", exist_ok=True)

# TODO: reemplaza por la ruta del archivo que tu pipeline de la Semana 7
# deja listo para modelar (una tabla del warehouse exportada a CSV, o el
# CSV limpio de tu etapa de calidad).
CARPETA = Path(__file__).parent
ARCHIVO = CARPETA / "obt_sales.csv.gz"
URL = "https://github.com/KeomaQuiroga/data-science-end-to-end/releases/download/v1.0-datos/obt_sales.csv.gz"

if not ARCHIVO.exists():
    print("Descargando datos (~490 MB), solo la primera vez...")
    urlretrieve(URL, ARCHIVO)
print("Datos descargados")
n = 1000000     # procesar los el primer millon
datos = pd.read_csv(ARCHIVO, nrows=n)
df = pd.DataFrame(datos)

# TODO: reemplaza por el nombre de la columna que tu modelo va a predecir.
OBJETIVO = "UNITS"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2), "\n")

# Paso 2: distribucion de la variable objetivo.
plt.figure()
plt.hist(datos[OBJETIVO], bins=40)
plt.title(OBJETIVO)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10), "\n")

# TODO, paso 4: elige la variable de entrada que esperas que mas influya
# en el objetivo y grafica su relacion con el (dispersion si es numerica,
# promedio del objetivo por grupo si es categorica).
ENTRADA = "CAT_ID"
plt.figure()
if pd.api.types.is_numeric_dtype(datos[ENTRADA]):
    plt.scatter(datos[ENTRADA], datos[OBJETIVO], s=8, alpha=0.5)
    plt.xlabel(ENTRADA)
    plt.ylabel(OBJETIVO)
else:
    datos.groupby(ENTRADA)[OBJETIVO].mean().plot.bar()
    plt.ylabel(f"promedio de {OBJETIVO}")
plt.title(f"{ENTRADA} vs {OBJETIVO}")
plt.savefig("figuras/relacion_entrada_objetivo.png")
plt.close()

# TODO, paso 5: agrega aqui las consultas que necesites para confirmar
# cada uno de tus 3 hallazgos (un filtro, un groupby, una correlacion).
# Correlacion de las variables numericas con el objetivo:
print(datos.groupby("CAT_ID")[OBJETIVO].agg(["count", "mean", "median"]).round(2))
print(datos.corr(numeric_only=True)[OBJETIVO].sort_values())
# Valores del objetivo que mas se repiten:
print(datos[OBJETIVO].value_counts().head(), "\n")

# 1. Cola larga
print("% de ceros:", round((datos[OBJETIVO] == 0).mean() * 100, 1))
print("media:", round(datos[OBJETIVO].mean(), 2), "| mediana:", datos[OBJETIVO].median())
print(f"IQR: q1={q1}, q3={q3}, limite superior={q3 + 1.5 * iqr}")
print("% de filas marcadas por el IQR:", round(len(fuera) / len(datos) * 100, 1))
print(datos[OBJETIVO].quantile([0.9, 0.99, 0.999]), "\n")

# 2. Lo que no se usa
constantes = [c for c in datos.columns if datos[c].nunique(dropna=False) == 1]
print("Columnas constantes en esta muestra:", constantes)
print("Correlacion de REVENUE con UNITS:", round(datos["REVENUE"].corr(datos[OBJETIVO]), 2), "\n")

# 3. Filas sin precio
print(pd.crosstab(datos["SELL_PRICE"].isna(), datos["IS_PRE_LAUNCH"]))
print(datos.groupby("IS_PRE_LAUNCH")[OBJETIVO].agg(["count", "mean"]).round(2))
activos = datos[~datos["IS_PRE_LAUNCH"]]
print("% de ceros sin las filas pre-lanzamiento:", round((activos[OBJETIVO] == 0).mean() * 100, 1), "\n")