import os
import sys 

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# TODO: reemplaza por la ruta del archivo que tu pipeline de la Semana 7
# deja listo para modelar (una tabla del warehouse exportada a CSV, o el
# CSV limpio de tu etapa de calidad).
datos = pd.read_csv('obt_sales.csv.gz')

# TODO: reemplaza por el nombre de la columna que tu modelo va a predecir.
OBJETIVO = "units"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

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
print(fuera.head(10))

# TODO, paso 4: elige la variable de entrada que esperas que mas influya
# en el objetivo y grafica su relacion con el (dispersion si es numerica,
# promedio del objetivo por grupo si es categorica).

if len(sys.argv) > 3:
    ENTRADA = sys.argv[3]
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
print(datos.corr(numeric_only=True)[OBJETIVO].sort_values())
# Valores del objetivo que mas se repiten:
print(datos[OBJETIVO].value_counts().head())