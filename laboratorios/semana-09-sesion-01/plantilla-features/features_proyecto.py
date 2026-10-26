import csv
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

# TODO: la misma ruta que usaste en la exploracion de la Semana 8.
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
df.to_csv("usuarios.csv", index=False, encoding="utf-8")
print(datos.info())
OBJETIVO = "UNITS"

# TODO, feature 1: crea la columna a partir de una o mas columnas existentes.
# unidad hacer 7 dias
datos["DATE"] = pd.to_datetime(datos["DATE"])
hace7d = datos[["ITEM_ID", "STORE_ID", "DATE", OBJETIVO]].copy()
hace7d["DATE"] = hace7d["DATE"] + pd.Timedelta(days=7)
hace7d = hace7d.rename(columns={OBJETIVO: "unidades_hace_7d"})
datos = datos.merge(hace7d, on=["ITEM_ID", "STORE_ID", "DATE"], how="left")
 
# TODO, feature 2.
# es comida
datos["es_snap_comida"] = ((datos["IS_SNAP"] == 1) & (datos["CAT_ID"] == "FOODS")).astype(int)

# Evidencia: compara el objetivo segun cada feature nueva.
# Si es numerica, su correlacion con el objetivo:
# print(datos[["TODO_feature_1", OBJETIVO]].corr())
# Si es categorica o binaria, el objetivo promedio por grupo:
# print(datos.groupby("TODO_feature_2")[OBJETIVO].mean())
print("\n--- Evidencia Feature 1 (Numerica) ---")
print("Correlacion con el objetivo:")
print(datos[["unidades_hace_7d", OBJETIVO]].corr())

print("\n--- Evidencia Feature 2 (Categorica/Binaria) ---")
print("Promedio de unidades vendidas por grupo SNAP Comida:")
print(datos.groupby("es_snap_comida")[OBJETIVO].mean())

print("\nVista previa de las primeras filas:")
print(datos[["ITEM_ID", "STORE_ID", "DATE", OBJETIVO, "unidades_hace_7d", "es_snap_comida"]].head())