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

DIA_FINAL = 1941        # ultimo dia con ventas
DIAS_MODELO = 33
DIA_INICIO = DIA_FINAL - DIAS_MODELO - 7 + 1

# leemos solo las columnas que nos interesa
COLUMNAS = ["DATE", "D_NUM", "ITEM_ID", "DEPT_ID", "CAT_ID", "STORE_ID", "UNITS", "SELL_PRICE",
            "WEEKDAY", "IS_WEEKEND", "EVENT_TYPE_1", "IS_EVENT_DAY", "IS_SNAP", "IS_PRE_LAUNCH", 
            "IS_CHRISTMAS_CLOSED", "IS_STORE_CLOSED"]

print("Leyendo el archivo por partes (unos minutos)...")
partes = []
for parte in pd.read_csv(ARCHIVO, usecols=COLUMNAS, dtype={"EVENT_TYPE_1": str},
                         chunksize=2_000_000):
    partes.append(parte[parte["D_NUM"] >= DIA_INICIO])
datos = pd.concat(partes, ignore_index=True)
print(f"Filas leidas: {len(datos):,}")

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

# quitamos calentamiento
datos = datos[datos["D_NUM"] >= DIA_INICIO + 7]

# el producto no estaba o la tienda estaba cerrada
no_demanda = datos["IS_PRE_LAUNCH"] | datos["IS_CHRISTMAS_CLOSED"] | datos["IS_STORE_CLOSED"]
datos = datos[~no_demanda].reset_index(drop=True)

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

datos.to_csv("proyecto_features.csv", index=False, encoding="utf-8")
print("Guardando nueva tablas")