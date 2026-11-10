import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

PROYECTO = Path(__file__).resolve().parent.parent
ARCHIVO = PROYECTO / "data" / "raw" / "obt_sales.csv.gz"
SALIDA = PROYECTO / "data" / "processed" / "proyecto_features.csv"
URL = "https://github.com/KeomaQuiroga/data-science-end-to-end/releases/download/v1.0-datos/obt_sales.csv.gz"

ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
SALIDA.parent.mkdir(parents=True, exist_ok=True)
if not ARCHIVO.exists():
    print("Descargando datos (~490 MB), solo la primera vez...")
    urlretrieve(URL, ARCHIVO)
print("Datos descargados")

DIA_FINAL = 1941        # ultimo dia con ventas en M5 (2016-05-22)
DIAS_MODELO = 33
DIA_INICIO = DIA_FINAL - DIAS_MODELO - 7 + 1

COLUMNAS = ["DATE", "D_NUM", "ITEM_ID", "DEPT_ID", "CAT_ID", "STORE_ID",
            "UNITS", "SELL_PRICE", "WEEKDAY", "IS_WEEKEND", "EVENT_TYPE_1",
            "IS_EVENT_DAY", "IS_SNAP", "IS_PRE_LAUNCH", "IS_CHRISTMAS_CLOSED",
            "IS_STORE_CLOSED"]

print("Leyendo el archivo por partes (unos minutos)...")
partes = []
for parte in pd.read_csv(ARCHIVO, usecols=COLUMNAS, dtype={"EVENT_TYPE_1": str},
                         chunksize=2_000_000):
    partes.append(parte[parte["D_NUM"] >= DIA_INICIO])
datos = pd.concat(partes, ignore_index=True)
print(f"Filas leidas: {len(datos):,}")

OBJETIVO = "UNITS"

# unidades vendidas por la misma serie (item x tienda) hace 7 dias.
datos["DATE"] = pd.to_datetime(datos["DATE"])
hace7d = datos[["ITEM_ID", "STORE_ID", "DATE", OBJETIVO]].copy()
hace7d["DATE"] = hace7d["DATE"] + pd.Timedelta(days=7)
hace7d = hace7d.rename(columns={OBJETIVO: "unidades_hace_7d"})
datos = datos.merge(hace7d, on=["ITEM_ID", "STORE_ID", "DATE"], how="left")

# dia SNAP y producto de comida.
datos["es_snap_comida"] = ((datos["IS_SNAP"] == 1) & (datos["CAT_ID"] == "FOODS")).astype(int)

# se quitan los 7 dias de calentamiento: solo servian para calcular el lag.
datos = datos[datos["D_NUM"] >= DIA_INICIO + 7]

# filas que no son demanda
no_demanda = datos["IS_PRE_LAUNCH"] | datos["IS_CHRISTMAS_CLOSED"] | datos["IS_STORE_CLOSED"]
datos = datos[~no_demanda].reset_index(drop=True)

print(f"Filas con features: {len(datos):,}  |  "
      f"lag de 7 dias disponible en el {datos['unidades_hace_7d'].notna().mean():.1%}")

# compara el objetivo segun cada feature nueva.
print("\n--- Evidencia Feature 1 (Numerica) ---")
print("Correlacion con el objetivo:")
print(datos[["unidades_hace_7d", OBJETIVO]].corr())

print("\n--- Evidencia Feature 2 (Categorica/Binaria) ---")
print("Promedio de unidades vendidas por grupo SNAP Comida:")
print(datos.groupby("es_snap_comida")[OBJETIVO].mean())

print("\nVista previa de las primeras filas:")
print(datos[["ITEM_ID", "STORE_ID", "DATE", OBJETIVO, "unidades_hace_7d", "es_snap_comida"]].head())

datos.to_csv(SALIDA, index=False)
print(f"\nGuardado: {SALIDA}")
