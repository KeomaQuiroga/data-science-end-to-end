import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARCHIVO = "proyecto_features.csv"          # lo genera features_proyecto.py

# 1. Datos y division de la anterior sesion
datos = pd.read_csv(ARCHIVO, parse_dates=["DATE"], dtype={"EVENT_TYPE_1": str})
OBJETIVO = "UNITS"

for col in ["IS_SNAP", "IS_WEEKEND", "IS_EVENT_DAY"]:
    datos[col] = datos[col].astype(int)

datos = datos.sort_values(["DATE", "STORE_ID", "ITEM_ID"]).reset_index(drop=True)

NUMERICAS = ["SELL_PRICE", "unidades_hace_7d"]
CATEGORICAS = ["DEPT_ID", "STORE_ID", "WEEKDAY", "EVENT_TYPE_1"]
YA_LISTAS = ["IS_SNAP", "IS_WEEKEND", "IS_EVENT_DAY", "es_snap_comida"]

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

fechas = datos["DATE"].unique()
corte = pd.Timestamp(fechas[-7])            # ultimos 7 dias = prueba
n_prueba = int((datos["DATE"] >= corte).sum())
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=n_prueba, shuffle=False)
print(f"Entrenamiento: {len(X_ent):,} filas  |  Prueba: {len(X_prueba):,} filas")


def nueva_preparacion():
    """El ColumnTransformer de la sesion pasada. Se crea uno nuevo por modelo, y cada
    Pipeline lo ajusta solo con entrenamiento: asi no hay fuga hacia prueba."""
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="no_event")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


# 2. Metrica WAPE = suma|error| / suma_unidades_reales
def wape(real, pred):
    return np.abs(np.asarray(real) - np.asarray(pred)).sum() / np.abs(np.asarray(real)).sum()
resultados = []

def registrar(nombre, pred_ent, pred_prueba, segundos):
    w_ent, w_prueba = wape(y_ent, pred_ent), wape(y_prueba, pred_prueba)
    resultados.append({
        "modelo": nombre,
        "wape_entrenamiento": round(w_ent, 4),
        "wape_prueba": round(w_prueba, 4),
        "brecha": round(w_prueba - w_ent, 4),
        "segundos_entrenamiento": round(segundos, 1),
    })
    print(f"{nombre:<38} WAPE ent {w_ent:.4f} | prueba {w_prueba:.4f} | "
          f"brecha {w_prueba - w_ent:+.4f} | {segundos:.1f} s")


# 3. Baselime - DummyRegressor
inicio = time.time()
registrar("Naive estacional (igual que hace 7 dias)",
          X_ent["unidades_hace_7d"], X_prueba["unidades_hace_7d"], time.time() - inicio)


# 4. Pipeline
modelos = {
    "DummyRegressor (media)": DummyRegressor(strategy="mean"),
    "Ridge (alpha=1)": Ridge(alpha=1.0),
    "Ridge (alpha=10000)": Ridge(alpha=10_000.0),
    "Random Forest sin regularizar": RandomForestRegressor(
        n_estimators=50, max_samples=0.3, n_jobs=-1, random_state=42),
    "Random Forest regularizado": RandomForestRegressor(
        n_estimators=50, max_samples=0.3, min_samples_leaf=200, max_depth=12,
        n_jobs=-1, random_state=42),
}

for nombre, modelo in modelos.items():
    pipe = Pipeline([("preparacion", nueva_preparacion()), ("modelo", modelo)])
    inicio = time.time()
    pipe.fit(X_ent, y_ent)
    segundos = time.time() - inicio
    registrar(nombre, pipe.predict(X_ent), pipe.predict(X_prueba), segundos)


# 5. Resultados
tabla = pd.DataFrame(resultados)
tabla.to_csv("resultados_proyecto.csv", index=False)
print("\nGuardado tabla resultados")
print(tabla.to_string(index=False))