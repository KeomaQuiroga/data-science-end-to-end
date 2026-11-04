import pandas as pd
from urllib.request import urlretrieve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TODO: el archivo de tu proyecto, con las 2 features de la sesion pasada
# ya agregadas (o el script que las agrega, antes de esta linea).
ARCHIVO = "proyecto_features.csv"
datos = pd.read_csv(ARCHIVO, parse_dates=["DATE"], dtype={"EVENT_TYPE_1":str})

OBJETIVO = "UNITS"

# binarizamos categorias
for col in ["IS_SNAP", "IS_WEEKEND", "IS_EVENT_DAY"]:
    datos[col] = datos[col].astype(int)

# ordenamos por fecha para la division
datos = datos.sort_values(["DATE", "STORE_ID", "ITEM_ID"]).reset_index(drop=True)

NUMERICAS = ["SELL_PRICE", "unidades_hace_7d"]
CATEGORICAS = ["DEPT_ID", "STORE_ID", "WEEKDAY", "EVENT_TYPE_1"]
YA_LISTAS = ["IS_SNAP", "IS_WEEKEND", "IS_EVENT_DAY", "es_snap_comida"]

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# TODO: decide como dividir. Si tus filas tienen fecha y vas a predecir
# el futuro, divide por fecha (entrenamiento antes de una fecha de corte,
# prueba despues) en vez de al azar.
fechas = datos["DATE"].unique()
corte = pd.Timestamp(fechas[-7])        # ultimos 7 dias
n_prueba = int((datos["DATE"] >= corte).sum())
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=n_prueba, shuffle=False)
 
fechas_ent = datos.loc[X_ent.index, "DATE"]
fechas_prueba = datos.loc[X_prueba.index, "DATE"]
print(f"Entrenamiento: {fechas_ent.min().date()} a {fechas_ent.max().date()}")
print(f"Prueba: {fechas_prueba.min().date()} a {fechas_prueba.max().date()}")

preparacion = ColumnTransformer([
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

X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
print("Entrenamiento:", X_ent_listo.shape)
print("Prueba:", X_prueba_listo.shape)
print(list(preparacion.get_feature_names_out()))

datos.to_csv("proyecto_division.csv", index=False, encoding="utf-8")
print("Guradada nueva tabla")
