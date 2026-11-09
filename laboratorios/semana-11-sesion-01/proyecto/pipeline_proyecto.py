import hashlib

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# el archivo del proyecto con las 2 features ya agregadas (lo genera features_proyecto.py).
ARCHIVO = "proyecto_features.csv"
OBJETIVO = "UNITS"
DIAS_PRUEBA = 7                     # ultimos 7 dias = prueba

NUMERICAS = ["SELL_PRICE", "unidades_hace_7d"]
CATEGORICAS = ["DEPT_ID", "STORE_ID", "WEEKDAY", "EVENT_TYPE_1"]
YA_LISTAS = ["IS_SNAP", "IS_WEEKEND", "IS_EVENT_DAY", "es_snap_comida"]
COLUMNAS = NUMERICAS + CATEGORICAS + YA_LISTAS


def huella(ruta=ARCHIVO):
    """SHA-256 del archivo de datos: si cambia un solo byte, cambia la huella."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_y_dividir(ruta=ARCHIVO):
    """Lee los datos y los divide por fecha: entrenamiento = dias anteriores al corte,
    prueba = los ultimos DIAS_PRUEBA dias. Devuelve los conjuntos y la informacion de la division."""
    datos = pd.read_csv(ruta, parse_dates=["DATE"], dtype={"EVENT_TYPE_1": str})

    # binarizamos categorias
    for col in ["IS_SNAP", "IS_WEEKEND", "IS_EVENT_DAY"]:
        datos[col] = datos[col].astype(int)

    # ordenamos por fecha para la division
    datos = datos.sort_values(["DATE", "STORE_ID", "ITEM_ID"]).reset_index(drop=True)

    X = datos[COLUMNAS]
    y = datos[OBJETIVO]

    fechas = datos["DATE"].unique()
    corte = pd.Timestamp(fechas[-DIAS_PRUEBA])
    n_prueba = int((datos["DATE"] >= corte).sum())
    X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=n_prueba, shuffle=False)

    fechas_ent = datos.loc[X_ent.index, "DATE"]
    fechas_prueba = datos.loc[X_prueba.index, "DATE"]
    division = {
        "division": "por_fecha",
        "fecha_corte": str(corte.date()),
        "entrenamiento_desde": str(fechas_ent.min().date()),
        "entrenamiento_hasta": str(fechas_ent.max().date()),
        "prueba_desde": str(fechas_prueba.min().date()),
        "prueba_hasta": str(fechas_prueba.max().date()),
        "filas_entrenamiento": len(X_ent),
        "filas_prueba": len(X_prueba),
    }
    return X_ent, X_prueba, y_ent, y_prueba, division


def nueva_preparacion():
    """El ColumnTransformer del pipeline. Se crea uno nuevo por modelo y se ajusta
    solo con entrenamiento: asi no hay fuga hacia prueba."""
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


class NaiveEstacional(BaseEstimator, RegressorMixin):
    """Linea base del dominio: "igual que hace 7 dias". No aprende nada: predice la
    columna unidades_hace_7d. Esta aqui (y no en el notebook) para que MLflow pueda
    guardarla y volver a cargarla como cualquier modelo de scikit-learn."""

    def __init__(self, columna="unidades_hace_7d"):
        self.columna = columna

    def fit(self, X, y=None):
        # No aprende nada, pero scikit-learn exige un atributo terminado en "_"
        # para considerar que el modelo ya esta ajustado.
        self.columna_ = self.columna
        return self

    def predict(self, X):
        return X[self.columna].fillna(0).to_numpy()


if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba, division = cargar_y_dividir()
    print(f"Entrenamiento: {division['entrenamiento_desde']} a {division['entrenamiento_hasta']}")
    print(f"Prueba: {division['prueba_desde']} a {division['prueba_hasta']}")

    preparacion = nueva_preparacion()
    X_ent_listo = preparacion.fit_transform(X_ent)
    X_prueba_listo = preparacion.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape)
    print("Prueba:", X_prueba_listo.shape)
    print(list(preparacion.get_feature_names_out()))
    print("Huella de los datos (SHA-256):", huella())