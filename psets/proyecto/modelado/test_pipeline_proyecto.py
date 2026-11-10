import numpy as np
import pandas as pd

from pipeline_proyecto import (
    CATEGORICAS, COLUMNAS, DIAS_PRUEBA, NUMERICAS, OBJETIVO,
    NaiveEstacional, dividir, nueva_preparacion,
)


def datos_de_prueba(n_dias=10):
    """2 productos x 2 tiendas x n_dias, ya ordenados como los deja cargar_datos()."""
    filas = []
    for dia, fecha in enumerate(pd.date_range("2016-05-01", periods=n_dias)):
        for tienda in ["CA_1", "TX_1"]:
            for item, depto in [("FOODS_1_001", "FOODS_1"), ("HOBBIES_1_001", "HOBBIES_1")]:
                filas.append({
                    "DATE": fecha, "STORE_ID": tienda, "ITEM_ID": item, "DEPT_ID": depto,
                    "UNITS": dia,                       # unidades = numero de dia: facil de seguir
                    "SELL_PRICE": 2.0, "unidades_hace_7d": max(dia - 7, 0),
                    "WEEKDAY": fecha.day_name(), "EVENT_TYPE_1": np.nan,
                    "IS_SNAP": 0, "IS_WEEKEND": int(fecha.dayofweek >= 5),
                    "IS_EVENT_DAY": 0, "es_snap_comida": 0,
                })
    return (pd.DataFrame(filas)
            .sort_values(["DATE", "STORE_ID", "ITEM_ID"]).reset_index(drop=True))


def test_el_objetivo_no_entra_al_modelo():
    assert OBJETIVO not in COLUMNAS
    for col in ["REVENUE", "IS_SALES_SPIKE", "IS_SUSPECTED_STOCKOUT_NB", "WEEK_IDX"]:
        assert col not in COLUMNAS

    X_ent, X_prueba, *_ = dividir(datos_de_prueba())
    assert OBJETIVO not in X_ent.columns
    assert OBJETIVO not in X_prueba.columns


def test_la_division_no_mezcla_el_futuro():
    datos = datos_de_prueba(n_dias=10)
    X_ent, X_prueba, y_ent, y_prueba, division = dividir(datos)
    fechas_ent = datos.loc[X_ent.index, "DATE"]
    fechas_prueba = datos.loc[X_prueba.index, "DATE"]

    assert fechas_ent.max() < fechas_prueba.min()
    assert fechas_prueba.nunique() == DIAS_PRUEBA
    assert len(X_prueba) == DIAS_PRUEBA * 4
    assert division["fecha_corte"] == "2016-05-05"      # 10 dias - 7 de prueba = corte en el dia 4


def test_la_preparacion_aprende_solo_de_entrenamiento():
    datos = datos_de_prueba(n_dias=10)
    X_ent, X_prueba, *_ = dividir(datos)
    X_ent = X_ent.copy()
    X_ent.loc[X_ent.index[0], "SELL_PRICE"] = np.nan        # un faltante en entrenamiento

    prep = nueva_preparacion().fit(X_ent)
    imputador = prep.named_transformers_["numericas"].named_steps["imputar"]
    escalador = prep.named_transformers_["numericas"].named_steps["escalar"]

    assert imputador.statistics_[NUMERICAS.index("unidades_hace_7d")] == 0
    assert escalador.mean_[NUMERICAS.index("unidades_hace_7d")] == 0


def test_acepta_una_categoria_que_no_vio_al_entrenar():
    datos = datos_de_prueba(n_dias=10)
    X_ent, X_prueba, *_ = dividir(datos)
    prep = nueva_preparacion().fit(X_ent)

    nueva = X_prueba.head(1).copy()
    nueva["STORE_ID"] = "WI_9"            # una tienda que no existia al entrenar
    nueva["EVENT_TYPE_1"] = "Sporting"    # un tipo de evento que no aparecio al entrenar

    resultado = prep.transform(nueva)     # no debe fallar
    assert resultado.shape[1] == prep.transform(X_ent.head(1)).shape[1]

    # la tienda desconocida queda con todas sus columnas de one-hot en 0.
    nombres = list(prep.get_feature_names_out())
    cols_tienda = [i for i, n in enumerate(nombres) if n.startswith("categoricas__STORE_ID_")]
    assert resultado[0, cols_tienda].sum() == 0


def test_el_naive_predice_lo_vendido_hace_7_dias():
    X = pd.DataFrame({c: [0, 0, 0] for c in COLUMNAS})
    X["unidades_hace_7d"] = [5, 0, np.nan]     # el ultimo no tiene historia
    pred = NaiveEstacional().fit(X).predict(X)
    # valor calculado a mano: lo mismo que hace 7 dias, y 0 si no hay dato.
    assert list(pred) == [5, 0, 0]
