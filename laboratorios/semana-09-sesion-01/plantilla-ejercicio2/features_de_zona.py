import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder        # interfaz de transformacion

viajes = pd.read_csv("../starter/viajes_taxi.csv")
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()
print(viajes.info())

ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

# TODO 1: crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
# va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
# JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
# y maxima de esos viajes.
print("\n---Tarifa fija---")
con_uno_manhattan = (viajes["zona_origen"] == "JFK Airport") & (viajes["zona_destino"].isin(ZONAS_MANHATTAN))
con_uno_jfk = (viajes["zona_origen"].isin(ZONAS_MANHATTAN)) & (viajes["zona_destino"] == "JFK Airport")

# condicion, si es es 1, si no es 0
viajes["es_tarifa_fija"] = np.where(con_uno_manhattan | con_uno_jfk, 1, 0)
total_unos = (viajes["es_tarifa_fija"] == 1).sum()
max_uno = viajes["tarifa"].max()
min_uno = viajes["tarifa"].min()

print(f"- Cantidad: {total_unos}")
print(f"- Maximo: {max_uno}")
print(f"- Minimo: {min_uno}")

# TODO 2: codifica zona_origen con one-hot (pd.get_dummies), con el
# prefijo "origen" y dtype=int. Imprime los nombres de las columnas
# nuevas.
print("\n---Codificacion---")
viajes_codificado = pd.get_dummies(viajes, columns=["zona_origen"], prefix="origen", dtype=int)
print(viajes_codificado.info())

# TODO 3: codifica zona_origen de forma ordinal: un numero entero por
# zona, en orden alfabetico, en una columna origen_ordinal.
print("\n---Ordinal---")
encoder = OrdinalEncoder()
viajes_codificado["origen_ordinal"] = encoder.fit_transform(viajes[["zona_origen"]])
print(viajes_codificado["origen_ordinal"])

# TODO 4: para cada columna original del dataset, decide si existe en el
# momento en que alguien pide el taxi (antes del viaje) e imprime la
# decision.
print("\n---Disponibilidad---")
cols_antes = {
    "id_viaje",
    "fecha_hora_inicio",
    "zona_origen",
    "zona_destino",
    "distancia_km",
    "pasajeros",
    "metodo_pago",
}

for c in viajes.columns:
    if c in cols_antes:
        print(f"{c} disponible antes del viaje")
    else:
        print(f"{c} disponible despues del viaje")