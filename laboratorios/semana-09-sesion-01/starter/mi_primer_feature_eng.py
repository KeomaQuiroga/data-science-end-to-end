import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression       # interfaz de aprendizaje
from sklearn.preprocessing import OrdinalEncoder        # interfaz de transformacion

viajes = pd.read_csv("viajes_taxi.csv")
viajes_clean = viajes.query("tarifa < 999 and duracion_min > 0")        # limpieza atipicos

# caracteristicas del dataset
print(viajes_clean.head())
print(viajes_clean.info(), "\n")

# fecha y hora
fecha_inicio = pd.to_datetime(viajes_clean["fecha_hora_inicio"])        # obtener la fecha
viajes_clean["fecha_hora_inicio"] = fecha_inicio
viajes_clean["hora"] = fecha_inicio.dt.hour     # crear hora
viajes_clean["dia_semana"] = fecha_inicio.dt.dayofweek      # crear semana

print(viajes_clean.head())
print(viajes_clean.info(), "\n")

# hora ciclica
viajes_clean["hora_sin"] = np.sin(2 * np.pi * viajes_clean["hora"] / 24)
viajes_clean["hora_cos"] = np.cos(2 * np.pi * viajes_clean["hora"] / 24)
print(viajes_clean[["fecha_hora_inicio", "hora", "hora_sin", "hora_cos"]], "\n")

# categorias nominales
viajes_codificado = pd.get_dummies(
    viajes_clean, 
    columns=["zona_origen", "zona_destino"], 
    prefix=["origen", "destino"]
)
print(viajes_codificado, "\n")

# categorias ordinales
encoder = OrdinalEncoder()
viajes_codificado["metodo_pago_ordinal"] = encoder.fit_transform(viajes_codificado[["metodo_pago"]]).ravel()
print(viajes_codificado, "\n")
# no ocupar get_dummies, ocupar los one hot encoder                                             