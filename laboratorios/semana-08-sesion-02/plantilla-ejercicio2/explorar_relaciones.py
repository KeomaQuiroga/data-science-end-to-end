import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# TODO 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
limpios = viajes.copy()
limpios = limpios.drop(limpios[limpios["tarifa"] == 999.99].index)       # tarifa
limpios = limpios.drop(limpios[limpios["duracion_min"] < 0.0].index)       # duracion
print(f"Filas antes: {len(viajes)} \t Filas ahora:{len(limpios)}")

# TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general y
# cuentalos con un filtro.
plt.figure()
plt.title("Distancia(km) vs. Tarifa")
plt.scatter(limpios["distancia_km"], limpios["tarifa"])
plt.show()
plt.close()

# TODO 3: propina promedio y cantidad de viajes con propina 0, por metodo de pago (groupby + agg).
resultado = limpios.groupby("metodo_pago").agg(
        propina_promedio=("propina", "mean"),
        sin_propina=("propina", lambda x: (x==0).sum())
    )
print(f"\n{resultado}")

# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina con la tarifa.
print(f"\n{limpios.corr(numeric_only=True)}")

# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
num_vacios = (limpios["pasajeros"] == None).sum()
num_cero = (limpios["pasajeros"] == 0.0).sum()
print(f"\nViajes vacio: {num_vacios} \t Viajes cero: {num_cero}")