# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tus 3 hallazgos de la exploración de la Semana 8, Sesión 2: las features de hoy salen de ahí.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**
Las que mejoran el negocio explican el comportamiento real del negocio que lo entiende todo el mundo. Por otro lado, las que mejoran una métrica son como una caja negra para personas sin conocimientos técnicos, no lo entienden los stakeholders y su justifiación suele ser más complicada.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**
Un conteo bruto no sirve si no se sabe si el cliente hizo 5 o 500 compras, debemos dividir el número de transacciones sobre transacciones totales para saber la fidelidad del cliente.

## Pregunta de apertura

**La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?**
Para la tarifa entre destinos se crean columnas adicionales, para ver desde donde partio y hacia donde va, dependiendo de la respuesta se completa con **True** o **False**. Se debe observar cuando los viajes cuestas más para así obtener la hora pico de los recorridos.

## Ejercicio 1: featuresu de fecha y hora

**Enunciado:** [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada, sin los errores confirmados. A partir de `fecha_hora_inicio`, construir hora, día de la semana, fin de semana, hora pico y la hora como ciclo, y comprobar si la hora pico aporta algo a la tarifa.

**Respuesta 1, antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?
Para un modelo que mide distancias queda a 23 espacios, para un reloj queda solo una hora, pues la hora 0 es media noche.

**Respuesta 2:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?
La hora te da el momento de cuando fue tomado el taxi, pero nos podemos guiar por la duracion. Podemos ver trayectos iguales (parten de punto A y van hacia punto B) o trayectos que siempre van a un mismo destino, y de esos trayectos ver la duración con respecto a la hora. Si no hay tráfico el viaje dura menos, si hay congestión el viaje dura más.

**La formulación completa, tal como la resuelvo yo:**

_(anota aquí el script y lo que mostró, mientras se revisa en clase)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/features_de_zona.py`](./plantilla-ejercicio2/features_de_zona.py) y córrela desde su carpeta. Completa los 4 `TODO`.

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?
- **Cantidad:** 300
- **Max:** 77.81
- **Min:** 1.6

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?
One hot crea 8 columnas, y la codificación ordinal me dio solo una columna. Lo ideal es usar one hot, pues la variable ordinal crea una jerarquía, le da más importancia a unos origenes que a otros.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?
- duracion_min
- tarifa
- propina
El modelo empezaría a alucinar o crear datos sin sentido, pues se le está entregando información futura (data leakage). Por ejemplo podría emprezar a tratar de predecir duración (puede haber tráfico o no), tarifa (el taxímetro) o la propina (depende del comportamiento del conductor).

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

_(tu respuesta)_

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

_(el código de cada feature)_

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

_(tu justificación)_

**Feature 2:**

_(tu justificación)_

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

_(tu respuesta)_
