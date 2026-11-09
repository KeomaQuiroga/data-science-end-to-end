# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**
Cuando un modelo muestra una exactitud alta, pero no predice nada; sucede principalmente con datasets desbalanceados. Ocupar métricas para clases pequeñas como recall (sensibilidad), precisión (predicciones positivas por el modelo) y AUC (para datasets desbalanceados).

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**
Al convertir un indicador único en el objetivo, el modelo encuentra atajos para maximizar la respuesta, lo que provoca sobreajuste.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

_(tu respuesta)_

## Ejercicio 1: tres formas de validar el mismo modelo

**Notebook:** [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1): Con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?**
Los errores si serán parecidos, no habrá mucha variación.

**Respuesta 2 (TODO 2): En la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?**
Se espera un error mayot, los primeros pliegues tendrán pocos datos históricos para predecir.

**Respuesta 3 (TODO 3): En la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?**
El mayor mae será con la zona menos frecuente y el menor mae será con la zona más repetida. Esto porque el modelo no puede aprender a generalizar de pocos datos.

**Respuesta 4 (TODO 4): La app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?**
El porcentaje de viajes subestimados en más de 20%, esto porque MAE y RMSE son métricas que no reflejan la regla del negocio.

**Lo que confirmó la tabla, esquema por esquema:**
                  validacion  mae_promedio  desviacion                       pliegues
aleatorio, gradient boosting        1.2288      0.0361 [1.28, 1.24, 1.16, 1.24, 1.23]
           aleatorio, bosque        1.2880      0.0291  [1.3, 1.27, 1.25, 1.34, 1.28]
 temporal, gradient boosting        1.3099      0.0693  [1.4, 1.21, 1.38, 1.27, 1.29]
 por zona, gradient boosting        2.2996      2.0648 [1.27, 1.31, 1.29, 1.19, 6.43]

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

**Solo o en pareja. Notebook:** [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1: ¿Qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?**
- **accuracy:** 0.9508
- **recall:** 0.00

**Respuesta 2: ¿Cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?**
- **accuracy:** 0.9712, mejora del 2.04%
- **recall:** 0.4610, mejora del 46.1%
El que mejor describe es el recall, esto porque el accuracy de la linea base es muy alto si lo vemos junto a su recall, significa un dataset desbalanceado.

**Respuesta 3: Con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?**
- **Fraudes no detectados:** 159
- **Falsas alarmas:** 14
- **Fraudes detectados:** 136
- **Transacciones correctas:** 5691
- **Umbral:** 0.1

**Si terminas antes: Si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.**
Al no tener datos completos de 20$, el modelo aumenta el umbral y se equivocará más al clasificar transacciones como fraudulentas.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**
Temporal, queremos saber el futuro en base a días pasados. Un esquema aleatorio comete trampas al entrenar con datos a futuro.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**
Compararemos con el wape promedio de cada modelo y esquema. 
| **Modelo**       | **Al azar** | **Temporal** |
|------------------|-------------|--------------|
| Naive estacional | 0.8542      | 0.8483       |
| Random forest    | 0.8411      | 0.8414       |
| Red neuronal     | 0.8333      | 0.9863       |

Y la desviación estándar promedio.
| **Modelo**       | **Al azar** | **Temporal** |
|------------------|-------------|--------------|
| Naive estacional | 0.0006      | 0.0367       |
| Random forest    | 0.0013      | 0.0322       |
| Red neuronal     | 0.0087      | 0.1053       |

Vemos que efectivamente a los modelos les va mejor frente a un esquema temporal, pero la diferencia se ve únicamente con la red neuronal. De igual forma, de acuerdo al wapa la red neuronal es el mejor modelo que se tiene.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**
Ahora el mejor modelo es la red neuronal, sin embargo, random forest gana con una desviación estándar más pequeña. Esto nos dice que los valores de RandomForest permanecen más pegados y el wape de la red neuronal es más errático. Se decide ocupar la red neuronal con más kfold (4 o 5) y en un entorno con GPU, caso contrario se escoge random forest por su rápidez.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**
El costo relativo asimétrico, dentro del pronóstico sale más caro subestimar que sobreestimar, si no hay producto no hay vento. Suponemos que subestimar cuesta 5 veces más que sobreestimar. El wape trata a los errores por igual, la nueva métrica no. Según wape la red neuronal es mejor, pero según el costo relativo el mejor modelo es random forest.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

_(tu respuesta)_
