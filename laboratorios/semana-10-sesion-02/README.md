# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**
Aprendizaje automático para problemas numéricos y LLM para problemas lingüisticos (analizar, entender y generar textos).

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**


## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?


## Ejercicio 1: una red neuronal frente al ensamblaje

**Archivo:** [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1): Sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?**
Significa que la red no está encontrando la respuesta dentro del número de iteraciones permitidas. Si le dejamos 1000 iteraciones suponiendo que son más que las de por defecto, es posible que la red si encuentre una respuesta o esté muy cerca de la misma.

**Respuesta 2 (TODO 2): Una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?**
- pesos_entrada = 25 * 64 + 64(bias) = 1664
- pesos_intermedio 64 * 1 + 1(bias) = 65
- resultado = pesos_entrada + pesos_intermedio = 1729

**Respuesta 3 (TODO 3): ¿Qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?**
La distancia_km, porque mide la distancia y eso define el costo del viaje. El modelo que explica es gradient boosting.

**Lo que confirmó la salida, TODO por TODO:**
gradient boosting    entrenamiento  1.0705   prueba  1.2754     0.43 s
red neuronal         entrenamiento  1.0882   prueba  1.1695     6.10 s

           modelo  mae_entrenamiento  mae_prueba  segundos
gradient boosting             1.0705      1.2754     0.426
     red neuronal             1.0882      1.1695     6.097

Parametros de la red: 1729
Las 3 columnas mas importantes para el ensamblaje:
numericas__distancia_km      0.9824
ya_listas__es_tarifa_fija    0.0120
ya_listas__es_hora_pico      0.0028

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

**Solo o en pareja. Archivo:** [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1: ¿Qué LLM usaste? Completa la tabla con los números de tu corrida.**
Se uso Claude Opus 5.5 high.
| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.8957 | 0.3s | 0$ | Si |
| Red neuronal | 0.9637 | 5s | 0$ | Difícil, pero aún es justificable |
| LLM | 0.765 | 0$ | 1.45$ | Sin garantía |

**Respuesta 2: Corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?**
No dan los mismo 20 resultados, solo coincide con los resultados fijos. Significa que cada viaje se cobra con diferente precio, algo que es injusto para el pasajero.

**Respuesta 3: Busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?**
La peor predicción del LLM fue de Lower Manhattan → Midtown, el LLM dijo 19.99 y la tarifa real era 17.65, un error de 2.34. Lo particular es que es hora pico, y ambos modelos esperaban una tarifa más alta por eso, pero el viaje salió más barato de lo normal. 
El ensamblaje predijo 19.52, con un error de 1.87, así que tampoco lo habría aprendido. Con 2380 filas aprende bien el patrón promedio, pero no puede anticipar una variación individual que no se explica con las columnas disponibles (el tráfico real de ese día, por ejemplo).

**Si terminas antes: El error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?**
El LLM no se mide sobre los 595 viajes, mientras que los modelos predicen los 595 en milisegundos. Además, el propio LLM varía en cuanto al precio de cada respuesta, así que la diferencia entre las soluciones es del mismo tamaño que el ruido.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.py`
**Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.**
Hay que recordar que al DummyRegressor:
- **wape_train:** 1.2146
- **wape_test:** 1.2079
- **brecha:** -0.0067
- **Segundos:** 1

| **Modelo**                    | **wape_train** | **wape_test** | **segundos** |
|-------------------------------|----------------|---------------|--------------|
| DummyRegressor                | 1.2146         | 1.2079        | 1.2          |
| Naive estacional              | 0.8542         | 0.9001        | 0.0          |
| Random forest sin regularizar | 0.6869         | 0.8392        | 43.8         |
| Red neuronal                  | 0.8223         | 0.8434        | 389.3        |

### Entregable 2: la prueba del LLM
**Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.**

id,UNITS_PRED,REAL
1,1,0
2,3,1
3,1,4
4,2,1
5,1,1
6,0,0
7,0,2
8,1,0
9,7,9
10,0,2

### Entregable 3: la decisión
**Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.**

| **Modelo**        | **Metrica**                                                                                                                                                                              | **Costo**                                                                                                                                                     | **Explicabilidad**                                                                                                                                                               |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Random forest** | El modelo es preciso en el entrenamiento y pruba,  pues su wape es el menor. Sin embargo, al tener un wape menor en la prueba,  es posible que este memorizando en lugar de generalizar. | Tiene un costo bajo al interactuar con CPU. Su costo en tiempo tambien es el mas bajo.                                                                        | Alta, es posible modificar los parametros del modelo y entender facilmente el funcionamiento de los arboles.                                                                     |
| **Red neuronal**  | El modelo es el mas equilibrado, con wape similares tanto para entrenamiento y para prueba.                                                                                              | Puede ser usado con una CPU, pero para sacar toda su potencia es  necesaria una GPU y librerias  especializadas como torch. Su costo en tiempo es mayor.      | Baja, actua como una caja negra de varias  multiplicaciones matematicas.                                                                                                         |
| **LLM**           | Presenta un wape muy bajo, es decir, es el peor de los 3 modelos. La principal razon se debe a que solo le dimos 10 filas, eso es el menos del 1% (~0.0046%) de los datos.               | Viene el costo de la membresia mensual o en su lugar el costo por token. En cuanto  a tiempo dependemos de la cantidad de los datos, con mas datos tarda mas. | Nula, es una arquitectura masiva de codigo cerrado, no comprendemos su funcionamiento y cada vez da un resultado diferente. Ademas de necesitar ser muy especifico en el prompt. |

El modelo que mejor rinde es el Random Forest, pues obtiene los mejores resultados, menor costo y su explicabilidad es sencilla. De igual forma, para evitar problemas de memorización es posible que se necesite mejorar un poco el modelo.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

_(tu respuesta)_
