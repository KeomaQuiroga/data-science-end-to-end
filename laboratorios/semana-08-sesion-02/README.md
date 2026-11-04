# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. Ábrelas desde el explorador de archivos de VS Code. También puedes correr el mismo código en un notebook de Jupyter (en el Codespace, crea un archivo `.ipynb` y elige el kernel de Python 3.11), donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

_(tu respuesta)_

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

_(tu respuesta)_

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

_(tu respuesta)_

## Ejercicio 1: resumen, distribución y atípicos

Archivo: [`starter/ejercicio1_explorar.py`](./starter/ejercicio1_explorar.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York, con fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. El script ya lee el archivo, arma los gráficos y los guarda en `figuras/`; faltan 4 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-08-sesion-02/starter
python3 ejercicio1_explorar.py
```

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

_(tu respuesta)_

**Respuesta 2 (TODO 2):** abre `figuras/hist_tarifa.png` después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

_(tu respuesta)_

**Respuesta 3 (TODO 3 y 4):** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

_(tu respuesta)_

**Lo que confirmó la salida, TODO por TODO:**

_(anota el límite superior, cuántos atípicos marcó y cómo se reparten)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Archivo: [`starter/ejercicio2_relaciones.py`](./starter/ejercicio2_relaciones.py), sobre el mismo `viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0. Al final, el script imprime una comprobación por TODO: completa los 5 hasta que todas digan `OK`.

```
python3 ejercicio2_relaciones.py
```

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

_(tu hallazgo, con el número que lo confirma)_

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

_(tu hallazgo, con el número que lo confirma)_

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

_(tu respuesta)_

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar, y la plantilla [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py).

### Paso 1: resumen y variable objetivo (5 minutos)

**Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.**
eda-proyecto  | Filas y columnas: (1000000, 40)
eda-proyecto  | DATE_KEY                                int64
eda-proyecto  | DATE                                      str
eda-proyecto  | D                                         str
eda-proyecto  | D_NUM                                   int64
eda-proyecto  | ITEM_ID                                   str
eda-proyecto  | DEPT_ID                                   str
eda-proyecto  | CAT_ID                                    str
eda-proyecto  | STORE_ID                                  str
eda-proyecto  | STATE_ID                                  str
eda-proyecto  | UNITS                                   int64
eda-proyecto  | SELL_PRICE                            float64
eda-proyecto  | REVENUE                               float64
eda-proyecto  | WM_YR_WK                                int64
eda-proyecto  | WM_FISCAL_YEAR                          int64
eda-proyecto  | WM_WEEK_OF_YEAR                         int64
eda-proyecto  | WEEK_SEQ                                int64
eda-proyecto  | WEEKDAY                                   str
eda-proyecto  | WDAY                                    int64
eda-proyecto  | IS_WEEKEND                               bool
eda-proyecto  | DAY_OF_MONTH                            int64
eda-proyecto  | MONTH                                   int64
eda-proyecto  | QUARTER                                 int64
eda-proyecto  | YEAR                                    int64
eda-proyecto  | EVENT_NAME_1                              str
eda-proyecto  | EVENT_TYPE_1                              str
eda-proyecto  | EVENT_NAME_2                          float64
eda-proyecto  | EVENT_TYPE_2                          float64
eda-proyecto  | IS_EVENT_DAY                             bool
eda-proyecto  | N_EVENTS                                int64
eda-proyecto  | IS_CHRISTMAS_CLOSED                      bool
eda-proyecto  | DATE_LY_364                               str
eda-proyecto  | DATE_KEY_LY_364                       float64
eda-proyecto  | IS_SNAP                                  bool
eda-proyecto  | IS_PRE_LAUNCH                            bool
eda-proyecto  | IS_STORE_CLOSED                          bool
eda-proyecto  | IS_SALES_SPIKE                           bool
eda-proyecto  | IS_SUSPECTED_STOCKOUT_CONSERVATIVE       bool
eda-proyecto  | IS_SUSPECTED_STOCKOUT_NB                 bool
eda-proyecto  | WEEK_IDX                                int64
eda-proyecto  | _LOADED_AT                                str
eda-proyecto  | dtype: object
eda-proyecto  | DATE_KEY                                    0
eda-proyecto  | DATE                                        0
eda-proyecto  | D                                           0
eda-proyecto  | D_NUM                                       0
eda-proyecto  | ITEM_ID                                     0
eda-proyecto  | DEPT_ID                                     0
eda-proyecto  | CAT_ID                                      0
eda-proyecto  | STORE_ID                                    0
eda-proyecto  | STATE_ID                                    0
eda-proyecto  | UNITS                                       0
eda-proyecto  | SELL_PRICE                             151406
eda-proyecto  | REVENUE                                151406
eda-proyecto  | WM_YR_WK                                    0
eda-proyecto  | WM_FISCAL_YEAR                              0
eda-proyecto  | WM_WEEK_OF_YEAR                             0
eda-proyecto  | WEEK_SEQ                                    0
eda-proyecto  | WEEKDAY                                     0
eda-proyecto  | WDAY                                        0
eda-proyecto  | IS_WEEKEND                                  0
eda-proyecto  | DAY_OF_MONTH                                0
eda-proyecto  | MONTH                                       0
eda-proyecto  | QUARTER                                     0
eda-proyecto  | YEAR                                        0
eda-proyecto  | EVENT_NAME_1                           920595
eda-proyecto  | EVENT_TYPE_1                           920595
eda-proyecto  | EVENT_NAME_2                          1000000
eda-proyecto  | EVENT_TYPE_2                          1000000
eda-proyecto  | IS_EVENT_DAY                                0
eda-proyecto  | N_EVENTS                                    0
eda-proyecto  | IS_CHRISTMAS_CLOSED                         0
eda-proyecto  | DATE_LY_364                                 0
eda-proyecto  | DATE_KEY_LY_364                        118831
eda-proyecto  | IS_SNAP                                     0
eda-proyecto  | IS_PRE_LAUNCH                               0
eda-proyecto  | IS_STORE_CLOSED                             0
eda-proyecto  | IS_SALES_SPIKE                              0
eda-proyecto  | IS_SUSPECTED_STOCKOUT_CONSERVATIVE          0
eda-proyecto  | IS_SUSPECTED_STOCKOUT_NB                    0
eda-proyecto  | WEEK_IDX                                    0
eda-proyecto  | _LOADED_AT                                  0
eda-proyecto  | dtype: int64
eda-proyecto  |           DATE_KEY       D_NUM  ...  DATE_KEY_LY_364    WEEK_IDX
eda-proyecto  | count   1000000.00  1000000.00  ...        881169.00  1000000.00
eda-proyecto  | mean   20137058.72     1139.26  ...      20130581.85      162.18
eda-proyecto  | std       14732.48      524.28  ...         11924.54       74.93
eda-proyecto  | min    20110929.00      244.00  ...      20110801.00       34.00
eda-proyecto  | 25%    20121118.00      660.00  ...      20120209.00       94.00
eda-proyecto  | 50%    20140807.00     1287.00  ...      20130810.00      183.00
eda-proyecto  | 75%    20150630.00     1614.00  ...      20140701.00      230.00
eda-proyecto  | max    20160406.00     1895.00  ...      20150408.00      270.00
eda-proyecto  | 
eda-proyecto  | [8 rows x 19 columns]
eda-proyecto  | Atipicos de UNITS segun IQR: 123214
eda-proyecto  |     DATE_KEY        DATE  ... WEEK_IDX               _LOADED_AT
eda-proyecto  | 11  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 27  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 36  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 53  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 54  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 58  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 62  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 64  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 67  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 70  20160318  2016-03-18  ...      267  2026-10-21 03:24:23.368
eda-proyecto  | 
eda-proyecto  | [10 rows x 40 columns]
eda-proyecto  | SELL_PRICE                           -0.145116
eda-proyecto  | IS_PRE_LAUNCH                        -0.126527
eda-proyecto  | IS_SUSPECTED_STOCKOUT_NB             -0.112844
eda-proyecto  | IS_SUSPECTED_STOCKOUT_CONSERVATIVE   -0.058009
eda-proyecto  | WDAY                                 -0.030285
eda-proyecto  | DAY_OF_MONTH                         -0.017681
eda-proyecto  | N_EVENTS                             -0.003865
eda-proyecto  | IS_EVENT_DAY                         -0.003865
eda-proyecto  | WM_WEEK_OF_YEAR                       0.002381
eda-proyecto  | MONTH                                 0.003847
eda-proyecto  | QUARTER                               0.006602
eda-proyecto  | DATE_KEY_LY_364                       0.017785
eda-proyecto  | YEAR                                  0.018280
eda-proyecto  | WM_FISCAL_YEAR                        0.018280
eda-proyecto  | IS_SNAP                               0.018428
eda-proyecto  | DATE_KEY                              0.018455
eda-proyecto  | WM_YR_WK                              0.018907
eda-proyecto  | D_NUM                                 0.019263
eda-proyecto  | WEEK_IDX                              0.019365
eda-proyecto  | WEEK_SEQ                              0.019365
eda-proyecto  | IS_WEEKEND                            0.039884
eda-proyecto  | IS_SALES_SPIKE                        0.122971
eda-proyecto  | REVENUE                               0.702844
eda-proyecto  | UNITS                                 1.000000
eda-proyecto  | EVENT_NAME_2                               NaN
eda-proyecto  | EVENT_TYPE_2                               NaN
eda-proyecto  | IS_CHRISTMAS_CLOSED                        NaN
eda-proyecto  | IS_STORE_CLOSED                            NaN
eda-proyecto  | Name: UNITS, dtype: float64
eda-proyecto  | UNITS
eda-proyecto  | 0    661500
eda-proyecto  | 1    144359
eda-proyecto  | 2     70927
eda-proyecto  | 3     38015
eda-proyecto  | 4     23008
eda-proyecto  | Name: count, dtype: int64

### Paso 2: relaciones (10 minutos)

**Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.**
La variable que más se espera que influya es CAT_ID o la categoría del producto.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

**Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).**

**Hallazgo 1:**
Presenta varios valores con 0, alrededor del 66% del la muesta. Es una característica que presenta sus clases desbalanceadas.

**Hallazgo 2:**
La correlación con `revenue` es de 0.7, pero la operación surge de `units * sell_price`. De igual forma `is_sales_spike` y `is_suspected_stockout_nb` son valores de futuro. Se deben quitar las columas que el modelo aún no debe haber aprendido.

**Hallazgo 3:**
La columna de `sell_price` tiene valores vacíosç, de igual forma tiene una correlación muy baja; no tiene tanta relación el precio y las unidades vendidas.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

_(tu respuesta)_
