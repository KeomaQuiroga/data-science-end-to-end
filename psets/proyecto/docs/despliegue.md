# Despliegue del modelo: por lotes (batch) semanal

**Decisión:** El modelo se sirve ocupará lotes una vez por semana, cada sábado después de que llega la entrega semanal de ventas. Se pronostican de una sola vez las unidades de los próximos 7 días para todas las combinaciones producto × tienda, y el resultado queda en una tabla que lee quien planifica la reposición. La predicción puede esperar hasta 24 horas sin que la decisión pierda valor.

## Qué decisión habilita, quién la usa y cuánto puede esperar
- **Decisión:** Cuánto pedir de cada producto en cada tienda para la próxima semana, el pronóstico de los próximos 7 días.
- **Quién la usa:** El planificador de inventario que arma el pedido semanal.
- **Cuánto puede esperar:** Hasta 24 horas, los datos llegan el sábado (en el pipeline, `load_sales_week` corre los sábados a las 06:00 UTC) y el pedido se arma una vez por semana. Si el pronóstico está listo el mismo sábado o el domingo, la decisión no cambia. La cadena completa tarda hoy unos 15–20 minutos (dbt + OBT + predicción), muy por debajo de ese margen.

## Cuántos casos hay que predecir, y si se conocen de antemano
- **30.490 series:** (3.049 productos × 10 tiendas) × 7 días = 213.430 predicciones por semana.
- **Se conocen todos de antemano:** El catálogo de productos y tiendas y las fechas de la semana siguiente son fijos. No hay que esperar a que alguien pida una predicción: se pueden calcular todas juntas, que es justo lo que hace bien un proceso por lotes.

## Qué datos necesitan las features al predecir, y si están disponibles

| Feature | De dónde sale | ¿Disponible el sábado para la semana siguiente? |
|---|---|---|
| `unidades_hace_7d` | Ventas de la misma serie 7 días antes | **Sí.** Para los días t+1 a t+7 se necesitan las ventas de t−6 a t, que ya llegaron en la entrega del sábado. Por eso el horizonte no puede pasar de 7 días |
| `SELL_PRICE` | Precio de la semana (`sell_prices`) | Sí, el precio de la semana se fija antes de que empiece |
| `WEEKDAY`, `IS_WEEKEND`, `EVENT_TYPE_1`, `IS_EVENT_DAY` | Calendario | Si, el calendario y los eventos se conocen con meses de anticipación |
| `IS_SNAP`, `es_snap_comida` | Calendario SNAP de cada estado | Si, los días SNAP se publican de antemano |
| `DEPT_ID`, `STORE_ID` | Catálogo | Si, no cambian |

## Qué se pierde con el patrón que descarto (en línea)
Con el lote semanal se pierde la capacidad de reaccionar dentro de la semana, si pasa algo imprevisto entre semana (una promoción no planificada, un quiebre de stock, un cambio de precio a mitad de semana), el pronóstico no se actualiza hasta el sábado siguiente.

Lo que no se pierde es lo que importa para esta decisión, el pedido se arma una vez por semana, así que una predicción más rápida no cambiaría ningún pedido.
