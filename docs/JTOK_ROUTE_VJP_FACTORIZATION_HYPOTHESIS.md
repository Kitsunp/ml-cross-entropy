# Hipótesis posterior: factorizar el VJP de pesos de ruta

Estado: derivación y lectura de código, no implementada ni medida. El VJP spline
compacto ya terminó su comparación real; esta nota propone la siguiente
hipótesis compartida y no convierte sus mejoras en resultados de factorización.

## Trabajo repetido en el consumidor actual

El backward wide ya necesita las contracciones del gradiente de superficie G
con las columnas de salida spline y residual. Además reconstruye un vector
completo de superficie por ruta para calcular el gradiente del peso de mezcla.
Ese segundo vector vive sobre hidden y vuelve a acumular todos los modos y
coordenadas de seed. El interés es eliminar esa reconstrucción usando las
contracciones ya necesarias, no cambiar el router ni su selección.

Para el experto e elegido en la ruta r:

```text
V_rh = sum_m M_rm W_spline_emh + sum_d z_d W_residual_edh
bar_p_r = sum_h G_h V_rh
a_rm = sum_h G_h W_spline_emh
b_rd = sum_h G_h W_residual_edh
bar_p_r = sum_m a_rm M_rm + sum_d b_rd z_d
bar_M_rm = p_r a_rm
bar_z_residual_rd = p_r b_rd
```

Esta es una identidad distributiva del VJP ya existente. Los a y b sin peso
permiten escribir los VJP ponderados y acumular bar_p como escalar. La reducción
de hidden permanece; se elimina la acumulación del vector V_r en ese paso.
Los productos guardados, gradientes spline posteriores y normalización de
superficie conservan sus propietarios actuales.

No se debe obtener a o b dividiendo VJP ponderados por p. Una ruta con p=0
puede tener bar_p distinto de cero; valores muy pequeños además amplificarían
redondeo o underflow. La planificación correcta contrae G sin ponderar y
aplica p después sólo a los consumidores que lo requieren.

## Integración prevista

El mismo kernel wide de JToK y JToK-M puede usar esta identidad, también con
múltiples tiles hidden. Cada tile suma su contribución escalar de bar_p; la
política actual de store único o atomic se conserva según ownership.
No hace falta otro workspace, una activación por experto, un nuevo objetivo,
un cambio de layout de pesos ni una división por probabilidades.

La salida interna de pesos de ruta debe seguir siendo correcta incluso en
JToK base, aunque su autograd registrado no la consuma. No se debe llenar con
datos indefinidos ni inferir que una salida no usada autoriza cambiar su contrato.

Los tests nuevos deberán cubrir p=0, pesos muy pequeños, rutas repetidas,
máscaras, tails hidden, tiles únicos/múltiples, todos los VJP registrados y
router. Se comparará el VJP completo con un oráculo independiente; no basta
ver que la loss es finita. Cambia el orden FP32, no hay igualdad bit a bit
prometida ni nueva cuantización de G.

## Coste y riesgos

La cuenta algebraica elimina actualizaciones de acumuladores vectoriales y
una contracción posterior con V_r, pero no prueba un speedup. El compilador
puede emitir más reducciones/temporales, cambiar registros o elegir otro
launch. Se debe registrar el autotune efectivo y medir entrenamiento real
10+10 con perfil, superponiendo sólo mejoras ya respaldadas.

Guardar la superficie del forward sería otra hipótesis distinta: el perfil
matricial observado separa recomputación de ~3.24 ms por paso en JToK y ~9.33 ms
en M. Requeriría checkpoints FP32 y lifetime explícito. No basta guardar la
superficie mezclada para recuperar cada V_r, y no se puede sobrescribir un
tensor guardado del forward en el backward. Por eso la factorización de bar_p
es una opción sin ese incremento de retención, no una promesa de eliminar
automáticamente toda la recomputación de superficie.
