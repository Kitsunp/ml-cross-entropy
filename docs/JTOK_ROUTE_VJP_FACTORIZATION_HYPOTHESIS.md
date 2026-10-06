# Hipótesis posterior: factorizar el VJP de pesos de ruta

Estado: implementación opt-in `JTOK_ROUTE_VJP_FACTOR=1`, desactivada por
defecto. Ocho casos nuevos aprobados en ejecuciones focalizadas; ambas
comparaciones reales 10+10 finalizaron con perfil. JToK no muestra ganancia
del paso completo; JToK-M muestra una ganancia corta de 0.628%, todavía no
una aceptación estable. Se conserva como opción investigable, no como mejora
universal. El VJP spline compacto sirve como control; sus mejoras no se
atribuyen a esta hipótesis.

## Trabajo repetido en el consumidor actual

El backward wide ya necesita las contracciones del gradiente de superficie G
con las columnas de salida spline y residual. Además reconstruye un vector
completo de superficie por ruta para calcular el gradiente del peso de mezcla.
Ese segundo vector vive sobre hidden y vuelve a acumular todos los modos y
coordenadas de seed. El interés es eliminar esa reconstrucción usando las
contracciones ya necesarias, no cambiar el router ni su selección.

En el perfil de ese control, este consumidor cuesta una mediana de 738.272 µs
por llamada en JToK y 1384.193 µs en JToK-M. Las 24 llamadas de dos pasos suman
17.754 y 33.016 ms, respectivamente. Su coste medio por paso es aproximadamente
8.877/16.508 ms, frente a 259.086/291.549 ms del paso completo estable. Son
poblaciones diferentes: sirven para priorizar, no para una igualdad causal.
Incluso eliminar todo este consumidor no bastaría para alcanzar los gates.

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

## Integración y contrato matemático conservado

El mismo kernel wide de JToK y JToK-M puede usar esta identidad, también con
múltiples tiles hidden. Cada tile suma su contribución escalar de bar_p; la
política actual de store único o atomic se conserva según ownership.
No hace falta otro workspace, una activación por experto, un nuevo objetivo,
un cambio de layout de pesos ni una división por probabilidades. La selección
opt-in requiere hidden>=256; la ruta estrecha se rechaza explícitamente.
El flag forma parte de la clave del autotuner. Se conservan sus resets de
gradientes y la restauración del workspace de superficie entre candidatos.

La salida interna de pesos de ruta debe seguir siendo correcta incluso en
JToK base, aunque su autograd registrado no la consuma. No se debe llenar con
datos indefinidos ni inferir que una salida no usada autoriza cambiar su contrato.

Los tests nuevos cubren p=0, pesos muy pequeños, rutas repetidas,
máscaras, tails hidden, tiles únicos/múltiples, todos los VJP registrados y
router. Se compara el VJP del consumidor con contracciones Double independientes;
también se verifica backward compilado fullgraph/max-autotune, el caso vacío,
y la superposición con compacto spline + split-N. No basta
ver que la loss es finita. Cambia el orden FP32, no hay igualdad bit a bit
prometida ni nueva cuantización de G.

La salida de JToK sigue siendo `delta * (1 + scaler * direction)`;
la de JToK-M sigue siendo `delta + residual_scale * scaler * direction`.
Ambas tienen conexión identidad, pero sólo la primera multiplica la corrección
por delta. Con un experto, JToK-M no se convierte automáticamente en JToK.
Compartir las contracciones no autoriza unificar esas formulaciones. El G
recibido por esta factorización ya contiene el VJP específico de cada salida;
se conserva ese contrato y se optimiza sólo su consumidor posterior.

## Cancelación y criterio numérico reproducible

La primera ejecución aprobó siete casos y rechazó uno. El perfil conserva cinco
llamadas del consumidor con cinco tokens, frente a las seis previstas si todos
los pares completasen: el primer caso se detuvo en el control, antes del nuevo
kernel. El valor de un bar_p casi nulo difería aproximadamente 8.9e-9 del
oráculo Double; dividir por un valor de aproximadamente 4.6e-8 produjo un
rechazo relativo de alrededor del 20%. No prueba una regresión del candidato.

Con una sola ruta, normalizar p*V hace la superficie casi invariante a la
escala de p. Los términos del VJP de p pueden cancelarse hasta el régimen EPS.
Por eso el cociente error/abs(bar_p) no es una medida bien condicionada allí.
Se mantiene 3e-5 como presupuesto, con escala de operaciones explícita:

```text
C_r = sum_hm abs(G_h W_spline_emh M_rm)
    + sum_hd abs(G_h W_residual_edh z_d)
abs(bar_p_computado - bar_p_Double) <= 3e-5 * max(C_r, 1e-12)
```

Los VJP de modos y residual mantienen la comparación relativa 3e-5 por fila.
Las filas p=0 y p=2^-30 mantienen además el criterio relativo 3e-5 para bar_p,
deben tener bar_p no nulo cuando corresponde, y las máscaras exigen ceros
exactos. No se cambia la fórmula de normalización, el EPS ni el dtype para
pasar el test. La comparación de todos los VJP registrados conserva 1e-3.
Sólo se repitió el caso fallido; pasó. Los otros siete no se reejecutaron.

```bash
python benchmark/profile_leviathan_compact_tests.py \
  --test-file test_jtok_route_vjp.py --test-name route_vjp \
  --profile .codex-tmp/route_vjp_correctness.trace.json
```

Los perfiles de fallo/corrección permanecen privados. El test y el criterio
quedan en código para evitar repetir un rechazo relativo mal condicionado.

## Resultados reales de la factorización

Control: compacto Leviathan + proyección split-N + VJP spline compacto.
Candidato: el mismo flujo con la factorización; sparse permanece desactivado.
Son comparaciones dentro de cada modo, no una equivalencia de arquitecturas.
Cada candidato hizo diez pasos reales completos y diez batches de validación;
no se reejecutaron los controles existentes. Backend Triton estricto, política
y fuentes privadas idénticas dentro de cada par, verificados en los registros
privados. Los perfiles completos y las verificaciones del modelo son privados.
Las poblaciones estables son los pasos 5–10; el profiler cubre sólo 3–4.

| Medida | JToK control → candidato | JToK-M control → candidato |
| --- | --- | --- |
| Paso compute, mediana ms | 259.085590 → 259.174217 | 291.549164 → 289.730154 |
| Compute, steps/s | 3.859728 → 3.858408 | 3.429953 → 3.451487 |
| Cambio de throughput | −0.034196% | +0.627829% |
| Compute p95, ms | 259.906587 → 259.902704 | 292.693326 → 291.154925 |
| Compute desviación poblacional, ms | 2.111879 → 1.031314 | 2.615489 → 1.221314 |
| Paso e2e, mediana ms | 269.216160 → 269.317909 | 301.345028 → 299.824376 |
| Productor VJP, mediana µs | 738.272 → 663.471 | 1384.193 → 1306.479 |
| Reducción de mediana del productor | 10.131903% | 5.614390% |
| Suma del productor en dos pasos, ms | 17.753728 → 16.029416 | 33.015992 → 31.348542 |
| Registros/hilo | 56 → 55 | 56 → 40 |
| Shared memory del productor, bytes | 2048 → 2048 | 2048 → 2048 |

En cada perfil hay 24 llamadas del productor: doce por paso activo. La
geometría registrada es idéntica: grid `[32768, 1, 1]`, block `[64, 1, 1]`
(dos warps), en los cuatro perfiles. Esto evita atribuir el cambio de registros
a un cambio de block; no mide ocupación efectiva, spills, caché ni bandwidth.
Los casos funcionales de ownership también cubren launches fijos de cuatro
warps, tiles únicos y múltiples; no son mediciones de throughput.

El ahorro agregado del productor es aproximadamente 0.862156/0.833725 ms
por paso perfilado para JToK/M. No son los mismos pasos de la población estable
y no prueban una igualdad causal con la diferencia del paso completo.
JToK es prácticamente neutro en el paso completo; M tiene una mejora observada
pequeña. Se mantiene opt-in, sin repetición idéntica de GPU ni promoción por
defecto. Los gates de 4.5/4.2 steps/s no se alcanzaron. Seis muestras de un solo
par no prueban estabilidad prolongada ni una significación estadística.

Los resúmenes públicos de ambos pares están en
`benchmark-results/leviathan-compact-20261006/*_route_vjp_factor_real_10x10.json`.
Sólo contienen tiempos completos y detalles de los kernels investigados.
La factorización sólo modifica backward: no se declara mejora de inferencia.

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
