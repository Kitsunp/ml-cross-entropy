# VJP spline compacto común: hipótesis y contrato

Estado: implementación opt-in `JTOK_COMPACT_SPLINE_VJP=1`, desactivada por
defecto. Trece casos nuevos aprobados en ejecuciones focalizadas, incluidos
backward compilado y rechazo de grid alterado. Comparaciones reales 10+10
terminadas: mejora observada corta del paso completo de +1.65% en JToK y +2.73%
en JToK-M. Ambos gates siguen no alcanzados; no se afirma estabilidad prolongada.
Se superpone al split de proyección publicado
en `d496780`, no al experimento sparse que regresó en JToK-M.

## Evidencia y límite de la hipótesis

En los perfiles reales del candidato matricial, el VJP spline anterior cuesta
aproximadamente 0.819 ms por llamada en JToK y 1.503 ms en JToK-M. Hay 12 llamadas
por paso perfilado. Esto justifica estudiarlo, pero incluso eliminar todo ese
coste no cubriría la distancia a los gates completos de 4.5 y 4.2 pasos/s.
No se presenta la compactación como garantía de cumplir el objetivo global.

La opción sparse previa conservaba todas las bases y cargas de coeficientes;
sólo omitía atomics cero. Esta implementación cambia ese consumidor completo:
construye un tile `[coordenadas,4]` con el soporte local y contrae sus pesos y
derivadas. La geometría se reutiliza del helper de Leviathan; no se implementa
otra base ni se cambia la función de entrenamiento.

## Matemática y ownership

Para una coordenada x, sean a_q los pesos cuadráticos sobre los knots realmente
almacenados, s=sum(a_q), b_q=a_q/max(s,1e-12) y c_q los coeficientes seleccionados.
En el régimen s>1e-12:

```text
phi = sum_q b_q c_q
db_q = (da_q - b_q sum_j da_j) / s
sum_q db_q = 0
dphi/dx = sum_q db_q (c_q - c_anchor)
bar_c_q += bar_phi b_q
bar_x += bar_phi dphi/dx
bar_phi = bar_mode * saved_mode * sign(phi) / (abs(phi) + 1e-9)
```

La última expresión, incluyendo su signo en cero y el EPS del producto,
conserva la política existente. También se conserva su derivada cero para
s<=1e-12. No se sustituye por una derivada distinta del clamp.

Tres vecinos bastan en aritmética real; los knots redondeados pueden activar
un cuarto halo. Ese halo entra tanto en la normalización como en los VJP.
El coeficiente central sirve de ancla para cancelar offsets constantes.
Esto reorganiza una derivada ya presente, no añade un objetivo o método nuevo.

El programa posee un token y una ruta. El kernel previo conserva los VJP de
superficie, pesos de mezcla, normalización y scaler. Aquí sólo se terminan
los VJP de seed y coeficientes. Distintas rutas y tokens comparten esos destinos:
sus atomics FP32 siguen siendo necesarios. No se afirma determinismo completo
ni se duplican el router o la reducción matricial de parámetros de salida.

## Ingeniería y restricciones

- Se mantienen los strides de coeficientes existentes; no hay migración de pesos.
- Tile de soporte de cuatro lanes contiguos por coordenada, hasta 128 coordenadas,
  cuatro warps y un stage. No se materializan activaciones densas por experto.
- Los contratos de dimensión, grid canónico y backend Triton son explícitos.
  Grids personalizados y combinaciones incompatibles fallan; no hay fallback.
- La opción sparse debe estar desactivada: el consumidor compacto ya posee
  las actualizaciones de soporte y no debe aparentar otra mejora independiente.
- El forward de modos queda intacto, incluido su producto guardado. Este primer
  experimento cambia sólo el VJP spline para poder atribuir resultados.
- El helper compartido expone opcionalmente su suma sin clamp. Su retorno por
  defecto y las expresiones usadas por Leviathan permanecen iguales.

Menos cargas lógicas no implica menos tráfico DRAM: los knots contiguos de una
coordenada pueden ocupar el mismo sector aunque sólo se pidan cuatro. El tile
reduce trabajo de bases y tamaño de reducciones, pero los gathers, atomics,
registros y ocupación pueden anular esa ventaja. Se medirá el paso completo,
incluidos todos los costes nuevos, no sólo la contracción aislada.

## Validación compatible con compilación

El guard de valores no puede leer punteros de FakeTensor durante el trazado.
Se separan contratos estructurales y una operación opaca de validación del grid.
Esta última verifica el tensor real con la caché/versionado ya existente y
devuelve una copia pequeña que el consumidor usa efectivamente. La copia no
puede eliminarse como trabajo muerto y mantiene el contrato sin alias del op.
Su asignación/copia forman parte del tiempo y perfil del candidato.
Los kernels numéricos mantienen su registro Triton componible; no se cambia
globalmente `torch.compile`, CUDA Graphs ni la política de cachés.

Antes de la primera ejecución compilada se llama a
`prepare_compact_jtok_grids(model)`, después del movimiento final de dtype y
dispositivo. Sólo valida grids congelados por el contrato estructural existente;
no cambia parámetros/buffers ni instala hooks. El runner real lo hace al inicio
del entrenamiento, antes de resetear memoria y medir. Una mutación posterior
invalida la caché y debe rechazarse, no aceptarse durante capture.

## Regresiones conservadas

1. El retorno opcional de nueve/diez valores del helper falló inicialmente en
   compilación Triton. Ramas `if/else` explícitas corrigen la selección de tupla.
   Los casos GPU de este consumidor reproducen la especialización nueva.
2. Cerca del borde exterior del soporte, la división densa puede sufrir
   cancelación catastrófica. La comparación estricta contra denso y Double se
   conserva en el dominio unitario de producción. Un caso exterior separado
   usa la base FP32 almacenada como oráculo y exige derivada cero para
   coeficientes constantes, también bajo el piso. No se ampliaron tolerancias
   para ocultar esa diferencia ni se promete paridad FP64 de ramas separadas
   por un ULP de distancia.
3. La lectura de data_ptr del guard provocó un fallo fullgraph antes del cambio
   de frontera. Las regresiones compiladas verifican backward finito y rechazo
   de un grid modificado después de poblar la caché.

Los perfiles/logs completos y manifiestos de fallos son privados. Los tests y
la explicación permiten reproducirlos sin fuentes del modelo ni dataset.

## Reproducción focalizada

```bash
python benchmark/profile_leviathan_compact_tests.py \
  --test-file test_jtok_compact.py \
  --test-file test_leviathan_candidate_provenance.py --test-name compact_jtok \
  --profile .codex-tmp/compact_jtok_correctness.trace.json
python benchmark/leviathan_candidate_provenance.py --source-root . \
  --include-compact-jtok --output .codex-tmp/compact_jtok_manifest.json
```

Los casos sintéticos sólo prueban funcionamiento y bordes. La aceptación exige
entrenamiento real perfilado 10+10, con `JTOK_PROJECTION_SPLIT=1`,
`JTOK_SPARSE_COEFF_UPDATES=0`, compacto Leviathan y la política congelada.
El control es la corrida matricial registrada del mismo modo. Rendimiento,
precisión, memoria y contribución de kernels se juzgan por separado.

## Resultado real y superposición

El control mantiene compacto Leviathan y split-N de proyección. El candidato
añade únicamente `JTOK_COMPACT_SPLINE_VJP=1`; sparse permanece en cero. Se
reutilizaron los controles existentes, sin repetir ejecuciones conocidas.
La comparabilidad efectiva se verificó con una auditoría privada adicional de
un valor por defecto sobrescrito explícitamente; no se afirma identidad de
todos los bytes de las fuentes. No se publican esa auditoría ni huellas privadas.

| Modo | Compute anterior → compacto VJP | Pasos/s anteriores → candidato | Ganancia throughput | Gate |
| --- | --- | --- | --- | --- |
| JToK | 263.371 → 259.086 ms | 3.796923 → 3.859728 | +1.6541% | 4.5: no alcanzado |
| JToK-M | 299.501 → 291.549 ms | 3.338890 → 3.429953 | +2.7273% | 4.2: no alcanzado |

Cada corrida contiene 10 optimizer steps reales y 10 batches de validación.
El gate usa los pasos estables 5–10; los dos pasos perfilados 3–4 son otra
población. P95 compute: 259.907 ms en JToK y 292.693 ms en M; desviación
poblacional 2.112 y 2.615 ms. E2E: 269.216 y 301.345 ms, con ganancias de
throughput de +1.5354% y +2.7946%. Compilación fría queda fuera de esas cifras.

| VJP spline investigado | Mediana anterior → compacto | Reducción mediana | Ahorro sumado por paso perfilado |
| --- | --- | --- | --- |
| JToK | 818.863 → 445.072 µs | 45.6475% | 4.5994 ms |
| JToK-M | 1502.843 → 786.736 µs | 47.6502% | 8.5282 ms |

En ambos modos hay 24 llamadas: 12 por cada paso perfilado. La ruta anterior
no aparece en el candidato, y la nueva no aparece en el control. Los dos
kernels de proyección split-N siguen presentes en ambas corridas.
La especialización spline cambia de 48 a 56 registros por thread, mientras
shared memory baja de 8192 a 512 bytes. Esto demuestra una reducción de trabajo
medida aun con más registros, no una mejora de ocupación o tráfico DRAM medida.
No se recogieron contadores de spills, caché, bandwidth u ocupación efectiva.

La copia del grid y el guard están incluidos en el paso completo. No forman
parte de la duración aislada del kernel spline; su atribución GPU independiente
no se considera probada únicamente por contar operaciones CPU del perfil.
No se interpreta una atribución incompleta como coste cero.

Las comprobaciones privadas de pérdidas finitas, diferencias numéricas y picos
de memoria se conservaron por separado; no hay afirmación de convergencia de
preentrenamiento a partir de diez pasos. Los perfiles completos y los rastros
visuales reducidos están guardados localmente, no en el repositorio.

La decisión es conservar el candidato opt-in y superponible, sin activar
sparse ni cambiar defaults. Resúmenes públicos sanitizados:
`benchmark-results/leviathan-compact-20261006/jtok_compact_spline_vjp_real_10x10.json`
y `jtokm_compact_spline_vjp_real_10x10.json` en el mismo directorio.
