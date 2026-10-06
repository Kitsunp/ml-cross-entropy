# JToK/JToK-M: actualización atómica de soporte no nulo

Estado: candidato experimental opt-in; nueve casos funcionales nuevos aprobados,
y dos candidatos reales 10+10 terminados. El VJP mejora en JToK y retrocede en
JToK-M; la ganancia del paso completo no está confirmada como estable.
Se apila sobre el compacto de Leviathan publicado en `d44c793`. No sustituye
las metas de 4.5 pasos/s en JToK ni 4.2 en JToK-M.

## Evidencia que motiva el cambio

En las corridas reales anteriores, el VJP vectorizado de spline ocupó cerca de
0.82 ms por llamada en JToK y 1.49 ms en JToK-M. Cada resumen contiene 24 eventos
de dos pasos perfilados, no 24 pasos independientes. La especialización utiliza
48 registros/hilo y 8 KiB de shared. Se mantienen privados los rastros completos.

El kernel calcula la base y su derivada normalizada para los 16 nudos. Su
contracción ya reutiliza la geometría entre modos; esa reutilización no es nueva.
El gradiente de cada coeficiente es `bar_C_q = bar_Phi * b_q`, con:

```text
Z = sum_q w_q; b_q = w_q / max(Z, EPS_basis)
Phi = sum_q b_q C_q
bar_Phi = bar_mode * mode * sign_policy(Phi) / (abs(Phi) + EPS_product)
bar_z = bar_z_residual + sum_m bar_Phi_m * Phi'_m
```

Para upstream finito, `b_q=0` aporta exactamente cero a `bar_C_q`. Las filas
inválidas también tienen upstream cero. Omitir esas actualizaciones no modifica
la fórmula, el cálculo de `Phi'`, EPS, signos ni pesos seleccionados del router.
Se conserva todo peso diminuto que no sea exactamente cero: no se introduce
un umbral de poda. No se afirma equivalencia general ante entradas no finitas.

## Implementación y reutilización

`JTOK_SPARSE_COEFF_UPDATES=1` añade una máscara al kernel vectorizado existente.
La condición del atomic incluye fila válida y `basis_raw != 0`. La base densa,
normalización, carga de coeficientes y ownership de reducciones no cambian.
La misma implementación opera con un experto o con las rutas seleccionadas
de JToK-M. La especialización y clave del autotuner incluyen la opción; no se
debe reutilizar una elección de recursos para otra variante sin medirla.

El default permanece desactivado. Solicitarlo fuera de la ruta wide/vectorizada
provoca un error explícito, no un éxito silencioso que hubiera ejecutado otra
ruta. JToK apagado no ejecuta esta lógica ni reserva buffers nuevos.

Este paso aísla el efecto de las atomics. La posible contracción de soporte
compacto queda como etapa siguiente: el layout `[experto,modo,coordenada,nudo]`
tiene nudo contiguo, distinto del rango contiguo de Leviathan. Tres cargas de
gather pueden tocar los mismos sectores que una carga densa de 16 nudos.
Por eso tres pesos matemáticos no implican `16/3` de ahorro de memoria o speedup.

Las atomics restantes aún acumulan FP32 en orden dependiente del scheduling.
Omitir sumandos cero conserva la matemática, pero no demuestra determinismo
bit a bit del resto de la reducción ni elimina sus colisiones por experto.

## Verificación realizada

Los casos sintéticos nuevos son sólo funcionales: bordes FP32/BF16 almacenados,
coeficientes positivos/con signo/cero, expertos repetidos y filas inválidas.
Se compara contra la variante densa y contra un VJP normalizado independiente.
El backward registrado compara todos los gradientes, incluido routing en mezcla.
La ejecución remota pasó dos contratos CPU (default y manifiesto nativo) y siete
casos GPU: cuatro VJP/oracle de bordes, dos backwards registrados plain/mixture
y un rechazo explícito de configuración no cubierta. Se conservaron ambos
perfiles. Triton emitió avisos AST de deprecación para una versión futura de
Python; no fueron fallos del runtime actual. La configuración pytest existente
se reutiliza para registrar sus marcas; no se repite el suite por un aviso.

La aceptación de rendimiento requiere entrenamiento real perfilado 10+10,
misma política y datos ya presentes, sin descarga. Se reutilizan los controles
anteriores del compacto Leviathan porque el único cambio GPU de este candidato
es la máscara de atomics; no se vuelven a ejecutar resultados idénticos conocidos.
Los manifiestos originales y el nuevo manifiesto con `jtok.py` deben distinguirse.

Reproducción desde un checkout completo, en el entorno CUDA de pruebas:

```bash
python benchmark/leviathan_candidate_provenance.py --source-root . \
  --include-jtok --output .codex-tmp/jtok_sparse_manifest.json
python benchmark/profile_leviathan_compact_tests.py \
  --test-file test_jtok_sparse_updates.py --test-name sparse \
  --profile .codex-tmp/jtok_sparse_regressions.trace.json
```

Para la corrida real se usa `remote_real_10x10_validation.py` con los argumentos
de datos preexistentes y entrenamiento privado, `--candidate-manifest` y
`--profile`. Se conserva `LEV_COMPACT_SPLINE=1`, BM64, BD1, BR64, split-N4,
cadena fusionada y política IEEE. Sólo se activa la nueva opción nativa.
Los artefactos privados contienen precisión y memoria; los públicos sólo tiempos
completos y agregados de los kernels investigados. No se promociona el default
antes de medir, ni se considera una ganancia de microkernel como cumplimiento
del gate completo.

## Primer resultado real: JToK

El candidato completó 10+10 con backend estricto y los mismos datos/semillas,
fuentes privadas y runtime que el control compacto anterior. Mediana compute
266.663→266.239 ms: 3.750050→3.756028 pasos/s, +0.1594%. El p95 pasó de
267.645 a 266.936 ms y la desviación de 1.981 a 2.232 ms. E2E cambió de
276.853 a 276.438 ms, +0.1502% de throughput. No alcanza 4.5 pasos/s.
La diferencia es pequeña frente a la dispersión; se registra como ganancia
observada corta, no mejora estable confirmada.

El VJP investigado tiene mediana 0.817119→0.775329 ms, aproximadamente -5.11%,
y suma 19.861574→18.625536 ms en los 24 eventos de dos pasos perfilados.
La suma ahorra cerca de 0.618 ms por paso perfilado; la mediana del paso completo
estable ahorra 0.424 ms. Son poblaciones diferentes, no una identidad causal.
El `dDelta` compacto de Leviathan se mantiene alrededor de 8.03–8.04 ms.

Ambos VJP usan 256 hilos por programa: los recursos son 48→70 registros/hilo
y 8192→512 bytes de shared. La configuración de ocho warps/una etapa permanece
igual según el conjunto existente y los bloques observados. Cambió la
especialización compilada de la máscara, no se demuestra un cambio de elección
de warps del autotuner. El incremento de registros podría perjudicar ocupación;
el menor shared puede favorecerla. No se infiere ocupación efectiva sin medirla.

La matemática sparse no elimina las bases, las cargas densas ni las contracciones
del VJP: sólo elimina actualizaciones cero. Además, éste representa una fracción
pequeña del paso completo. Éstas son razones para esperar una ganancia moderada,
no una prueba de que toda la diferencia de tiempo tenga una sola causa.
Los picos privados de memoria no aumentaron y las pérdidas fueron finitas;
diferencias de precisión del modelo quedan en el registro privado.

## Resultado real: JToK-M y decisión conjunta

La comparación dentro de JToK-M también confirmó fuentes privadas, runtime,
semillas y datos idénticos; 10 pasos de optimizador y 10 validaciones completos,
sin fallback. Mediana compute 310.466→310.148 ms: 3.220963→3.224270 pasos/s,
+0.1027%. E2E 320.492→320.389 ms, +0.0322%. No alcanza 4.2 pasos/s.
El p95 compute cambia 312.204→312.567 ms y la desviación 1.798→2.793 ms.
La diferencia de medianas de 0.318 ms es menor que esta dispersión; no demuestra
un incremento estable del throughput ni permite atribuirlo al VJP modificado.

El VJP investigado empeora: mediana 1.490656→1.514737 ms, +1.6155%; suma
35.802219→36.380130 ms en 24 eventos de dos pasos perfilados. Conserva el
mismo cambio de recursos observado en JToK: 48→70 registros/hilo, 8 KiB→512 B
de shared, 256 hilos/programa. La matemática y el código compartidos no implican
la misma distribución de colisiones, scheduling o sensibilidad a registros.
El pico allocated privado no aumentó; las pérdidas fueron finitas. Esto valida
funcionamiento corto, no convergencia ni estabilidad de larga duración.

| Evidencia | JToK | JToK-M |
| --- | --- | --- |
| Mediana VJP, control→sparse | 0.817119→0.775329 ms | 1.490656→1.514737 ms |
| Cambio de latencia VJP | −5.1143% | +1.6155% |
| Pasos/s compute, control→sparse | 3.750050→3.756028 | 3.220963→3.224270 |
| Ganancia observada del paso | +0.1594% | +0.1027% |
| Gate alcanzado | No: 4.5 requerido | No: 4.2 requerido |

Decisión: mantener la opción explícita para investigación, no promocionar el
default ni activarla automáticamente en mezcla. La regresión de JToK-M queda
versionada junto al resultado favorable de JToK. El control recomendado para
la siguiente hipótesis sigue siendo el compacto Leviathan con esta opción en 0.
No se repiten estos controles ya registrados para ocultar una diferencia pequeña.

## Por qué no escala igual entre niveles

La contribución cero que se elimina es una identidad algebraica; el ahorro
físico depende del kernel compilado. La máscara cuesta predicados y aumenta
la presión de registros. Reducir shared puede ayudar, pero no basta para
deducir ocupación efectiva; tampoco un conteo de atomics revela por sí solo
sectores de caché, tráfico DRAM, contención real o spills. Esas causas concretas
son hipótesis pendientes de contadores físicos, no conclusiones del perfil actual.

Una estimación de dilución usa `T = T_otro + T_VJP`. Si el VJP ocupa fracción
`f` y se reduce su latencia en fracción `s`, el speedup ideal del paso sería
`1 / (1 - f*s)`, manteniendo todo lo demás constante. En JToK, el total VJP
perfilado equivale a aproximadamente 9.93 ms por paso frente a 266.66 ms
compute estable: cerca del 3.7%. Una mejora local del 5% sólo puede producir
del orden del 0.2% global bajo esas hipótesis. No es una predicción exacta:
perfilado y gate usan poblaciones distintas y el profiler perturba la ejecución.

En JToK-M la misma máscara tiene regresión local: una ligera ganancia global
no la convierte en una mejora causal de ese kernel. Éste es el motivo para
comparar cada nivel dentro de su propio control y conservar ambos resultados.
El cambio compacto anterior de Leviathan sí redujo trabajo en un consumidor
mayor, `dDelta`; sus cargas/layout y su peso en el paso no son los de este VJP.

## Artefactos y reproducción sin repetir entrenamiento

Antes de agregar los eventos se deduplicaron las anotaciones anidadas del
profiler: dos pasos de optimizador, 12 VJP por paso, sin eventos omitidos ni
contados dos veces. El primer paso activo está fuertemente perturbado por el
arranque de captura; no se usa su tiempo completo como gate. En el segundo
paso activo, la mediana VJP JToK es 0.816575→0.769410 ms (−5.7760%) y la de
JToK-M 1.497823→1.510242 ms (+0.8291%). El signo local no desaparece al separar
esa fase, pero dos pasos perfilados no prueban estabilidad de larga duración.

En ese segundo paso JToK, compute perfilado cambia 280.932→281.634 ms a pesar
del VJP más rápido. Esto refuerza que el tiempo total incluye costes ajenos a
esa máscara, perturbación de captura y variación. Los resúmenes incluyen este
diagnóstico por paso, separado de los seis pasos estables 5–10. No se asigna
todo el residual de tiempos a sincronizaciones o a registros sin evidencia.

Los resúmenes públicos están en
`benchmark-results/leviathan-compact-20261006/jtok_sparse_vjp_real_10x10.json` y
`benchmark-results/leviathan-compact-20261006/jtokm_sparse_vjp_real_10x10.json`.
Contienen únicamente tiempos completos y agregados de kernels investigados.
Los originales Chrome Trace, pérdidas, memoria y huellas privadas permanecen
en `.codex-tmp`, fuera de publicación. Hay un rastro reducido del VJP de cada
brazo para visualizarlo localmente; sus huecos no significan GPU ociosa.

Para analizar un rastro existente, sin CUDA ni entrenamiento nuevo:

```bash
python benchmark/analyze_leviathan_kernel_trace.py \
  .codex-tmp/remote-compact-20261006/real_jtokm_compact_sparsev1.trace.json \
  --target _jtok_backward_token_projection_grad_block_kernel \
  --kernel-trace .codex-tmp/remote-compact-20261006/real_jtokm_compact_sparsev1_vjp.trace.json
```

El comando funcional anterior cubre siete casos GPU y un contrato CPU de la
opción. El noveno caso es la procedencia del manifiesto nativo, ejecutable con
el mismo wrapper, `--cpu-only --test-file test_leviathan_candidate_provenance.py
--test-name native_jtok`. El test adicional del exportador reducido se ejecutó
por separado; no se cuenta como prueba numérica del kernel.

Para repetir una corrida real sólo cuando haga falta evidencia nueva, definir
localmente `PRIVATE_TRAIN_SCRIPT`, `REAL_TRAIN_SPLIT` y `REAL_VALIDATION_SPLIT`
con los archivos existentes; no descargar datos. Usar salidas nuevas, conservar
las huellas privadas y no sustituir un control registrado por otra configuración.
Desde el checkout que contiene el candidato completo:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
WANDB_MODE=disabled LEV_COMPACT_SPLINE=1 LEV_DDELTA_SPLITS=4 \
LEV_DDELTA_BM=64 LEV_DDELTA_BD=1 LEV_DDELTA_BR=64 \
LEV_FUSE_CHAIN_DDELTA=1 LEV_DOT_IEEE=1 JTOK_SPARSE_COEFF_UPDATES=1 \
python benchmark/remote_real_10x10_validation.py \
  --mode jtokm --batch-size 64 --train-script "$PRIVATE_TRAIN_SCRIPT" \
  --train-data-dir "$REAL_TRAIN_SPLIT" \
  --validation-data-dir "$REAL_VALIDATION_SPLIT" \
  --candidate-manifest .codex-tmp/jtok_sparse_manifest.json \
  --output .codex-tmp/new_jtokm_sparse_real.json \
  --profile .codex-tmp/new_jtokm_sparse_real.trace.json
```

Para JToK, cambiar únicamente el modo y los nombres de salida. La política del
entrenamiento original continúa congelada: compile/max-autotune, caches,
optimizador, dtype y pérdidas no se cambian para favorecer este candidato.
