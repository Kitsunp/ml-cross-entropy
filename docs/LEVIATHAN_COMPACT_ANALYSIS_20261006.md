# Leviathan: recapitulación y análisis del candidato compacto

Fecha de corte: 6 de octubre de 2026. Estado: candidato experimental opt-in;
las parejas reales cortas observan +2.29% de throughput en Leviathan, +0.98%
en JToK y +1.28% en JToK-M. No se ha demostrado estabilidad prolongada ni
aceleración de los consumidores nativos JToK/JToK-M por soporte compacto.
Este documento publica únicamente información del kernel investigado y tiempos
totales del paso. No contiene fuentes privadas, sus huellas, rutas personales,
datos de conexión ni perfiles completos.

## Conclusión ejecutiva

La propuesta reorganiza el método cuadrático que ya existe. No añade parámetros,
no cambia la arquitectura y no introduce otro objetivo de entrenamiento.
Combina soporte local de la spline, una contracción de su derivada por diferencias
y cambios de planificación para acortar la vida de los temporales. Conserva el
último cambio de `dDelta`: reducción split-N determinista y su reducer final.

Los controles locales y las nuevas comprobaciones remotas avalan paridad numérica
dentro de las tolerancias declaradas. Una compilación local del backward fusionado compacto
pasó de 255 a 243 registros por hilo al cambiar la planificación, con el mismo
tile compacto. Frente al control denso, la comparación final emplea también
otro tile; no permite atribuir todo el cambio de recursos a la fórmula.
Ni esos recursos ni las pruebas sintéticas demuestran mayor throughput real.
La nueva pareja real completa diez pasos de entrenamiento y diez batches de
validación por brazo: 4.513453 frente a 4.616969 pasos/s. Se separan compilación,
profiler activo y seis pasos estables; la evidencia y sus límites figuran abajo.

La geometría está separada del layout de coeficientes para reutilizarla en
JToK/JToK-M. Sus consumidores nativos todavía no se han portado a ella.

## 1. Qué se conserva y qué cambia

La línea histórica parte de `ml-cross-entropy-jtok-pr`, commit `5a4072d`.
El HEAD local de esta revisión es `2252907`, que incorpora split-N de `dDelta`;
el candidato incluye las modificaciones adicionales de esta publicación.
No debe identificarse el candidato exclusivamente por ese HEAD: las ejecuciones
exigen un manifiesto de las ocho fuentes públicas afectadas.

| Área | Antes | Candidato |
| --- | --- | --- |
| Interpolación cuadrática | Evaluación densa de los nudos | Soporte local y halo de precisión flotante |
| VJP de coordenadas | Intermediarios de la base y su derivada | Contracción directa con diferencias de coeficientes |
| Cargas de coeficientes | Expresión densa | Cargas bidimensionales; vecinos en bucle no desenrollado |
| Temporales del backward | Algunas cargas adelantadas | Cargas diferidas para reducir solapamiento |
| `dDelta` | Producto/reducción split-N | Misma reducción; reconstrucción de pesos en registros |
| Selección de implementación | Ruta densa | `LEV_COMPACT_SPLINE=1`, guardas estrictas y modo guardado |
| Configuración global | Política existente | Sin cambios solicitados de modelo, dtype, pérdida u optimizer |

La ruta densa sigue siendo la predeterminada. El soporte compacto se limita a
la configuración validada: spline de grado 2, rejilla canónica unitaria,
`d_seed=128`, 16 nudos y rango 64. Forzar la ruta dot no amplía estas condiciones.
La hipótesis de coordenadas en `[0,1]` procede de la transformación existente.

## 2. Derivación matemática

Sea `w_q(t)` el peso cuadrático sin normalizar, `Z(t)=sum_q w_q(t)` y
`b_q(t)=w_q(t)/Z(t)`. En el dominio soportado `Z` es positivo; se conserva el
clamp numérico de la implementación. La derivada de la base normalizada es:

```text
b'_q = (w'_q - b_q Z') / Z
sum_q b_q = 1
sum_q b'_q = 0
```

No basta con utilizar `w'_q` como derivada de `b_q`: la normalización importa,
especialmente cerca de los extremos. Ese error se corrigió también en la
referencia usada por los tests.

Para un coeficiente `S_qr = 1 + Delta_qr`, la interpolación por rango es
`Phi_r = sum_q b_q S_qr`. Tomando un nudo activo `a` como ancla:

```text
dPhi_r/dt = sum_(q != a) b'_q (Delta_qr - Delta_ar)
bar_t = sum_r bar_Phi_r * dPhi_r/dt
bar_Delta_qr = sum_tokens b_q * bar_Phi_r
```

La primera identidad cancela el desplazamiento constante `1` y evita formar
un intermediario completo para `dB`. Con coeficientes constantes, la derivada
es cero. La segunda conserva el VJP de coordenadas; la tercera conserva
`dDelta = B^T bar_Phi`, implementado mediante el producto existente y split-N.

La equivalencia es algebraica en aritmética real, no identidad bit a bit en
FP32/BF16: cambian asociaciones y redondeos. Se verifican tolerancias, valores
finitos, bordes y el gradiente compartido, no una supuesta igualdad exacta general.

### Tres nudos matemáticos; cuatro posibles contribuciones flotantes

Una spline cuadrática tiene hasta tres pesos no nulos en una rejilla exacta.
La rejilla FP32 almacenada puede introducir un cuarto peso o una cuarta
derivada muy pequeños en una frontera. Suprimirlos produjo una discrepancia
reproducible. La geometría actual lee la rejilla real, corrige el inicio del
soporte y conserva ese halo en la normalización y en la contracción.

El cuarto coeficiente se carga sólo si el tile contiene contribución de halo.
Ese control exige predicados y una reducción; su coste tampoco es gratuito.
No se ha implementado ni se presupone una compresión BF16 sin pérdida.

### Rejilla almacenada BF16: corrección necesaria para entrenamiento real

El entrenamiento convierte el buffer inicial FP32 a BF16. Promoverlo después
a FP32 conserva el redondeo BF16; no recupera la `linspace` inicial. El primer
candidato real fue rechazado antes de entrenar porque la guarda exigía esa
`linspace` FP32 nueva. Se conserva como fallo reproducible, no como benchmark.

La guarda corregida acepta únicamente representaciones canónicas exactas:
`linspace` FP32, su conversión a BF16/FP16 y su promoción posterior, además de
la representación FP64 original de la referencia. No utiliza `allclose` para
aceptar rejillas arbitrarias, y el dispositivo sigue leyendo los nudos guardados.

Para K=16, el error escalado de redondear a BF16 está acotado por aproximadamente
`15/512 = 0.02930`, más el pequeño redondeo inicial FP32; es menor que 1/2.
Si `L=floor(15*t-0.5)` y `f=15*t-0.5-L`, el halo izquierdo sólo puede activarse
cuando `f < delta`; el derecho cuando `f > 1-delta`. Con `delta < 1/2` no pueden
activarse ambos simultáneamente. Corregir L si aparece el vecino anterior y
conservar cuatro slots cubre el soporte; los nudos más lejanos quedan fuera.
La referencia rechaza representaciones cuantizadas cuyo límite conservador de
soporte no se ha probado; la producción continúa restringida a K=16.

La validación se cachea sobre el buffer congelado original, no sobre una nueva
copia FP32 por llamada. Así la promoción de BF16 no causa otra comprobación
con sincronización en cada paso. Los checkpoints directos guardan esa identidad;
los checkpoints compilados ya conservaban el buffer original.

## 3. Ingeniería y costes que deben medirse

`spline_support_kernels.py` separa geometría de contracción. Leviathan consume
inmediatamente los pesos en registros. La contracción evita un tensor lógico
`[tokens, soporte, rango]`: mantiene el ancla y procesa un vecino por iteración.
El backward compacto con `BD=1` difiere la carga de la proyección de salida;
la ruta no fusionada difiere los temporales de LayerNorm hasta necesitarlos.

El cambio intenta reducir cálculo redundante y presión de temporales, pero
también introduce gathers dependientes de la coordenada y control del halo.
Los coeficientes densos pueden reutilizarse entre muchos tokens: pasar de 16
nudos a tres no demuestra una reducción física de tráfico de `16/3`, ni un
speedup equivalente. Caché, coalescencia y reutilización pueden cambiar el balance.

Se conserva la reducción matricial de `dDelta`; no se la denomina
incondicionalmente Tensor Core, porque la política IEEE FP32 también determina
su implementación. Los metadatos de registros y shared memory no bastan para
inferir ocupación efectiva ni ausencia de spills. El backward no fusionado
mantiene otro kernel de cadena con alta presión de registros.

No se añaden buffers globales densos para la base compacta. El cache de
validación de rejilla evita comprobaciones repetidas con sincronización del host,
retiene la identidad del almacenamiento e invalida modificaciones normales.
No promete detectar escrituras mediante `.data` o punteros externos. Una rejilla
desconocida durante captura se rechaza. El modo guardado en forward determina
backward aunque cambie después la variable de entorno; no se acepta fallback
silencioso para un forward compacto.

## 4. Evidencia local conservada

Resultados funcionales sintéticos, no validación de preentrenamiento.
Entorno local registrado: RTX 5090, Torch `2.13.0+cu130`, Triton `3.7.1`.
Todos los experimentos nuevos de esta línea requieren perfilamiento.

### Paridad del último probe local registrado

Probe de 257 tokens, BF16, compacto `BM=64`, split-N `S=4`:

| Comprobación | Error relativo L2 observado | Criterio |
| --- | ---: | ---: |
| Salida compacta frente a densa | 2.45780e-4 | <= 3e-4 |
| Mayor error entre gradientes | 3.71055e-4 | <= 1e-3 |
| Gradiente de `Delta` | 1.07714e-4 | <= 1e-3 |
| Mayor error fusionado frente a no fusionado | 1.90439e-5 | <= 1e-3 |
| Gradiente compartido de seed, oracle FP32 | 0 | <= 1e-3 |

No se observaron valores no finitos en esos casos. El test anterior de `S=4`
frente a `S=1` pasó antes del último ajuste de planificación; no se presenta como
una verificación nueva de la versión final. Se incluye en el suite remoto nuevo.

### Recursos de las especializaciones compiladas de ese probe

| Kernel investigado | Tile M | Registros/hilo | Shared memory |
| --- | ---: | ---: | ---: |
| Forward denso | — | 176 | 40 KiB |
| Forward compacto | — | 174 | 40 KiB |
| Backward denso fusionado de `dDelta` | 128 | 255 | 44 KiB |
| Backward compacto fusionado de `dDelta` | 64 | 243 | 20 KiB |
| Backward compacto no fusionado de `dDelta` | 64 | 165 | 20 KiB |
| Cadena compacta no fusionada | — | 255 | 512 bytes |

Son recursos de especializaciones concretas, no una medida universal del modelo.
El compacto con `BM=64` tenía 255 registros antes del último cambio de
planificación y 243 después: esa comparación mantiene el tile.
La comparación denso/compacto cambia también `BM=128` a `BM=64`.
El candidato no ha promovido automáticamente ese tile a la política predeterminada.

## 5. Fallos y controles de regresión

| Hallazgo | Causa y corrección | Reproducción conservada |
| --- | --- | --- |
| Compilación de potencia | `**` no admitido en aquella especialización Triton; multiplicación explícita | Tests de contracción del dispositivo |
| Discrepancia en frontera, aproximadamente 7.15e-6 | Halo de la rejilla FP32 omitido; se preserva | Semipuntos, extremos y `nextafter` |
| Referencia FP64 incompatible en forma | Broadcasting de la expectativa; shape corregido | Referencia independiente de soporte |
| Error de seed aproximadamente 3.65e-3 | Oracle acumulaba en BF16; producción acumula FP32 y castea una vez | Oracle FP32 y test compilado de seed |
| Corrida antigua etiquetada como compacta cargó código distinto | No ejecutó el candidato; se excluye | Manifiesto obligatorio y verificación antes/después |
| Posible cambio de modo entre forward/backward | No debe reinterpretar el checkpoint | Test compilado con cambio de variable tras forward |
| Callback de tiempos falló tras el primer paso real | `TrainingArguments` ocultaba la ruta del runner; se captura la ruta externa | Test CPU del callback real extraído por AST |
| Candidato real rechazó la rejilla antes de entrenar | FP32→BF16→FP32 no equivale a una `linspace` nueva; guarda canónica exacta y caché del buffer original | Tests CPU de soporte cuantizado, contracción GPU y seed compilado BF16 |
| Override dot aceptaba arquitecturas fuera del contrato compacto | La guarda añadida valida capacidad también con `LEV_DOT=1`; no cambia el kernel GPU | Mock CPU SM80/SM90 falla antes y pasa después; SM120 permitido |

Hubo 17 tests focalizados distintos con resultado satisfactorio en la etapa
local intermedia. Esa cifra no certifica por sí sola cada edición posterior.
La validación remota comprobó los contratos en un runtime distinto, no repitió
una corrida idéntica conocida. Las correcciones posteriores se verificaron
con selecciones nuevas y focalizadas, conservando los resultados previos.

## 6. Resultados reales históricos: contexto, no prueba del candidato

Estos registros corresponden a la implementación anterior, con el protocolo
perfilado de diez pasos de entrenamiento y diez batches de validación.
Se publica exclusivamente la mediana del paso completo después del primero:

| Variante histórica | ms/paso completo | pasos/s | Meta vigente |
| --- | ---: | ---: | --- |
| Leviathan, split-N 4 | 203.407 | 4.916 | Control, sin gate de 4.5 |
| Leviathan JToK, split-N 4 | 252.006 | 3.968 | >= 4.5; no alcanzada |
| Leviathan JToK-M, split-N 4 | 292.223 | 3.422 | >= 4.2; no alcanzada |

El cambio previo split-N redujo la mediana observada de `dDelta` de 12.086 ms
a 10.949 ms, aproximadamente 9.4%, con sólo dos eventos medidos por corrida.
Es evidencia limitada del cambio previo, no del soporte compacto.
No se mezclan cifras sin profiler con estas cifras.

La corrida antigua etiquetada como compacta de aproximadamente 204.931 ms/paso
queda excluida de las conclusiones del candidato por procedencia incorrecta.

Las metas actuales equivalen a <= 222.222 ms/paso para JToK y <= 238.095 ms/paso
para JToK-M. Respecto a aquellos registros requerirían ahorrar aproximadamente
29.8 ms y 54.1 ms por paso, respectivamente. Esas diferencias superan el tiempo
histórico completo de `dDelta`: optimizar sólo ese kernel no garantiza las metas.
Es un límite de atribución, no una predicción para el entorno nuevo.

## 7. Extensión a JToK y JToK-M

La pieza compartible ahora es la geometría: índices activos, pesos normalizados
y derivadas normalizadas, incluido el halo. La contracción de Leviathan sigue
dependiendo de su layout y no es un reemplazo directo del consumidor JToK.

En JToK, un consumidor debe usar sus strides de modo/coordenada/nudo y su propia
contracción. En JToK-M debe añadirse además el índice y máscara de experto.
La intención es mantener un núcleo común de geometría, contracción de modos y
VJP. Sólo selección, pesos de mezcla, máscaras por experto y sus reducciones
específicas pertenecen al adaptador JToK-M. Un mismo cambio del núcleo deberá
probarse con y sin mezcla, evitando dos formulaciones que diverjan sin necesidad.
Si varios consumidores reciben de verdad la misma coordenada desde un nodo
compartido, autograd ya suma sus VJP en ese nodo; no se adjudica al candidato
la eliminación de múltiples backward del bridge sin demostrar que existían.
El beneficio potencial reutilizable es evitar reconstruir bases y derivadas,
no cambiar automáticamente la estructura de autograd.

Antes de portar: conservar layouts/dtypes y despacho, probar acumulación de seed
y expertos, evitar nuevas atomics no deterministas y validar cada variante en
entrenamiento real. Aún no hay resultados que permitan declarar aceleración de
sus kernels nativos por esta geometría.

Para coeficientes nativos `C`, distintos de `1+Delta`, se conserva:

```text
Phi_m = sum_q b_q C_mq
dPhi_m/dz = sum_(q != a) b'_q (C_mq - C_ma)
bar_Phi_m = bar_mode_m * mode_m * sign(Phi_m) / (abs(Phi_m) + EPS)
bar_C_mq = bar_Phi_m * b_q
bar_z = bar_z_residual + sum_m bar_Phi_m * dPhi_m/dz
```

El cociente de producto por `Phi` no es un sustituto válido de esta derivada:
perdería EPS y la política en cero. La anulación por diferencias exige la
derivada de la base normalizada, no sólo la derivada del peso sin normalizar.
El layout nativo es `[experto, modo, coordenada, nudo]`, nudo contiguo; el de
Leviathan es rango contiguo. Reutilizar la geometría no exige imponer el mismo
patrón de carga ni la misma organización de programas.

El perfil nuevo del control JToK contiene, dentro de los kernels investigados,
24 eventos del evaluador vectorizado de modos: mediana 0.961088 ms, suma 23.059646 ms;
24 del VJP de spline por bloque: mediana 0.820160 ms, suma 19.875007 ms. Son totales
del intervalo perfilado, no un tiempo por paso: no se mezclan con la población
estable. El evaluador usa 23 registros/hilo y 16 bytes de shared; el VJP, 48 y
8 KiB. La mejora siguiente debe atender también paralelismo, cargas/atomics y
latencia, no asumir que estos kernels tienen la misma saturación de registros
que el backward de Leviathan.

Una hipótesis pequeña es enmascarar atomics cuyo peso de base sea exactamente
cero o cuya fila sea inválida, porque su VJP finito aporta cero. La hipótesis
más amplia usa soporte local para la contracción y sus derivadas, manteniendo
el reparto existente por token/ruta y la reutilización entre modos. Ninguna
está implementada ni aceptada aún; sus tests y medición real son obligatorios.

## 8. Comparación remota actual y reproducción

El entorno nuevo dispone de RTX 5090, Torch `2.14.1+cu132`, Triton `3.8.0` y
Transformers `5.19.0`. Las fuentes privadas también difieren de las históricas:
es obligatorio obtener un control nuevo. Sus huellas se mantienen sólo en los
artefactos privados. Se localizaron y validaron los splits reales existentes
antes de lanzar el control; no se descargó ningún dataset ni se sustituyó el
gate por sintéticos.

Se preparó una copia aislada de CCE instalado y se superpusieron únicamente las
ocho fuentes públicas del candidato; el resto del paquete remoto permanece
igual para ambos brazos. La instalación activa y el entrenamiento privado no
se sobrescriben. Se verifica el manifiesto de lo realmente importado.

### Resultados remotos nuevos del 6 de octubre

Se completaron 17 tests perfilados, con 12 tests fuera de la selección focalizada.
El probe de 257 y 513 tokens terminó correctamente, con fuentes públicas
verificadas antes y después. Ambos perfiles se descargaron para análisis local;
no se descargaron datos. Son controles funcionales sintéticos, no entrenamiento.

| Tokens | Salida relativa L2 | Mayor gradiente relativo L2 | `Delta`, split 4 vs 1 | Seed relativo L2 |
| --- | ---: | ---: | ---: | ---: |
| 257 | 1.80338e-4 | 3.40619e-4 | 3.47410e-7 | 4.99371e-5 |
| 513 | 1.22848e-4 | 3.36637e-4 | 2.54418e-5 | 0 |

La mayor discrepancia de gradiente fusionado/no fusionado fue 2.31018e-5.
Todos los valores comprobados fueron finitos y las tolerancias se cumplieron.
El seed de 257 tokens no fue bit a bit idéntico al oracle: se informa el error
observado, sin trasladar la igualdad exacta de otro caso a éste.

El perfil nuevo mostró las siguientes especializaciones:

| Kernel investigado | Registros/hilo | Shared memory |
| --- | ---: | ---: |
| Forward denso | 170 | 40 KiB |
| Forward compacto | 121 | 40 KiB |
| Backward fusionado denso, `BM=128` | 255 | 44 KiB |
| Backward fusionado compacto, `BM=64` | 227 | 20 KiB |
| Backward no fusionado compacto de `dDelta` | 157 | 20 KiB |
| Cadena compacta no fusionada | 255 | 1 KiB |

La diferencia respecto a la compilación local confirma que los recursos
dependen del compilador/runtime. No demuestra menor duración del paso completo,
ni se utiliza para declarar que ya se cumplieron las metas JToK/JToK-M.
Triton emitió ocho avisos de deprecación de su generador AST para una versión
futura de Python; no hubo errores en el runtime empleado.

### Nuevos casos funcionales de rejilla cuantizada

Tras la corrección pasaron siete casos nuevos GPU: seis contracciones con
rejillas BF16/FP16 y coeficientes aleatorios, constantes o con signo; y el
backward compilado de seed compartido con buffer BF16. Tres casos CPU focalizados
comprobaron la caché y los límites conservadores. Los casos FP32 ya conocidos
no se repitieron. Los fallos previos y posteriores conservan sus perfiles.

El probe de paridad se extendió con `--grid-storage bf16`, usando los valores
almacenados y verificando salida y todos los VJP de parámetros contra la ruta
densa. Para 257 tokens: salida relativa L2 1.38376e-4 y mayor gradiente 2.28652e-4;
para 513: 9.77026e-5 y 2.74682e-4. Todo fue finito, dentro de 3e-4 para salida y
1e-3 para gradientes. Son datos sintéticos de corrección, no evidencia de velocidad.

### Pareja real Leviathan `10+10`

Ambos brazos usaron la misma semilla efectiva 42 y data seed 42, fuentes privadas,
runtime, muestras, batch, secuencia, optimizador y protocolo. El control corresponde
a la revisión anterior de la guarda; la corrección cambia sólo el camino compacto
y la identidad del buffer guardado, sin editar el código GPU denso. Se conservan
ambos manifiestos y no se duplicó el control sólo por ese cambio host opt-in.

El paso 1 contiene compilación fría; el 2 es warmup; el profiler está activo en
3–4. La población primaria estable contiene exactamente los pasos 5–10, seis
observaciones. Los tiempos siguientes incluyen forward, pérdida, backward y
optimizer; E2E añade preparación/entrada del batch y se informa por separado.

| Medida del paso completo | Control denso | Compacto corregido |
| --- | ---: | ---: |
| Mediana compute | 221.560 ms | 216.592 ms |
| Throughput compute | 4.513453 pasos/s | 4.616969 pasos/s |
| p95 compute | 223.341 ms | 218.222 ms |
| Desviación poblacional compute | 2.037 ms | 2.018 ms |
| Mediana E2E | 231.730 ms | 226.641 ms |
| Throughput E2E | 4.315360 pasos/s | 4.412257 pasos/s |
| p95 E2E | 233.455 ms | 228.551 ms |
| Desviación poblacional E2E | 4.959 ms | 4.501 ms |
| Primer paso compute, compilación incluida | 379.064 s | 353.959 s |

El throughput compute observado aumenta 2.2935%; la latencia mediana baja 2.2421%.
E2E mejora 2.2454% en throughput. No se combinan muestras de compute con E2E ni
con pasos perturbados por el profiler. La compilación fría tampoco es el gate.
Los dos brazos terminaron los diez optimizer steps y diez validaciones reales,
con backend estricto. Pérdidas, sus diferencias y memoria del modelo quedan
registradas en el análisis privado, sin publicar componentes ajenos al kernel.

### Atribución a los kernels investigados del mismo entrenamiento

Se cuentan exclusivamente eventos GPU `cat=kernel`, no rangos CPU del mismo
nombre. Cada mediana siguiente se apoya sólo en dos eventos del profiler.

| Kernel | Control | Compacto | Registros/hilo control→compacto | Shared control→compacto |
| --- | ---: | ---: | ---: | ---: |
| `_lev_fused_dot` | 3.383437 ms | 3.202176 ms | 128→124 | 40→40 KiB |
| `_lev_bwd_ddelta_dot_kernel`, fusionado | 11.420582 ms | 8.026531 ms | 255→227 | 44→20 KiB |
| `_lev_bwd_ddelta_split_reduce_kernel` | 11.696 µs | 11.312 µs | 40→40 | 0→0 |

El backward investigado reduce aproximadamente 29.72% su duración; el forward
aproximadamente 5.36%. El reducer sigue esencialmente igual. La suma del ahorro
de estas medianas es aproximadamente 3.576 ms, mientras que el ahorro del paso
compute estable es 4.968 ms: no se adjudica toda la diferencia a estos kernels,
porque son poblaciones distintas y hay perturbación/variabilidad del profiler.
Menos registros y shared respaldan la hipótesis de planificación; no demuestran
por sí solos ocupación efectiva, spills ni tráfico físico de memoria.

Este resultado mide conjuntamente soporte compacto, VJP por diferencias y
tile BM=64. No permite identificar la contribución aislada de cada cambio.
Una sola pareja corta no satisface la verificación de estabilidad prolongada;
no se promociona todavía como default ni se declara que JToK alcance su gate.

Los scripts de reproducción del repositorio son:

```bash
python benchmark/leviathan_candidate_provenance.py \
  --source-root . --output .codex-tmp/candidate_manifest.json

python benchmark/profile_leviathan_compact_tests.py \
  --profile .codex-tmp/regressions.trace.json

python benchmark/leviathan_compact_correctness.py \
  --tokens 257 513 --block-m 64 --checks parity splits unfused seed \
  --candidate-manifest .codex-tmp/candidate_manifest.json \
  --profile .codex-tmp/correctness.trace.json \
  --output .codex-tmp/correctness.json

# Sólo los casos nuevos de rejilla cuantizada:
python benchmark/profile_leviathan_compact_tests.py \
  --test-file test_leviathan_compact_kernels.py --test-name quantized \
  --profile .codex-tmp/quantized.trace.json

python benchmark/leviathan_compact_correctness.py \
  --tokens 257 513 --block-m 64 --grid-storage bf16 --checks parity \
  --candidate-manifest .codex-tmp/candidate_manifest.json \
  --profile .codex-tmp/bf16-parity.trace.json \
  --output .codex-tmp/bf16-parity.json
```

Los resultados y perfiles brutos permanecen privados. Para el gate real,
`benchmark/remote_real_10x10_validation.py` requiere `--profile`, datos existentes
y diez optimizer steps más diez batches de validación. Se reutilizan construcción,
pérdida, compilación y optimizer del entrenamiento privado. Ambos brazos deben
tener las mismas fuentes, seed, datos, batch, longitud, dtype y política.
Controles breves de logging/dataloader deben ser idénticos en ambos y no se
equiparan automáticamente al throughput continuo de producción.

El control previsto usa `LEV_COMPACT_SPLINE=0`, `BM=128`, `BD=1`, `BR=64`,
split-N 4 y cadena fusionada. El candidato usa compacto y `BM=64`, con las demás
opciones iguales. Esto mide el conjunto fórmula+tile. La atribución por separado
requiere otro experimento justificado, no duplicar corridas existentes.

Tras las mediciones se cerró también el bypass de arquitectura bajo el override
diagnóstico dot. Es una guarda host comprobada en CPU, no una nueva versión del
kernel GPU medido. Se conservan los manifiestos originales de las seis corridas;
no se adjudican timings nuevos a esa edición ni se repite entrenamiento conocido.

Se conserva `torch.compile(max-autotune)`, `force_disable_caches`, AdEMAMix, CCE,
MEAP/MiLe/MU/NITP y la política actual. La métrica histórica del harness, mediana
después del primer paso, se conserva en los registros pero no se mezcla con la
nueva población estable 5–10. El cambio de instrumentación y su definición están
declarados; ambos brazos actuales usan exactamente esa misma instrumentación.
No se cambia a backend Torch ni se deshabilita cudagraphs para declarar éxito.

Los perfiles completos de ejecución CPU/CUDA se conservan privados y descargados,
no sólo los manifiestos. Un índice privado relaciona cada rastro con su corrida,
alcance, resultado y comando de análisis. `analyze_leviathan_kernel_trace.py`
puede producir un Chrome Trace reducido mediante `--kernel-trace`, conservando
únicamente tiempos y recursos de los kernels seleccionados, con streams
anonimizados y sin metadatos de otros componentes. Ese rastro puede abrirse en
un visor compatible. Los huecos no significan que la GPU estuviera ociosa.

Se reutilizan los probes de gradientes/split-N, integración de seed, MEAP y el
harness real ya existentes; no se sustituye entrenamiento real por los antiguos
probes sintéticos. Las parejas JToK y JToK-M terminaron secuencialmente, cada
candidato condicionado al éxito de su control para evitar ejecuciones caras
si falla el contrato. Todos los brazos completaron entrenamiento real 10+10.

### JToK/JToK-M: registro antes/después del mismo cambio Leviathan

| Variante | Compute control→compacto | Pasos/s control→compacto | Ganancia de throughput | Meta |
| --- | ---: | ---: | ---: | --- |
| JToK | 269.265→266.663 ms | 3.713808→3.750050 | +0.9759% | >=4.5; no alcanzada |
| JToK-M | 314.438→310.466 ms | 3.180274→3.220963 | +1.2794% | >=4.2; no alcanzada |

En JToK, p95 compute 270.711→267.645 ms y desviación 1.805→1.981 ms; E2E
279.154→276.853 ms, ganancia de throughput +0.8310%. En JToK-M, p95 compute
315.782→312.204 ms y desviación 1.346→1.798 ms; E2E 324.477→320.492 ms,
ganancia +1.2433%. Son seis observaciones estables por brazo, no intervalos
de confianza ni evidencia de estabilidad prolongada.

| Kernel Leviathan dentro del entrenamiento | JToK control→compacto | JToK-M control→compacto |
| --- | ---: | ---: |
| `dDelta` fusionado | 11.776686→8.044934 ms | 12.076902→8.037787 ms |
| Forward | 3.343888→3.178572 ms | 3.350877→3.189102 ms |
| Reducer split-N | 0.011472→0.011168 ms | 0.011840→0.011696 ms |

Cada mediana usa dos eventos GPU. Los recursos de esas especializaciones son
iguales a los de la pareja Leviathan real: `dDelta` 255→227 registros/hilo,
shared 44→20 KiB; forward 128→124 registros, shared 40 KiB.
Los picos privados de memoria asignada y reservada no aumentaron en ninguna
pareja. Las pérdidas reales fueron finitas y sus diferencias se conservan
privadas; no se usa este control corto como prueba de convergencia.

Los kernels nativos de JToK/JToK-M permanecieron sin cambios. En JToK-M el VJP
de proyección por bloques tiene mediana 1.488126→1.490656 ms, con 48 registros
y 8 KiB de shared; el gradiente de proyección 2.123310→2.124750 ms, con 80
registros y 2 KiB. El evaluador de modos por rutas queda en 0.501487→0.501184 ms,
24 registros y 16 bytes. Cada uno tiene 24 eventos, dentro de dos pasos
perfilados. No muestran una aceleración nativa sistemática por el cambio
Leviathan; los costes y despachos de JToK-M no se deducen del perfil JToK.
La proyección aparece con dos especializaciones en un mismo nombre: no se
interpreta su mediana mezclada como un kernel representativo único.

### Por qué no se conserva el mismo porcentaje de mejora

Si el tiempo del núcleo modificado es `L` y el resto es `R`, con aceleración
local `s`, el modelo aditivo ideal da:

```text
T_antes = L + R
T_después = L/s + R
S_total = 1 / ((1-f) + f/s),    f = L/(L+R)
ahorro_ms = L * (1 - 1/s)
```

El porcentaje se diluye cuando crece R, incluso si el mismo kernel ahorra los
mismos milisegundos. La fórmula exige que R y las interacciones no cambien;
no explica por sí sola cualquier diferencia observada. El compacto mantiene
`dDelta` alrededor de 8.03–8.04 ms en las tres integraciones, pero el control
denso varía entre 11.42 y 12.08 ms. Es evidencia de dependencia del contexto,
no una medición de ancho de banda, spills o presión de caché.

El ahorro estable del paso completo es 4.968 ms en Leviathan, 2.602 ms en JToK
y 3.972 ms en JToK-M. La suma de las medianas de los tres kernels perfilados
ahorra 3.576, 3.897 y 4.201 ms respectivamente. Estas poblaciones son distintas;
el residuo no identifica un componente ni demuestra una causa.

Se hizo también un contraste dentro del último paso con profiler activo,
identificando una sola anotación externa por paso: los wrappers pueden anidar
varias `ProfilerStep` del mismo nombre. Los tres eventos GPU investigados se
asignan una sola vez y no se solapan en las trazas observadas.

| Último paso perfilado | Ahorro compute completo | Ahorro GPU de los tres kernels | Diferencia no atribuida |
| --- | ---: | ---: | ---: |
| Leviathan | 4.658 ms | 3.627 ms | +1.032 ms |
| JToK | 3.496 ms | 4.205 ms | -0.710 ms |
| JToK-M | 5.978 ms | 4.214 ms | +1.764 ms |

Una observación por brazo tampoco es prueba causal. El primer paso activo del
profiler tiene tiempos compute mucho mayores que el siguiente y la población
estable. En JToK, promediar esos dos pasos activos incluso invertiría el signo
de la mejora completa. Por ello se separa arranque del profiler, ejecución GPU
seleccionada y gate estable; no se fabrica una conclusión con esa media.

Queda sin medir directamente la ocupación efectiva, los spills y el tráfico
L2/DRAM. El siguiente experimento debe responder a una hipótesis concreta del
consumidor nativo compartido, conservando la semántica EPS y las rutas/expertos,
no prometer `16/3` de speedup ni trasladar un porcentaje entre variantes.

### Protección de publicación reproducible

Los registros privados mantienen las huellas necesarias para comparar fuentes.
Los JSON públicos se generan con campos seleccionados: tiempos completos y
agregados de los kernels investigados, nunca el rastro original.
`scripts/verify_publication_boundary.py --staged` bloquea fuentes privadas,
sus huellas, rastros completos —también escondidos dentro de otro JSON—,
datos de conexión y rutas personales en los documentos/resultados.
`--tree` conserva el rechazo aunque una fuente privada ya estuviese confirmada.
Sólo se exceptúa la utilidad pública preexistente, sin permitir cambios de
contenido; CRLF/LF no cuenta como edición de Python.
Se incorporan comprobaciones pre-commit/pre-push y CI, sin instalar hooks
automáticamente. Ningún detector por patrones garantiza descubrir todo secreto;
siguen siendo obligatorios exportación acotada y revisión del diff.

## 9. Criterio de decisión

Aceptar una mejora exige paridad, valores finitos, memoria aceptable, ausencia
de fallback, estabilidad y una comparación real válida. Los microbenchmarks
y recursos sirven para atribución, no para el gate. Diez pasos son un control
corto de ingeniería, no una demostración de convergencia a largo plazo.

La memoria se juzga como un trade-off, no como veto automático: se registran
ganancia de throughput, cambios absolutos/relativos de memoria asignada y
reservada, capacidad disponible y corrección. Un incremento puede merecer la
pena si la ganancia lo justifica o abre una mejora de recursos comprobable;
no se acepta por una promesa sin medición. El ledger privado conserva también
los resultados negativos y evita repetir ejecuciones idénticas conocidas.

Si la versión compacta no mejora o incumple un contrato, se informa como no
alcanzado y se conserva la ruta anterior. No se promociona el candidato por una
fórmula atractiva o por un resultado sintético. Ningún archivo de este reporte
autoriza publicar fuentes privadas, sus huellas o rastros completos.
