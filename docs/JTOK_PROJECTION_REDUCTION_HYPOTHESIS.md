# Hipótesis siguiente: reducción matricial común de proyecciones JToK/JToK-M

Estado: candidato implementado opt-in; diez pruebas focalizadas nuevas pasan.
Las comparaciones reales perfiladas 10+10 terminaron en JToK y JToK-M: mejora
observada del paso completo de +1.25% y +3.66%, respectivamente. Los gates
de 4.5 y 4.2 pasos/s no se alcanzaron; no se afirma estabilidad de largo plazo.
La comparación sparse terminó: mejora el VJP en JToK, pero lo empeora en
JToK-M; ninguna ganancia global estable está confirmada. Esta hipótesis parte
del compacto Leviathan con sparse desactivado, sin convertir esa regresión
en requisito de la siguiente etapa.
Este documento se limita a los kernels investigados; no publica fuentes privadas
ni detalles de componentes ajenos al objetivo.

## Por qué investigar este consumidor

El perfil real JToK-M anterior muestra cerca de 2.125 ms por llamada del kernel
de gradiente de proyección, con 24 eventos en dos pasos activos. En JToK el mismo
tipo de reducción cuesta cerca de 0.517 ms por llamada. La priorización debe
mantener la fórmula compartida, pero no suponer el mismo beneficio en ambos.

El kernel actual ya comparte el tile del gradiente de superficie y la metadata
de rutas entre modos/coordenadas. No se propone esa reutilización como novedad.
Todavía recorre cada coordenada de seed de forma serial, reduce sobre tokens y
emite una actualización atómica por coordenada y columna. El nuevo interés es
su organización matricial y el ownership de la reducción.

## Identidad matemática

Sean `G_nh` el VJP de la superficie, `p_nr` el peso seleccionado, `e_nr` el
experto de esa ruta, `M_nrm` su producto de modo y `z_nd` la coordenada:

```text
alpha_ne = sum_r [e_nr=e] p_nr
beta_nem = sum_r [e_nr=e] p_nr M_nrm
bar_W_residual_edh = sum_n alpha_ne z_nd G_nh
bar_W_spline_emh = sum_n beta_nem G_nh
A_ne = concat(beta_ne, alpha_ne * z_n)
bar_W_e = transpose(A_e) @ G
```

La identidad también cubre expertos repetidos: se suman todas las rutas con la
misma identidad, no se toma sólo la primera. El experto único es el mismo caso
con una ruta y peso uno. No cambia la derivada de selección/top-k ni la mezcla;
sus VJP siguen teniendo un único dueño en el kernel previo.

`A_e` es una identidad algebraica, no autorización para materializar un tensor
denso `[tokens,expertos,features]`. Se reconstruye sólo el tile del experto
actual a partir de la metadata compacta `[tokens,top_k]`. El gradiente de
superficie ya existe y se reutiliza, sin reconstruirlo por otra ruta.

## Planificación implementada

Un programa posee `(experto, split_tokens, tile_features, tile_hidden)`. Construye
un pequeño tile A y lo contrae con el tile G. Las features de modo y residuales
pueden compartir la planificación sin imponer que sus layouts sean idénticos.
Primero se conserva FP32 y precisión IEEE: no se introduce una cuantización
BF16 nueva de `alpha*z`, `beta` o G para obtener velocidad artificialmente.

El split-N fija rangos disjuntos de tokens. Cada programa escribe su parcial
sin atomic y un reducer suma splits en un orden fijo. Esto da ownership claro
y determinismo de esa reducción bajo la misma especialización, no igualdad
bit a bit con la acumulación atómica anterior ni determinismo de todo el modelo.

El workspace de esta implementación es:

```text
expertos * splits * features * hidden * sizeof(FP32)
features = modos + d_seed
```

No depende de `tokens*expertos*modos*hidden`. El número de splits debe equilibrar
paralelismo, duración por programa, lecturas repetidas y memoria. Como ejemplo
de geometría de kernel —no del modelo completo—, E=5, features=132, H=512, S=4
requieren 5,406,720 bytes sin padding; con features padded=144 serían 5,898,240.
Es sólo la cuenta del scratch, no un pico de VRAM medido: también importan sus
lifetimes, buffers de salida y retención por CUDA Graphs.

## Costes que pueden impedir la mejora

La primera implementación usa tiles F16×H64×N32, hasta ocho splits, FP32/IEEE
y un reducer fijo. El scratch almacena únicamente features y hidden reales,
sin padding de la salida: para E=5, F=132 y H=512 son 10,813,440 bytes con S=8.
Se limita explícitamente a 128 MiB antes de asignar; un rechazo no es fallback.
La opción `JTOK_PROJECTION_SPLIT=1` requiere el backward wide, y permanece en 0
por defecto. El camino anterior y su autotuner se conservan cuando está apagada.
No se activa sparse para esta comparación: se reutiliza el compacto registrado.

Como límite de prioridad, unos 25.5 ms de proyección por paso frente a 310.5 ms
compute representan ~8.2%. Aun su eliminación ideal no alcanzaría por sí sola
los 238.1 ms equivalentes al gate de 4.2 pasos/s de JToK-M. Se investiga un
consumidor concreto, no una promesa de resolver todo el paso con una sola fórmula.

- `tl.dot` IEEE no implica automáticamente uso de Tensor Cores ni gran speedup.
- Tiling por features puede releer G más veces que el bucle actual, que lo retiene.
- Repetir scans por experto sigue costando cuando el routing es disperso.
- Accumuladores feature×hidden grandes pueden causar presión de registros/spills.
- Un split-N demasiado pequeño puede dejar pocos programas activos; uno grande
  aumenta escritura de parciales y coste del reducer.
- Bajar atomics no prueba menor tráfico DRAM: la caché y el tamaño de sectores
  pueden dominar. Los contadores físicos no se infieren de un conteo algebraico.
- El padding y la retención de temporales pueden elevar el pico aunque la cuenta
  aislada del scratch sea pequeña.

## Gate antes de integrar

Se necesita probar el kernel común con experto único, mezcla, rutas repetidas,
expertos vacíos, filas inválidas y tiles incompletos. La referencia usa la
fórmula FP32 anterior; se comparan gradientes antes y después del cast final.
Después debe conservar todos los VJP registrados de JToK/JToK-M, no duplicar
gradientes ni cambiar pérdidas, y medirse en entrenamiento real perfilado 10+10.

La mejora aislada, su coste de memoria y la mejora del paso completo se registran
por separado. Un incremento de memoria puede ser justificable por velocidad o
mejor ownership de recursos, pero no se acepta por una promesa algebraica.
Se reutilizan controles ya válidos y se evita repetir corridas conocidas.

## Verificación funcional del primer candidato

La ejecución remota perfilada pasó 10 casos en 10.71 s: contratos CPU de default,
workspace y procedencia; tres geometrías contra oráculo Double con FP32 de salida,
tiles incompletos, rutas duplicadas, expertos vacíos y reducción repetible; dos
backwards registrados que conservan todos los VJP, incluido router en mezcla;
rechazo explícito de la ruta narrow y gradientes vacíos inicializados a cero.
Los sintéticos son funcionales, no evidencia de throughput. Los avisos de AST
de Triton afectan una versión futura de Python; no hubo error del kernel actual.

Reproducción desde un checkout completo con el entorno CUDA activado:

```bash
python benchmark/profile_leviathan_compact_tests.py \
  --test-file test_jtok_projection.py \
  --test-file test_leviathan_candidate_provenance.py --test-name projection \
  --profile .codex-tmp/projection_correctness.trace.json
python benchmark/leviathan_candidate_provenance.py --source-root . \
  --include-projection --output .codex-tmp/projection_manifest.json
```

El manifiesto incluye el kernel nuevo y su dispatcher, no fuentes privadas.
La comparación real 10+10 reutiliza el runner de entrenamiento existente con
`JTOK_PROJECTION_SPLIT=1`, `JTOK_SPARSE_COEFF_UPDATES=0`, compacto Leviathan,
`--candidate-manifest` y `--profile`. Las dos corridas nuevas deben conservar
el mismo flujo privado y la política de los controles registrados.

## Resultado real del candidato

Se reutilizó el control compacto registrado de cada modo, sin repetirlo.
La igualdad de fuentes privadas, entorno y semillas se comprobó dentro de
cada par. Ambos candidatos completaron 10 pasos de entrenamiento y 10 batches
de validación con datos reales, sin cambiar la política global ni usar fallback.
El throughput principal es el inverso de la mediana compute de los pasos 5–10;
los dos pasos perfilados son otra población, utilizada sólo para atribución.

| Modo | Compute anterior → candidato | Pasos/s anteriores → candidato | Ganancia throughput | Gate |
| --- | --- | --- | --- | --- |
| JToK | 266.663 → 263.371 ms | 3.750050 → 3.796923 | +1.2499% | 4.5: no alcanzado |
| JToK-M | 310.466 → 299.501 ms | 3.220963 → 3.338890 | +3.6612% | 4.2: no alcanzado |

El p95 compute del candidato fue 264.642 ms en JToK y 301.060 ms en JToK-M;
la desviación poblacional, 1.515 y 1.651 ms. Son seis observaciones por corrida,
no intervalos de confianza ni repeticiones independientes. E2E mejoró
276.853 → 273.350 ms y 320.492 → 309.767 ms, respectivamente.

La familia de proyección incluye productor y reducer, emparejados por stream;
no se compara el productor aislado con todo el kernel anterior:

| Modo | Mediana GPU por llamada anterior → familia nueva | Reducción | Ahorro sumado por paso perfilado |
| --- | --- | --- | --- |
| JToK | 516.543 → 309.919 us | 40.0013% | 2.4901 ms |
| JToK-M | 2124.751 → 1224.509 us | 42.3693% | 10.8093 ms |

Hay 24 llamadas o pares por brazo: 12 por cada uno de dos pasos perfilados,
no 24 pasos independientes. El kernel anterior está ausente del candidato
y los nuevos están ausentes del control: no se ejecutaron ambas reducciones.
El mayor ahorro absoluto en M explica por qué la misma ingeniería tiene más
peso allí; no implica un porcentaje idéntico en todo el entrenamiento.

El productor nuevo usa 179 registros/hilo y 10,240 bytes de shared; el reducer,
39 y cero. Los anteriores usan 166/2,048 en JToK y 80/2,048 en M. La mejora
no procede de reducir estos recursos. No se midieron spills, tráfico DRAM,
caché ni ocupación efectiva con contadores físicos. El scratch parametrizado
existe aunque no domine el pico global; sus lifetimes siguen siendo un coste.
La precisión funcional se acota con el oráculo Double y los VJP registrados
descritos arriba; una corrida corta finita no prueba convergencia.

El analizador suma las duraciones GPU de cada par y excluye huecos CPU; no
calcula una unión de intervalos. La exportación muestra pequeños solapamientos
aparentes entre duración y timestamp: 22 pares de J, máximo 0.224 us, y 24 de M,
máximo 0.288 us. Se registran sin corregirlos ni atribuirlos a concurrencia real;
su causa no está probada. Una regresión numérica mínima reproduce el problema
del antiguo rechazo estricto del analizador y pasa con el registro explícito.

Reanálisis offline de un perfil privado ya existente, sin nueva ejecución CUDA:

```bash
python benchmark/analyze_leviathan_kernel_trace.py PRIVATE_PROFILE \
  --target _jtok_projection_split_kernel \
  --target _jtok_projection_split_reduce_kernel \
  --pair-first _jtok_projection_split_kernel \
  --pair-second _jtok_projection_split_reduce_kernel
```

Se conserva la opción desactivada por defecto. La próxima hipótesis priorizada
es reutilizar el soporte cuadrático compacto de Leviathan en el VJP spline
común, en vez de limitarse a omitir atomics cero. Debe respetar los knots
realmente almacenados, normalización, derivadas, EPS y routing. Aún no está
implementada ni medida; menos coeficientes lógicos no garantiza menos sectores
de memoria, menos registros ni un speedup de igual magnitud.
