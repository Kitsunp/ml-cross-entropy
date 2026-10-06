# Hipótesis siguiente: reducción matricial común de proyecciones JToK/JToK-M

Estado: derivación de ingeniería, no implementación aceptada ni speedup medido.
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

## Propuesta de planificación, pendiente de comprobación

Un programa posee `(experto, split_tokens, tile_features, tile_hidden)`. Construye
un pequeño tile A y lo contrae con el tile G. Las features de modo y residuales
pueden compartir la planificación sin imponer que sus layouts sean idénticos.
Primero se conserva FP32 y precisión IEEE: no se introduce una cuantización
BF16 nueva de `alpha*z`, `beta` o G para obtener velocidad artificialmente.

El split-N fija rangos disjuntos de tokens. Cada programa escribe su parcial
sin atomic y un reducer suma splits en un orden fijo. Esto da ownership claro
y determinismo de esa reducción bajo la misma especialización, no igualdad
bit a bit con la acumulación atómica anterior ni determinismo de todo el modelo.

El workspace sería proporcional a:

```text
expertos * splits * features_padded * hidden * sizeof(FP32)
features = modos + d_seed
```

No depende de `tokens*expertos*modos*hidden`. El número de splits debe equilibrar
paralelismo, duración por programa, lecturas repetidas y memoria. Como ejemplo
de geometría de kernel —no del modelo completo—, E=5, features=132, H=512, S=4
requieren 5,406,720 bytes sin padding; con features padded=144 serían 5,898,240.
Es sólo la cuenta del scratch, no un pico de VRAM medido: también importan sus
lifetimes, buffers de salida y retención por CUDA Graphs.

## Costes que pueden impedir la mejora

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
