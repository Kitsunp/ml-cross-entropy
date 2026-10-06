# Resultados remotos reales `10+10` — 2026-09-10

Este registro contiene únicamente resultados sanitizados. No incluye rutas de
conexión, IPs, usuarios, claves, tokens, datos personales ni contenido del
dataset. Los datos usados ya estaban presentes en el entorno remoto; no se
descargó ningún dataset.

## Protocolo

- Flujo: estructura real del `train.py` remoto, incluyendo forward, pérdida,
  backward y `optimizer step`.
- Datos: splits tokenizados reales preexistentes.
- Pasos: exactamente `10` de entrenamiento y `10` batches de validación.
- Batch: `64`; secuencia: `512`; semilla: `1729`.
- Backend JToK: Triton; fallback de referencia: rechazado explícitamente.
- El gate de `4.5 steps/s` aplica únicamente a `Leviathan + JToK`; Leviathan
  solo conserva su baseline propio y no se evalúa contra ese gate.
- La meta separada de `Leviathan + JToK-M` es `>=4.2 steps/s` (`<=238.095 ms`
  por paso).
- El perfilamiento se ejecuta para diagnóstico, pero las trazas crudas son
  privadas. Sólo se publica, cuando se especifica, el agregado del kernel
  investigado y el tiempo total del paso completo.
- Compilación: `torch.compile` con el modo configurado por `train.py`;
  compilación fría separada del throughput estable.

## Procedencia reproducible

| Artefacto | SHA-256 |
|---|---|
| árbol Python CCE | `620adc7fd00dc20bc3eabb5d04011e45cb1a51aac949ebd1520c3b610ff4851c` |

Entorno: PyTorch `2.14.0+cu132`, CUDA `13.2`, Triton `3.8.0`, Transformers
`5.17.0`, GPU RTX 5090, compute capability `12.0`.

## Resultados

| Variante | Mediana estable | Pasos completados | Validación | Criterio / estado |
|---|---:|---:|---:|---|
| Leviathan (`off`, control) | `4.970983 steps/s` (`201.167 ms`) | `10/10` | `10/10` | Baseline propio |
| Leviathan (`off`, split-N=4, perfilado) | `4.916241 steps/s` (`203.407 ms`) | `10/10` | `10/10` | Kernel candidato válido; sin gate JToK |
| Leviathan + JToK | `4.042266 steps/s` (`247.386 ms`) | `10/10` | `10/10` | Gate `>=4.5`: **no alcanzado** |
| Leviathan + JToK (`split-N=4`, perfilado) | `3.968158 steps/s` (`252.006 ms`) | `10/10` | `10/10` | Gate `>=4.5`: **no alcanzado** |
| Leviathan + JToK-M (`split-N=4`) | `3.422044 steps/s` (`292.223 ms`) | `10/10` | `10/10` | Meta `>=4.2`: **no alcanzado** |

La compilación fría fue aproximadamente `361.5 s` para Leviathan y `457.1 s`
para JToK. Se reporta por separado y no se usa como throughput estable.
JToK quedó aproximadamente `18.68%` por debajo del baseline controlado, por lo
que la siguiente investigación se concentra en el backward de Leviathan,
empezando por `_lev_bwd_ddelta_dot_kernel`.

## Resultado del kernel split-N

La variante `LEV_DDELTA_SPLITS=4` se validó dentro del entrenamiento real de
Leviathan (`mode=off`), con perfilamiento diagnóstico, sin descargar datos y
sin permitir fallback. El agregado del kernel investigado se publica sólo en
su alcance explícito; no se publican métricas de componentes no investigados.

Este resultado valida la mejora del kernel de Leviathan. No declara todavía
que JToK ni JToK-M cumplan su propio criterio: ambos requieren sus corridas
reales `10+10` con perfilamiento y la misma política congelada.

En la integración JToK con el mismo candidato, `_lev_bwd_ddelta_dot_kernel`
registró `10.717850 ms` por invocación. Aun con ese ahorro aislado, el paso
completo quedó en `3.968158 steps/s`; por tanto el resultado correcto para el
gate de JToK es **no alcanzado**, sin convertir la mejora de microkernel en
éxito global.

### Procedencia del candidato

- Fuente canónica: `ml-cross-entropy-jtok-pr`, commit base `5a4072d`.
- `backward_dot_kernels.py`: `41956e305ea6ff4a1c0326cb7b2b41847dcd6b7b96cf7609dc8f9caebd3ab965`.
- `backward_kernels.py`: `21548ff05407b694cac44f51ee226331d6d037ed9e68e918d861ab1d333c11ca`.
- Árbol Python CCE verificado después de la corrida: `f79fc331d878b49d17b5ba81fdb4ff3cd86505525480f91e60b5101fe3ed4398` (`52` archivos).

El arnés se ajustó para registrar, en las siguientes corridas, tanto el SHA
del árbol CCE como los SHA de los archivos críticos; así la procedencia no
depende sólo de un nombre de ejecución. Las huellas de las fuentes privadas
se conservan únicamente en registros locales excluidos del repositorio: no se
publican ni sus contenidos ni sus fingerprints. Las trazas crudas permanecen
fuera del control de versiones.
