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
- Compilación: `torch.compile` con el modo configurado por `train.py`;
  compilación fría separada del throughput estable.

## Procedencia reproducible

| Artefacto | SHA-256 |
|---|---|
| `train.py` | `5d43cd58096aaebb0b04eb59a9f1b82163d02089ee18fca83fb6bd1f2175c48d` |
| `modeling_neollm.py` | `26914cf05982e015fa10fec29385dddb0e434e97269a09694ffb06f40b519187` |
| `configuration_neollm.py` | `d58cfb55ac2dc1056037ef7b0dcd8d38da617dadad0f23d85ce31cc06082706c` |
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

La compilación fría fue aproximadamente `361.5 s` para Leviathan y `457.1 s`
para JToK. Se reporta por separado y no se usa como throughput estable.
JToK quedó aproximadamente `18.68%` por debajo del baseline controlado, por lo
que la siguiente investigación se concentra en el backward de Leviathan,
empezando por `_lev_bwd_ddelta_dot_kernel`.

## Resultado del kernel split-N

La variante `LEV_DDELTA_SPLITS=4` se validó dentro del entrenamiento real de
Leviathan (`mode=off`), con perfilamiento diagnóstico, sin descargar datos y
sin permitir fallback. Frente al control perfilado comparable, el kernel
`_lev_bwd_ddelta_dot_kernel` pasó de `12.0860675 ms` a `10.9490925 ms` por
invocación: reducción de `9.407%`. La pérdida de evaluación fue `8.881209`,
frente a `8.881250` del control, y no hubo valores no finitos en el smoke de
corrección.

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
depende sólo de un nombre de traza.
