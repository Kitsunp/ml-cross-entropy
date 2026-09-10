# Fallos reproducibles de validación remota

Este registro contiene sólo diagnósticos sanitizados. No se deben añadir IPs,
puertos, usuarios, comandos de conexión, rutas de claves, tokens, datos
personales ni trazas sin revisar.

## 2026-09-10 — Verificación SSH detenida por cambio de identidad del host

- **Clase:** `HostKeyVerificationError`.
- **Resultado:** no se ejecutó ningún comando remoto ni se descargó ningún dato.
- **Causa observable:** la clave de host presentada no coincidió con la entrada
  local existente; la verificación estricta rechazó la conexión.
- **Reproducción segura:** intentar una conexión normal con verificación
  estricta al entorno remoto configurado, sin desactivar la comprobación.
- **Resolución requerida:** confirmar el nuevo host remoto y actualizar la
  confianza SSH mediante el procedimiento administrado por el usuario. No se
  debe borrar o reemplazar `known_hosts` automáticamente.
- **Estado:** pendiente; no es un fallo del kernel ni evidencia de rendimiento.

## 2026-09-10 — Importación del `train.py` remoto sin módulos hermanos

- **Clase:** `ModuleNotFoundError`.
- **Resultado:** el control Leviathan no llegó a construir el modelo ni ejecutó
  pasos de entrenamiento o validación; no se descargó ningún dato.
- **Causa observable:** el arnés cargó el archivo remoto por ruta absoluta sin
  exponer su directorio a los imports de módulos hermanos, por lo que no pudo
  resolver `configuration_neollm`.
- **Resolución:** el arnés ahora inserta temporalmente el directorio del
  `train.py` en `sys.path` y conserva un test de regresión para esa condición.
- **Estado:** corregido en el working tree; pendiente de repetir en remoto.

## 2026-09-10 — Baseline descartado por guard de no-fallback ausente

- **Clase:** `ValidationContractGap`.
- **Resultado:** el run real completó `10+10`, pero no se aceptó como evidencia
  principal porque el arnés todavía no había activado explícitamente el guard
  que rechaza el fallback de referencia.
- **Causa observable:** faltaba aplicar `_require_leviathan_triton` al modelo
  construido por el `train.py` remoto.
- **Resolución:** el arnés ahora marca el guard en los objetos de configuración
  y conserva un test de regresión; el control debe repetirse antes de comparar.
- **Estado:** corregido en el working tree y pendiente de repetir en remoto.

## 2026-09-10 — Primer smoke split-N con eje Triton no soportado

- **Clase:** `TritonCompilationError`.
- **Resultado:** el smoke sintético falló antes de producir una comparación
  numérica; no se ejecutó entrenamiento real ni se descargó ningún dato.
- **Causa observable:** la primera implementación intentó usar
  `program_id(3)`, pero Triton 3.8 en el entorno de validación sólo admite los
  ejes 0, 1 y 2.
- **Resolución:** codificar `(split, head)` en el eje 0 del grid, con tamaño
  `H * N_SPLITS`, manteniendo tres ejes de lanzamiento.
- **Verificación posterior:** el smoke sintético reproducible pasó para
  `N=257` y `N=513` con `N_SPLITS=4`; los gradientes coincidieron con `N_SPLITS=1`
  dentro de la comparación registrada y no hubo valores no finitos.
- **Estado:** corregido y verificado.
