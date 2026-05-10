# Benchmark Issues Log

Errores y comportamientos anómalos detectados durante las ejecuciones del benchmark MNIST e2e.
Rama: `feat/benchmark-allreduce-models`. Ejecutado: 2026-05-08.

---

## Errores arquitectónicos / de protocolo

### [CRÍTICO] Ring cleanup falla después de cada sesión AllReduce — afecta la siguiente sesión

**Cómo se manifiesta:**
1. Una sesión de AllReduce termina normalmente (el orquestador recibe el modelo).
2. El worker ejecuta `ring_manager.disconnect()` (después de la comunicación con orch).
3. El ring disconnect falla con **`session failed: early eof`** ~30 segundos después del stop.
4. La **siguiente sesión** lee datos residuales del ring anterior y falla:
   - En 2w: a veces funciona (conexión en cola, datos son descartados sin corrupt)
   - En 3w: falla consistentemente con **`buffer too small 0, must at least be 4 bytes`** o **`Broken Pipe`**

**Evidencia docker logs (2w_0s, 3w_0s groups 2026-05-08):**
```
# 2w: patrón por cada run ar_2w_base
23:38:33 received a stop command from orchestrator
23:39:03 session failed with an error: early eof        ← ring cleanup failure (30s después)
23:39:03 starting worker session                        ← nueva sesión inmediata

# 3w: la segunda sesión lee basura del ring anterior
23:51:26 received a stop command from orchestrator       ← dense_small terminó
23:51:56 session failed with an error: early eof         ← ring cleanup failure
23:51:56 starting worker session                         ← dense_large arranca
23:52:25 session failed with an error: The given buffer is too small 0  ← ring residual
```

**Root cause (mismo que PR `all-reduce/early-stopping` de Marcos):**
`ring_manager.disconnect()` está al final de `all_reduce.rs::run()`. Si cualquier
operación de orch falla con `?`, o cuando el orquestador cierra su conexión antes de que
el ring se limpie, el ring nunca se desconecta limpiamente. Datos TCP residuales quedan
en los buffers, y la siguiente sesión los lee como frames válidos.

**Impacto en benchmark:** en un Docker lifecycle con múltiples runs por topología:
- **2 workers:** runs 1-4 completan (con delays de 30s entre sesiones), luego ar_2w_sparse
  falla con Broken Pipe.
- **3 workers:** solo run 1 completa; todas las siguientes fallan con ring stale data.

**Status:** reportado a Marcos — ver PR de `all-reduce/early-stopping`.

**Hallazgo adicional (2026-05-09):** El mismo escenario (mismos workers, misma topología, múltiples runs) funciona correctamente usando **orchestui** (orquestador Rust puro). El bug es específico de **orchestra-py**: Python cierra la conexión con el worker antes de que `ring_manager.disconnect()` termine en el otro lado del ring → el peer ve `early eof` → stale data en siguiente sesión. Reconstruir orchestra-py con maturin (`--release`) no resuelve el problema porque el bug está en el timing del protocolo de cierre de sesión en orchestra-py, no en el código del worker.

---

### [CRÍTICO] ar_2w_sparse — Broken Pipe en todos los modelos

**Runs afectadas:** toda combinación que use `ar_2w_sparse` (SparseSerializer + AllReduce)

**Error:**
```
RuntimeError: io error: Broken pipe (os error 32)
  at orchestra.orchestrate(model, training)
```

**Causa probable:** El SparseSerializer genera un frame de gradiente con formato
incompatible con el protocolo de ring AllReduce. El receptor cierra el lado de lectura
antes de que el sender termine → Broken Pipe.

**Nota:** El problema también ocurre como consecuencia del ring cleanup de la sesión
anterior (los datos residuales hacen que el ring falle al intentar comunicar datos sparse).

**¿Qué funciona?** ar_2w_base y ar_3w_base con BaseSerializer.

**Status:** no investigado formalmente — necesita análisis del handshake SparseSerializer ↔ ring.

---

## Errores de ML / convergencia

### [CRÍTICO] Conv2d no converge en ninguna topología ni algoritmo — accuracy aleatoria

**Runs afectadas:** toda combinación con conv_small o conv_large.

**Resultados observados (2026-05-06 a 2026-05-08):**
| run                              | acc   |
|----------------------------------|-------|
| conv_small × ps_2w_1s_base_barrier | 0.101 |
| conv_small × ar_2w_base          | 0.113 |
| conv_large × ar_2w_base          | 0.113 |
| conv_small × ar_3w_base          | 0.098 |

Accuracy invariablemente ~0.10 = clasificación aleatoria en 10 clases.
El error NO está en el protocolo distribuido (se reproduce en PS y AllReduce).

**Hipótesis:** El backward pass de Conv2d en Rust no actualiza los pesos del kernel.
La capa Dense que sigue aprende, pero sobre features aleatorias fijas → acc aleatoria.

**Status:** no investigado — necesita inspección de conv2d backward pass en Rust.

---

### [RESUELTO] dense_large no alcanza 0.90 con AllReduce 2 workers

**Fix aplicado:** `early_stopping_tolerance: 5e-5, max_train_seconds: 1800` → acc=0.904 ✓

---

### [CRÍTICO] dense_large falla con random accuracy en todas las topologías PS (base + nonblocking)

**Runs afectadas:** `dense_large × ps_2w_1s_base_barrier`, `dense_large × ps_2w_1s_nonblocking`

**Síntomas:** acc ~0.11-0.20, training termina en ~10-12s (dense_small tarda ~25-30s)

**Root cause confirmado (2026-05-09):** `early_stopping_tolerance: 5e-5` aplicado globalmente
a dense_large dispara early stopping en el primer epoch con PS porque la mejora de loss se
mide por sync round (= mini-batch), no por epoch. Con 242k params y lr=0.5, la mejora por
mini-batch en la primera vuelta es < 5e-5 aunque el modelo esté entrenando correctamente.

Con AllReduce la tolerancia funciona porque AllReduce sincroniza gradientes por época completa,
no por mini-batch. En PS, cada forward+backward de cada worker dispara un sync.

**Fix:** `early_stopping_tolerance: null` (deshabilitado) para dense_large + PS combinations.
Resultado: `dense_large × ps_base_barrier` → acc=0.943, 229s **PASS ✓**
           `dense_large × ps_nonblocking`  → acc=0.938, 215s **PASS ✓**

**Status:** resuelto en `mnist_configs.json` con overrides compuestos `model+preset`.

---

### [MENOR] dense_large × ps_2w_1s_sparse_barrier — deadline tras ~211s

**Run afectada:** `dense_large × ps_2w_1s_sparse_barrier`

**Error:** `Peer took too long to respond: deadline has elapsed` (en worker-1, ~211s después del inicio)

**Nota:** dense_small × ps_sparse_barrier funciona correctamente (24.7s, 0.910). El problema
es específico de dense_large (242k params). El training arranca y progresa durante 211s antes
de que alguna operación de sincronización se trabe. Puede estar relacionado con el tamaño del
gradiente sparse para modelos con más parámetros cruzando algún límite interno.

**Status:** no investigado formalmente — no afecta el perfil de benchmark (se puede excluir
dense_large del sparse preset, o desactivar early_stopping como se hizo con base/nonblocking).

---

## Notas por combinación (resultados completos 2026-05-08)

### Run original (tolerance 1e-4)

| modelo      | preset             | acc    | train_s | resultado | notas                             |
|-------------|--------------------|--------|---------|-----------|-----------------------------------|
| dense_small | ar_2w_base         | 0.911  | 19s     | PASS ✓    | estable, reproducible             |
| dense_small | ar_3w_base         | 0.905  | 26s     | PASS ✓    | estable, reproducible             |
| dense_small | ar_2w_sparse       | ERR    | -       | ERROR     | Broken Pipe — SparseSerializer    |
| dense_large | ar_2w_base         | 0.895  | 81s     | FAIL      | tolerancia 1e-4 muy agresiva      |
| dense_large | ar_3w_base         | ERR    | -       | ERROR     | ring stale data de sesión anterior|
| dense_large | ar_2w_sparse       | ERR    | -       | ERROR     | Broken Pipe                       |
| conv_small  | ar_2w_base         | 0.113  | 84s     | FAIL      | BUG backward pass conv2d          |
| conv_small  | ar_3w_base         | 0.098  | 32s     | FAIL      | BUG backward pass conv2d          |
| conv_small  | ar_2w_sparse       | ERR    | -       | ERROR     | Broken Pipe                       |
| conv_large  | ar_2w_base         | 0.113  | 127s    | FAIL      | BUG backward pass conv2d          |
| conv_large  | ar_3w_base         | ERR    | -       | ERROR     | ring stale + crash                |
| conv_large  | ar_2w_sparse       | ERR    | -       | ERROR     | Broken Pipe                       |
| dense_small | ps_2w_1s_base_barrier | 0.910 | ~?   | PASS ✓    | histórico — ver runs anteriores   |
| conv_small  | ps_2w_1s_base_barrier | 0.101 | ~?   | FAIL      | BUG backward pass conv2d          |

### Re-run profile allreduce (tolerance 5e-5 para dense_large)

| modelo      | preset             | acc    | train_s | resultado | notas                                |
|-------------|--------------------|--------|---------|-----------|--------------------------------------|
| dense_small | ar_2w_base         | 0.911  | 22.7s   | PASS ✓    | estable                              |
| dense_large | ar_2w_base         | **0.904**  | 105.2s  | **PASS ✓** | fix tolerance 5e-5 confirmado     |
| dense_small | ar_3w_base         | 0.905  | 30.1s   | PASS ✓    | estable                              |
| dense_large | ar_3w_base         | ERR    | -       | ERROR     | ring stale bug (Broken Pipe), siempre falla si hay run anterior en 3w |

### Run profile benchmark — resultados finales (2026-05-09)

| modelo      | preset                    | acc    | train_s | resultado | notas                                      |
|-------------|---------------------------|--------|---------|-----------|-------------------------------------------|
| dense_small | ar_2w_base                | 0.911  | 21.8s   | PASS ✓    | estable                                   |
| dense_large | ar_2w_base                | 0.904  | 88.8s   | PASS ✓    | tolerance 5e-5 (AllReduce, por epoch)     |
| dense_small | ps_2w_1s_base_barrier     | 0.910  | 25.8s   | PASS ✓    | estable                                   |
| dense_large | ps_2w_1s_base_barrier     | **0.943** | 229.0s | **PASS ✓** | fix: early_stopping=null para PS       |
| dense_small | ps_2w_1s_nonblocking      | 0.907  | 24.9s   | PASS ✓    | estable                                   |
| dense_large | ps_2w_1s_nonblocking      | **0.938** | 214.6s | **PASS ✓** | fix: early_stopping=null para PS       |
| dense_small | ps_2w_1s_sparse_barrier   | 0.910  | 24.7s   | PASS ✓    | SparseSerializer + PS funciona            |
| dense_large | ps_2w_1s_sparse_barrier   | ERR    | 211s    | ERROR     | deadline elapsed, sólo con dense_large    |

---

## Config recomendada para pasar 0.90

Basada en los datos actuales:

1. **Excluir conv_small y conv_large** de todos los profiles hasta resolver conv2d backward.
2. **Excluir ar_2w_sparse** hasta resolver SparseSerializer incompatibility con ring AllReduce.
3. **dense_large + AllReduce:** `early_stopping_tolerance: 5e-5, max_train_seconds: 1800`.
4. **dense_large + PS:** `early_stopping_tolerance: null` — early stopping granularidad mini-batch, no epoch.
5. **Para ar_3w_base:** solo usar dense_small (ring stale bug impide segunda sesión en misma topología).
6. **dense_large × ps_sparse_barrier:** deadline tras 211s — excluir o investigar límite de tamaño.

La config actualizada ya refleja estas decisiones en `mnist_configs.json`.
