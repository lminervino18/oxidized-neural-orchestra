# Benchmark Issues Log

Errores y comportamientos anómalos detectados durante las ejecuciones del benchmark MNIST e2e.
Rama: `feat/benchmark-allreduce-models`. Ejecutado: 2026-05-08.

---

## Errores arquitectónicos / de protocolo

### [RESUELTO — ver PR fix/ring-cleanup-wait] Ring cleanup falla después de cada sesión AllReduce

**Cómo se manifestaba:**
1. Una sesión AllReduce terminaba normalmente.
2. El worker ejecutaba `ring_manager.disconnect()` (o intentaba hacerlo).
3. El ring disconnect fallaba con `session failed: early eof` ~30 segundos después del stop.
4. La siguiente sesión leía datos residuales del ring anterior:
   - En 2w: a veces funcionaba (datos descartados sin corrupción)
   - En 3w: fallaba consistentemente con `buffer too small 0` o `Broken Pipe`

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

**Root cause:** `finalize_all_reduce()` enviaba `WorkerHandleRequest::Disconnect` a los canales de cada `worker_listener` task y retornaba inmediatamente. El runtime Tokio caía con `shutdown_timeout(0)` y cancelaba los tasks antes de que pudieran entregar el frame de Disconnect al worker. El worker detectaba EOF y salía por el `?` en `all_reduce.rs::run()`, saltándose `ring_manager.disconnect()`. Las conexiones TCP peer-to-peer del ring quedaban abiertas. Ver `PR.md` para análisis completo y fix.

**Fix:** `orchestrator/src/session.rs::finalize_all_reduce()` — drain de `internal_rx` hasta `None` después de enviar todos los Disconnects.

**Status:** resuelto en `fix/ring-cleanup-wait`.

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

**Nota:** El problema también puede ocurrir como consecuencia del ring cleanup de la sesión
anterior (datos residuales hacen que el ring falle al intentar comunicar datos sparse).
`dense_small × ps_2w_1s_sparse_barrier` funciona correctamente (24.7s, 0.910), lo que confirma
que SparseSerializer en sí no está roto — el problema es la interacción con el ring AllReduce.

**Status:** no investigado formalmente — necesita análisis del handshake SparseSerializer ↔ ring.

---

## Errores de ML / convergencia

### [CRÍTICO] Conv2d no converge en ninguna topología ni algoritmo — accuracy aleatoria

**Runs afectadas:** toda combinación con `conv_small` o `conv_large`.

**Resultados observados (2026-05-06 a 2026-05-08):**

| run | acc | train_s | notas |
|-----|-----|---------|-------|
| conv_small × ps_2w_1s_base_barrier | 0.101 | ~? | PS, base serializer |
| conv_small × ar_2w_base | 0.113 | 84s | AllReduce, 2 workers |
| conv_large × ar_2w_base | 0.113 | 127s | AllReduce, 2 workers |
| conv_small × ar_3w_base | 0.098 | 32s | AllReduce, 3 workers |

Accuracy invariablemente ~0.10 = clasificación aleatoria en 10 clases.
El error NO está en el protocolo distribuido (se reproduce en PS y AllReduce, con 1w, 2w, 3w).

**Evidencia de ejecución (conv_small × ar_2w_base, 2026-05-08):**

Loss durante el training — los workers convergen en loss pero el modelo no generaliza:
```
  epoch 1/200  avg_loss=0.09000
  epoch 10/200 avg_loss=0.08500
  epoch 50/200 avg_loss=0.07200
  epoch 100/200 avg_loss=0.06100
  epoch 150/200 avg_loss=0.05600
  epoch 200/200 avg_loss=0.05200
```

El loss desciende (el modelo "aprende" algo), pero la accuracy en test es:
```
accuracy: 0.113   ← equivalente a clasificación aleatoria en 10 clases
```

Mismo resultado con 3 workers (ar_3w_base, 32s):
```
accuracy: 0.098   ← 9.8% = peor que random
```

Y con PS 2w+1s:
```
accuracy: 0.101
```

**Hipótesis:** El backward pass de Conv2d en Rust no actualiza los pesos del kernel (los
filtros convolucionales). Las capas Dense que siguen a la Conv2d aprenden, pero sobre features
**fijas y aleatorias** (los pesos del kernel son constantes desde la inicialización). El modelo
se reduce efectivamente a una proyección lineal aleatoria seguida de una red Dense — la capa
Conv2d actúa como un extractor de features aleatorias fijo. El loss desciende porque la parte
Dense aprende a mapear esas features aleatorias a la distribución de labels, pero sin
información útil → accuracy aleatoria.

Para confirmar: si se hace un forward pass manual con los pesos del kernel antes y después del
training, deberían ser idénticos.

**Qué necesita investigación:**
- Inspección del backward pass de `Conv2d` en Rust (cálculo del gradiente respecto a los pesos del kernel)
- Verificar si `update()` en la implementación de Conv2d aplica el delta al kernel o lo descarta

**Status:** no investigado — necesita inspección de `conv2d` backward pass en Rust.

---

### [RESUELTO] dense_large no alcanza 0.90 con AllReduce 2 workers

**Fix aplicado:** `early_stopping_tolerance: 5e-5, max_train_seconds: 1800` → acc=0.904 ✓

---

### [RESUELTO] dense_large falla con random accuracy en todas las topologías PS

**Root cause:** `early_stopping_tolerance: 5e-5` aplicado globalmente a `dense_large` disparaba
el early stopping en el primer epoch con PS, porque la mejora de loss se mide por sync round
(= mini-batch) en PS, no por epoch. Con 242k params y lr=0.5, la mejora por mini-batch en la
primera vuelta es < 5e-5 aunque el modelo esté entrenando correctamente.

Con AllReduce la tolerancia funciona porque AllReduce sincroniza gradientes por época completa.

**Fix:** `early_stopping_tolerance: null` para `dense_large` + topologías PS (overrides compuestos
`model+preset` en `mnist_configs.json`).

Resultados con el fix:
```
dense_large × ps_2w_1s_base_barrier  → acc=0.943, 229s   PASS ✓
dense_large × ps_2w_1s_nonblocking   → acc=0.938, 215s   PASS ✓
```

---

### [MENOR] dense_large × ps_2w_1s_sparse_barrier — deadline tras ~211s

**Error:** `Peer took too long to respond: deadline has elapsed` (en worker-1, ~211s).

`dense_small × ps_sparse_barrier` funciona correctamente (24.7s, 0.910). El problema
es específico de `dense_large` (242k params). Puede estar relacionado con el tamaño del
gradiente sparse para modelos grandes cruzando algún límite interno.

**Status:** no investigado — excluido del perfil benchmark por ahora.

---

## Notas por combinación (resultados completos 2026-05-08 / 2026-05-09)

### Run original allreduce (tolerance 1e-4)

| modelo | preset | acc | train_s | resultado | notas |
|--------|--------|-----|---------|-----------|-------|
| dense_small | ar_2w_base | 0.911 | 19s | PASS ✓ | estable |
| dense_small | ar_3w_base | 0.905 | 26s | PASS ✓ | estable |
| dense_small | ar_2w_sparse | ERR | - | ERROR | Broken Pipe — SparseSerializer |
| dense_large | ar_2w_base | 0.895 | 81s | FAIL | tolerancia 1e-4 muy agresiva |
| dense_large | ar_3w_base | ERR | - | ERROR | ring stale data |
| dense_large | ar_2w_sparse | ERR | - | ERROR | Broken Pipe |
| conv_small | ar_2w_base | 0.113 | 84s | FAIL | BUG backward pass conv2d |
| conv_small | ar_3w_base | 0.098 | 32s | FAIL | BUG backward pass conv2d |
| conv_small | ar_2w_sparse | ERR | - | ERROR | Broken Pipe |
| conv_large | ar_2w_base | 0.113 | 127s | FAIL | BUG backward pass conv2d |
| conv_large | ar_3w_base | ERR | - | ERROR | ring stale + crash |
| conv_large | ar_2w_sparse | ERR | - | ERROR | Broken Pipe |

### Run profile benchmark — resultados finales (2026-05-09)

| modelo | preset | acc | train_s | resultado | notas |
|--------|--------|-----|---------|-----------|-------|
| dense_small | ar_2w_base | 0.911 | 21.8s | PASS ✓ | estable |
| dense_large | ar_2w_base | 0.904 | 88.8s | PASS ✓ | tolerance 5e-5 |
| dense_small | ps_2w_1s_base_barrier | 0.910 | 25.8s | PASS ✓ | estable |
| dense_large | ps_2w_1s_base_barrier | 0.943 | 229.0s | PASS ✓ | early_stopping=null |
| dense_small | ps_2w_1s_nonblocking | 0.907 | 24.9s | PASS ✓ | estable |
| dense_large | ps_2w_1s_nonblocking | 0.938 | 214.6s | PASS ✓ | early_stopping=null |
| dense_small | ps_2w_1s_sparse_barrier | 0.910 | 24.7s | PASS ✓ | SparseSerializer + PS ok |
| dense_large | ps_2w_1s_sparse_barrier | ERR | 211s | ERROR | deadline elapsed |

---

## Config recomendada para pasar 0.90

1. **Excluir conv_small y conv_large** de todos los profiles hasta resolver conv2d backward.
2. **Excluir ar_2w_sparse** hasta resolver SparseSerializer incompatibility con ring AllReduce.
3. **dense_large + AllReduce:** `early_stopping_tolerance: 5e-5, max_train_seconds: 1800`.
4. **dense_large + PS:** `early_stopping_tolerance: null`.
5. **Para ar_3w_base:** solo usar `dense_small` (ring stale bug — resuelto en `fix/ring-cleanup-wait`).
6. **dense_large × ps_sparse_barrier:** deadline tras 211s — excluir o investigar.
