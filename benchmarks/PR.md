# PR: fix/ring-cleanup-wait

Fix de timing en el cleanup del ring AllReduce entre sesiones consecutivas sobre los mismos nodos.

---

## Problema

Al correr múltiples sesiones AllReduce secuencialmente con los mismos workers (Docker containers reutilizados), la segunda sesión fallaba invariablemente con errores de ring stale data:

```
# 2 workers
session failed with an error: early eof
starting worker session
session failed with an error: Broken pipe (os error 32)

# 3 workers
session failed with an error: early eof
starting worker session
session failed with an error: The given buffer is too small 0, must at least be 4 bytes
```

El error no ocurría con `orchestui` (orquestador Rust puro con interacción manual), solo con `orchestra-py`. La diferencia era de timing: con `orchestui` el humano tarda segundos en lanzar la siguiente sesión, tiempo suficiente para que las conexiones TCP del ring se cierren solas por timeout del OS. Con `orchestra-py` la siguiente sesión arrancaba milisegundos después — las conexiones TCP entre workers seguían vivas y el nuevo ring leía datos residuales de la sesión anterior.

---

## Root cause

`finalize_all_reduce()` en `orchestrator/src/session.rs` enviaba `WorkerHandleRequest::Disconnect` a los canales de cada `worker_listener` task y retornaba **inmediatamente**, sin esperar a que los tasks procesaran el mensaje.

Secuencia exacta:

```
1. finalize_all_reduce()
   ├── PullParams → recibe parámetros del worker líder
   └── for tx in wk_txs: tx.send(Disconnect)  ← encola en el channel, no espera

2. event_tx.send(TrainingEvent::Complete).await  ← orchestra-py recibe Complete

3. El async block del runtime termina
   └── block_on() retorna → el std::thread sale → self.runtime dropa

4. Tokio runtime: shutdown_timeout(0)
   └── Todos los tasks spawneados se CANCELAN
      └── worker_listener tasks: el Disconnect está en la queue
          pero el task es cancelado antes de leerlo
          → worker_handle.disconnect().await NUNCA se llama
```

Consecuencia: el worker nunca recibe el frame de Disconnect. Detecta EOF en su conexión con el orquestador (porque el runtime cayó) y sale por el `?` en el loop principal de `all_reduce.rs::run()`, saltándose `ring_manager.disconnect()`. Las conexiones TCP peer-to-peer del ring quedan abiertas.

El fix en orchestra-py (`wait()` drena `rx` después de `Complete`) era un no-op: `event_tx` ya se había dropeado cuando el async block terminó, entonces `rx.blocking_recv()` devolvía `None` inmediatamente sin bloquear nada.

---

## Fix

`orchestrator/src/session.rs` — `finalize_all_reduce()`:

```rust
// antes
for tx in wk_txs {
    let _ = tx.send(WorkerHandleRequest::Disconnect).await;
}
params

// después
for tx in wk_txs.iter() {
    let _ = tx.send(WorkerHandleRequest::Disconnect).await;
}

// Cada worker_listener tiene un clone de internal_tx.
// Drenar hasta None significa que todos los tasks terminaron su disconnect.
while internal_rx.recv().await.is_some() {}

params
```

Por qué funciona: `spawn_worker_listeners()` clona `internal_tx` una vez por cada worker. El `internal_tx` original se dropea al final de la función (sale de scope). Cada `worker_listener` task tiene exactamente un clone de `internal_tx`. Cuando el task retorna (después de ejecutar `worker_handle.disconnect().await`), ese clone se dropea. Cuando **todos** los clones se dropean, `internal_rx.recv()` devuelve `None`. En ese punto, todos los workers recibieron su Disconnect limpiamente y el ring fue desconectado antes de que el runtime caiga.

---

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `orchestrator/src/session.rs` | `finalize_all_reduce()`: drain de `internal_rx` después de enviar todos los Disconnects |
| `orchestra-py/src/session.rs` | `wait()`: drain de `rx` después de `Complete` (inocuo pero documentado) |

---

## Verificación

Smoke e2e (PS, container reuse — regression):

```
dense_tiny  × ps_1w_1s_base_barrier  → PASS  9.16s   0.882
dense_small × ps_1w_1s_base_barrier  → PASS  36.91s  0.882  [reused]
dense_tiny  × ps_2w_1s_base_barrier  → PASS  4.91s   0.874
dense_small × ps_2w_1s_base_barrier  → PASS  21.63s  0.872  [reused]
4/4 passed.
```

AllReduce 3w — 3 sesiones back-to-back sin delay (caso que antes fallaba siempre):

```
==================================================
  ar_3w  (3 workers, 3 back-to-back sessions)
==================================================
  session 1/3: 8.7s  OK
  session 2/3: 8.1s  OK     ← antes: buffer too small 0 / Broken Pipe
  session 3/3: 8.4s  OK
ALL PASS
```
