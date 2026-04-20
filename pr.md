# feat: node runtime bootstrap (Etapa 1)

## Qué

Reemplaza los dos binarios dedicados (`worker`, `parameter_server`) por un único host genérico: `node`.

El nodo escucha conexiones entrantes. Por cada conexión lee el primer mensaje del orchestrator y despacha:

- `CreateWorker` → bootstrapea una sesión de worker en una tarea local independiente
- `CreateServer` → acepta las N conexiones de workers inline, luego corre la sesión del parameter server en una tarea local independiente

Cuando cada sesión termina, el nodo sigue escuchando nuevas conexiones.

## Decisión de diseño

`ParameterServerWorker` y `Box<dyn Server>` no son `Send`, por lo que `tokio::spawn` no es viable. Las sesiones se despachan con `tokio::task::LocalSet` + `spawn_local`, que da concurrencia a nivel de tarea sin requerir bounds `Send` en los traits existentes.

## Qué viene después

- **Etapa 2 — registry de entidades**: múltiples entidades concurrentes por nodo con registry explícito, creación/destrucción dinámica y scheduler
