# Stage 2 — Sesiones concurrentes en el nodo

## Problema

Un nodo ya podía actuar como worker o como parameter server según el primer mensaje recibido (Stage 1). El problema era que si un nodo bootstrapeaba un parameter server, el loop principal quedaba bloqueado aceptando las conexiones de los workers de esa sesión secuencialmente. Si llegaba una segunda sesión mientras tanto, las conexiones se mezclaban y el nodo no tenía forma de saber a qué server pertenecía cada una.

## Solución: routing por session ID

### Protocolo nuevo

Se agregaron dos variantes a `Command`:

- `ServerReady { session_id: u64 }` — respuesta del nodo al orchestrator cuando termina de buildear el server. Le comunica el ID de esa sesión.
- `JoinServer { session_id: u64 }` — primer mensaje que manda un worker cuando conecta al nodo del server. Le dice a qué sesión pertenece.

### Flujo

**Lado orchestrator / server node:**

1. Orchestrator conecta al nodo y manda `CreateServer(spec)`.
2. El nodo buildea el server, genera un `session_id` incremental, responde `ServerReady { session_id }`.
3. El nodo crea un `mpsc::channel` y registra `session_id → (workers_restantes, Sender)` en un `HashMap` local al main loop.
4. Spawnea una tarea (`tokio::spawn`) que espera exactamente `nworkers` conexiones del channel y llama `pserver.spawn(rx, tx)` por cada una. Cuando tiene todas, corre el server.

**Lado worker node:**

1. El orchestrator inyecta el `session_id` en el `AlgorithmSpec::ParameterServer` de cada worker spec (campo nuevo: `server_session_ids: Vec<u64>`).
2. Cuando el worker buildea su middleware, por cada server: conecta TCP, manda `JoinServer { session_id }`, extrae los raw halves con `into_inner()`, y crea el canal real con el serializer que corresponda.

**Lado nodo — routing:**

Cuando llega una conexión con `JoinServer { session_id }`:
1. Busca el ID en el registry.
2. Extrae los raw halves del `OnoReceiver` / `OnoSender` con `into_inner()` — seguro porque el receiver no tiene read-ahead buffering.
3. Manda los halves al channel correspondiente.
4. Decrementa el contador de workers restantes. Si llega a 0, borra la entrada del registry.
