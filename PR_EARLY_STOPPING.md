End-to-end early stopping with manual cancel


## Diseño

El sistema tiene dos caminos de detención, ambos desembocando en el mismo mecanismo:

```
[Python / TUI]
    │
    │  CancelHandle::stop()          ← stop manual
    │  EarlyStoppingConfig           ← convergencia automática
    ▼
[Orchestrator — event loop]
    │  evalúa delta de loss por sync round (ConvergenceTracker)
    │  si |prev − curr| < tolerance → detener
    ▼
Command::StopAfterEpoch  ──────────► [Workers]
                                          │
                                          │  biased select! → procesa señal
                                          │  antes de iniciar otro epoch
                                          ▼
                                     finaliza limpio en boundary de epoch
    ◄─────────────────────────────────────┘
[Orchestrator] lee params finales → TrainingEvent::Complete { model, reason }
```

El motivo de detención (`MaxEpochsReached`, `EarlyStopping`, `ManualStop`) queda representado explícitamente en el tipo y se propaga hasta la TUI y Python.

## Cambios por capa

### Protocolo (`comms`)
- Nuevo `Command::StopAfterEpoch` — el orchestrator lo manda a los workers cuando decide detener.

### Dominio de config (`orchestrator/configs`)
- `EarlyStoppingConfig { tolerance: f32 }` — el invariante `tolerance > 0` se valida en construcción, no en runtime.
- Campo `early_stopping: Option<EarlyStoppingConfig>` en `TrainingConfig` (opcional, `serde(default)`).

### Orchestrator (`orchestrator/session`)
- `StopReason` — enum que representa el motivo de finalización.
- `CancelHandle::pair()` — el caller crea el par `(handle, receiver)`, retiene el handle y pasa el receiver a `event_listener`.
- `ConvergenceTracker` — evalúa convergencia al recibir reporte de todos los workers por sync round. Usa el último valor del slice (correcto con `offline_epochs > 0`).
- `event_listener` refactorizado: loop centralizado con `biased select!` sobre `[cancel_rx, worker_events]`. Cuando dispara el stop, hace broadcast de `StopAfterEpoch` a todos los workers via los `NetTx` que ahora retiene el event loop.
- `TrainingEvent::Complete` pasa de tuple a struct: `Complete { model, reason }`.

### Worker
- `biased select!` — prioriza el canal de control sobre `pull_params`. Si llegó `StopAfterEpoch` mientras entrenaba, lo procesa antes de arrancar otro epoch.

### Python FFI (`orchestra-py`)
- `parameter_server()` y `all_reduce()` aceptan `early_stopping_tolerance: float | None`.
- `Session.stop()` — cancela el training en curso desde Python sin bloquear.

### TUI (`orchestui`)
- Tecla `x` durante el training muestra un popup de confirmación antes de mandar el stop.
- Header muestra la tolerancia configurada y el motivo de finalización (`FINISHED · early stop`, `FINISHED · stopped`).
- `training.json` y `training.example` incluyen el campo `early_stopping`.

## Lo que no cubre este PR

- All-reduce: la finalización de sesión no está implementada en esa rama, así que el early stopping queda pendiente para cuando se complete.
- Tests de integración del loop de convergencia.
