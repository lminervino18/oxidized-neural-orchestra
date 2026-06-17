# Oxidized Neural Orchestra — Roadmap conceptual, decisiones y obstáculos

> Documento de apoyo para el informe. Reconstruye, a partir de **989 commits**, **108 pull requests** (incluidas las cerradas/descartadas) y **80 issues** (Sep 2025 → Jun 2026), el recorrido conceptual del proyecto, las **decisiones importantes** (de arquitectura y de alcance), los **debates** que las motivaron y los **obstáculos** (técnicos, conceptuales, de arquitectura y de scope). Al final, la **fundamentación bibliográfica** que respalda cada decisión, con enlaces.
>
> O.N.O es un sistema distribuido de entrenamiento de redes neuronales escrito **desde cero en Rust**, con tres algoritmos (Parameter Server, All-Reduce en anillo, Strategy Switch) y tres interfaces (TUI `orchestui`, módulo Python `orchestra-py` vía PyO3, y binario headless `orchestrator`). FIUBA — Ingeniería en Informática.

---

## 1. De la propuesta al sistema: qué planeamos vs. qué construimos

La [propuesta original](../proposal/objetivos.md) fijó cinco objetivos y un plan de **16 tareas** sobre dos cuatrimestres. El núcleo del plan —sistema base parametrizable en Rust, los tres algoritmos, la interfaz Python, la simulación con Docker y el análisis comparativo— **se cumplió**. Las tareas de optimización (comunicación, sincronización) también se materializaron (cuantización f16, gradientes sparse, store lock-free). Lo que quedó **fuera de alcance** fue la *tolerancia a fallos* y la *optimización de carga en configuraciones heterogéneas* (tarea 13), descoladas conscientemente hasta tener el MVP andando (ver §5).

El [estado del arte](../proposal/situacion.md) ya identificaba la tensión conceptual que gobernó todo el proyecto: **el equilibrio entre velocidad de convergencia y costo de comunicación**. Esa tensión es exactamente lo que mide el [suite de benchmarks](../../benchmarks/README.md) y lo que justifica que existan tres algoritmos en vez de uno.

---

## 2. Roadmap conceptual por fases

Las fases fusionan la evidencia de commits, PRs e issues. Las fechas son aproximadas (rango de actividad).

### Fase 0 — Propuesta, papeleo y un primer arranque descartado (Ago–Nov 2025)
Casi cinco meses de documentación: README, referencias semanales, [borrador de arquitectura](../architecture-draft.md) y la propuesta TPP formal, más una pelea larga por dejar CI en verde. El **primer diseño de arquitectura fue descartado** (PR #4): un orchestrator↔worker basado en **actores Actix** sobre TCP bloqueante, *thread-per-connection*. Se abandonó por sus propias limitaciones declaradas. Único código real del período: un spike de red multithread (`07f7c84`).

### Fase 1 — Capa de comunicación (Nov 2025 – Ene 2026)
El spike de red se tira a la basura: `6ea5292` *"replaced entire implementation with a simple application layer protocol."* Nace la capa `comms` real (PR #6): **async, con la concurrencia empujada al nivel de nodo** en lugar de incrustada en el protocolo. Se establecen el tipo `Msg`/`Payload` y la serialización a mano.

### Fase 2 — Parameter Server + Worker (Dic 2025 – Feb 2026)
Primer algoritmo distribuido. Tanto el **Parameter Server** (PRs #9→#10→#14) como el **Worker** (PRs #13→#15→#16) llegan a su diseño definitivo después de **tres intentos cerrados cada uno**. Se consolida el diseño durable: un único buffer plano `Vec<f32>` con una **vista de modelo stateless** sobre offsets/shapes, ciclo de 3 fases (recibir→computar→enviar). PR #26 toma la decisión rectora: *"el Worker debe ser estrictamente infraestructura."* Aparecen los optimizadores (SGD, momentum, Adam) y el **HogWild!** lock-free detrás de un trait `Store` (PR #50).

### Fase 3 — Motor de machine learning (Ene–Mar 2026)
Módulo `machine_learning` basado en capas. Backpropagation hecho a mano (con una saga de bugs de gradientes, ver §4). Rename global Weights→Parameter, export de modelo vía **safetensors**, capas **Conv2d**, **CrossEntropy** y **Softmax** (la pieza que faltaba para que CrossEntropy convergiera).

### Fase 4 — Orchestrator, TUI e interfaces (Feb–Mar 2026)
El crate `orchestrator` y una **`Session` orientada a eventos** con la TUI real (PR #64, ratatui) reemplazan un dashboard mock. Se decide la **interfaz Python**: se rechaza el enfoque por feature-flag (PR #80) en favor de un crate wrapper separado `orchestra-py` (PR #81), luego migrado a PyO3 0.28.

### Fase 5 — Multi-server, datasets y eficiencia de red (Mar 2026)
El Parameter Server se generaliza a **muchos servidores** (PR #65, issue #60). Distribución del dataset y el **split samples/labels** (PRs #86→#109), que además destrabó Conv. Reducción de tráfico: **codificación f16** de gradientes (PR #97 rechazada → #106 aceptada) y **protocolo de gradientes sparse** (PRs #101/#102/#108/#149).

### Fase 6 — All-Reduce en anillo + nodo genérico (Mar–May 2026)
Segundo algoritmo: **ring All-Reduce** (scatter-reduce + all-gather, PR #100 rechazada → #113/#121). Se **unifica el runtime**: un único binario **`node`** cuyo rol (worker o server) viene en el spec (PR #115). Transporte en capas (`Framer`/`TimeOuter`/`Retryer`) como costura para tolerancia a fallos futura (PR #116).

### Fase 7 — Early stopping y la saga de deadlocks (Abr–May 2026)
El early stopping de punta a punta (PR #114) desató una cascada de deadlocks de coordinación distribuida (PRs #124/#132/#136/#142; issues #122/#128/#131). Ver §4 — es el corazón "sistemas distribuidos" del proyecto.

### Fase 8 — Strategy Switch, topología y benchmarks (May–Jun 2026)
Tercer algoritmo: **Strategy Switch** arranca como All-Reduce y **promueve workers a parameter servers** cuando los gradientes convergen (PRs #145/#150/#153/#154). **Selección de topología** vía estadísticas de ping + **TSP** para ordenar el anillo y ubicar los servers (PRs #157/#159). Suite de benchmarks sobre MNIST (PR #127, issue #146).

### Fase 9 — Endurecimiento y exposición (Jun 2026)
Bugs de correctitud tardíos de alto impacto: orden de capas en PS, conteo de workers vivos, estabilidad numérica, exposición de optimizadores/activaciones por FFI y TUI, y validación de dimensiones. Los últimos commits (17-jun) todavía corrigen bugs fundamentales de promediado de gradientes y orden de capas.

---

## 3. Decisiones importantes y los debates que las motivaron

Esta sección es el núcleo del informe: cada decisión, **dónde se debatió**, las **alternativas** consideradas, una **cita textual** del debate y el **paper** que la fundamenta (tabla completa en §6).

### Arquitectura del sistema distribuido

**3.1 — Async con la concurrencia a nivel de nodo, no en el protocolo (PR #6).**
Se revirtió un primer diseño concurrente dentro de `comms`. La concurrencia por tarea dentro del protocolo *"hacía todo mucho más complicado."*
> *"Cambié la implementación para que sea async y le saqué la parte concurrente… va a ser mejor tenerlo más arriba. Que va a ser algo que manejen los nodos."*

Esto convirtió a `comms` en una frontera de protocolo fina y secuencial — regla que más tarde mataría a la PR #97 y daría forma a #96/#108.

**3.2 — El Worker como infraestructura pura; contratos en `ml_core` (PR #26).**
> *"El Worker debe ser estrictamente infraestructura."*

El worker se vuelve genérico sobre `S: TrainStrategy`, empujando el ML concreto a un crate de contratos compartidos. Es la separación definitiva **infraestructura vs. dominio ML** del proyecto.

**3.3 — Modelo stateless sobre buffer plano (PRs #13/#16).**
Un único `Vec<f32>` con vista computacional stateless (offsets/shapes), slices zero-copy, buffers reusables — elegido sobre duplicar el estado del modelo, para permitir **I/O de red zero-copy**. Reemplazó el modelo de actores Actix descartado (PR #4).

**3.4 — Tres algoritmos detrás de un único binario `node` (PR #115).**
La decisión de unificación más fuerte: en vez de binarios separados `worker`/`parameter_server`, **un solo binario cuyo rol viene en el spec** que el orchestrator le entrega al conectarse. Requirió `LocalSet`/`spawn_local` porque `ParameterServerWorker` contiene un `Box<dyn Trainer>` no-`Send`.

**3.5 — HogWild! vs. actualización con locks, unificadas tras un trait (PR #50, issue #38).**
Un trait `Store` respalda tanto `BlockingStore` (con lock) como `WildStore`:
> *"following the HogWild! paper on embracing race conditions when updating the parameters."*

Lock-free vs. lockeado pasa a ser una elección polimórfica en runtime (config `store: blocking|wild`). **Fundamento directo: Hogwild! (Niu et al., 2011).**

**3.6 — Trucos de codificación/precisión SOLO en `comms` (PRs #97→#106, #96, #108).**
La regla arquitectónica más recurrente. La codificación f16 fue **rechazada** cuando filtró `half`/`Cow` a otros módulos:
> *"No deberíamos acoplar todos los módulos a una lib que usamos para la comunicación… el protocolo debería estar implementado ahí exclusivamente."*

Diseño acordado (PR #106): *"Workers y servers nunca tocan f16, siempre operan en f32"*; la conversión vive solo en el serializer/deserializer.

**3.7 — Split worker/middleware por algoritmo vs. trait `Runtime` (PR #100 rechazada → #113).**
El reviewer quería lógica genérica de sistemas distribuidos en `Worker::run()` y lo específico de cada algoritmo en runtimes separados. El autor cerró la PR:
> *"La cierro. Abro otra por simplicidad. Confíen."*

Resultado: `ParameterServerWorker`/`AllReduceWorker` explícitos + split de middlewares.

**3.8 — Interfaz PyO3: crate separado sobre feature-flag (PR #80 rechazada → #81).**
PR #80 anotaba los tipos existentes con `#[cfg_attr(feature="python-ffi", pyclass)]` — más limpio en teoría (única fuente de verdad). Se descartó por **restricciones duras del compilador**: `#[pyclass]` no soporta genéricos (`TrainingConfig<A>` no compila), las variantes de enum complejas necesitan formas incompatibles, y `OrchestratorError` requeriría `Clone + pyclass`. PR #81 construyó el wrapper `orchestra-py` aparte, manteniendo `orchestrator` Rust puro y liberando el GIL con `py.allow_threads`.

**3.9 — Ubicación del optimizador: workers computan GD plano, los servers tienen el optimizador (PRs #121/#134, issue #133).**
Los workers siempre computan gradientes locales con descenso plano; los optimizadores con estado (Adam, momentum) viven en el server (PS) para no mandar estado por mensaje. En All-Reduce los buffers internos del optimizador tendrían basura al momento de la reducción.

**3.10 — Selección de topología por latencia máxima, no por suma (PR #157, issue #160).**
El orden del anillo all-reduce se resuelve como **TSP** sobre el ping máximo por arista; la ubicación de PS minimiza la arista **máxima**:
> *"es mejor tener 2 conexiones 'más o menos' que una muy buena y una muy mala."*

**3.11 — Particionado por arquitectura, no equitativo (issue #60, PRs #65/#155).**
Con particionado equitativo los parámetros de una capa pueden quedar a caballo de dos servers → slice no contiguo → alocación forzada. Se eligió **particionar siguiendo los límites de capa** + una lista de orden 0/1 para que el worker tome el slice del server correcto por capa.

**3.12 — Tipo float: alias en vez de genéricos (issue #40).**
Se debatió largo si `machine_learning` debía ser genérico sobre `num-traits::Float`. **Se rechazaron los genéricos** (plagarían el código con `Trainer<F: Float>`, etc.); se usa `pub type Param = f32;` manteniendo la frontera comms/orchestrator en f32. *"hay cosas que urgen más."*

---

## 4. Obstáculos

Clasificados por naturaleza. Cada uno cita su evidencia (commit/PR/issue).

### 4.1 Obstáculos técnicos

- **Backpropagation hecho a mano que no convergía** (commits 18-ene: `f509b59` transpuesta mal en producto externo, `c9d471e` signo del exponente de sigmoid, `5807e52` resta mal en `cost_prime`, `9d3a584` gradientes invertidos + último gradiente faltante; cierra `97456d9` *"fix: convergence"*). Matemática de gradientes derivada y depurada empíricamente.
- **La saga de Conv2d (~2 meses)**: padding que no andaba (`de9af6d`), backward pisando el gradiente del kernel (`25f7423`), buffer dilatado acumulando basura (`60195ae`), no andaba con múltiples filtros (`882e0a0`). El autor llamó "frankestein" a la implementación (PR #103). Causa raíz latente: dataset samples/labels sin separar (issue #94, arrays no contiguos → `unreachable!`).
- **Max-pooling backward** (`ff8a9fe` *"wip: max pooling backward (?????????????)"*): mismo gradiente escrito en todos los planes de batch/canal, enmascarado por tests de 1-batch/1-canal (PR #188).
- **NaN/Inf en MNIST**: estabilización log-sum-exp / softmax con max-shift (`d3c3db9`), guardas de batch vacío.
- **Endianness no estandarizado** (issue #87): números reinterpretados con `bytemuck` y mandados crudos, sin acuerdo de byte-order emisor/receptor.
- **Bugs de alineación al particionar el dataset binario** (`9723864`, `ebfbaf0`, `19bdb84`).
- **TUI corrompida por `println!` perdidos** (`d35f60a` y otros).
- **Panic de borrow en PyO3** (PR #147): `wait(&mut self)` retenía `borrow_mut()` durante la llamada bloqueante; `stop()` desde otro hilo paniqueaba. Se arregló con `PyRefMut` + `drop` temprano.

### 4.2 Obstáculos conceptuales (ML / numéricos)

- **Conv2d no converge** (issue #141): el loss baja pero la accuracy de test queda en ~0.10 (azar para 10 clases) — prueba de que el bug estaba en el ML, no en el protocolo. Resolución en un comentario filoso:
  > *"no tiene sentido usar sigmoid con cross entropy."*
- **CrossEntropy da loss negativo o NaN sin salida de probabilidad** (issues #171/#111): CE no aplicaba softmax interno ni validaba que la entrada fueran probabilidades. Decisión: **fundir softmax dentro de CE**.
  > *"ojo con sigmoid + cross entropy: la red aprende a escupir 0s."*
- **Gradientes que se desvanecen** (issue #148): `delta_out` de Conv se achica y luego NaN, peor con más filtros. Propuesta: loss scaling estilo fp16, f64 solo en la reducción, y clipping + guardas de no-finitos.
- **Explosión del loss** (issue #163): salta a NaN con MSE y con CrossEntropy.

### 4.3 Obstáculos de arquitectura

- **Dos reescrituras completas tempranas**: la capa de comms inicial descartada (`6ea5292`), y la abstracción `ParameterEngine`/`Executor` borrada entera — `ade263f` *"restart: remove everything, didn't work."* Las capas de abstracción iniciales no encajaron y forzaron un reinicio limpio.
- **Reversión del multiplexado de sesiones (la mayor corrección de rumbo)**: se construyó routing por session-ID (`8047803`/`857a8d7`) y días después se **arrancó de cuajo** — `655f800` *"simplify NodeRouter to single-role bootstrap, remove session multiplexing."* La complejidad no pagaba.
- **Filtración de f16 entre módulos** (PR #97 cerrada): violaba la regla de frontera de `comms`; se resolvió limpio en #106.
- **Forma de la abstracción de all-reduce** (PR #100 cerrada): `Runtime` trait vs. split por algoritmo.
- **Suposiciones de membresía estática vs. terminación dinámica** (issues #122/#128/#131, PRs #124/#132): el contador del barrier es fijo en N workers; si uno hace early-stop y se desconecta, el barrier espera para siempre al que se fue. El bug canónico de "membresía estática vs. terminación dinámica".
- **Violaciones de única-fuente-de-verdad** (issue #175): dos funciones de particionado distintas — los workers se promueven a servers con `adapt_param_gens`, pero el modelo se rearma con `adapt_servers`; coinciden solo en `nservers=1`.
- **`LossRecorder` de tamaño estático** (issues #170/#177): dimensionado con `addrs.len()` (incluyendo servers, que no reportan loss) → el early stopping nunca dispara; en Strategy Switch el conteo de workers cambia en runtime y necesita ser dinámico.
- **Bug de orden de capas en PS** (PRs #155/#179/#189): los servers guardaban capas ordenadas por tamaño, desalineadas con el orden del modelo. **Enmascarado** porque todas las redes de benchmark eran monótonas decrecientes (784→128→64→10), así que el orden por tamaño coincidía con el del modelo por casualidad.

### 4.4 Obstáculos de scope / proceso

- **Tolerancia a fallos diferida post-MVP** (issue #75, épica): *"estas features van al final, una vez que tengamos el MVP andando."* Quedó abierta en #181–#186 (la mayor área inconclusa).
- **Patrón cerrar-y-reabrir**: el equipo cerraba ramas WIP rugosas y abría limpias en vez de iterar en el lugar (PS 9→10→14, Worker 13→15→16, comms 11→12, dataset 86→109, f16 97→106, all-reduce 100→113). Disciplina de proceso, pero cuesta de seguir en el historial.
- **Scope creep declarado** (PR #34): *"ya se que no iría en esta PR pero lo metí porque tenía ganas."*
- **Tests de integración abandonados** (PR #112): sin estándar de repo acordado para configs Python/Docker → reemplazados por la suite de benchmarks (PR #127).
- **Dolor del refactor de tolerancia a fallos** (PR #116): `session.rs` quedó en un handler de 100+ líneas y ~5 conflictos de merge — *"un quilombo."*
- **Multithreading del ML deliberadamente último** (issue #144): *"quería dejarlo para lo último porque es mejor optimizar single-thread y recién ahí ir por multi."* Sigue abierto.
- **Streaming de dataset desde disco** (issue #85): abierto y cerrado sin aterrizar el diseño.

---

## 5. Lo que quedó abierto

- **Tolerancia a fallos** (#75, #181–#186): reconexión desde checkpoint, redistribución ante caída de server, capa `Reconnector` con heartbeats UDP (#186). La frontera más grande sin cerrar.
- **Conv apiladas** (#176), **softmax dentro de CE** (#171), **gradientes que se desvanecen** (#148): brechas de ML conocidas.
- **Multithreading del motor ML** (#144).
- **Streaming de datasets grandes** (#85).

---

## 6. Fundamentación bibliográfica

Mapeo de cada decisión/técnica con la literatura que la respalda. La columna **Origen** distingue lo que el equipo ya había referenciado (`semana N` / `bib`) de lo que se incorpora ahora como fundamento de una decisión que no tenía cita explícita (`agregado`).

### 6.1 Paradigma y algoritmos distribuidos

| Decisión / técnica | Referencia | Enlace | Origen |
|---|---|---|---|
| Paralelismo de datos; tensión convergencia/comunicación | Ben-Nun & Hoefler, *Demystifying Parallel and Distributed Deep Learning* (2018) | https://arxiv.org/abs/1802.09941 | semana 1 |
| Panorama general | Dehghani & Yazdanparast, *A Survey From Distributed ML to Distributed DL* (2023) | https://arxiv.org/abs/2307.05232 | semana 1 |
| **Parameter Server** | Li et al., *Scaling Distributed Machine Learning with the Parameter Server* (OSDI '14) | https://www.istc-cc.cmu.edu/publications/papers/2013/ps.pdf | semana 4 / bib |
| Parameter Server a escala (cloud) | Amazon, *Herring: Rethinking the Parameter Server at Scale* (2021) | https://assets.amazon.science/ba/69/0a396bd3459294ad940a705ad7f5/herring-rethinking-the-parameter-server-at-scale-for-the-cloud.pdf | semana 4 |
| **All-Reduce** (operación colectiva) | Li, Davis & Jarvis, *An Efficient Task-based All-Reduce for ML* (MLHPC '17) | https://doi.org/10.1145/3146347.3146350 | bib |
| **Ring All-Reduce** (anillo bandwidth-óptimo) | Patarasuk & Yuan, *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations* (JPDC 2009) | https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf | agregado |
| Ring All-Reduce aplicado a DL | Sergeev & Del Balso, *Horovod* (2018) | https://arxiv.org/abs/1802.05799 | agregado |
| **Strategy Switch** | Provatas et al., *Strategy-Switch: From All-Reduce to Parameter Server* (IEEE Access 2025) | https://doi.org/10.1109/ACCESS.2025.3528248 | semana 2 / bib |

### 6.2 Sincronización y eficiencia de comunicación

| Decisión / técnica | Referencia | Enlace | Origen |
|---|---|---|---|
| **`WildStore`** (actualización lock-free) | Niu, Recht, Ré & Wright, *Hogwild!* (2011) | https://arxiv.org/abs/1106.5730 | post-investigación |
| **Gradientes sparse** (gradient dropping) | Aji & Heafield, *Sparse Communication for Distributed Gradient Descent* (2017) | https://arxiv.org/abs/1704.05021 | semana 36 |
| Compresión de gradientes (referencia de DGC) | Lin et al., *Deep Gradient Compression* (2017) | https://arxiv.org/abs/1712.01887 | semana 36 |

### 6.3 Motor de ML (decisiones sin cita previa — fundamento agregado)

| Decisión / técnica | Referencia | Enlace | Origen |
|---|---|---|---|
| **Optimizador Adam** | Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014) | https://arxiv.org/abs/1412.6980 | agregado |
| **SGD con momentum** | Sutskever et al., *On the Importance of Initialization and Momentum in Deep Learning* (ICML 2013) | https://proceedings.mlr.press/v28/sutskever13.html | agregado |
| **Inicialización Kaiming** | He et al., *Delving Deep into Rectifiers* (2015) | https://arxiv.org/abs/1502.01852 | agregado |
| **Inicialización Xavier/Glorot** | Glorot & Bengio, *Understanding the Difficulty of Training Deep Feedforward NN* (AISTATS 2010) | https://proceedings.mlr.press/v9/glorot10a.html | agregado |
| **CNN / LeNet-5 / MNIST** | LeCun et al., *Gradient-Based Learning Applied to Document Recognition* (1998) | http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf | agregado |
| Red "Nielsen MNIST" (benchmark) | Nielsen, *Neural Networks and Deep Learning* (cap. 6) | http://neuralnetworksanddeeplearning.com/chap6.html | agregado |
| **Softmax + CrossEntropy** (fundir softmax en CE) | Goodfellow, Bengio & Courville, *Deep Learning* (2016), cap. 6.2 | https://www.deeplearningbook.org/contents/mlp.html | agregado |
| **Early stopping** | Prechelt, *Early Stopping — But When?* (1998) | https://doi.org/10.1007/3-540-49430-8_3 | agregado |

### 6.4 Frameworks y lenguaje de referencia

| Tema | Referencia | Enlace | Origen |
|---|---|---|---|
| Framework de referencia | Paszke et al., *PyTorch* (2019) | https://arxiv.org/abs/1912.01703 | bib |
| Framework de referencia | TensorFlow (2021) | https://doi.org/10.5281/zenodo.4758419 | bib |
| Concurrencia segura (*fearless concurrency*) | The Rust Programming Language — *Fearless Concurrency* | https://doc.rust-lang.org/book/ch16-00-concurrency.html | agregado |

> **Nota de honestidad académica.** Las referencias marcadas como `agregado` son la **literatura canónica** que sustenta técnicas que el sistema ya implementaba (Adam, Kaiming, ring all-reduce, early stopping, etc.) pero que no tenían una cita explícita en las notas semanales. No reemplazan el criterio con que se tomaron las decisiones; documentan su fundamento teórico, como corresponde a un informe formal.

---

*Generado a partir del análisis del historial completo del repositorio (commits, PRs e issues) y la documentación de `docs/`.*
