# Notas de los expositores

Generado desde `index.html` con `python3 tools/export_notes.py`. No editar a mano.

## Resumen de tiempos

- Alejo Ordoñez: 11:00
- Marcos Bianchi Fernández: 11:00
- Lorenzo Minervino: 10:30
- [a completar]: 00:30
- Los tres: 02:00
- Diapositivas principales: 35:00
- Demo: 10:00
- Total: 45:00 (objetivo 45:00)

## 01 · Portada

- Expositor sugerido: Alejo Ordoñez
- Tiempo estimado: 0:45 (acumulado al terminar: 00:45)
- Idea principal: Presentarnos y anticipar en una frase qué construimos: una base común para comparar estrategias de entrenamiento distribuido.
- Desarrollo oral:

Saludo al jurado. Nombres del equipo, tutor y co-tutor.

Frase de apertura sugerida: «Construimos O.N.O., un sistema distribuido de entrenamiento escrito íntegramente en Rust, con su propia biblioteca de redes neuronales, para comparar estrategias de distribución sin que la implementación sesgue la comparación».

Anticipar la estructura: tres bloques de exposición (problema y fundamentos, estrategias y diseño, evidencia y conclusiones) y una demo al final.

- Transición: «Empecemos por el problema: por qué distribuir y por qué compararlo es difícil.»
- Respaldo: docs/report/caratula.md · resumen.md

## 02 · El costo de entrenar

- Expositor sugerido: Alejo Ordoñez
- Tiempo estimado: 1:15 (acumulado al terminar: 02:00)
- Idea principal: Distribuir no es un lujo: es la forma accesible de sumar cómputo y memoria con el hardware que ya se tiene.
- Desarrollo oral:

El costo tiene dos dimensiones: el tiempo (días sobre una sola máquina) y la memoria (datos, modelo y estados intermedios pueden exceder un dispositivo).

Paso 1: el escenario real. Rara vez se tiene un equipo de gran porte; sí varias máquinas modestas. Sumar hardware existente es más accesible que comprar uno que lo iguale.

Paso 2: por eso se distribuye: reducir tiempo repartiendo cómputo y superar la restricción de memoria repartiendo datos o parámetros.

- Transición: «Distribuir es la respuesta conocida. Lo que nos interesó es otra cosa: cómo elegir entre las distintas formas de distribuir.»
- Respaldo: docs/report/problema.md (párrafo 1) · introduccion.md

## 03 · El problema: comparar estrategias

- Expositor sugerido: Alejo Ordoñez
- Tiempo estimado: 2:00 (acumulado al terminar: 04:00)
- Idea principal: El problema concreto no es distribuir, sino comparar estrategias sin que la implementación contamine la medición.
- Desarrollo oral:

Las bibliotecas de referencia ofrecen paralelismo de datos, pero exponen una única llamada que envuelve toda la sincronización.

Paso 1: dentro de esa llamada quedan tomadas cuatro decisiones que nos interesa variar de forma independiente: el régimen de sincronización (atado a la estrategia), la ubicación del optimizador con estado, el particionado de parámetros y la codificación de los mensajes.

Paso 2: además, esas implementaciones están optimizadas para producción y no aplican el mismo trabajo de bajo nivel a cada estrategia (solapamiento cómputo/comunicación, kernels especializados). Al comparar dos estrategias sobre dos implementaciones, la diferencia mezcla algoritmo y calidad de implementación. Ese sesgo no se puede cuantificar.

- Transición: «De ahí sale el objetivo del trabajo.»
- Respaldo: docs/report/problema.md · estado_del_arte.md (Dehghani y Yazdanparast, 2023)

## 04 · Objetivo y alcance

- Expositor sugerido: Alejo Ordoñez
- Tiempo estimado: 1:15 (acumulado al terminar: 05:15)
- Idea principal: El producto es un sistema de experimentación: su valor está en la evidencia que permite producir.
- Desarrollo oral:

Objetivo: construir una base común, parametrizable y de código abierto en la que las tres estrategias compartan capa de comunicación, motor de redes neuronales y plano de control.

Paso 1, alcance: paralelismo de datos; los tres algoritmos de referencia; motor propio sin frameworks de deep learning; dos interfaces; suite reproducible. Fuera: tolerancia a fallos (se asumió entorno local) y otros esquemas de paralelismo.

Si preguntan por la optimización de carga en configuraciones heterogéneas: estaba en la propuesta y no llegó a abordarse (desvío de alcance documentado en Riesgos).

- Transición: «Para entender las estrategias hace falta fijar dos conceptos: paralelismo de datos y sincronización.»
- Respaldo: docs/report/introduccion.md · problema.md (último párrafo) · riesgos.md (Desvíos de alcance)

## 05 · Fundamentos: paralelismo de datos

- Expositor sugerido: Alejo Ordoñez
- Tiempo estimado: 2:00 (acumulado al terminar: 07:15)
- Idea principal: Toda estrategia que comparamos resuelve el mismo paso: cómo agregar los gradientes de N workers.
- Desarrollo oral:

La literatura clasifica la concurrencia según qué se particiona: datos, modelo, pipeline o híbrido (Ben-Nun y Hoefler 2019; Dehghani y Yazdanparast 2023). O.N.O. trabaja solo con paralelismo de datos.

Recorrer los pasos: (1) cada worker recibe una partición distinta; (2) calcula el gradiente sobre ella con su réplica; (3) los gradientes se agregan, en O.N.O. siempre por promedio; (4) todas las réplicas aplican la misma actualización θ ← θ − η·ḡ.

Remarcar: el paso 3 es exactamente donde difieren Parameter Server y All-Reduce. La diapositiva es conceptual; los detalles reales vienen en el bloque 2.

- Transición: «La otra pregunta es cuándo se agrega: si todos esperan a todos o no.»
- Respaldo: docs/report/estado_del_arte.md (Paralelismo de datos, de modelo y de pipeline)

## 06 · Sincrónico vs. asincrónico

- Expositor sugerido: Alejo Ordoñez
- Tiempo estimado: 1:15 (acumulado al terminar: 08:30)
- Idea principal: El régimen de sincronización es un trade-off entre coherencia de réplicas y tiempo ocioso.
- Desarrollo oral:

Sincrónico: hay una barrera por ronda; el nodo más lento fija el ritmo del clúster (zonas rayadas = espera).

Paso 1: asincrónico: cada gradiente se aplica apenas llega; desaparece la espera, pero un worker puede empujar un gradiente calculado sobre parámetros viejos. Cuando las réplicas divergen por particiones sesgadas se habla de client drift (McMahan et al., 2017).

Hogwild! (Niu et al., 2011) lleva esto al extremo: acepta condiciones de carrera y prueba convergencia bajo actualizaciones dispersas. Vuelve en resultados.

- Transición: «Con estos conceptos, veamos cómo está construido O.N.O.»
- Respaldo: docs/report/estado_del_arte.md (Parameter Server y All-Reduce) · problema.md (régimen de sincronización)

## 07 · Arquitectura general

- Expositor sugerido: Alejo Ordoñez
- Tiempo estimado: 1:30 (acumulado al terminar: 10:00)
- Idea principal: La arquitectura agnóstica de rol es lo que hace posible cambiar de estrategia en medio de un entrenamiento.
- Desarrollo oral:

Todos los nodos ejecutan el mismo binario, `node`. No hay imagen de worker ni de servidor.

Paso 1: el orchestrator (plano de control, embebido en la TUI y en la biblioteca de Python) conecta con cada nodo mediante un handshake Connect/Accept en el que cada extremo declara identificador y rol; el receptor construye un handle tipado (WorkerHandle, ParamServerHandle…) que solo expone los mensajes válidos para ese rol, y el compilador rechaza los demás.

Paso 2: la especificación que recibe cada nodo define si es worker o servidor. Paso 3: los gradientes y parámetros viajan entre nodos; el orchestrator queda fuera del plano de datos.

Crates: comms, machine_learning, parameter_server, worker, orchestrator, node, orchestui, orchestra-py.

- Transición: «Adentro de cada nodo corre el motor de redes neuronales, y una decisión suya condiciona todo lo demás.»
- Respaldo: docs/report/solucion.md (Componentes generales · Nodo agnóstico de rol) · README.md (Architecture)

## 08 · Motor: modelo como buffer plano

- Expositor sugerido: Alejo Ordoñez
- Tiempo estimado: 1:00 (acumulado al terminar: 11:00)
- Idea principal: Una decisión del motor de redes neuronales habilita la comunicación sin copias de todo el sistema.
- Desarrollo oral:

La biblioteca está construida sobre ndarray: capas densas, convolucionales, max pooling, reshape y activaciones; inicializaciones Xavier-Glorot y Kaiming-He; optimizadores detrás de un mismo trait.

La decisión clave: el modelo es un buffer contiguo; las capas son vistas sobre tramos. La alternativa (un árbol de objetos con sus pesos) obligaba a serializar cada capa y, en Rust, a acoplar el motor con comms para prestar referencias mutables.

Paso 1: particionar entre servidores es delimitar rangos. Paso 2: enviar es reinterpretar memoria (bytemuck).

Limitación honesta: los modelos son secuencias de capas; arquitecturas con conexiones de salto como U-Net no se pueden expresar hoy.

- Transición: Traspaso: «Marcos va a contar cómo se sincronizan esas réplicas en cada estrategia.»
- Respaldo: docs/report/solucion.md (Motor de Redes Neuronales · Modelo como buffer plano)

## 09 · Parameter Server

- Expositor sugerido: Marcos Bianchi Fernández
- Tiempo estimado: 2:00 (acumulado al terminar: 13:00)
- Idea principal: En PS, el estado del optimizador vive en el servidor y el cuello de botella es su enlace de red.
- Desarrollo oral:

Ciclo implementado (worker/src/workers/parameter_server.rs y parameter_server/src/synchronization): (1) el worker hace pull de los parámetros; (2) entrena sobre su partición; (3) empuja el gradiente; (4) el servidor acumula; con `barrier` espera a los N_W y el líder aplica la actualización, con `non_blocking` aplica en cuanto llega. El servidor promedia (suma / N_W) y aplica su propia instancia del optimizador con estado.

Paso 5: con más de un servidor, el particionado es por capa completa. El orchestrator asigna capas de mayor a menor tamaño al servidor menos cargado. Ningún servidor necesita información de otro porque los optimizadores operan elemento a elemento.

Almacenamiento: BlockingStore (lock por fragmento y doble buffer de gradientes) o WildStore (estilo Hogwild!, sin protección).

- Transición: «La alternativa sin servidor central es All-Reduce.»
- Respaldo: docs/report/solucion.md (Parameter Server · Particionado respetando límites de capa · Actualización sin bloqueos)

## 10 · Ring All-Reduce

- Expositor sugerido: Marcos Bianchi Fernández
- Tiempo estimado: 2:15 (acumulado al terminar: 15:15)
- Idea principal: All-Reduce reparte la comunicación en un anillo sin centro: cada worker termina con el mismo gradiente promediado.
- Desarrollo oral:

Variante implementada: Ring All-Reduce de Patarasuk y Yuan (2009), en worker/src/middlewares/worker_ring.rs. El diagrama replica el orden del código con N_W = 4.

Paso 1: cada worker parte su gradiente en N_W fragmentos. Paso 2, reduce-scatter: en cada iteración el worker p envía un fragmento al siguiente y suma el que recibe del anterior; tras N_W−1 iteraciones, cada worker tiene la suma completa de un fragmento distinto (el p+1). Paso 3, all-gather: esos fragmentos reducidos recorren el anillo copiándose, sin sumar. Paso 4: cada worker divide por N_W y aplica localmente el optimizador. El promedio importa: un error real del proyecto fue sumar sin promediar (ver apéndice A5).

El orden del anillo no es arbitrario: lo decide el orchestrator a partir de la latencia medida (diapositiva de coordinación).

- Transición: «¿Cuánto cuesta sincronizar en cada caso? Lo derivamos analíticamente.»
- Respaldo: docs/report/solucion.md (Ring All-Reduce) · código: worker/src/middlewares/worker_ring.rs

## 11 · Costo de sincronización

- Expositor sugerido: Marcos Bianchi Fernández
- Tiempo estimado: 1:30 (acumulado al terminar: 16:45)
- Idea principal: El modelo analítico predice qué esperar antes de medir: PS con un servidor escala linealmente; AR queda acotado.
- Desarrollo oral:

Expresiones de elaboración propia (Resultados): T_total = N_epochs · (T_epoch/N_W + d_sync) + d_init.

Paso 1: PS con un servidor: ida de parámetros y vuelta de gradientes por una sola placa de red: 2·N_W·S/R. Crece con los workers.

Paso 2: Ring AR: se hace dar la vuelta a 1/N_W del gradiente dos veces: 2(1 − 1/N_W)·S/R, acotado por 2S/R.

Curvas con S/R ≈ 0,1 s, el valor de la campaña (LeNet-5 en f16 a 10 Mbps). Con N_S servidores: 2·(N_W/N_S)·S/R (apéndice A3).

Consecuencia: aun en AR la aceleración queda acotada, porque al sumar workers el cómputo tiende a cero pero d_sync tiende a 2S/R.

- Transición: «¿Y si se pudiera combinar lo mejor de los dos regímenes? Esa es la idea de Strategy-Switch.»
- Respaldo: docs/report/resultados.md (secciones PS sincrónico, Ring All-Reduce, Multi-PS) · simulaciones/experimentacion.md (S, R)

## 12 · Strategy-Switch: la idea

- Expositor sugerido: Marcos Bianchi Fernández
- Tiempo estimado: 1:00 (acumulado al terminar: 17:45)
- Idea principal: Strategy-Switch busca un buen mínimo con All-Reduce y lo sigue bajando con PS asincrónico.
- Desarrollo oral:

Propuesto por Provatas et al. (2025). Fase 1: All-Reduce sincrónico, que preserva la coherencia mientras los workers todavía no acordaron un mínimo. Fase 2: Parameter Server asincrónico, que elimina la espera.

Paso 1: el disparador es la variación relativa media de la pérdida en las últimas w = 6 rondas: cuando varió menos de 1 % por ronda, se cambia. Ventanas más grandes demoraban el cambio; más chicas lo adelantaban por ruido.

Paso 2: diferencia con el paper: calculamos s sobre la pérdida de entrenamiento máxima entre workers (evita una pasada de validación cuyo costo se sumaría al tiempo medido, y es conservador).

La curva de la figura es ilustrativa: muestra el mecanismo de la ventana, no un resultado.

- Transición: «Lo interesante es cómo se implementa ese cambio en un sistema distribuido que ya está corriendo.»
- Respaldo: docs/report/estado_del_arte.md (Strategy-Switch) · solucion.md (Strategy-Switch) · código: orchestrator/src/sessions/switch_tracker.rs

## 13 · Strategy-Switch en O.N.O.

- Expositor sugerido: Marcos Bianchi Fernández
- Tiempo estimado: 2:00 (acumulado al terminar: 19:45)
- Idea principal: Qué cambia, qué lo dispara, quién decide y cómo se transiciona, tal como está implementado.
- Desarrollo oral:

Fase 1: los workers entrenan en anillo y reportan su pérdida al orchestrator en cada ronda.

Paso 1: el EventListener del orchestrator toma la pérdida máxima de la ronda y la registra en el SwitchTracker (ventana de 6, umbral 0,01).

Paso 2: cuando s ≤ 0,01, el orchestrator difunde las acciones precalculadas: Upgrade a los nservers nodos que serán servidores (elegidos al inicio por latencia) y Switch al resto, con las direcciones y el orden de los servidores.

Paso 3: el nodo promovido toma de sus parámetros actuales los rangos de su fragmento y arranca como servidor; el resto deja el anillo y se conecta como worker de PS. No se reinicia nada: es posible por la arquitectura agnóstica de rol.

Paso 4: continúa como Parameter Server con el synchronizer y el store configurados (asincrónico en el paper y en la campaña del informe). No hay selección automática de estrategia: el sistema solo decide el momento del cambio.

- Transición: «Para que esto funcione con buen rendimiento hubo decisiones en la capa de comunicación.»
- Respaldo: docs/report/solucion.md (Strategy-Switch · Nodo agnóstico de rol) · código: orchestrator/src/sessions/event_listener.rs, orchestrator/src/configs/adapter.rs, worker/src/workers/all_reduce.rs

## 14 · Comunicación: dos planos y compresión

- Expositor sugerido: Marcos Bianchi Fernández
- Tiempo estimado: 1:15 (acumulado al terminar: 21:00)
- Idea principal: Separar control de datos y comprimir sin perder información acumulada.
- Desarrollo oral:

Control: comandos poco frecuentes en JSON, inspeccionables. Datos: concentran el volumen; cada copia cuesta, así que viajan por reinterpretación de memoria y en media precisión.

Compresión (diagrama, pasos): (1) solo viajan los componentes por encima de un umbral, elegido para que pase una fracción r; el umbral se estima muestreando hasta 16384 componentes y quickselect (Aji y Heafield 2017), en tiempo esperado lineal. (2) Lo que no se envía queda en un residual persistente. (3) En la ronda siguiente el residual se suma al gradiente nuevo: un componente chico pero constante termina cruzando el umbral (error feedback, Lin et al. 2018).

Si la codificación dispersa pesa más que el gradiente denso en f16, el sistema cae a la versión densa: comprimir nunca resulta más caro.

Limitación: no se midió de forma aislada el compromiso entre tasa de compresión y convergencia (trabajo futuro).

- Transición: «Queda la coordinación: dónde ubicar a cada nodo y dónde aplicar el optimizador.»
- Respaldo: docs/report/solucion.md (Separación de planos · Compresión de gradientes con error feedback) · trabajos_futuros.md

## 15 · Coordinación: topología y optimizador

- Expositor sugerido: Marcos Bianchi Fernández
- Tiempo estimado: 1:00 (acumulado al terminar: 22:00)
- Idea principal: Ninguna diferencia de tiempo debería venir del orden en que el usuario escribió las direcciones.
- Desarrollo oral:

Antes de entrenar, cada par de nodos intercambia diez pings y se toma el mayor RTT. Se mide una sola vez (limitación en redes con carga variable).

Paso 1: All-Reduce: el orden del anillo que minimiza la latencia total es un circuito hamiltoniano de costo mínimo (viajante), resuelto en forma exacta con programación dinámica sobre subconjuntos (Held y Karp, 1962), hasta usize::BITS nodos. Parameter Server: se eligen los k servidores que minimizan la mayor latencia worker–servidor, por backtracking.

Paso 2: ubicación del optimizador: los workers acumulan un gradiente plano; el optimizador con estado se aplica después de agregar (en el servidor en PS; en cada worker sobre el promedio en AR). Si cada worker aplicara Adam sobre su gradiente local, acumularían momentos distintos y las réplicas divergirían.

- Transición: Traspaso: «Lorenzo va a mostrar cómo se usa el sistema y qué evidencia produjo.»
- Respaldo: docs/report/solucion.md (Selección de topología por latencia medida · Ubicación del optimizador) · código: orchestrator/src/calculator/

## 16 · Interfaces: TUI y Python

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 1:15 (acumulado al terminar: 23:15)
- Idea principal: Dos interfaces sobre el mismo orchestrator: una para ver el entrenamiento en vivo y otra para experimentar programáticamente.
- Desarrollo oral:

orchestui (ratatui + crossterm) embebe el orchestrator: se le indican model.json y training.json, se conecta a los nodos, asigna roles y muestra en vivo la pérdida promedio y por worker, la tabla de workers, el progreso y un log de eventos.

Paso 1: orchestra-py expone el mismo sistema como módulo nativo de Python vía PyO3 y maturin. El fragmento muestra la API real (funciones strategy_switch y orchestrate). Se construyó como crate separado: anotar los tipos existentes con #[pyclass] chocaba con genéricos, variantes de enum y Clone.

La suite de benchmarks del repositorio está escrita sobre esta interfaz.

- Transición: «La TUI no solo monitorea: permite ver la topología que eligió el sistema.»
- Respaldo: docs/report/solucion.md (Interfaces de Usuario · Interfaz de Python como crate separado) · orchestui/README.md · orchestra-py/local.py

## 17 · La TUI muestra la topología

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 1:00 (acumulado al terminar: 24:15)
- Idea principal: La topología que dibuja la TUI es la que construyó el orchestrator, y se actualiza cuando Strategy-Switch promueve un nodo.
- Desarrollo oral:

Capturas reales tomadas con tmux sobre una ejecución local (4 procesos node en la misma máquina, configs en presentation/captures/configs/). No forman parte de la campaña experimental.

Izquierda: el anillo de All-Reduce con la pérdida y la época de cada worker. Paso 1: Parameter Server, con workers a la izquierda y el servidor a la derecha («→ grads, ← params»). Paso 2: Strategy-Switch en el instante del cambio: el encabezado indica «switching W3 → PS» y la topología pasó a bipartita.

- Transición: «Pasemos a la evidencia: cómo medimos.»
- Respaldo: orchestui/src/ui/components/topology.rs · presentation/captures/raw/

## 18 · Cómo medimos

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 1:30 (acumulado al terminar: 25:45)
- Idea principal: Fijamos todo salvo la estrategia, y forzamos una relación modelo/red que haga visibles las diferencias.
- Desarrollo oral:

Entorno: clúster simulado en una máquina de 16 núcleos; cada contenedor restringido a un núcleo; Pumba limita la salida a 10 Mbps.

Paso 1: ¿por qué 10 Mbps? Con modelos útiles para MNIST, S/R real es demasiado chico para distinguir estrategias. En vez de inflar el modelo, se bajó R: S ≈ 1 Mbit en f16, d_trans ≈ 0,1 s. Referencia: T_1 = 600 s sin distribución.

Paso 2: antes de comparar tiempos hay que validar corrección: pruebas unitarias (la retropropagación se contrasta con valores calculados aparte), de integración (protocolos) y de aceptación (convergencia igual a la secuencial), todas en CI. Aun así, tres errores silenciosos aparecieron recién con los benchmarks (apéndice A5).

- Transición: «Primero, los algoritmos sincrónicos.»
- Respaldo: docs/report/experimentacion.md (Pruebas automatizadas · Entorno de experimentación y reproducibilidad)

## 19 · Resultados: sincrónicos

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 1:15 (acumulado al terminar: 27:00)
- Idea principal: Lo medido coincide con lo que predice el modelo analítico.
- Desarrollo oral:

Paso 1: Parameter Server tiene su mínimo en N_W = 6 y desde ahí crece: el servidor único concentra el tráfico (d_sync = 2·N_W·S/R).

Paso 2: All-Reduce no se degrada porque su delay está acotado; pero el tiempo deja de bajar, porque el término de cómputo tiende a cero y la sincronización no.

Lectura aproximada de la figura, por si preguntan: AR cerca de 290 s desde 6 workers; PS cerca de 340 s en 6 y algo más de 400 s en 16. Los valores exactos están en la figura del informe.

Aclarar: el eje Y de la figura no indica unidad; el informe trabaja en segundos (T_1 = 600 s).

- Transición: «¿Qué pasa si se rompe la barrera de sincronización?»
- Respaldo: docs/report/resultados.md (Resultados obtenidos) · figura execution_time_per_nodes_ar_vs_ps.png

## 20 · Resultados: asincrónicos

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 0:45 (acumulado al terminar: 27:45)
- Idea principal: En velocidad pura, los dos asincrónicos ganan.
- Desarrollo oral:

Ambos rompen la barrera estricta y el tiempo sigue bajando a medida que se suman workers, a diferencia de All-Reduce.

Paso 1: el objetivo de un algoritmo distribuido es acelerar garantizando convergencia; si no, el entrenamiento pierde sentido. Por eso miramos exactitud.

- Transición: «Veamos la exactitud final sobre el conjunto de prueba.»
- Respaldo: docs/report/resultados.md (Resultados obtenidos) · figura execution_times_async_parameter_server_and_strategy_switch.png

## 21 · Resultados: exactitud

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 1:30 (acumulado al terminar: 29:15)
- Idea principal: Strategy-Switch fue el mejor balance de los evaluados en esta campaña; la salvedad sobre WildStore es parte del resultado.
- Desarrollo oral:

PS asincrónico cae fuertemente al aumentar workers (en la figura, a valores cercanos al azar desde 8 workers). Es consistente con client drift: al principio del entrenamiento cada worker empuja en la dirección que le dicta su partición.

Paso 1: Strategy-Switch usa la desincronización solo cuando los mínimos ya se consensuaron: exactitud similar a la de los sincrónicos y velocidad similar a la del asincrónico.

Paso 2: decirlo antes de que lo pregunten: la variante asincrónica se ejecutó con WildStore (estilo Hogwild!, escrituras sin protección). Estos datos no permiten separar el client drift del efecto de esas escrituras. Repetir con BlockingStore es trabajo futuro.

Alcance: estas conclusiones valen para MNIST, LeNet-5 y este entorno simulado.

- Transición: «La misma base permitió también un estudio sobre una sola variable: los epochs offline.»
- Respaldo: docs/report/resultados.md (Resultados obtenidos, salvedad final) · conclusiones_capitulo4.md · riesgos.md

## 22 · Estudio derivado: epochs offline

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 1:15 (acumulado al terminar: 30:30)
- Idea principal: La base sirve también para aislar el efecto de una decisión de configuración.
- Desarrollo oral:

Estudio derivado de una monografía de Aprendizaje Profundo: AR y PS sincrónico, 2 a 10 nodos, E de 0 a 4; 11 h 15 min de ejecución en una máquina de 4 núcleos con red hogareña emulada.

Paso 1: la exactitud decrece monótonamente con E; un solo epoch offline (la mitad de sincronizaciones) ya cuesta cerca de un punto. La degradación crece con la cantidad de nodos (particiones más chicas y sesgadas).

Paso 2: las curvas de tiempo se solapan: en una sola máquina, ahorrar comunicación no compensa la contención de CPU. La tercera hipótesis (más inestabilidad en AR) no se confirmó.

Línea abierta: activar epochs offline recién cuando la pérdida se estabiliza, con un criterio análogo al de Strategy-Switch.

- Transición: «Con todo esto, qué aporta el trabajo y dónde están sus límites.»
- Respaldo: docs/report/estudio_derivado.md

## 23 · Aportes y limitaciones

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 1:15 (acumulado al terminar: 31:45)
- Idea principal: Ser explícitos con el alcance fortalece las conclusiones.
- Desarrollo oral:

Aportes: los tres algoritmos publicados implementados sobre la misma capa de comunicación, el mismo motor y el mismo plano de control, con instrumentación para producir mediciones; ese es el problema que planteamos.

Paso 1, limitaciones: las conclusiones valen para el problema, el modelo y el entorno medidos; para generalizar hacen falta otros datasets, modelos de otros tamaños y una red física. Otras: WildStore, sin tolerancia a fallos, topología fija tras la medición inicial, solo secuencias de capas, datasets en memoria, heterogeneidad de hardware no abordada, consumo energético y costo económico no medidos.

- Transición: «Para cerrar la exposición, las tres conclusiones.»
- Respaldo: docs/report/conclusiones.md · trabajos_futuros.md · impacto.md

## 24 · Conclusiones

- Expositor sugerido: Lorenzo Minervino
- Tiempo estimado: 0:45 (acumulado al terminar: 32:30)
- Idea principal: El sistema cumplió su propósito: atribuir diferencias al algoritmo y no a la implementación.
- Desarrollo oral:

(1) Con ancho de banda restringido, PS sincrónico con un servidor escala mal; Ring AR resuelve el escalado pero su aceleración queda acotada.

Paso 1: (2) eliminar la barrera reduce radicalmente el tiempo, pero en este experimento destruyó la convergencia (con la salvedad de WildStore).

Paso 2: (3) Strategy-Switch se comporta sincrónicamente en la fase crítica y conmuta después del consenso: exactitud de los sincrónicos en una fracción del tiempo. Empíricamente, la mejor alternativa de esta comparativa.

- Transición: «Ahora vamos a mostrarlo funcionando.»
- Respaldo: docs/report/conclusiones_capitulo4.md · conclusiones.md

## 25 · Ahora, la demo

- Expositor sugerido: [a completar]
- Tiempo estimado: 0:30 (acumulado al terminar: 33:00)
- Idea principal: [a completar por el equipo]
- Desarrollo oral:

[a completar: qué se muestra, en qué orden, qué ejecutar y qué resultado observar]

Guion en blanco: presentation/GUION_DEMO.md. Material de respaldo opcional: apéndice A1 (grabación real de Strategy-Switch en la TUI) y A2 (capturas reales).

- Transición: [a completar]
- Respaldo: [a completar]

## 26 · Cierre

- Expositor sugerido: Los tres
- Tiempo estimado: 2:00 (acumulado al terminar: 35:00)
- Idea principal: Volver al mensaje central después de la demo y abrir preguntas.
- Desarrollo oral:

Retomar en una frase lo que se vio en la demo y conectarlo con la contribución: la misma base ejecuta las tres estrategias y permite compararlas.

Recordar las tres ideas de la diapositiva. El repositorio es público y se mantiene al menos un año después de la defensa.

Preparar respuestas: WildStore vs. BlockingStore; por qué 10 Mbps; por qué la pérdida de entrenamiento y no la de validación; tolerancia a fallos; heterogeneidad no abordada; uso de herramientas de IA declarado en el informe. Para volver a una diapositiva: tecla G (índice) o número + Enter.

- Transición: —
- Respaldo: docs/report/conclusiones.md · anexos.md (repositorio)

## Apéndice · A1 · Grabación real de Strategy-Switch

- Expositor sugerido: —
- Tiempo estimado: 0:00
- Idea principal: Material de respaldo opcional, por si se quiere mostrar el cambio de estrategia sin ejecutar en vivo.
- Desarrollo oral:

Secuencia real de pantallas (4 nodos locales, subconjunto de 10.000 muestras de MNIST, configs en presentation/captures/configs/). Mostrar el anillo, el instante «switching → PS» y la topología bipartita posterior. Si se usa, aclarar que es una grabación.

- Transición: —
- Respaldo: presentation/captures/rec/ · presentation/captures/configs/training_strategy_switch.json

## Apéndice · A2 · Capturas reales de la TUI

- Expositor sugerido: —
- Tiempo estimado: 0:00
- Idea principal: Capturas de respaldo de las pantallas principales.
- Desarrollo oral:

Menú, ayuda de configuración y dashboard al finalizar una ejecución de Strategy-Switch (pérdida promedio, pérdida por worker, tabla de workers, parámetros finales y eventos).

Ojo: en esa ejecución la tabla muestra 99–100 epochs con max_epochs = 60; ver README, sección «Diferencias detectadas».

- Transición: —
- Respaldo: presentation/captures/raw/

## Apéndice · A3 · Derivación de d_sync

- Expositor sugerido: —
- Tiempo estimado: 0:00
- Idea principal: Respaldo para preguntas sobre el modelo analítico.
- Desarrollo oral:

Con N_S = N_W, Multi-PS alcanza 2S/R, el mismo límite que AR cuando N_W crece.

- Transición: —
- Respaldo: docs/report/resultados.md

## Apéndice · A4 · Detención temprana y disparador

- Expositor sugerido: —
- Tiempo estimado: 0:00
- Idea principal: Respaldo: dos criterios distintos sobre la misma señal.
- Desarrollo oral:

Ambos usan la pérdida de entrenamiento máxima para evitar una pasada de evaluación por ronda y para ser conservadores.

- Transición: —
- Respaldo: docs/report/solucion.md (Plano de Control y Orquestación) · código: orchestrator/src/sessions/

## Apéndice · A5 · Lecciones: errores silenciosos

- Expositor sugerido: —
- Tiempo estimado: 0:00
- Idea principal: Respaldo: la lección central de ingeniería del trabajo.
- Desarrollo oral:

Conviene tener una medición end-to-end desde temprano. Otra lección: «make it work, make it right, make it fast».

- Transición: —
- Respaldo: docs/report/riesgos.md (Lecciones aprendidas)

## Apéndice · A6 · Trabajo futuro

- Expositor sugerido: —
- Tiempo estimado: 0:00
- Idea principal: Respaldo para preguntas sobre cómo seguir.
- Desarrollo oral:

Varias líneas tienen el diseño especificado en el repositorio. También quedan medir costo económico y consumo energético.

- Transición: —
- Respaldo: docs/report/trabajos_futuros.md · impacto.md
