\newpage
# Metodología aplicada

Esta sección actualiza lo planteado en la propuesta, contrastando lo previsto con lo efectivamente ejecutado.

## Gestión y roles

Se estableció un compromiso de 500 horas por estudiante a lo largo de dos cuatrimestres, lo que representa un promedio de unas 16 horas semanales por persona sobre 32 semanas. El Ing. Ricardo A. Veiga cumplió la función de tutor y el Dr. Ing. J. Ignacio Alvarez-Hamelin la de co-tutor y orientador en el campo de desempeño, en el marco del CoNexDat, Grupo de Redes Complejas y Comunicación de Datos de la Facultad de Ingeniería.

Los tres estudiantes participaron de todas las etapas de análisis, implementación y pruebas, y cada integrante concentró su foco principal en un conjunto de componentes, según se detalla en *Aportes individuales*. Esa especialización, que emergió del desarrollo y no estaba impuesta por la propuesta, tuvo un efecto secundario relevante: la revisión cruzada de código se volvió el principal mecanismo de difusión de conocimiento entre componentes, y en varios casos las discusiones de revisión terminaron modificando decisiones de diseño y no solo detalles de implementación.

La coordinación se sostuvo en dos planos. Con los tutores se mantuvieron reuniones virtuales para informar el avance, definir prioridades y planificar próximas etapas, adecuando los encuentros según los hitos y necesidades del proyecto dentro del marco acordado. Entre los integrantes, la coordinación fue continua y de cadencia diaria, apoyada en un canal de mensajería para las decisiones rápidas y en las issues del repositorio para las que requerían registro.

## Proceso de desarrollo

El método de desarrollo fue incremental, organizado en iteraciones con entregables intermedios. La unidad de entrega fue en forma de *pull requests*: cada funcionalidad se desarrolló en una rama propia que se integraba a la rama principal con previa revisión de los demás integrantes. A lo largo del proyecto se produjeron +1100 commits, +120 pull requests y +85 issues, entre septiembre de 2025 y julio de 2026.

El código y todos los artefactos versionables se gestionaron con Git sobre un repositorio alojado en GitHub, y los mensajes de los commits siguen la convención de *Conventional Commits*. Los entregables intermedios que marcaron el avance fueron, en orden: la capa de comunicación funcionando de punta a punta, el primer entrenamiento distribuido completo con Parameter Server emulando una compuerta lógica XOR, el motor de redes neuronales convergiendo sobre MNIST, la interfaz de terminal mostrando un entrenamiento en vivo, All-Reduce en anillo, la libreria ffi de Python, y finalmente la suite de benchmarks con sus resultados.

## Seguimiento, tickets y gestión del alcance

El seguimiento del proceso, la gestión de tickets y el registro de errores se realizaron con las *issues* del repositorio, categorizadas mediante etiquetas por tipo (*feature*, *bug*, *enhancement*, *documentation*) y por componente del sistema (*parameter-server*, *comms*, *worker*, *ffi*, *tui*, entre otras). Las issues se usaron además, y esto excedió lo previsto en la propuesta, como espacio de debate de diseño: varias de las decisiones de arquitectura documentadas en este informe se resolvieron enteramente en el hilo de comentarios de un issue, con enumeración de alternativas y su descarte razonado. Ese registro fue el insumo principal para reconstruir el proceso.

Los errores se clasificaron por criticidad, distinguiendo los que impedían el avance del trabajo de los que admitían corrección diferida. La gestión del alcance se materializó en dos momentos explícitos: el diferimiento de las funcionalidades no esenciales hasta tener un producto mínimo funcionando, tomado al inicio bajo el criterio de que *"estas features van al final, una vez que tengamos el MVP andando"*, y una sesión de definición de alcance previa al cierre en la que se descartaron de forma deliberada las líneas que no iban a llegar. Ambos se detallan en la sección de riesgos.

## Automatización

La construcción y las pruebas del sistema se ejecutan con las herramientas del ecosistema de Rust (`cargo test`), y los entornos de simulación se levantan de forma automatizada mediante Docker y un `Makefile` autodocumentado, lo que hace reproducible el despliegue de las distintas configuraciones de nodos.

La construcción, las pruebas, el análisis estático, la ejecución de entornos multinodo y de benchmarks completos quedaron totalmente automatizados y de ejecución desatendida: el conjunto final de 38 configuraciones se ejecutó de un solo comando durante ~6 horas sin intervención. La suite de tests se ejecuta cada vez que se commitea en un Pull Requests que mergea en la rama principal mediante github actions, es obligatorio que la totalidad de las pruebas pase de forma satisfactoria para habilitar el merge.

## Criterios de aceptación

Cada pull request se sometió a revisión de código por parte de otro integrante antes de integrarse. Una entrega se consideró aceptada cuando se cumplían de forma conjunta tres condiciones: la revisión de código fue aprobada, la totalidad de las pruebas automatizadas asociadas se ejecutaba con éxito, y la funcionalidad requerida quedaba demostrada mediante una prueba de aceptación. Para las funcionalidades que involucran coordinación entre nodos, esa prueba consistió en una ejecución completa de entrenamiento distribuido sobre el entorno simulado, verificando que la convergencia obtenida fuera consistente con la del entrenamiento secuencial equivalente.

El criterio funcionó para las fallas visibles, pero resultó insuficiente para detectar errores de corrección distribuida silenciosos, aquellos que no provocan fallos sino que degradan la calidad del modelo. Este tipo de errores se solventó con la introducción de benchmarks y se detalla en las lecciones aprendidas.

## Artefactos e Indicadores

Como herramienta de gestión se utilizó GitHub Projects, cada ticket nació en el `Backlog`, siguió por una etapa de `En Progreso`, eventualmente cuando el feature alcanza un estado estable se pasa a `En Revisión` y finalmente quedo archivada en `Entregada`.

Algunos de los artefactos de gestión que utilizamos durante el proceso fueron minutas de reunión para una mejor organización previa a lars reuniones con los tutores, diagramas de los temas atacados en las reuniones si lo ameritaban, por ejemplo diagramas de robustez para los distintos algoritmos distribuidos que se implementaron, diagramas de ejecución del flujo de entrenamiento agnostico del tipo de entrenamiento mostrando la comunicación entre orquestador y nodos trabajadores.

Sobre la calidad del producto se definieron los siguientes indicadores: la proporción de cambios requieridos luego de publicar un **pull request** como listo para revisión, y el desvío entre horas planificadas y efectivamente dedicadas. Este último funcionó también como indicador de costos.

Sobre la calidad del producto se consideraron la proporción de pruebas exitosas sobre el total, la cantidad de advertencias del análisis estático y resultados obtenidos de la ejecución de benchmarks. En cuanto al análisis estático, se mantuvo en 0 warnings de forma sostenida durante toda la evolución del proyecto.

## Riesgos Iniciales

- **Complejidad de la implementación distribuida en Rust.** La coordinación entre nodos y el manejo de la concurrencia son intrínsecamente complejos y propensos a errores sutiles, como bloqueos o condiciones de carrera. Como mitigación, el desarrollo es incremental y se apoya en las garantías de *fearless-concurrency* del lenguaje y en una cobertura de pruebas unitarias y de integración que acompaña cada avance.

- **Disponibilidad de hardware para simulaciones representativas.** Reproducir un entorno distribuido realista requiere de múltiples máquinas, algo no siempre disponible. Para mitigarlo, las simulaciones se ejecutan sobre contenedores Docker con límites de recursos por nodo, lo que permite emular configuraciones heterogéneas de forma controlada; se asume que los resultados así obtenidos son una aproximación y no un reemplazo exacto de un despliegue sobre máquinas físicas separadas.

- **Dependencia de trabajos previos.** Parte del trabajo se apoya en algoritmos y resultados publicados, como *Strategy-Switch*, cuya interpretación e implementación pueden diferir de lo documentado. La mitigación consiste en implementar versiones de referencia de los algoritmos base y validarlas antes de construir mejoras sobre ellas.

- **Amplitud del alcance y estimación de tiempos.** El trabajo abarca tanto el sistema base como varias líneas de optimización (comunicación, sincronización y carga en configuraciones heterogéneas), lo que dificulta la estimación del esfuerzo. Para mitigarlo, se prioriza primero un sistema base funcional junto con las implementaciones de referencia, y las optimizaciones se abordan de forma incremental según el avance, con reuniones semanales de seguimiento para reajustar prioridades.
