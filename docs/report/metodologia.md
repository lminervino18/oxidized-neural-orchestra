\newpage
# Metodología aplicada

Esta sección actualiza lo planteado en la propuesta, contrastando lo previsto con lo efectivamente ejecutado.

## Gestión y roles

Se estableció un compromiso de 500 horas por estudiante a lo largo de dos cuatrimestres, lo que representa un promedio de unas 16 horas semanales por persona sobre 32 semanas. El Ing. Ricardo A. Veiga cumplió la función de tutor y el Dr. Ing. J. Ignacio Alvarez-Hamelin la de co-tutor y orientador en el campo de desempeño, en el marco del CoNexDat, Grupo de Redes Complejas y Comunicación de Datos de la Facultad de Ingeniería.

Los tres estudiantes participaron de todas las etapas de análisis, implementación y pruebas. Sin perjuicio de ello, cada integrante concentró su foco principal en un conjunto de componentes, según se detalla en *Aportes individuales*. Esa especialización, que no estaba impuesta por la propuesta sino que emergió del desarrollo, tuvo un efecto secundario que conviene señalar: la revisión cruzada de código se volvió el principal mecanismo de difusión de conocimiento entre componentes, y en varios casos las discusiones de revisión terminaron modificando decisiones de diseño y no solo detalles de implementación.

Se mantuvieron reuniones periódicas en formato virtual con los tutores, con la cadencia comprometida en el Acta de Acuerdo de una periodicidad igual o menor a las dos semanas, para informar el avance, definir prioridades y planificar la etapa siguiente.

## Proceso de desarrollo

El método de desarrollo fue **incremental**, organizado en iteraciones con entregables intermedios. La unidad de entrega fue la *pull request*: cada funcionalidad se desarrolló en una rama propia que se integraba a la rama principal previa revisión. A lo largo del proyecto se produjeron **989 commits, 108 pull requests y 80 issues**, entre septiembre de 2025 y julio de 2026.

El código y todos los artefactos versionables se gestionaron con **Git** sobre un repositorio alojado en **GitHub**. Los mensajes de los commits siguen la convención de *Conventional Commits*, lo que mantiene un historial legible y trazable. Los entregables intermedios que marcaron el avance fueron, en orden: la capa de comunicación funcionando de punta a punta, el primer entrenamiento distribuido completo con Parameter Server, el motor de redes neuronales convergiendo sobre MNIST, la interfaz de terminal mostrando un entrenamiento en vivo, el segundo algoritmo (All-Reduce en anillo), y finalmente la suite de benchmarks con sus resultados.

Un patrón de proceso que el equipo adoptó de forma sistemática, y que corresponde declarar porque condiciona la lectura del historial, es el de **cerrar una rama de trabajo en curso y abrir una nueva y limpia** en lugar de iterar sobre la existente. Ocurrió en los componentes centrales del sistema: el servidor de parámetros y el worker llegaron a su diseño definitivo después de tres intentos cerrados cada uno, y lo mismo sucedió con la capa de comunicación, la distribución del conjunto de datos, la codificación en media precisión y la abstracción de All-Reduce. Como disciplina produce revisiones acotadas y un historial de la rama principal limpio; su costo, discutido en *Riesgos materializados y lecciones aprendidas*, es que la traza del razonamiento queda repartida entre pull requests cerradas.

## Seguimiento, tickets y gestión del alcance

El seguimiento del proceso, la gestión de tickets y el registro de errores se realizaron con las *issues* del repositorio en GitHub, categorizadas mediante etiquetas por tipo (*bug*, *enhancement*, *documentation*) y por componente del sistema (*parameter-server*, *comms*, *worker*, *ffi*, *tui*, entre otras). Las issues se usaron además, y esto excedió lo previsto en la propuesta, como **espacio de debate de diseño**: varias de las decisiones de arquitectura documentadas en este informe se resolvieron enteramente en el hilo de comentarios de una issue, con enumeración de alternativas y su descarte razonado. Ese registro fue el insumo principal para reconstruir el proceso.

Los errores se clasificaron por criticidad, distinguiendo los que impedían el avance del trabajo de los que admitían corrección diferida. La gestión del alcance se materializó en dos momentos explícitos: el diferimiento de la tolerancia a fallos hasta tener un producto mínimo funcionando, tomado al inicio bajo el criterio de que *"estas features van al final, una vez que tengamos el MVP andando"*, y una sesión de definición de alcance previa al cierre en la que se descartaron de forma deliberada cinco líneas de trabajo que no iban a llegar. Ambos se detallan en la sección de riesgos.

## Automatización

La construcción y las pruebas del sistema se ejecutan con las herramientas del ecosistema de Rust (`cargo build`, `cargo test`, `cargo clippy` para análisis estático), y los entornos de simulación se levantan de forma automatizada mediante **Docker** y un `Makefile` autodocumentado, lo que hace reproducible el despliegue de las distintas configuraciones de nodos. Para el código Python se incorporó `ruff` como analizador estático y formateador.

El grado de automatización alcanzado debe declararse con honestidad. La construcción, las pruebas, el análisis estático, el levantamiento de entornos multinodo y la ejecución de campañas completas de benchmarks están **totalmente automatizados y son de ejecución desatendida**: la campaña final de 38 configuraciones se ejecutó de un solo comando durante 5 horas y 39 minutos sin intervención. En cambio, la **integración continua no llegó a funcionar**. La propuesta ya la señalaba como una cuestión no definida cuya adopción dependería de la evolución del proyecto; lo que ocurrió es peor que no haberla adoptado: se escribió un flujo de trabajo de integración continua que **nunca se ejecutó porque residía en un directorio con el nombre mal escrito**, y el error recién se detectó en la limpieza final del repositorio, a dos semanas del cierre. La verificación previa a la integración quedó, en los hechos, a cargo de la ejecución local de las pruebas por parte de quien abría la pull request y de quien la revisaba.

## Criterios de aceptación

Cada pull request se sometió a revisión de código por parte de otro integrante antes de integrarse. Una entrega se consideró aceptada cuando se cumplían de forma conjunta tres condiciones: la revisión de código fue aprobada, la totalidad de las pruebas automatizadas asociadas se ejecutaba con éxito, y la funcionalidad requerida quedaba demostrada mediante una prueba de aceptación. Para las funcionalidades que involucran coordinación entre nodos, esa prueba de aceptación consistió en una ejecución completa de entrenamiento distribuido sobre el entorno simulado en Docker, verificando que la convergencia obtenida fuera consistente con la del entrenamiento secuencial equivalente.

Este criterio funcionó, pero la experiencia del proyecto muestra que era **insuficiente para detectar errores de corrección distribuida silenciosos**, aquellos que no provocan fallos sino que degradan la calidad del modelo. Se retoma en las lecciones aprendidas.

## Artefactos e indicadores

Como artefactos de gestión se llevaron minutas de las reuniones periódicas, el registro del alcance y el registro de riesgos, revisado en las reuniones de seguimiento. A ellos se sumaron dos artefactos técnicos no previstos que resultaron centrales: el **documento canónico del esquema de configuración**, creado para eliminar la divergencia entre tres copias distintas de la misma documentación, y el **borrador de arquitectura**, que sostuvo las discusiones de diseño en las etapas iniciales.

Sobre la calidad del proceso se definieron tres indicadores: el tiempo entre apertura y cierre de una issue, la proporción de pull requests que requieren correcciones tras la revisión, y el desvío entre horas planificadas y efectivamente dedicadas. Este último funcionó también como indicador de costos, dado que el trabajo se desarrolló sobre hardware propio y con herramientas de código abierto, de modo que el costo del proyecto se reduce a las horas-persona invertidas.

Sobre la calidad del producto se consideraron la proporción de pruebas exitosas sobre el total y la cantidad de advertencias reportadas por el análisis estático, esta última mantenida en cero mediante `clippy` de forma sostenida. La **cobertura de pruebas**, en cambio, se relevó de manera irregular y no se sostuvo como indicador continuo. Ese incumplimiento tuvo consecuencias concretas y verificables: la variante de actualización sin bloqueos de los parámetros nunca tuvo cobertura, y el error que la invalidaba permaneció oculto hasta una auditoría tardía; y un error en la propagación hacia atrás del *max-pooling* quedó enmascarado porque las pruebas existentes usaban un único canal y un único elemento de lote.
