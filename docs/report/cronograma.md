\newpage
# Cronograma de las actividades realizadas

El plan de actividades de la propuesta fijó dieciséis tareas repartidas en cuatro hitos sobre dos cuatrimestres, con una carga estimada de 1500 horas de equipo. Esta sección contrasta ese plan con la ejecución real, que se extendió entre agosto de 2025 y julio de 2026.

## Fases efectivamente recorridas

El desarrollo puede reconstruirse en once fases. Las fechas son aproximadas y corresponden a rangos de actividad, ya que varias fases se solaparon.

| Fase | Período           | Contenido                                                           |
|------|-------------------|---------------------------------------------------------------------|
| 0    | Ago–Nov 2025      | Propuesta, relevamiento bibliográfico y un primer diseño descartado |
| 1    | Nov 2025–Ene 2026 | Capa de comunicación                                                |
| 2    | Dic 2025–Feb 2026 | Parameter Server y Worker                                           |
| 3    | Ene–Mar 2026      | Motor de redes neuronales                                           |
| 4    | Feb–Mar 2026      | Orchestrator, interfaz de terminal e interfaz de Python             |
| 5    | Mar 2026          | Multi-servidor, conjuntos de datos y eficiencia de red              |
| 6    | Mar–May 2026      | All-Reduce en anillo y unificación del nodo                         |
| 7    | Abr–May 2026      | Early stopping y la saga de bloqueos                                |
| 8    | May–Jun 2026      | Strategy Switch, topología y benchmarks                             |
| 9    | Jun 2026          | Endurecimiento y exposición de funcionalidad                        |
| 10   | Jun–Jul 2026      | Consolidación del benchmark y auditoría de corrección               |
| 11   | Jul 2026          | Optimización final y documentación                                  |

: Fases del desarrollo efectivamente recorridas.

## Contraste entre lo planificado y lo ejecutado

**Lo que se cumplió según lo previsto.** El núcleo del plan se completó: el sistema base parametrizable en Rust con su biblioteca de redes neuronales propia (tarea 4), los tres algoritmos (tareas 5 a 7), las optimizaciones de comunicación y sincronización (tareas 8 y 9), las dos interfaces (tarea 10), las pruebas (tarea 11), la simulación sobre distintas configuraciones de nodos (tarea 12) y el análisis de resultados con sus gráficos (tarea 14).

**Lo que llevó más tiempo del estimado.** Tres bloques se desviaron de forma significativa. El primero es la fase 0: casi cinco meses dedicados a documentación, relevamiento y trámites antes de escribir código productivo, con un primer diseño de arquitectura descartado por completo. El segundo es el motor de redes neuronales (tarea 4), cuya derivación manual de la propagación hacia atrás y la implementación de las capas convolucionales insumieron alrededor de dos meses de depuración de gradientes, muy por encima de lo previsto para un componente que la propuesta trataba como parte de una tarea mayor. El tercero es la saga de bloqueos de coordinación desatada por el early stopping (fase 7), que no figuraba en el plan bajo ninguna tarea y consumió un mes completo.

**Lo que no se ejecutó.** La tarea 13, optimización de carga de cómputo en configuraciones heterogéneas, con 150 horas estimadas, no se abordó: dependía de disponer de máquinas físicas con capacidades distintas, algo que el entorno de simulación no permite representar de forma convincente.

**Lo que se ejecutó sin estar planificado.** Aparecieron cuatro líneas no previstas: la selección de topología mediante estadísticas de latencia, la auditoría de corrección distribuida que surgió de la construcción de la suite de benchmarks, la optimización del motor de convolución, y las dos monografías que constituyen los estudios experimentales de este informe, que excedieron en profundidad lo que la tarea 14 contemplaba.

**Una observación sobre la estimación.** La distribución de horas de la propuesta asignaba 200 horas a investigar las implementaciones distribuidas de TensorFlow y PyTorch (tareas 2 y 3). En la práctica ese estudio se realizó de forma mucho más acotada y en modalidad de consulta puntual, y el esfuerzo se redistribuyó hacia la implementación y la depuración del motor de aprendizaje profundo. La lección de estimación es la habitual y no por conocida menos real: el esfuerzo se subestima sistemáticamente en aquello que no se conoce de antemano, que en este trabajo fue todo lo relativo a la corrección numérica del entrenamiento y a la coordinación distribuida.

## Carga horaria efectiva

| Nro. | Tarea                                                           | Horas estimadas | Horas reales     |
|------|-----------------------------------------------------------------|-----------------|------------------|
| 1    | Leer y analizar los trabajos previos                            | 100             | 60               |
| 2    | Investigar la implementación distribuida de TensorFlow          | 50              | 5                |
| 3    | Investigar la implementación distribuida de PyTorch             | 50              | 25               |
| 4    | Desarrollar el sistema distribuido en Rust                      | 150             | 400              |
| 5    | Implementar Parameter Server                                    | 100             | 200              |
| 6    | Implementar All-Reduce                                          | 100             | 50               |
| 7    | Implementar Strategy-Switch                                     | 100             | 50               |
| 8    | Optimizaciones de comunicación entre nodos                      | 150             | 100              |
| 9    | Optimizaciones de sincronización de las copias del modelo       | 150             | 100              |
| 10   | Interfaz de funciones externas en Python e interfaz de terminal | 50              | 100              |
| 11   | Pruebas unitarias y de integración                              | 50              | 100              |
| 12   | Simulación sobre distintas configuraciones de nodos             | 100             | 150              |
| 13   | Optimizaciones de carga en configuraciones heterogéneas         | 150             | 0 (no ejecutada) |
| 14   | Análisis de resultados y gráficos                               | 100             | 100              |
| 15   | Documentación del código y del proceso                          | 50              | 100              |
| 16   | Informe final                                                   | 50              | 50               |
|      | Total                                                           | 1500            | 1590             |

: Contraste entre la carga horaria estimada en la propuesta y la efectivamente dedicada.

## Carga horaria por integrante

| Tarea                                                           | Lorenzo Minervino | Alejo Ordóñez | Marcos Bianchi |
|-----------------------------------------------------------------|-------------------|---------------|----------------|
| Cantidad de horas insumidas                                     |       545         |      520      |      525       |
| Leer y analizar los trabajos previos                            |        20         |       25      |       15       |
| Investigar la implementación distribuida de TensorFlow          |         5         |        0      |        0       |
| Investigar la implementación distribuida de PyTorch             |         0         |       20      |        5       |
| Desarrollar el sistema distribuido en Rust                      |        90         |      110      |      200       |
| Implementar Parameter Server                                    |        10         |       90      |      100       |
| Implementar All-Reduce                                          |         0         |       20      |       30       |
| Implementar Strategy-Switch                                     |         0         |       15      |       35       |
| Optimizaciones de comunicación entre nodos                      |         0         |       50      |       50       |
| Optimizaciones de sincronización de las copias del modelo       |        80         |       20      |        0       |
| Interfaz de funciones externas en Python e interfaz de terminal |       100         |        0      |        0       |
| Pruebas unitarias y de integración                              |        60         |       20      |       20       |
| Simulación sobre distintas configuraciones de nodos             |       100         |       30      |       20       |
| Optimizaciones de carga en configuraciones heterogéneas         |         0         |        0      |        0       |
| Análisis de resultados y gráficos                               |        40         |       50      |       10       |
| Documentación del código y del proceso                          |        10         |       60      |       30       |
| Informe final                                                   |        30         |       10      |       10       |
| Total                                                           |                   |               |                |
