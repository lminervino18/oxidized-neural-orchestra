\newpage
# Aportes individuales

El desarrollo del trabajo se planteó como una responsabilidad compartida: las etapas de análisis, revisión de código y pruebas fueron llevadas adelante por los tres integrantes en todas las tareas. Sin perjuicio de ello, cada integrante concentró su foco principal en un conjunto de componentes, que se detalla a continuación y que se corresponde con lo previsto en la propuesta.

**Alejo Ordoñez (Padrón 108397).** Núcleo de aprendizaje automático: el modelo, las capas, los optimizadores y las funciones de pérdida. Las capas convolucionales y de *max-pooling*, incluida la reformulación de la convolución como multiplicación de matrices que produjo la mejora final de rendimiento del motor. El manejo y la distribución de los conjuntos de datos. La monografía sobre *Strategy-Switch*, que constituye el tercer estudio experimental del informe.

**Lorenzo Minervino (Padrón 107863).** El módulo del Worker y la estrategia *All-Reduce* en anillo. La interfaz de funciones externas en Python y la interfaz interactiva de terminal. La suite de benchmarks y el análisis experimental de resultados. La monografía comparativa entre *Parameter Server* y *All-Reduce*, que constituye el primer estudio experimental del informe.

**Marcos Bianchi Fernández (Padrón 108921).** Los algoritmos de entrenamiento distribuido y su coordinación. El módulo de comunicaciones y el protocolo entre nodos, incluidas las técnicas de reducción de tráfico. El módulo del Servidor de parámetros. La monografía sobre el impacto de las épocas offline, que constituye el segundo estudio experimental del informe.


|                                                                 | Alejo Ordoñez | Lorenzo Minervino | Marcos Bianchi Fernández |
|-----------------------------------------------------------------|---|---|---|
| **Cantidad de horas insumidas**                                 | 520 | 545 | 525 |
| Leer y analizar los trabajos previos                            | 25 | 20 | 15 |
| Investigar la implementación distribuida de TensorFlow          | 0 | 5 | 0 |
| Investigar la implementación distribuida de PyTorch             | 20 | 0 | 5 |
| Desarrollar el sistema distribuido en Rust                      | 120 | 70 | 210 |
| Implementar Parameter Server                                    | 90 | 10 | 100 |
| Implementar All-Reduce                                          | 10 | 20 | 20 |
| Implementar Strategy-Switch                                     | 15 | 0 | 35 |
| Optimizaciones de comunicación entre nodos                      | 50 | 0 | 50 |
| Optimizaciones de sincronización de las copias del modelo       | 20 | 80 | 0 |
| Interfaz de funciones externas en Python e interfaz de terminal | 0 | 100 | 0 |
| Pruebas unitarias y de integración                              | 20 | 60 | 20 |
| Simulación sobre distintas configuraciones de nodos             | 30 | 100 | 20 |
| Optimizaciones de carga en configuraciones heterogéneas         | 0 | 0 | 0 |
| Análisis de resultados y gráficos                               | 50 | 40 | 10 |
| Documentación del código y del proceso                          | 60 | 10 | 30 |
| Informe final                                                   | 10 | 30 | 10 |

: Horas dedicadas por cada integrante a cada tarea del plan de actividades.
