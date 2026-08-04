\newpage
# Aportes individuales

El desarrollo del trabajo se planteó como una responsabilidad compartida: las etapas de análisis, revisión de código y pruebas fueron llevadas adelante por los tres integrantes en todas las tareas. Sin perjuicio de ello, cada integrante concentró su foco principal en un conjunto de componentes, que se detalla a continuación y que se corresponde con lo previsto en la propuesta.

**Alejo Ordoñez (Padrón 108397).** Núcleo de aprendizaje automático: el modelo, las capas, los optimizadores y las funciones de pérdida. Las capas convolucionales y de *max-pooling*, incluida la reformulación de la convolución como multiplicación de matrices que produjo la mejora final de rendimiento del motor. El manejo y la distribución de los conjuntos de datos. La monografía sobre *Strategy-Switch*, que constituye el tercer estudio experimental del informe.

**Lorenzo Minervino (Padrón 107863).** El módulo del Worker y la estrategia *All-Reduce* en anillo. La interfaz de funciones externas en Python y la interfaz interactiva de terminal. La suite de benchmarks y el análisis experimental de resultados. La monografía comparativa entre *Parameter Server* y *All-Reduce*, que constituye el primer estudio experimental del informe.

**Marcos Bianchi Fernández (Padrón 108921).** Los algoritmos de entrenamiento distribuido y su coordinación. El módulo de comunicaciones y el protocolo entre nodos, incluidas las técnicas de reducción de tráfico. El módulo del Servidor de parámetros. La monografía sobre el impacto de las épocas offline, que constituye el segundo estudio experimental del informe.

| | Alejo Ordoñez | Lorenzo Minervino | Marcos Bianchi Fernández |
|---|---|---|---|
| **Cantidad de horas insumidas** | 520 | 545 | 525 |
| Preparación de la propuesta | X | X | X |
| Relevamiento del estado del arte | X | X | X |
| Diseño de la arquitectura del sistema | X | X | X |
| Capa de comunicaciones y protocolo | | | X |
| Motor de redes neuronales | X | | |
| Capas convolucionales y max-pooling | X | | |
| Manejo y distribución de datasets | X | | |
| Parameter Server | | | X |
| Worker y All-Reduce en anillo | | X | |
| Strategy-Switch | | X | X |
| Orchestrator y coordinación | | X | X |
| Optimizaciones de comunicación | | | X |
| Optimizaciones de sincronización | | X | X |
| Optimización del motor de aprendizaje | X | | |
| Interfaz de funciones externas en Python | | X | |
| Interfaz interactiva de terminal | | X | |
| Entornos de simulación con Docker | | X | X |
| Definición de casos de prueba | X | X | X |
| Revisión de código | X | X | X |
| Suite de benchmarks | | X | |
| Análisis experimental de resultados | | X | |
| Monografía: All-Reduce frente a Parameter Server | | X | |
| Monografía: impacto de las épocas offline | | | X |
| Monografía: Strategy-Switch | X | | |
| Documentación técnica del sistema | X | X | X |
| Elaboración del Informe Final | X | X | X |

: Distribución del trabajo individual por tipo de tarea.
