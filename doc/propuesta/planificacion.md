\newpage
# Planificación
## Gestión
Se establece un compromiso por parte de cada estudiante para dedicar un total de 500 horas al desarrollo del trabajo profesional. Esto representa, en promedio 15 horas semanales por persona a la ejecución de las tareas asignadas. Este compromiso se mantendrá a lo largo de 32 semanas (dos cuatrimestres). Además, se tiene previsto llevar a cabo encuentros periódicos en formato virtual entre los miembros del equipo y los tutores, cada semana. El propósito de estas reuniones es informar el avance y desarrollo del proyecto en curso. Asimismo, se abordarán aspectos como la definición de prioridades en las labores a realizar y la planificación requerida para la próxima etapa del proceso.

## Tareas
Las principales tareas para llevar a cabo el desarrollo de este trabajo son:

1. Leer y analizar los trabajos previos, actuales y que surjan sobre el entrenamiento distribuido de modelos de aprendizaje profundo.
2. Investigar sobre la implementación de TensorFlow distribuido.
3. Investigar sobre la implementación distribuida de Pytorch.

Tanto Tensorflow como Pytorch son implementaciones previas del tipo de sistema que se desarrollará en este trabajo.

4. Desarrollar el sistema distribuido que sirva como base para la implementación y análisis de los algoritmos actuales en Rust. Este sistema proveerá una base sobre la cual poder probar distintos algoritmos de entrenamiento distribuido de modelos de machine learning. La idea es hacerlo tan *parametrizable* como sea posible para facilitar la posterior investigación y el desarrollo de estrategias que optimicen los tiempos de ejecución.
5. Implementar *Parameter Server* sobre el sistema desarrollado en 4.
6. Implementar *All-Reduce* sobre el sistema desarrollado en 4.
7. Implementar *Strategy-Switch* sobre el sistema desarrollado en 4. y utilizando las implementaciones de los algoritmos en 5. y 6.

Estos tres últimos puntos refieren al punto de partida de la implementación del sistema funcional y servirán como referencia para la comparación con las futuras mejoras que se estudien y desarrollen.

8. Estudiar sobre optimizaciones de comunicación entre nodos e implementarlas.
9. Estudiar sobre optimizaciones de sincronización de las copias del modelo en los distintos nodos e implementarlas.
10. Implementar una interfaz funcional externa para poder usar el sistema en Python. La idea de este punto es proveer una API fácil de usar para aquellos usuarios que trabajen con modelos de machine learning en este lenguaje, siendo que Python es el lenguaje más popular para este tipo de proyectos.
11. Testear el sistema desarrollado, con tests unitarios y de integración.
12. Simular la ejecución de los algoritmos implementados sobre el sistema distribuido base en distintas configuraciones de nodos; esto es, lograr una métrica que muestre el rendimiento del sistema, según los parámetros que este use, para distintas combinaciones de máquinas, con distintas capacidades de cómputo, que trabajen en la ejecución.
13. Estudiar sobre optimizaciones de carga de cómputo en configuraciones heterogéneas e implementarlas.
13. Analizar los resultados obtenidos de la comparación de los algoritmos, documentar y volcar el análisis utilizando gráficos en Python. Esto abarca también los resultados obtenidos las simulaciones mencionadas en 12.
14. Documentar el código generado y el proceso de desarrollo (decisiones que se tomaron, inconvenientes encontrados, etc.).
15. Realizar un informe detallado de la evolución del trabajo y los resultados obtenidos.

<!-- ## Carga horaria
| Tarea                                                                 | Duración (hs) | Responsable |
|-----------------------------------------------------------------------|---------------|-------------|
| 1. Revisión bibliográfica sobre entrenamiento distribuido en Deep Learning | 90       | A, L y M    |
| 2. Estudio de implementaciones existentes (TensorFlow, PyTorch, Horovod, etc.) | 90      | A, L y M    |
| 3. Desarrollo del sistema distribuido en Rust                         | 510           | A, L y M    |
| 4. Implementación y comparación de algoritmos de entrenamiento distribuido | 300      | A, L y M    |
| 5. Creación de la interfaz externa en Python                          | 120            | A, L y M    |
| 6. Simulación sobre distintas configuraciones del sistema distribuido | 120            | A, L y M    |
| 7. Análisis de datos comparativos usando Python                       | 150            | A, L y M    |
| 8. Informe detallado de evolución y resultados                        | 120            | A, L y M    |
| **Total**                                                             | **1500**       | A, L y M    | -->

## Carga horaria
<!-- | Tarea | Duración (horas) | Responsable(s) |
| ---- | -: | - |
| 1. Leer y analizar los trabajos previos, actuales y que surjan | 100 | A, L y M |
| 2. Investigar sobre la implementación de TensorFlow distribuido | 60 | A, L y M |
| 3. Investigar sobre la implementación distribuida de Pytorch | 60 | A, L y M |
| 4. Desarrollar el sistema distribuido en Rust | 150 | A, L y M |
| 5. Implementar *Parameter Server* | 100 | A, L y M |
| 6. Implementar *All-Reduce* | 100 | A, L y M |
| 7. Implementar *Strategy-Switch* | 100 | A, L y M |
| 8. Estudiar sobre optimizaciones de comunicación entre nodos e implementarlas | 150 | A, L y M |
| 9. Estudiar sobre optimizaciones de sincronización de las copias del modelo en los distintos nodos e implementarlas | 150 | A, L y M |
| 10. Implementar una interfaz funcional externa para poder usar el sistema en Python | 60 | A, L y M |
| 11. Testear el sistema desarrollado, con tests unitarios y de integración | 50 | A, L y M |
| 12. Simular la ejecución de los algoritmos en distintas configuraciones de nodos | 100 | A, L y M |
| 13. Estudiar sobre optimizaciones de carga de cómputo en configuraciones heterogéneas e implementarlas | 150 | A, L y M |
| 14. Analizar los resultados obtenidos y volcar el análisis utilizando gráficos en Python | 100 | A, L y M |
| 15. Documentar el código generado y el proceso de desarrollo  | 50 | A, L y M |
| 16. Escribir el informe final | 60 | A, L y M |
| **Total** | **1500** | | -->

\begin{tabular}{|p{0.05\textwidth}|p{0.6\textwidth}|p{0.1\textwidth}|p{0.15\textwidth}|}
\hline
\textbf{Nro.} & \textbf{Tarea} & \textbf{Duración (horas)} & \textbf{Responsables} \\ \hline
1 & Leer y analizar los trabajos previos, actuales y que surjan & 100 & A, L y M \\ \hline
2 & Investigar sobre la implementación de \textit{TensorFlow} distribuido & 50 & A, L y M \\ \hline
3 & Investigar sobre la implementación distribuida de \textit{Pytorch} & 50 & A, L y M \\ \hline
4 & Desarrollar el sistema distribuido en Rust & 150 & A, L y M \\ \hline
5 & Implementar \textit{Parameter Server} & 100 & A, L y M \\ \hline
6 & Implementar \textit{All-Reduce} & 100 & A, L y M \\ \hline
7 & Implementar \textit{Strategy-Switch} & 100 & A, L y M \\ \hline
8 & Estudiar sobre optimizaciones de comunicación entre nodos e implementarlas & 150 & A, L y M \\ \hline
9 & Estudiar sobre optimizaciones de sincronización de las copias del modelo en los distintos nodos e implementarlas & 150 & A, L y M \\ \hline
10 & Implementar una interfaz funcional externa para poder usar el sistema en Python & 50 & A, L y M \\ \hline
11 & Testear el sistema desarrollado, con tests unitarios y de integración & 50 & A, L y M \\ \hline
12 & Simular la ejecución de los algoritmos en distintas configuraciones de nodos & 100 & A, L y M \\ \hline
13 & Estudiar sobre optimizaciones de carga de cómputo en configuraciones heterogéneas e implementarlas & 150 & A, L y M \\ \hline
14 & Analizar los resultados obtenidos y volcar el análisis utilizando gráficos en Python & 100 & A, L y M \\ \hline
15 & Documentar el código generado y el proceso de desarrollo  & 50 & A, L y M \\ \hline
16 & Escribir el informe final & 50 & A, L y M \\ \hline
 & \textbf{Total} & \textbf{1500} & \\ \hline
\end{tabular}
