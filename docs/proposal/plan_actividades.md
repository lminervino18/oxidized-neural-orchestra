\newpage
# Plan de actividades
El proceso de construcción del producto sigue el método incremental descrito en la Metodología, del que este plan es la concreción: la gestión del alcance, de los tiempos, de los riesgos, de los indicadores de calidad y de las reuniones con los tutores se rige por lo allí establecido. Las tareas que siguen se abordan en iteraciones sucesivas, priorizando primero el sistema base y las implementaciones de referencia, y dejando las líneas de optimización para las iteraciones posteriores.

Las principales tareas para llevar a cabo el desarrollo de este trabajo son:

1. Leer y analizar los trabajos previos, actuales y que surjan sobre el entrenamiento distribuido de modelos de aprendizaje profundo.
2. Investigar sobre la implementación de TensorFlow distribuido.
3. Investigar sobre la implementación distribuida de Pytorch.

Tanto TensorFlow como PyTorch son implementaciones previas del tipo de sistema que se desarrollará en este trabajo.

4. Desarrollar el sistema distribuido que sirva como base para la implementación y análisis de los algoritmos actuales en Rust. Esto comprende el núcleo de redes neuronales (capas, optimizadores, funciones de pérdida y la representación de tensores), el manejo y la distribución de los conjuntos de datos, el middleware de comunicación entre nodos y los procesos de servidor y de trabajador. Este sistema proveerá una base sobre la cual poder probar distintos algoritmos de entrenamiento distribuido de modelos de machine learning. La idea es hacerlo tan *parametrizable* como sea posible para facilitar la posterior investigación y el desarrollo de estrategias que optimicen los tiempos de ejecución.
5. Implementar *Parameter Server* sobre el sistema desarrollado en 4.
6. Implementar *All-Reduce* sobre el sistema desarrollado en 4.
7. Implementar *Strategy-Switch* sobre el sistema desarrollado en 4. y utilizando las implementaciones de los algoritmos en 5. y 6.

Estos tres últimos puntos refieren al punto de partida de la implementación del sistema funcional y servirán como referencia para la comparación con las futuras mejoras que se estudien y desarrollen.

8. Estudiar sobre optimizaciones de comunicación entre nodos e implementarlas.
9. Estudiar sobre optimizaciones de sincronización de las copias del modelo en los distintos nodos e implementarlas.
10. Implementar una interfaz funcional externa para poder usar el sistema en Python, y una interfaz interactiva de terminal que permita configurar, lanzar y monitorear en vivo los entrenamientos. La idea de este punto es proveer una API fácil de usar para aquellos usuarios que trabajen con modelos de machine learning en este lenguaje, siendo que Python es el lenguaje más popular para este tipo de proyectos.
11. Testear el sistema desarrollado, con tests unitarios y de integración.
12. Simular la ejecución de los algoritmos implementados sobre el sistema distribuido base en distintas configuraciones de nodos; esto es, lograr una métrica que muestre el rendimiento del sistema, según los parámetros que este use, para distintas combinaciones de máquinas, con distintas capacidades de cómputo, que trabajen en la ejecución.
13. Estudiar sobre optimizaciones de carga de cómputo en configuraciones heterogéneas e implementarlas.
14. Analizar los resultados obtenidos de la comparación de los algoritmos, documentar y volcar el análisis utilizando gráficos en Python. Esto abarca también los resultados obtenidos en las simulaciones mencionadas en 12.
15. Documentar el código generado y el proceso de desarrollo (decisiones que se tomaron, inconvenientes encontrados, etc.).
16. Realizar un informe detallado de la evolución del trabajo y los resultados obtenidos.

## Entregables e hitos de avance
El trabajo se organiza en torno a cuatro hitos, cada uno asociado a un entregable verificable:

1. **Relevamiento del estado del arte** (tareas 1 a 3). Entregable: un documento de análisis de los trabajos previos y de las implementaciones distribuidas de TensorFlow y PyTorch, que fundamenta las decisiones de diseño del sistema.
2. **Sistema base con implementaciones de referencia** (tareas 4 a 7, 11). Entregable: el sistema distribuido en Rust, con *Parameter Server*, *All-Reduce* y *Strategy-Switch* funcionando sobre él, acompañado de su conjunto de pruebas unitarias y de integración. Este hito coincide con la entrega de medio término del Trabajo Profesional.
3. **Optimizaciones e interfaces** (tareas 8 a 10, 13). Entregable: las optimizaciones de comunicación, de sincronización y de carga en configuraciones heterogéneas, junto con la biblioteca de Python construida sobre la FFI y la interfaz interactiva de terminal.
4. **Experimentación e informe final** (tareas 12, 14 a 16). Entregable: las simulaciones sobre las distintas configuraciones de nodos, el análisis comparativo de los algoritmos con sus gráficos, la documentación del código y del proceso de desarrollo, y el informe final del trabajo.

El avance de cada hito se revisa en las reuniones semanales con los tutores, y el pasaje de un hito al siguiente requiere que los entregables del anterior hayan superado los criterios de aceptación definidos en la Metodología.

## Carga horaria
La columna de responsables indica el o los integrantes que concentran el foco principal de cada tarea, de acuerdo con lo detallado en los Aportes individuales; las etapas de análisis, revisión de código y pruebas involucran a los tres integrantes en todas las tareas. Se identifica con **A** a Alejo Ordoñez, con **L** a Lorenzo Minervino y con **M** a Marcos Bianchi Fernández. Las horas indicadas corresponden al esfuerzo total del equipo en cada tarea.

\begin{tabular}{|p{0.05\textwidth}|p{0.6\textwidth}|p{0.1\textwidth}|p{0.15\textwidth}|}
\hline
\textbf{Nro.} & \textbf{Tarea} & \textbf{Duración (horas)} & \textbf{Responsables} \\ \hline
1 & Leer y analizar los trabajos previos, actuales y que surjan & 100 & A, L y M \\ \hline
2 & Investigar sobre la implementación de \textit{TensorFlow} distribuido & 50 & A, L y M \\ \hline
3 & Investigar sobre la implementación distribuida de \textit{Pytorch} & 50 & A, L y M \\ \hline
4 & Desarrollar el sistema distribuido en Rust & 150 & A, L y M \\ \hline
5 & Implementar \textit{Parameter Server} & 100 & M \\ \hline
6 & Implementar \textit{All-Reduce} & 100 & L \\ \hline
7 & Implementar \textit{Strategy-Switch} & 100 & L y M \\ \hline
8 & Estudiar sobre optimizaciones de comunicación entre nodos e implementarlas & 150 & M \\ \hline
9 & Estudiar sobre optimizaciones de sincronización de las copias del modelo en los distintos nodos e implementarlas & 150 & L y M \\ \hline
10 & Implementar la interfaz de funciones externas en Python y la interfaz interactiva de terminal & 50 & L \\ \hline
11 & Testear el sistema desarrollado, con tests unitarios y de integración & 50 & A, L y M \\ \hline
12 & Simular la ejecución de los algoritmos en distintas configuraciones de nodos & 100 & L \\ \hline
13 & Estudiar sobre optimizaciones de carga de cómputo en configuraciones heterogéneas e implementarlas & 150 & A y M \\ \hline
14 & Analizar los resultados obtenidos y volcar el análisis utilizando gráficos en Python & 100 & L \\ \hline
15 & Documentar el código generado y el proceso de desarrollo  & 50 & A, L y M \\ \hline
16 & Escribir el informe final & 50 & A, L y M \\ \hline
 & \textbf{Total} & \textbf{1500} & \\ \hline
\end{tabular}
