\newpage
# Anexos

## Anexo A: Repositorio del proyecto

El código fuente del trabajo, junto con su documentación técnica y el historial completo de su desarrollo, se encuentra alojado en el siguiente repositorio público, que se mantendrá disponible al menos hasta un año después de la defensa:

\begin{center}
\texttt{https://github.com/lminervino18/oxidized-neural-orchestra}
\end{center}

El repositorio está organizado como un *workspace* de Cargo, la herramienta de construcción de Rust, y se divide en los siguientes módulos:

- `machine_learning`: el núcleo de redes neuronales, que comprende las capas, los optimizadores, las funciones de pérdida y la representación de tensores.
- `comms`: el middleware de comunicación entre nodos.
- `parameter_server`: la implementación de la estrategia *Parameter Server*.
- `worker`: el runtime del nodo trabajador, que contiene además la implementación en anillo de *All-Reduce*.
- `orchestrator`: el plano de control, que asigna los roles de servidor y de trabajador y dirige la ejecución de un entrenamiento.
- `node`: el proceso de cómputo, agnóstico del rol que le sea asignado.
- `orchestra-py`: la biblioteca de Python, construida sobre la interfaz de funciones externas.
- `orchestui`: la interfaz interactiva de terminal.
- `benchmarks`: la suite de experimentación y los generadores de gráficos.
- `docker`: los scripts que levantan los entornos de simulación multinodo.
- `docs`: la propuesta, el presente informe y la documentación de diseño del sistema.

## Anexo B: Documentación técnica

Dentro del repositorio se encuentran disponibles los siguientes documentos, referenciados a lo largo de este informe:

- `docs/config-schema.md`: la referencia canónica del esquema de configuración de un entrenamiento. Es la única fuente de verdad del formato aceptado por las tres interfaces del sistema.
- `docs/architecture-draft.md`: el borrador de arquitectura que sostuvo las discusiones de diseño de las etapas iniciales.
- `docs/report/roadmap.md`: la reconstrucción detallada del recorrido conceptual del proyecto a partir del historial completo del repositorio, con las decisiones de arquitectura, los debates que las motivaron y su fundamentación bibliográfica. Es el documento de apoyo del que se derivan las secciones de *Metodología aplicada*, *Cronograma* y *Riesgos materializados y lecciones aprendidas*.
- `benchmarks/README.md`: la documentación de la suite de experimentación, con las configuraciones de cada campaña y las instrucciones para reproducirlas.
- Los `README.md` de cada crate, con la documentación de sus abstracciones principales.

## Anexo C: Reproducibilidad de los experimentos

Los estudios experimentales presentados en este informe son reproducibles a partir del material versionado en el repositorio. Para cada campaña se conservan las configuraciones que la generaron, los resultados crudos y los resultados procesados, junto con los scripts que producen las figuras y las tablas a partir de ellos.

El material del primer estudio se encuentra en `doc/monography_all_reduce_vs_ps/`, e incluye el arnés de experimentación autocontenido, los archivos de configuración de los cuatro experimentos, los resultados en formato crudo y procesado, y los generadores de figuras y tablas. La campaña completa de benchmarks del sistema se ejecuta mediante `benchmarks/run_issue_benchmarks.py`; su última corrida completa abarcó 38 configuraciones sin fallos, con un tiempo total de cómputo de 5 horas y 39 minutos, y con tres repeticiones en las suites de throughput y de escalabilidad para reportar media y desvío.

Las semillas de inicialización de los parámetros están fijadas en todas las configuraciones. Debe tenerse presente, no obstante, que las mediciones de tiempo dependen del hardware y de la carga de la máquina en la que se ejecuten, de modo que la reproducibilidad exacta alcanza a las métricas de convergencia y de exactitud, y no a las de rendimiento.

## Anexo D: Monografías

Los dos primeros estudios experimentales de este informe fueron desarrollados originalmente como monografías individuales en el marco de la materia Aprendizaje Profundo, y se presentan aquí en forma condensada y reorientada al sistema. Los documentos completos, con su desarrollo extendido, se encuentran disponibles en el repositorio.
