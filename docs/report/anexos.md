\newpage
# Anexos

## Anexo A: Repositorio del proyecto

El código fuente, su documentación técnica y el historial completo del desarrollo están alojados en el siguiente repositorio público, que se mantendrá disponible al menos hasta un año después de la defensa:

\begin{center}
\texttt{https://github.com/lminervino18/oxidized-neural-orchestra}
\end{center}

Está organizado como un *workspace* de Cargo, la herramienta de construcción de Rust, y se divide en los siguientes módulos:

- `machine_learning`: el motor de redes neuronales, con las capas, los optimizadores, las funciones de pérdida y la representación de tensores.
- `comms`: la capa de comunicación entre nodos y su protocolo propio.
- `parameter_server`: la implementación de *Parameter Server*.
- `worker`: el runtime del nodo trabajador, que incluye la implementación en anillo de *All-Reduce*.
- `orchestrator`: el plano de control, que asigna los roles y dirige la ejecución de un entrenamiento.
- `node`: el proceso de cómputo, agnóstico del rol que le sea asignado.
- `orchestra-py`: la biblioteca de Python, construida sobre la interfaz de funciones externas.
- `orchestui`: la interfaz interactiva de terminal.
- `benchmarks`: la suite de experimentación y los generadores de gráficos.
- `docker`: los scripts que levantan los entornos de simulación multinodo.
- `docs`: la propuesta, el presente informe y la documentación de diseño del sistema.

## Anexo B: Documentación técnica

Dentro del repositorio se encuentran los siguientes documentos, referenciados a lo largo de este informe:

- `docs/config-schema.md`: la referencia canónica del esquema de configuración de un entrenamiento, y única fuente de verdad del formato que aceptan las interfaces del sistema.
- `docs/architecture-draft.md`: el borrador de arquitectura que sostuvo las discusiones de diseño de las etapas iniciales.
- `docs/report/roadmap.md`: la reconstrucción del recorrido conceptual del proyecto a partir del historial del repositorio, con las decisiones de arquitectura, los debates que las motivaron y su fundamentación bibliográfica. Es el documento de apoyo del que derivan *Metodología aplicada*, *Cronograma* y *Riesgos materializados y lecciones aprendidas*.
- `benchmarks/README.md`: la documentación de la suite de experimentación, con las configuraciones de cada campaña y las instrucciones para reproducirlas.
- Los `README.md` de cada crate, con la documentación de sus abstracciones principales.

## Anexo C: Reproducibilidad de los experimentos

La campaña de benchmarks del sistema es reproducible a partir del material versionado en el repositorio. Se ejecuta mediante `benchmarks/run_issue_benchmarks.py`, cubre 38 configuraciones con tres repeticiones en las suites de throughput y de escalabilidad, que se reportan en media y desvío, y conserva las configuraciones que generaron cada corrida junto con los resultados crudos, los procesados y los generadores de figuras.

Los dos primeros estudios se desarrollaron originalmente como monografías individuales en el marco de la materia Aprendizaje Profundo, sobre un arnés de experimentación propio que no forma parte de este repositorio. Sus configuraciones, sus resultados y el criterio de comparación de cada uno se documentan en *Experimentación y validación*, y las monografías completas se ponen a disposición del jurado como documentos separados.

Las semillas de inicialización de los parámetros están fijadas en todas las configuraciones. Debe tenerse presente, no obstante, que las mediciones de tiempo dependen del hardware y de la carga de la máquina en la que se ejecuten, de modo que la reproducibilidad exacta alcanza a las métricas de convergencia y de exactitud, y no a las de rendimiento.
