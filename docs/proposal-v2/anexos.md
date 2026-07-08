\newpage
# Anexos

## Anexo A: Repositorio del proyecto
El código fuente del trabajo, junto con su documentación técnica y el historial completo de su desarrollo, se encuentra alojado en el siguiente repositorio público:

\begin{center}
\texttt{https://github.com/lminervino18/oxidized-neural-orchestra}
\end{center}

El repositorio está organizado como un *workspace* de Cargo, la herramienta de construcción de Rust, y se divide en los siguientes módulos:

- `machine_learning`: el núcleo de redes neuronales, que comprende las capas, los optimizadores, las funciones de pérdida y la representación de tensores.
- `comms`: el middleware de comunicación entre nodos.
- `parameter_server`: la implementación de la estrategia *Parameter Server*, en sus variantes sincrónica y asincrónica.
- `worker`: el runtime del nodo trabajador, que contiene además la implementación en anillo de *All-Reduce*.
- `orchestrator`: el plano de control, que asigna los roles de servidor y de trabajador y dirige la ejecución de un entrenamiento.
- `node`: el proceso de cómputo, agnóstico del rol que le sea asignado.
- `orchestra-py`: la biblioteca de Python, construida sobre la interfaz de funciones externas.
- `orchestui`: la interfaz interactiva de terminal.
- `docker`: los scripts que levantan los entornos de simulación multinodo.
- `docs`: la presente propuesta y la documentación de diseño del sistema.
