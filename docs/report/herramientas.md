\newpage

# Herramientas externas utilizadas

**Rust** se eligió como lenguaje del sistema principal por varios motivos:

1. Control de bajo nivel sin renunciar a garantías de seguridad.
2. Buen modelo de concurrencia.
3. Relevancia y crecimiento en la industria.
4. Familiaridad y gusto por programar en él.
5. Es muy performante.
6. Su expresabilidad.

**Python** Se utilizó para tres propósitos:

1. Interfaz de funciones externas del sistema.
2. Por ser un lenguaje dominante en la industria del *Deep Learning*.
3. Familiaridad con el lenguaje.

**Docker** Se utilizó para simular la ejecución del sistema sobre múltiples nodos y para automatizar el despliegue. Se acompaña de `cargo-chef` para cachear las capas de dependencias en la construcción de la imagen, lo que reduce de forma sustancial el tiempo de reconstrucción.

**Git, GitHub, GitHub Projects y GitHub Actions** Control de versiones y gestión del proceso: ramas por funcionalidad, *pull requests* con revisión obligatoria, e *issues* para el seguimiento de tickets, errores y debates de diseño. Se hizo uso de Actions para CI/CD del repositorio.

**Pumba** Se utilizó para simular una red real durante los benchmarks del sistema en entornos conteinerizados de Docker. Es facil de usar y muy parametrizable.

La totalidad de las herramientas y bibliotecas utilizadas son de código abierto, con licencias permisivas (MIT o Apache 2.0 en su mayoría). El sistema desarrollado se publica bajo licencia abierta, de modo que pueda ser retomado y extendido por terceros, según el criterio establecido en la propuesta. No se incurrió en costos de licenciamiento.

## Bibliotecas de Rust

Estas son algunas de las bibliotecas más importantes del proyecto.

| Biblioteca                   | Uso                                                                                                  |
|------------------------------|------------------------------------------------------------------------------------------------------|
| `tokio`                      | Runtime asincrónico, TCP, tareas y canales.                                                          |
| `ndarray`                    | Operaciones sobre arreglos numéricos n-dimensionales; provee la multiplicación de matrices del motor |
| `rayon`                      | Paralelismo de datos en el almacenamiento del servidor de parámetros                                 |
| `serde` y `serde_json`       | Serialización del plano de control y de los archivos de configuración                                |
| `bytemuck`                   | Reinterpretación de memoria sin copias entre valores numéricos y bytes                               |
| `half`                       | Tipo de media precisión para la compresión de gradientes                                             |
| `uuid`                       | Identidad estable de nodos                                                                           |
| `safetensors`                | Exportación del modelo entrenado                                                                     |
| `log` y `tracing-subscriber` | Registro estructurado con filtrado por variable de entorno                                           |
| `ratatui` y `crossterm`      | Interfaz interactiva de terminal                                                                     |
| `pyo3`                       | Interfaz de funciones externas con Python                                                            |
| `anyhow`                     | Manejo de errores en la interfaz de terminal                                                         |


## Bibliotecas de Python

| Biblioteca    | Uso                                                                |
|---------------|--------------------------------------------------------------------|
| `maturin`     | Construcción de la extensión nativa que expone el sistema a Python |
| `numpy`       | Lectura de los conjuntos de datos binarios en la evaluación        |
| `matplotlib`  | Generación de todas las figuras de los experimentos                |
| `safetensors` | Carga de los pesos entrenados por el sistema para su evaluación    |
| `torch`       | Línea de base de referencia                                        |
| `ruff`        | Análisis estático y formato                                        |

El uso de Pytorch se restringió a benchmarks para comparar la implementación de O.N.O. contra una base conocida.

## Herramientas de inteligencia artificial generativa

En cumplimiento de la obligación de declararlo explícitamente, se deja constancia de que durante el desarrollo del trabajo se utilizaron **modelos grandes de lenguaje** como asistencia, en los siguientes usos: consulta y discusión de conceptos de sistemas distribuidos y de aprendizaje profundo, asistencia en la depuración de errores, revisión y sugerencia de mejoras sobre código ya escrito, generación de código auxiliar en tareas accesorias como scripts de despliegue y de graficado, y asistencia en la redacción y revisión de la documentación y de este informe.

Los integrantes del equipo asumen la responsabilidad plena sobre la totalidad del contenido generado, se apropian del mismo, lo conocen en profundidad y están en condiciones de defenderlo en el acto de Defensa. Delegar la ejecución de una tarea en una herramienta no exime de la responsabilidad sobre su resultado. En particular, se deja constancia de que las decisiones de arquitectura documentadas en este informe, los criterios metodológicos de los estudios experimentales y la interpretación de sus resultados son producto del trabajo y del debate del equipo, y su registro en las *issues* y *pull requests* del repositorio deja constancia de esto.
