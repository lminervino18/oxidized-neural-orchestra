\newpage

# Herramientas externas utilizadas

**Rust** se eligió como lenguaje del sistema principal por varios motivos:

- Control de bajo nivel sin renunciar a garantías de seguridad.
- Buen modelo de concurrencia.
- Relevancia y crecimiento en la industria.
- Familiaridad y preferencia de los integrantes.

**Python** Se utilizó para tres propósitos:

- Interfaz de funciones externas del sistema.
- Por ser un lenguaje dominante en la industria del *Deep Learning*.
- Familiaridad con el lenguaje.

**Docker** Se utilizó para simular la ejecución del sistema sobre múltiples nodos y para automatizar el despliegue.

**Git, GitHub, GitHub Projects y GitHub Actions** Control de versiones y gestión del proceso: ramas por funcionalidad, *pull requests* con revisión obligatoria, e *issues* para el seguimiento de tickets, errores y debates de diseño. Se hizo uso de Actions para CI/CD del repositorio.

**Pumba** Se utilizó para simular delays de comunicación por red durante los benchmarks del sistema en entornos contenerizados de Docker. Es fácil de usar y muy parametrizable.

La totalidad de las herramientas y bibliotecas utilizadas son de código abierto, con licencias permisivas (MIT o Apache 2.0 en su mayoría). El sistema desarrollado se publica bajo licencia abierta, de modo que pueda ser retomado y extendido por terceros, según el criterio establecido en la propuesta. No se incurrió en costos de licenciamiento.

## Bibliotecas de Rust

Estas son algunas de las bibliotecas más importantes del proyecto.

| Biblioteca                   | Uso                                                                                                  |
|------------------------------|------------------------------------------------------------------------------------------------------|
| `tokio`                      | Runtime asincrónico, TCP, tareas y canales.                                                          |
| `ndarray`                    | Operaciones sobre arreglos numéricos n-dimensionales; provee la multiplicación de matrices del motor. |
| `rayon`                      | Paralelismo de datos en el almacenamiento del servidor de parámetros.                                 |
| `serde` y `serde_json`       | Serialización del plano de control y de los archivos de configuración, respectivamente.               |
| `bytemuck`                   | Reinterpretación de memoria sin copias entre valores numéricos y bytes.                               |
| `half`                       | Tipo de media precisión para la compresión de gradientes.                                             |
| `uuid`                       | Identidad estable de nodos.                                                                           |
| `safetensors`                | Exportación del modelo entrenado.                                                                     |
| `log` y `tracing-subscriber` | Logging estructurado con filtrado por variable de entorno.                                           |
| `ratatui` y `crossterm`      | Interfaz interactiva de terminal.                                                                     |
| `pyo3`                       | Interfaz de funciones externas con Python.                                                            |
| `anyhow`                     | Manejo de errores en la interfaz de terminal.                                                         |


## Bibliotecas de Python

| Biblioteca    | Uso                                                                 |
|---------------|---------------------------------------------------------------------|
| `maturin`     | Construcción de la extensión nativa que expone el sistema a Python. |
| `numpy`       | Lectura de los conjuntos de datos binarios en la evaluación.        |
| `matplotlib`  | Generación de todas las figuras de los experimentos.                |
| `safetensors` | Carga de los pesos entrenados por el sistema para su evaluación.    |
| `torch`       | Línea de base de referencia.                                        |

El uso de Pytorch se restringió a benchmarks para comparar la implementación del motor de redes neuronales de O.N.O. contra una base conocida.
