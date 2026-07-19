\newpage
# Herramientas externas utilizadas

En cumplimiento de las pautas de ética profesional, se declaran a continuación todas las herramientas empleadas en el desarrollo del trabajo, junto con las consideraciones que motivaron su elección.

La propuesta fijó tres tecnologías desde el inicio, Rust, Python y Docker, y dejó explícitamente abiertas las decisiones restantes, estableciendo tres criterios para tomarlas: que no introdujeran un costo de comunicación o de serialización capaz de distorsionar la comparación entre estrategias, que se integraran con el sistema de construcción y de pruebas del ecosistema de Rust sin requerir dependencias externas a él, y que su licencia fuera de código abierto de modo que el sistema pudiera ser retomado por terceros. Los tres criterios se sostuvieron.

## Lenguajes y entornos

**Rust.** Se eligió como lenguaje del sistema principal por tres motivos. El primero es el control de bajo nivel que ofrece sin renunciar a garantías de seguridad, lo que resultó determinante para un sistema cuyo camino crítico exige manipular memoria sin copias. El segundo es la *concurrencia sin miedo*: el compilador verifica estáticamente una clase completa de errores de concurrencia. El tercero es la relevancia creciente del lenguaje en el ámbito de los sistemas.

Corresponde evaluar esa elección a posteriori, ya que la sección de lecciones aprendidas la retoma en detalle. Las garantías del lenguaje cumplieron exactamente lo que prometen: ninguna condición de carrera sobre memoria compartida llegó a producción. No cubrieron, ni podían cubrir, los bloqueos de coordinación distribuida ni los errores de derivación de gradientes, que fueron las dos fuentes reales de dificultad. El único punto en el que el sistema salió de las garantías del lenguaje, la implementación sin bloqueos del almacenamiento de parámetros, es también el único que resultó incorrecto.

El ecosistema se utilizó completo: `cargo` para construcción y gestión de dependencias, `cargo test` como arnés de pruebas, `cargo clippy` para análisis estático y `cargo fmt` para formato. No se incorporó ninguna herramienta externa a él, según el criterio fijado en la propuesta.

**Python.** Se utilizó para tres propósitos: la interfaz de funciones externas del sistema, por ser el lenguaje dominante en aprendizaje profundo; la suite de experimentación y el análisis de resultados; y los scripts de generación y despliegue del entorno contenedizado. El código Python se verifica con `ruff` como analizador estático y formateador.

**Docker.** Se utilizó para simular la ejecución del sistema sobre múltiples nodos y para automatizar el despliegue. Es la herramienta que hizo posible la experimentación, y también la fuente de la limitación transversal del trabajo, ya que un clúster simulado sobre una única máquina no reproduce la latencia ni el ancho de banda de una red real. Se acompaña de `cargo-chef` para cachear las capas de dependencias en la construcción de la imagen, lo que reduce de forma sustancial el tiempo de reconstrucción.

**Git y GitHub.** Control de versiones y gestión del proceso: ramas por funcionalidad, *pull requests* con revisión obligatoria, e *issues* para el seguimiento de tickets, errores y debates de diseño.

## Bibliotecas de Rust

La elección de cada una respondió a los criterios enunciados. Todas son de código abierto.

| Biblioteca | Uso |
|---|---|
| `tokio` | Runtime asincrónico, TCP, tareas y canales. Base de todo el sistema distribuido |
| `ndarray` | Operaciones sobre arreglos numéricos n-dimensionales; provee la multiplicación de matrices del motor |
| `ndarray-rand` | Inicialización aleatoria de tensores |
| `rayon` | Paralelismo de datos en el almacenamiento del servidor de parámetros |
| `serde` y `serde_json` | Serialización del plano de control y de los archivos de configuración |
| `bytemuck` | Reinterpretación de memoria sin copias entre valores numéricos y bytes |
| `half` | Tipo de media precisión para la compresión de gradientes |
| `uuid` | Identidad estable de nodos |
| `futures` | Composición concurrente de operaciones de red |
| `rand` y `rand_distr` | Muestreo del umbral disperso, barajado y distribuciones de inicialización |
| `parking_lot` | Primitivas de sincronización de la barrera dinámica |
| `trait-variant` y `async-trait` | Traits asincrónicos |
| `safetensors` | Exportación del modelo entrenado |
| `log` y `tracing-subscriber` | Registro estructurado con filtrado por variable de entorno |
| `ratatui` y `crossterm` | Interfaz interactiva de terminal |
| `pyo3` | Interfaz de funciones externas con Python |
| `approx` | Comparaciones aproximadas en las pruebas |
| `anyhow` | Manejo de errores en la interfaz de terminal |

Tres de estas elecciones merecen justificarse porque tenían alternativas reales.

**`ndarray` en lugar de un framework de aprendizaje profundo.** Es la decisión que define el alcance del trabajo. Se eligió una biblioteca que provee *únicamente* operaciones sobre arreglos numéricos, sin diferenciación automática, sin capas y sin optimizadores, porque el objetivo del trabajo incluía implementar el motor de aprendizaje desde cero. Usar un framework existente habría hecho el trabajo considerablemente más simple y considerablemente menos valioso.

**`bytemuck` para la serialización del plano de datos.** Aquí operó de forma directa el primer criterio de la propuesta. Una biblioteca de serialización de propósito general habría introducido una copia y un costo de codificación en el camino crítico, exactamente el tipo de sobrecarga que podría distorsionar la comparación entre estrategias que comunican de maneras distintas. La reinterpretación de memoria elimina esa variable.

**`tracing-subscriber` en reemplazo de `env_logger`.** Se migró tardíamente, manteniendo la fachada genérica de registro en las bibliotecas. La razón fue concreta: la interfaz de terminal se corrompía con salidas de registro dirigidas a la salida estándar, y el registro debía dirigirse a la salida de error de forma consistente.

## Bibliotecas de Python

| Biblioteca | Uso |
|---|---|
| `maturin` | Construcción de la extensión nativa que expone el sistema a Python |
| `numpy` | Lectura de los conjuntos de datos binarios en la evaluación |
| `matplotlib` | Generación de todas las figuras de los experimentos |
| `safetensors` | Carga de los pesos entrenados por el sistema para su evaluación |
| `torch` | Línea de base de referencia |
| `ruff` | Análisis estático y formato |

El uso de **PyTorch** requiere una aclaración explícita, porque podría malinterpretarse dado que el trabajo declara no usar frameworks de aprendizaje profundo. PyTorch **no participa del sistema en ninguna forma**: no se utiliza para entrenar, ni para calcular gradientes, ni para representar modelos. Se lo emplea exclusivamente como **línea de base externa**, entrenando en un solo proceso el mismo modelo con la misma receta de hiperparámetros, para tener un punto de referencia contra el cual contrastar la exactitud que alcanza el motor propio. Es un instrumento de medición, y su presencia en el repositorio es una dependencia opcional del entorno de experimentación, separada de las dependencias de ejecución del sistema.

## Documentación y edición del informe

El presente informe y la propuesta que lo precedió se escriben en Markdown y se componen a PDF mediante **Pandoc**, con procesamiento de citas y bibliografía en formato BibTeX bajo el estilo APA 6. Las monografías que dieron origen a los estudios experimentales se compusieron en LaTeX con la clase de artículo de IEEE. Los diagramas se generaron con **Mermaid** y **Matplotlib**.

## Herramientas de inteligencia artificial generativa

En cumplimiento de la obligación de declararlo explícitamente, se deja constancia de que durante el desarrollo del trabajo se utilizaron **modelos grandes de lenguaje** como asistencia, en los siguientes usos: consulta y discusión de conceptos de sistemas distribuidos y de aprendizaje profundo, asistencia en la depuración de errores, revisión y sugerencia de mejoras sobre código ya escrito, generación de código auxiliar en tareas accesorias como scripts de despliegue y de graficado, y asistencia en la redacción y revisión de la documentación y de este informe.

Los integrantes del equipo asumen la responsabilidad plena sobre la totalidad del contenido generado, se apropian del mismo, lo conocen en profundidad y están en condiciones de defenderlo en el acto de Defensa. Delegar la ejecución de una tarea en una herramienta no exime de la responsabilidad sobre su resultado. En particular, se deja constancia de que las decisiones de arquitectura documentadas en este informe, los criterios metodológicos de los estudios experimentales y la interpretación de sus resultados son producto del trabajo y del debate del equipo, y su registro en las *issues* y *pull requests* del repositorio es verificable de forma independiente.

## Consideración sobre licencias

La totalidad de las herramientas y bibliotecas utilizadas son de código abierto, con licencias permisivas (MIT o Apache 2.0 en su mayoría). El sistema desarrollado se publica bajo licencia abierta, de modo que pueda ser retomado y extendido por terceros, según el criterio establecido en la propuesta. No se incurrió en costos de licenciamiento, y el trabajo se desarrolló sobre hardware propio de los integrantes.
