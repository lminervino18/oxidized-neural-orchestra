\newpage
# Solución propuesta
Se propone implementar un sistema distribuido de entrenamiento de modelos de aprendizaje profundo que sirva como base para la investigación de los trabajos previos, actuales y que surjan sobre esta temática. La idea es que sea un sistema *parametrizable*, de modo de facilitar la posterior comparación entre estrategias de distribución y el desarrollo de mejoras que optimicen los tiempos de ejecución.

Sobre esa base se implementarán los algoritmos que hoy sirven como punto de partida del área: *Parameter Server*, *All-Reduce* y *Strategy-Switch*. Estas implementaciones funcionarán como referencia para la comparación con las futuras mejoras que se estudien y desarrollen, tanto en la comunicación entre nodos como en la sincronización de las copias del modelo. En términos generales, el desarrollo del trabajo conlleva:

1. Desarrollar un sistema distribuido de entrenamiento de modelos de aprendizaje profundo en Rust.
2. Implementar y comparar los distintos algoritmos que se utilicen para la ejecución del entrenamiento distribuido, sobre el sistema implementado.
3. Proveer una biblioteca en Python, construida sobre una interfaz de funciones externas (FFI), que permita configurar y utilizar el sistema desarrollado desde ese lenguaje.
4. Desarrollar una interfaz interactiva de terminal (TUI) que permita configurar, lanzar y monitorear en vivo los entrenamientos distribuidos sobre el sistema.
5. Simular la ejecución sobre distintas configuraciones del sistema distribuido, para así obtener datos que se puedan analizar, y obtener comparaciones de los distintos algoritmos que se implementen, usando Python.
6. Confeccionar un informe detallado de la evolución del trabajo y los resultados obtenidos.

## Tecnologías
Las tecnologías que van a ser utilizadas para el desarrollo de este proyecto son:

- **Rust**: Se opta por el lenguaje de programación Rust para la implementación del sistema principal, porque ofrece: en primer lugar, la capacidad de editar código a bajo nivel, por la robustez del lenguaje, siendo que los requerimientos mínimos de compilación son más estrictos que la mayoría del resto de lenguajes, y que ofrece *fearless-concurrency*, haciendo chequeos estáticos de posibles problemas con la concurrencia de los programas, y por la relevancia que está cobrando en estos últimos tiempos.
- **Python**: Se va a usar Python para el análisis de datos obtenidos a partir de las comparaciones de los distintos algoritmos de machine-learning distribuido que serán llevados a cabo en el desarrollo del trabajo, y, por ser uno de los lenguajes más utilizados en la industria del aprendizaje profundo, para implementar una Interfaz de Funciones Externas (en inglés Foreign Function Interface, FFI) del resultado del sistema principal.
- **Docker**: Se hará uso de Docker para simular la ejecución del sistema en distintos entornos, y para agilizar la automatización de esto último.
