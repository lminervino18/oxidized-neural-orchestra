\newpage
# Tecnologías
Las tecnologías que van a ser utilizadas para el desarrollo de este proyecto son:

- **Rust**: Se opta por el lenguaje de programación Rust para la implementación del sistema principal, porque ofrece: en primer lugar: la capacidad de editar código a bajo nivel, por la robustez del lenguaje, siendo que los requerimientos mínimos de compilación son más estrictos que la mayoría del resto de lenguajes, y que ofrece *fearless-concurrency*, haciendo checkeos estáticos de posibles problemas con la concurrencia de los programas, y por la relevancia que está cobrando en estos últimos tiempos.
- **Python**: Se va a usar Python para el análisis de datos obtenidos a partir de las comparaciones de los distintos algoritmos de machine-learning distribuido que serán llevados a cabo en el desarrollo del trabajo, y, por ser uno de los lenguajes más utilizados en la industria del aprendizaje profundo, para implementar una Interfaz de Funciones Externas (en inglés Foreign Function Interface, FFI) del resultado del sistema principal.
<!-- - **C/C++**: Se implementará, así como en Python, una interfaz que permita la utilización del sistema desarrollado en Rust, en los lenguajes C y C++. -->
- **Docker**: Se hará uso de Docker para simular la ejecución del sistema en distintos entornos, y para agilizar la automatización de esto último.
