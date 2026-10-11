\newpage
## Metodología de experimentación y validación

Antes de presentar los resultados conviene fijar cómo se los obtuvo. Esta sección describe los dos pilares sobre los que se apoya toda medición del capítulo: las *pruebas automatizadas*, que verifican que el sistema hace lo que dice hacer, y el *entorno de experimentación*, el clúster simulado sobre el que se ejecutan tanto esas pruebas como los ensayos cuyos resultados se analizan en las Secciones \ref{sec:resultados} y \ref{sec:estudio-derivado}.

### Pruebas automatizadas

El sistema se verificó con tres tipos de pruebas. Las pruebas unitarias ejercitan cada componente de forma aislada y son las que sostienen el motor de redes neuronales: la corrección de la propagación hacia atrás no es observable a simple vista, y la única manera práctica de detectar un gradiente mal derivado es contrastarlo contra un valor calculado de forma independiente. Las pruebas de integración validan la interacción entre nodos durante una ejecución distribuida, en particular los protocolos de coordinación. Las pruebas de aceptación comprueban que cada funcionalidad requerida se comporta como fue especificada, ejecutando un entrenamiento completo sobre el entorno simulado y verificando que la convergencia obtenida sea consistente con la del entrenamiento secuencial equivalente.

Las dos primeras se escribieron con el arnés de pruebas nativo de Rust y se ejecutan mediante `cargo test`; las de aceptación se levantan sobre los entornos contenerizados descritos en la Sección \ref{sec:entorno}. Las tres son automatizadas y de ejecución desatendida, y se versionan junto con el código, de modo que cada cambio pueda validarse antes de integrarse.

La suite completa se ejecuta con éxito en cada integración y su aprobación es condición obligatoria para incorporar cambios a la rama principal; en particular, las pruebas de aceptación confirman que la convergencia de las estrategias distribuidas coincide con la del entrenamiento secuencial equivalente. Esta verificación no alcanza, sin embargo, a los errores de corrección distribuida silenciosos, aquellos que no rompen ninguna prueba pero degradan la calidad del modelo; su detección motivó la incorporación de los ensayos de rendimiento y convergencia que se presentan en la Sección \ref{sec:resultados}.

### Entorno de experimentación y reproducibilidad {#sec:entorno}

El despliegue se realiza con una única imagen de Docker, coherente con el agnosticismo de rol del sistema: no hay imagen de worker ni imagen de servidor. Un conjunto de scripts genera la configuración del clúster, ajusta la resolución de nombres y levanta el entorno a partir de un único parámetro con la cantidad de nodos.

Todas las mediciones se realizaron sobre clústeres simulados en una única máquina física: un contenedor por nodo, red *bridge* y puertos publicados. Cada contenedor se restringe a una porción fija de CPU, para emular nodos independientes, y el ancho de banda de salida de cada contenedor se limita con Pumba, que aplica sobre su interfaz de red un límite de tasa de transmisión. El capítulo reúne dos conjuntos de ensayos, ejecutados en máquinas distintas: la campaña de comparación de estrategias, cuyos resultados se presentan en la Sección \ref{sec:resultados}, y el estudio de epochs offline de la Sección \ref{sec:estudio-derivado}. La Tabla siguiente resume el entorno de cada uno.

| Característica       | Comparación de estrategias                   | Estudio de epochs offline                                                         |
|----------------------|----------------------------------------------|-----------------------------------------------------------------------------------|
| Procesador           | 16 núcleos                                   | Intel Core i5-8350U (4 núcleos físicos, 8 hilos lógicos; 1,7 GHz base, 3,6 GHz turbo) |
| Memoria              | No registrada                                | 8 GB DDR4                                                                         |
| CPU por contenedor   | 1 núcleo                                     | No registrada                                                                     |
| Red entre nodos      | 10 Mbit/s por contenedor (Pumba)             | Emulación de red hogareña: retardo, *jitter* y ancho de banda (Pumba)             |
| Software             | Docker, Pumba 1.1.7                          | Ubuntu 24.04 LTS, Docker 29.6.1, Rust 1.96.1, CPython 3.14.0, Pumba 1.1.7         |

: Entornos de ejecución de los dos conjuntos de ensayos del capítulo.

Para la campaña de comparación de estrategias, el ancho de banda se fijó en $R = 10$ Mbit/s, deliberadamente bajo, de modo que la relación entre el tamaño del modelo y la velocidad de la red produzca diferencias observables entre estrategias sin recurrir a un modelo artificialmente grande.

Esa campaña se corrió sobre el conjunto de datos MNIST completo, con la arquitectura convolucional LeNet-5, de 61706 parámetros; ambos fueron introducidos por LeCun et al. [-@lecun1998gradient], que es la referencia de los dos. Como los parámetros y los gradientes viajan por la red como flotantes de 16 bits, el modelo ocupa $S \approx 1$ Mbit ($\approx 0{,}12$ MB) a los efectos de la comunicación. El entrenamiento de referencia en una sola máquina, sin distribución, tarda $T_1 = 600$ segundos. El resto de la configuración se resume en la Tabla siguiente.

| Parámetro            | Valores                                                                        |
|----------------------|--------------------------------------------------------------------------------|
| Dataset              | MNIST completo                                                                 |
| Modelo               | LeNet-5                                                                        |
| Función de pérdida   | Entropía cruzada                                                               |
| Optimizador          | Adam [@kingma2014adam] ($\alpha=0{,}01$; $\beta_1=0{,}9$; $\beta_2=0{,}999$; $\epsilon=10^{-8}$) |
| Tamaño de mini-batch | 64                                                                             |
| Epochs máximos       | 48                                                                             |
| $N_W$                | 2, 4, 6, 8, 12, 16                                                             |

: Configuración de la campaña de comparación de estrategias. Parameter Server se ejecuta siempre con un único servidor.

Además de esa campaña, el repositorio incluye una suite de experimentación escrita en Python haciendo uso de la interfaz que provee el sistema para el lenguaje, y está organizada de forma declarativa en cuatro conjuntos de ensayos que miden convergencia, velocidad de ejecución, velocidad de convergencia y escalabilidad. Los resultados se persisten de forma incremental, con repeticiones en media y desvío, y la documentación se regenera automáticamente en cada corrida. Documenta además de forma explícita sus criterios de equidad y sus fuentes de ruido: justifica el presupuesto fijo de nodos, aclara qué significa un epoch en cada ensayo, y explica por qué ciertas variantes se comparan por exactitud y no por velocidad.
