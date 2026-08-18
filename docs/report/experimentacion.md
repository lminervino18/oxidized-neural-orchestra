\newpage
## Experimentación y validación

La validación se apoyó en dos frentes. El primero es el de las pruebas automatizadas, que verifican que el sistema hace lo que dice hacer. El segundo, y el que da sentido al proyecto, es el de los estudios experimentales: como O.N.O. fue concebido para comparar estrategias de distribución de forma controlada, la validación última consiste en demostrar que esa base produce evidencia comparable.

### Pruebas automatizadas

El sistema se verificó con tres tipos de pruebas. Las pruebas unitarias ejercitan cada componente de forma aislada y son las que sostienen el motor de redes neuronales: la corrección de la propagación hacia atrás no es observable a simple vista, y la única manera práctica de detectar un gradiente mal derivado es contrastarlo contra un valor calculado de forma independiente. Las pruebas de integración validan la interacción entre nodos durante una ejecución distribuida, en particular los protocolos de coordinación. Las pruebas de aceptación comprueban que cada funcionalidad requerida se comporta como fue especificada, ejecutando un entrenamiento completo sobre el entorno simulado y verificando que la convergencia obtenida sea consistente con la del entrenamiento secuencial equivalente.

Las dos primeras se escribieron con el arnés de pruebas nativo de Rust y se ejecutan mediante `cargo test`; las de aceptación se levantan sobre los entornos contenerizados descritos más abajo. Las tres son automatizadas y de ejecución desatendida, y se versionan junto con el código, de modo que cada cambio pueda validarse antes de integrarse.

### Entorno de experimentación y reproducibilidad

El despliegue se realiza con una única imagen de Docker, coherente con el agnosticismo de rol del sistema: no hay imagen de trabajador ni imagen de servidor. Un conjunto de scripts genera la configuración del clúster, ajusta la resolución de nombres y levanta el entorno a partir de un único parámetro con la cantidad de nodos.

Todas las mediciones se realizaron sobre clústeres simulados en una única máquina física: un contenedor por nodo, red *bridge* y puertos publicados.

La suite de experimentación está escrita en Python haciendo uso de la interfaz que provee el sistema para el lenguage, y está organizada de forma declarativa en cuatro conjuntos de ensayos que miden convergencia, velocidad de ejecución, velocidad de convergencia y escalabilidad. Los resultados se persisten de forma incremental, con repeticiones en media y desvío, y la documentación se regenera automáticamente en cada corrida. Documenta además de forma explícita sus criterios de equidad y sus fuentes de ruido: justifica el presupuesto fijo de nodos, aclara qué significa una época en cada ensayo, y explica por qué ciertas variantes se comparan por exactitud y no por velocidad.

### Estudio derivado: impacto de las épocas offline

#### Pregunta e hipótesis

Permitir que los nodos operen de forma autónoma durante varias épocas posterga el intercambio de parámetros, disminuyendo la dependencia de la red y priorizando el cómputo local. Sin embargo, demorar la sincronización introduce divergencia entre los pesos de los nodos, un fenómeno conocido como *client drift* [@mcmahan2017federated]. O.N.O. expone este mecanismo como un parámetro de configuración, las épocas offline ($E$), lo que permite estudiarlo de forma directa.

La pregunta es: ¿Cómo impacta la variación de las épocas offline $E$ en la velocidad de convergencia, el tiempo de comunicación y la eficacia del modelo entrenado bajo distintas topologías de red en un entorno distribuido con restricciones de red reales?

#### Metodología

Se ejecutaron entrenamientos sobre una misma arquitectura, conjunto de datos e hiperparámetros, variando únicamente el algoritmo distribuido, la cantidad de nodos y el valor de $E$. Todos los ensayos se realizaron de forma secuencial sobre la misma máquina, sumando un total de 11 horas y 15 minutos de ejecución.

| Categoría        | Parámetro            | Valores                                                                        |
|------------------|----------------------|--------------------------------------------------------------------------------|
| Modelo y datos   | Dataset              | MNIST completo [@lecun1998gradient]                                            |
|                  | Arquitectura         | LeNet-5                                                                        |
|                  | Función de pérdida   | Entropía cruzada                                                               |
| Optimización     | Optimizador          | Adam ($\alpha=0{,}01$; $\beta_1=0{,}9$; $\beta_2=0{,}999$; $\epsilon=10^{-8}$) |
|                  | Tamaño de mini-batch | 64                                                                             |
|                  | Épocas máximas       | 48                                                                             |
| Infraestructura  | Serialización        | Dispersa ($r=0{,}5$)                                                           |
| Variables libres | Algoritmo            | All-Reduce, Parameter Server sincrónico                                        |
|                  | Nodos                | 2, 4, 6, 8, 10                                                                 |
|                  | Épocas offline       | 0, 1, 2, 3, 4                                                                  |

: Configuración de los ensayos. En Parameter Server, la mitad de los nodos son workers y la otra mitad servidores.

Al parametrizar $E>0$, los nodos entrenan aislados durante múltiples ciclos y el tráfico de red se reduce de forma aproximada inversamente proporcional a $E$. Al alcanzar la etapa de sincronización, la consolidación se realiza por promedio directo de los gradientes parciales, de modo que bajo descenso por gradiente la actualización responde a

$$W_{t + E} = W_{t} - \frac{\lambda}{N} \sum_{i=1}^{N} G_{i}^{(E)}$$

Los ensayos se realizaron con la serialización dispersa que implementa el sistema, siguiendo el enfoque de Aji y Heafield [@aji2017sparse]. El parámetro $r$ acota el muestreo del gradiente que se transmite: se calcula la cardinalidad efectiva del mensaje como

$$k = \operatorname{round}\bigl(|g|\cdot(1-r)\bigr),$$

y ese valor $k$ se utiliza para escoger los $k$ elementos de mayor magnitud absoluta del tensor de gradientes local, fijando un umbral mínimo. Los elementos descartados se acumulan en un gradiente residual que se considera en el mensaje siguiente. Con $r=0{,}5$, la mitad de los componentes queda fuera de cada mensaje.

El entorno de ejecución fue un procesador Intel Core i5-8350U (4 núcleos físicos, 8 hilos lógicos, 1,7 GHz base y 3,6 GHz turbo) con 8 GB de memoria DDR4, sobre Ubuntu 24.04 LTS, Docker 29.6.1, Rust 1.96.1, CPython 3.14.0 y Pumba 1.1.7.

Con Pumba se logró emular una red hogareña típica en cuestión de delay, jitter y ancho de banda.

#### Resultados: tiempo de ejecución

La variación de $E$ no indujo mejoras significativas en los tiempos de ejecución: las curvas de cada configuración se solapan de forma estricta en todos los ensayos. Dado que el sistema se ejecuta en una única máquina, la supresión del costo de comunicación no alcanza a compensar la sobrecarga de cómputo, contrariamente a la expectativa inicial.

![Comparación de tiempos de ejecución bajo distintas cantidades de épocas offline con All-Reduce.](figures/oe_ar_time.png){width=70%}

La escalabilidad del rendimiento se estanca al superar los 4 nodos, y en All-Reduce la elevación a 10 nodos degrada el desempeño global por la alta contención de CPU.

![Comparación de tiempos de ejecución bajo distintas cantidades de épocas offline con Parameter Server.](figures/oe_ps_time.png){width=70%}

En Parameter Server, en cambio, la configuración de 10 nodos (5 workers y 5 servidores) reduce el tiempo respecto de configuraciones menores. La explicación es contraintuitiva pero consistente con el entorno: mientras que en All-Reduce los procesos operan en cómputo continuo, el esquema sincrónico de Parameter Server introduce estados de inactividad obligatorios mediante barreras, y esa inactividad disminuye la contención, permitiendo una alternancia más eficiente de los hilos de ejecución sobre los cuatro núcleos físicos disponibles. All-Reduce resulta, no obstante, intrínsecamente más veloz en configuraciones de pocos nodos, por su distribución balanceada de responsabilidades sin dependencias centralizadas.

#### Resultados: exactitud

El impacto de $E$ se manifiesta con mayor claridad en la exactitud, por ser una métrica independiente de las limitaciones del hardware de ejecución. En ambos algoritmos, la exactitud final decrece de forma monótona al incrementar $E$. La introducción de una única época offline ($E=1$), que reduce la frecuencia de sincronización global a la mitad, ya provoca un deterioro inmediato de aproximadamente un punto porcentual.

![Comparación de exactitud bajo distintas cantidades de épocas offline con Parameter Server.](figures/oe_ps_acc.png){width=70%}

La degradación se acentúa de forma proporcional al número de nodos. Al incrementar las entidades distribuidas, las particiones locales del conjunto de datos se reducen y se vuelven potencialmente sesgadas; la falta de sincronización frecuente exacerba el impacto de esos sesgos locales, perjudicando la convergencia global.

![Comparación de exactitud bajo distintas cantidades de épocas offline con All-Reduce.](figures/oe_ar_acc.png){width=70%}

En All-Reduce la pérdida de exactitud es más severa que en Parameter Server. Esto se explica por el diseño experimental: bajo las condiciones evaluadas, All-Reduce duplica el número de workers activos al prescindir de servidores dedicados, lo que fragmenta aún más el conjunto de datos e incrementa el aislamiento operativo. La integración tardía de parámetros fuerza entonces correcciones abruptas del gradiente global.

#### Resultados: inestabilidad de la pérdida

A partir del historial de entrenamiento se seleccionaron trayectorias representativas que confirman que el incremento de $E$ introduce inestabilidad en la función de pérdida, evidenciada mediante oscilaciones y picos proporcionales al valor de $E$.

![Trayectoria de pérdida con entropía cruzada exhibiendo ruido estocástico por cómputo offline en Parameter Server (2 nodos).](figures/oe_ps_loss_n2.png){width=70%}

La dinámica del optimizador presenta mayores dificultades en las épocas iniciales, la fase asociada a las correcciones de mayor magnitud, y luego se estabiliza al aproximarse a un mínimo local. Sin embargo, los valores de convergencia residual son sistemáticamente más altos a mayor $E$. El fenómeno sugiere que el desacoplamiento prolongado restringe la capacidad del optimizador para alcanzar el mínimo, induciendo un comportamiento análogo al de una tasa de aprendizaje excesivamente alta.

![Trayectoria de pérdida con entropía cruzada exhibiendo ruido estocástico por cómputo offline en All-Reduce (2 nodos).](figures/oe_ar_loss_n2.png){width=70%}

En All-Reduce se observa un patrón similar, aunque con oscilaciones relativamente más atenuadas, lo que revierte la hipótesis inicial que preveía mayor inestabilidad en el esquema descentralizado.

#### Discusión y limitaciones

La ventaja de introducir épocas offline es estrictamente reducir el tiempo de entrenamiento. En un entorno local simulado, reducir el costo de comunicación no compensa el costo de obtener un peor modelo: en estos experimentos no se evidenció ninguna ventaja de utilizar un $E$ mayor a 0.

En contraposición, en un despliegue multimáquina sobre una red física, retrasar la agregación de gradientes puede resultar crítico para amortizar el tiempo de comunicación. La decisión depende de tres factores: el tamaño del modelo, la velocidad de la red y la cantidad de nodos. Con modelos más grandes, los mensajes que contienen gradientes y parámetros son también más grandes. Para entrenamientos con mucho dato y poco modelo, el cuello de botella es el cómputo de gradientes; en cambio, para entrenamientos con poco dato y mucho modelo, donde la comunicación toma protagonismo, incrementar $E$ puede ser útil para reducir los tiempos de ejecución.

Un resultado del estudio que merece señalarse es que la desestabilización afecta negativamente al modelo de forma predecible, lo que abre una línea concreta: evaluar estrategias de activación dinámica de las épocas offline, introduciéndolas exclusivamente en fases avanzadas del entrenamiento, cuando la optimización principal ha concluido, de modo que la exploración independiente de un worker en torno a un mínimo local pueda guiar favorablemente al resto tras la sincronización.

#### Conclusión del estudio

El uso de épocas offline en O.N.O. altera el balance entre comunicación y fidelidad algorítmica. En la infraestructura simulada, incrementar $E$ perjudicó al modelo entrenado de manera predecible pero sin aportar ganancias de velocidad, debido a las limitaciones del hardware empleado. Queda planteado validar el sistema bajo configuraciones multimáquina reales, comparando arquitecturas de modelos de distintos tamaños al variar $E$.

### Síntesis de la validación
La observación transversal es que el sistema cumplió su propósito: validada la corrección de las estrategias por la vía de su convergencia, las diferencias de tiempo, throughput y escalabilidad admiten atribuirse al algoritmo y no a su implementación, que era exactamente el problema que este trabajo se propuso resolver.
