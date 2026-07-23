\newpage
# Solución implementada

Se construyó Oxidized Neural Orchestra (O.N.O.), un sistema distribuido de entrenamiento de modelos de aprendizaje profundo. Comprende una biblioteca de redes neuronales, una capa de comunicación con protocolo propio, un plano de control que coordina la ejecución, los tres algoritmos de distribución y tres interfaces de uso.

Esta sección describe la arquitectura resultante. El énfasis está puesto en las decisiones de diseño y en las razones que las motivaron, por encima de los detalles de implementación: en un trabajo cuyo objetivo era construir una base de comparación, son esas decisiones las que determinan si la base sirve para lo que fue construida.

## Visión general

El sistema se organiza como un *workspace* de Cargo con ocho crates y unas 27 000 líneas de código, más una suite de experimentación en Python. La dependencia entre ellos es estrictamente jerárquica: `comms` no depende de ningún otro crate del sistema, `machine_learning` depende únicamente de `comms` y no contiene nada relativo a redes, y sobre ambos se construyen los runtimes de los nodos y el plano de control.

| Crate | Rol |
|---|---|
| `comms` | Capa de red: protocolo, transporte, handles tipados, compresión, reparto del dataset |
| `machine_learning` | Motor de redes neuronales, sin networking |
| `parameter_server` | Runtime del servidor de parámetros: almacenamiento y sincronización |
| `worker` | Runtime del nodo trabajador en sus dos variantes, más los middlewares de topología |
| `orchestrator` | Esquema de configuración, validación, cálculo de topología y sesión de entrenamiento |
| `node` | El único binario desplegable, agnóstico del rol |
| `orchestui` | Interfaz interactiva de terminal |
| `orchestra-py` | Biblioteca de Python sobre la interfaz de funciones externas |

: Crates del workspace y su responsabilidad.

La decisión rectora de la arquitectura es el agnosticismo de rol. Todas las máquinas del clúster ejecutan el mismo binario, `node`; el rol que cada una cumplirá, trabajador o servidor de parámetros, se determina en tiempo de ejecución a partir de la especificación que el orquestador le entrega al conectarse. La configuración de un entrenamiento lleva una única lista plana de direcciones más la cantidad de servidores deseada, y nunca declara qué dirección será servidor: eso lo decide el orquestador a partir de las latencias que mide.

La alternativa evidente era mantener binarios separados de trabajador y de servidor. Se eligió la unificación porque el sistema debía poder cambiar la topología en tiempo de ejecución: Strategy-Switch requiere que un nodo que arrancó como trabajador se convierta en servidor a mitad del entrenamiento, y eso es imposible si el rol está fijado en el binario desplegado. El costo fue obligar a usar tareas de ejecución local en el runtime asincrónico, porque uno de los runtimes contiene un objeto de entrenamiento que no puede moverse entre hilos.

La segunda decisión estructural es la separación entre infraestructura y dominio. El trabajador no sabe nada de redes neuronales: es genérico sobre una estrategia de entrenamiento, y todo lo relativo al aprendizaje vive detrás de contratos compartidos. Del lado de la red, la contracara de esa regla es que los trucos de codificación y de precisión viven exclusivamente en `comms`. Los trabajadores y los servidores nunca tocan media precisión y siempre operan en precisión simple; la conversión ocurre únicamente en el serializador y el deserializador.

## Capa de comunicación

### Protocolo de dos planos

El protocolo separa el plano de control del plano de datos y los serializa de manera distinta, porque las exigencias de uno y otro son opuestas.

El plano de control transporta veinte variantes de comando, entre ellas el establecimiento de conexión, la creación de nodo con su especificación, la medición de latencia, el reporte de pérdidas, la orden de detención y la promoción de un nodo a servidor. Son mensajes poco frecuentes y estructuralmente ricos, de modo que se serializan como JSON: la ergonomía y la evolucionabilidad del formato pesan más que los bytes.

El plano de datos transporta gradientes, parámetros y fragmentos del conjunto de datos. Son mensajes grandes y frecuentes, y se serializan reinterpretando la memoria directamente, sin ninguna biblioteca de serialización de por medio. El resultado es un camino de datos sin copias, y esa propiedad se apoya en tres detalles concretos. El envío emite una única llamada al sistema vectorizada sobre el encabezado y el tensor, de modo que el tensor nunca se copia a un buffer intermedio. El buffer de recepción está respaldado por un vector de enteros de 32 bits en lugar de bytes, para garantizar el alineamiento a cuatro bytes que la reinterpretación de memoria requiere. Y el mensaje de parámetros expone una referencia mutable, lo que permite al receptor modificarlos en el propio buffer de lectura sin copiarlos.

### Transporte y reducción del tráfico

El transporte se define como un *trait* asincrónico con dos operaciones, envío y recepción, pensado para apilarse siguiendo el patrón decorador. Al cierre del trabajo tiene una única capa, la de enmarcado de mensajes, de modo que el sistema hereda su fiabilidad de TCP. La discusión sobre las capas de reintento y vencimiento que llegó a tener, y las razones por las que se retiraron, se retoma en *Riesgos materializados y lecciones aprendidas*.

Sobre ese transporte se aplican dos técnicas complementarias de reducción del volumen transmitido. La primera es la codificación de gradientes en media precisión, que reduce a la mitad los bytes enviados; se aplica únicamente a los gradientes, mientras que los parámetros y los fragmentos de datos viajan en precisión simple.

La segunda es un protocolo de gradientes dispersos que sigue el enfoque de Aji y Heafield [@aji2017sparse]: se transmiten únicamente los componentes de mayor magnitud del gradiente y los descartados se acumulan en un residuo que se considera en el mensaje siguiente. El umbral que separa unos de otros no se calcula de forma exacta, ya que hacerlo exigiría ordenar el gradiente completo en cada paso; se estima muestreando hasta 16 384 valores y aplicando una selección parcial de costo lineal. El formato serializado codifica tramos contiguos por encima del umbral mediante desplazamientos relativos, lo que evita transmitir un índice por cada valor.

La elección entre ambas es adaptativa: se calcula la versión dispersa y solo se utiliza si su tamaño resulta efectivamente menor que el de la versión densa en media precisión; en caso contrario se degrada a esta última. Un gradiente denso, que es lo que produce una red convolucional, no se beneficia de la codificación dispersa, y el sistema lo detecta en lugar de suponerlo.

### Handles tipados

Toda la capa de comunicación usa el sistema de tipos para hacer irrepresentables los estados inválidos. Cada rol tiene su propio handle con únicamente las operaciones que le corresponden, y la asignación de rol a un nodo recién conectado consume el handle sin rol y devuelve uno de trabajador o de servidor, lo que hace imposible por construcción operar sobre un nodo al que todavía no se le asignó un papel. El mismo mecanismo sostiene la promoción en Strategy-Switch, donde el handle de trabajador se consume y devuelve un handle de servidor reutilizando la misma conexión, de modo que la transición no puede dejar al sistema en un estado intermedio. La misma idea se aplica a los hiperparámetros con rangos acotados, que se representan mediante tipos que validan en el borde de deserialización.

## Motor de redes neuronales

El motor se construyó sobre `ndarray`, que provee las operaciones sobre arreglos numéricos n-dimensionales y la representación de los tensores. La propagación hacia adelante y hacia atrás, los optimizadores, las inicializaciones y las funciones de pérdida se implementaron en el marco de este trabajo, sin recurrir a ningún framework de aprendizaje profundo.

### El modelo como buffer plano

La decisión de diseño más consecuente del motor es que las capas no almacenan sus parámetros. Los parámetros se pasan como un fragmento plano de memoria en cada llamada, y la capa los reinterpreta con la forma que le corresponde. Las capas sí conservan estado de trabajo, que es la entrada de la pasada y los buffers precomputados para la propagación hacia atrás. Ese estado se dimensiona durante la primera época y luego se reutiliza, de modo que después de ese calentamiento una pasada de entrenamiento no reserva memoria dinámica.

El modelo completo vive entonces en un único buffer plano, sobre el cual las capas operan como vistas definidas por desplazamientos y formas. Esa representación habilita tres cosas que de otro modo serían imposibles: que la entrada y salida de red no requiera copias, ya que el buffer del modelo es directamente el buffer que se transmite; que el modelo pueda partirse entre varios servidores entregándole a cada uno un rango del buffer; y que la misma implementación funcione sin cambios tanto cuando una sola entidad posee todas las capas como cuando están repartidas. El componente que unifica los tres algoritmos no serializa nada: rebana uno o varios buffers planos y entrega a cada capa su ventana, guiado bajo Parameter Server por un vector de asignación que indica a qué servidor pertenece cada capa. El mismo modelo secuencial corre sin modificaciones en ambos regímenes, y es el punto de polimorfismo entre algoritmos del sistema.

### Componentes

Las capas se modelan como un enumerado cerrado con despacho estático y no como un *trait* con despacho dinámico, decisión coherente con el objetivo de rendimiento. Se implementaron capas densas, convolucionales, de *max-pooling*, de reordenamiento de forma y las activaciones sigmoide, tangente hiperbólica, ReLU y softmax. La capa de reordenamiento, que es de costo nulo, es el adaptador que permite convivir a las capas densas con las convolucionales, y el constructor la inserta automáticamente en las fronteras correspondientes, de modo que el lenguaje de configuración no necesita exponer el concepto.

La convolución concentró buena parte del esfuerzo del trabajo. Su implementación final se basa en reformular la convolución como una multiplicación de matrices, lo que permite delegar el trabajo pesado en las rutinas de álgebra lineal optimizadas y aprovechar la localidad de caché. Se llegó a ella después de probar y descartar dos alternativas, la transformada rápida de Fourier y la paralelización con hilos, según se detalla en las lecciones aprendidas. La mejora fue de 2,27 veces en el tiempo total de entrenamiento.

Se implementaron dos funciones de pérdida, error cuadrático medio y entropía cruzada, que calculan el valor y el gradiente en una misma pasada. La entropía cruzada acota su argumento para protegerse del logaritmo de cero.

Los optimizadores implementados son descenso por gradiente, descenso con momento clásico y Adam [@kingma2014adam; @sutskever2013momentum]. Las inicializaciones cubren los esquemas de Xavier-Glorot [@glorot2010understanding], Kaiming-He [@he2015delving] y LeCun, en sus variantes uniforme y normal. Los inicializadores no son generadores por tensor sino generadores con presupuesto que producen valores sobre demanda, lo cual es consecuencia natural de que el modelo sea un buffer plano; un mecanismo adicional resuelve los pedidos que cruzan la frontera entre dos generadores, y es el que permite inicializar cada capa con su esquema propio sobre una única tira de memoria.

La ubicación del optimizador merece explicarse, porque es una decisión de diseño y no un detalle. En el bucle interno, cada worker calcula el gradiente de un mini-batch y lo aplica localmente mediante descenso por gradiente simple, sin estado. El optimizador configurado, que puede tener estado, se aplica una sola vez por paso de sincronización, cuando ya están agregados los gradientes de todos los workers: en el servidor bajo Parameter Server, y en cada nodo bajo All-Reduce. La razón es que un optimizador con estado en el bucle interno acumularía momentos contra una posición del modelo que la sincronización va a sobrescribir, dejando ese estado inconsistente con el punto en el que el modelo efectivamente queda tras agregar.

El modelo entrenado se exporta en formato `safetensors`, siguiendo la convención de nombres de PyTorch para facilitar la interoperabilidad. El layout interno de las capas densas es la traspuesta del que usa PyTorch, de modo que un cargador debe transponer.

## Los tres algoritmos

### Parameter Server

El sistema implementa un Parameter Server multi-servidor con particionado del modelo. La decisión de diseño relevante es cómo se reparten las capas entre los servidores.

El particionado equitativo por cantidad de parámetros deja los parámetros de una misma capa repartidos entre dos servidores, con lo cual el fragmento deja de ser contiguo y fuerza una reserva de memoria. Se optó por particionar respetando los límites de capa, repartiendo capas completas entre los servidores mediante un algoritmo *greedy* de empaquetado que asigna primero las más grandes. Junto con la asignación viaja al trabajador un vector de orden que le indica, para cada capa, a qué servidor debe pedirle su fragmento.

Esa decisión tiene una consecuencia en el reensamblado final del modelo: como las capas se reparten por tamaño y no por orden, concatenar ingenuamente lo que devuelve cada servidor produciría un modelo barajado, de modo que el reensamblado recorre explícitamente los desplazamientos por capa.

Del lado del servidor, el almacenamiento y la sincronización se abstraen detrás de dos *traits* independientes, lo que convierte el régimen de operación en una elección de configuración en tiempo de ejecución en lugar de una decisión de compilación. El almacenamiento con bloqueo está fragmentado internamente y usa doble buffer: los trabajadores siguen acumulando sobre el buffer activo mientras el congelado se consume, y la actualización usa una operación atómica de comparación e intercambio que la vuelve de ejecución única, de modo que si otro hilo ya está actualizando, la llamada no hace nada.

El sincronizador con barrera es el que da la semántica sincrónica utilizada en los experimentos. Su contador es dinámico: un trabajador que termina su entrenamiento y se retira lo decrementa permanentemente, y si con eso la barrera se completa, la avanza él mismo antes de irse. Esa lógica está atada a la destrucción del objeto, de modo que ocurre necesariamente, y resuelve el problema de membresía estática frente a terminación dinámica que se discute en las lecciones aprendidas.

La segunda variante de almacenamiento sigue el enfoque sin bloqueos de *Hogwild!* [@recht2011hogwild]. Está implementada, pero no se utilizó en los benchmarks por las razones de corrección y de validez de supuestos que se detallan en las lecciones aprendidas.

### Ring All-Reduce

La implementación sigue la formulación en anillo de Patarasuk y Yuan [@patarasuk2009bandwidth], en dos fases de $n-1$ pasos cada una. En la primera, cada nodo termina con un fragmento del gradiente reducido sobre todos los participantes; en la segunda, ese fragmento ya reducido se propaga por el anillo. El costo total es de $2(n-1)$ pasos, cada uno moviendo aproximadamente $1/n$ del modelo, lo que da el costo óptimo en ancho de banda.

Tres decisiones concretas la sostienen. El fragmentado es balanceado: los fragmentos difieren a lo sumo en un elemento, en lugar de dejar una cola corta, de modo que ningún paso del anillo es sistemáticamente más lento que los demás. El envío y la recepción de cada paso son concurrentes, y lo mismo vale para la construcción del anillo, donde aceptar la conexión entrante y establecer la saliente también se hacen en paralelo; hacerlo de forma secuencial provoca que todos los nodos intenten enviar antes de recibir y los buffers de los sockets se llenen sin que nadie los drene. Y el gradiente reducido se promedia dividiendo por la cantidad de trabajadores, de modo que la tasa de aprendizaje efectiva no escale con el tamaño del anillo.

### Strategy Switch

La implementación de Strategy-Switch [@provatas2025strategyswitch] separa la decisión, que vive en el orquestador, de la ejecución, que ocurre en los nodos. El criterio de conmutación mantiene una ventana deslizante de las últimas pérdidas y calcula su cambio relativo medio; cuando ese valor cae por debajo de un umbral, es decir, cuando la curva de pérdida se aplana, se dispara la transición. La ventana y el umbral son los valores indicados en el trabajo original.

El plan se precomputa antes de entrenar. La topología completa del Parameter Server, la ubicación de los servidores y el particionado de las capas se calculan en tiempo de configuración, y para cada nodo se construye por anticipado la acción que le corresponderá; lo único dinámico es el momento en que se ejecuta. La razón es que calcular la topología requiere las mediciones de latencia, que ya se hicieron al conectar, y hacerlo en caliente introduciría una pausa en el entrenamiento.

La promoción de un trabajador a servidor transfiere los parámetros ya entrenados: el nodo rebana sus pesos según los rangos recibidos y arranca como servidor sembrado con ellos, no con valores aleatorios. La ligadura entre cada promoción y su fragmento se hace por dirección del servidor y no por orden de aparición en el recorrido de la topología.

La transición es unidireccional, de All-Reduce a Parameter Server, y de un solo disparo por sesión.

## Plano de control

El orquestador es el plano de control del sistema. Recibe la configuración del entrenamiento, la valida, se conecta a los nodos, mide la topología de la red, deriva de allí la asignación de roles y especificaciones, y conduce la sesión.

### Selección de topología

Una de las líneas no previstas en el plan original, y de las más interesantes, es la selección de topología a partir de mediciones. Antes de comenzar, el orquestador realiza rondas de medición de latencia entre todos los pares de nodos y construye con ellas un grafo, cuyas aristas se pesan con el RTT máximo observado y no con el promedio. Ambos problemas de optimización se plantean, en consecuencia, sobre el peor caso. El criterio quedó resumido en la discusión que lo originó: *"es mejor tener 2 conexiones más o menos que una muy buena y una muy mala"*, que es exactamente la diferencia entre optimizar la suma y optimizar el máximo.

El orden del anillo de All-Reduce se resuelve como un problema del viajante de comercio, mediante programación dinámica sobre subconjuntos. La ubicación de los servidores de parámetros se resuelve por *backtracking* sobre las combinaciones de $k$ servidores: en Parameter Server los nodos forman un grafo bipartito completo $K_{n,m}$ con $n$ trabajadores y $m$ servidores, de modo que todas las conexiones son entre trabajadores y servidores, y se elige la combinación cuya conexión máxima resulte la menor.

Ambos son algoritmos exactos y no heurísticas, lo cual es defendible dado el régimen al que apunta el sistema, de decenas de nodos, aunque el del ciclo degenera para tamaños grandes.

### Sesión orientada a eventos

La ejecución de un entrenamiento se modela como una sesión orientada a eventos. El orquestador expone un canal de eventos y mantiene una tarea por trabajador que traduce los mensajes del protocolo en eventos de dominio: pérdidas publicadas, trabajador terminado, entrenamiento completo, nodo promoviéndose. Ese diseño es lo que permite que las tres interfaces del sistema consuman exactamente la misma información sin duplicar lógica.

El punto de agregación es el manejo de pérdidas: se registra la última pérdida de cada trabajador, se exige que todos los trabajadores vivos hayan reportado, y recién entonces se alimentan los dos criterios que observan la curva, el de detención temprana y el de conmutación de estrategia. Que la condición sea sobre los trabajadores vivos y no sobre una cantidad fija es, otra vez, consecuencia de la membresía dinámica.

La detención temprana [@prechelt1998early] se implementó como un criterio de tolerancia: cuando tres pérdidas agregadas consecutivas se mueven por debajo de un umbral configurable, se considera que el entrenamiento convergió y se difunde la orden de detención.

## Interfaces

El sistema expone tres formas de uso, todas sobre el mismo plano de control y compartiendo el mismo esquema de configuración canónico.

La interfaz interactiva de terminal permite configurar, lanzar y monitorear un entrenamiento en vivo. Presenta un asistente de configuración con sugerencias y validación, y un panel que muestra la evolución de la pérdida, el progreso por trabajador y una vista de topología. En esa vista, las partículas que representan el flujo de datos entre nodos se mueven al ritmo real del entrenamiento, siguiendo una media móvil del intervalo entre reportes, y permiten ver cómo un trabajador se convierte en servidor durante una conmutación de estrategia. El entrenamiento corre en un hilo de fondo para que la interfaz nunca se bloquee.

La biblioteca de Python se construyó sobre PyO3, como un crate envoltorio separado que mantiene el orquestador como Rust puro. La alternativa, más limpia en teoría, era anotar los tipos existentes del orquestador con los atributos de PyO3 detrás de una bandera de compilación; se descartó por restricciones del compilador, ya que esos atributos no admiten tipos genéricos ni las variantes complejas de enumerados que el esquema de configuración utiliza. El entrenamiento libera el bloqueo global del intérprete y se ejecuta en un hilo dedicado, de modo que el proceso de Python actúa efectivamente como orquestador.

El binario de ejecución desatendida completa el conjunto: no presenta interfaz interactiva, toma un archivo de configuración y ejecuta el entrenamiento hasta el final. Es el que utiliza la suite de benchmarks.

## Estado de la implementación

El sistema compila sin advertencias y su conjunto de pruebas pasa con 87 pruebas en verde. El motor de aprendizaje propio converge a 0,983 de exactitud sobre MNIST con la red de referencia y a 0,980 con LeNet-5, frente a 0,989 y 0,990 de una implementación de PyTorch con la misma receta de hiperparámetros, lo que sitúa la implementación propia a alrededor de medio punto porcentual de un framework maduro. Los defectos conocidos al cierre y las líneas que quedaron abiertas se detallan en *Trabajos futuros*.
