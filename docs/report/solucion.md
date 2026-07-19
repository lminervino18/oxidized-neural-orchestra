\newpage
# Solución implementada

Se construyó **Oxidized Neural Orchestra** (O.N.O.), un sistema distribuido de entrenamiento de modelos de aprendizaje profundo escrito íntegramente en Rust. El sistema comprende una biblioteca de redes neuronales desarrollada desde cero, una capa de comunicación con protocolo propio, un plano de control que coordina la ejecución, los tres algoritmos de distribución y tres interfaces de uso.

Esta sección describe la arquitectura resultante. El énfasis está puesto en las **decisiones de diseño y en las razones que las motivaron**, por encima de los detalles de implementación: en un trabajo cuyo objetivo era construir una base de comparación, son esas decisiones las que determinan si la base sirve o no para lo que fue construida.

## Visión general y decisión rectora

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

La **decisión rectora de toda la arquitectura es el agnosticismo de rol**. Todas las máquinas del clúster ejecutan el mismo binario, `node`; el rol que cada una cumplirá, trabajador o servidor de parámetros, se determina en tiempo de ejecución a partir de la especificación que el orquestador le entrega al conectarse. La configuración de un entrenamiento lleva una **única lista plana de direcciones** más la cantidad de servidores deseada, y nunca declara qué dirección será servidor: eso lo decide el orquestador a partir de las latencias que mide.

Esta decisión se tomó frente a la alternativa evidente, que era mantener binarios separados de trabajador y de servidor. Se eligió la unificación porque el sistema debía poder **cambiar la topología en tiempo de ejecución**: Strategy-Switch requiere que un nodo que arrancó como trabajador se convierta en servidor a mitad del entrenamiento, y eso es imposible si el rol está fijado en el binario que se desplegó. El costo de la decisión fue técnico y concreto: obligó a usar tareas de ejecución local en el runtime asincrónico, porque uno de los runtimes contiene un objeto de entrenamiento que no puede moverse entre hilos.

Una segunda decisión estructural, tomada muy temprano y sostenida hasta el final, es la **separación estricta entre infraestructura y dominio**. El trabajador es "estrictamente infraestructura": no sabe nada de redes neuronales, es genérico sobre una estrategia de entrenamiento, y todo lo relativo al aprendizaje vive detrás de contratos compartidos. La contracara de esa regla, del lado de la red, es que **los trucos de codificación y de precisión viven exclusivamente en `comms`**. Esta última regla fue la más disputada del proyecto y llegó a rechazarse una implementación completa de codificación en media precisión por violarla, filtrando tipos de la biblioteca de media precisión hacia otros módulos. El diseño finalmente aceptado establece que los trabajadores y los servidores nunca tocan media precisión y siempre operan en precisión simple; la conversión ocurre únicamente en el serializador y el deserializador.

## Capa de comunicación

### Protocolo de dos planos

La decisión de diseño central de la capa de comunicación es un **protocolo híbrido**, que separa el plano de control del plano de datos y los serializa de manera completamente distinta.

El **plano de control** transporta veinte variantes de comando: establecimiento de conexión, creación de nodo con su especificación, medición de latencia, reporte de pérdidas, pedido de parámetros, reparto del dataset, orden de detención, conmutación de estrategia, promoción de un nodo a servidor y desconexión. Son mensajes poco frecuentes y estructuralmente ricos, de modo que se serializan como JSON: la ergonomía y la evolucionabilidad del formato pesan más que los bytes.

El **plano de datos** transporta gradientes, parámetros y fragmentos del conjunto de datos. Aquí el criterio se invierte por completo: son mensajes enormes y frecuentes, y se serializan **reinterpretando la memoria directamente**, sin ninguna biblioteca de serialización de por medio. El resultado es un camino de datos sin copias.

Esa ausencia de copias no es una afirmación de intención sino una propiedad construida deliberadamente, y se apoya en tres detalles que vale la pena señalar porque muestran el nivel al que se trabajó el problema. El primero es que el envío emite una **única llamada al sistema vectorizada** sobre el encabezado y el tensor, de modo que el tensor nunca se copia a un buffer intermedio. El segundo es que el buffer de recepción está respaldado por un vector de enteros de 32 bits en lugar de bytes, **con el único propósito de garantizar el alineamiento a cuatro bytes** que la reinterpretación de memoria requiere para ser válida. El tercero es que el mensaje de parámetros es un préstamo **mutable**, lo que permite al receptor modificar los parámetros en el propio buffer de lectura sin copiarlos.

Un detalle menor pero elegante: el indicador que marca el último mensaje de gradiente de una fase no ocupa ningún byte, ya que se codifica en el bit menos significativo del campo que identifica el tipo de mensaje.

Esa señalización merece una nota, porque es una decisión de diseño generalizable a la que el proyecto llegó por la vía del error. Originalmente el trabajador señalizaba el fin del entrenamiento **cerrando el socket**, y el servidor leía ese cierre como una falla de escritura. La corrección movió la señal al protocolo: **el fin de una fase distribuida es información de aplicación, no un evento de transporte**.

### Reducción del tráfico

Se implementaron dos técnicas complementarias para reducir el volumen de datos que circula por la red.

La primera es la **codificación de gradientes en media precisión**, que reduce a la mitad los bytes transmitidos. Se aplica únicamente a los gradientes; los parámetros y los fragmentos de datos viajan en precisión simple.

La segunda es un **protocolo de gradientes dispersos** que sigue el enfoque de Aji y Heafield [@aji2017sparse]: se transmiten únicamente los componentes de mayor magnitud del gradiente y los descartados se acumulan en un residuo que se considera en el mensaje siguiente. El umbral que separa unos de otros no se calcula de forma exacta, ya que hacerlo exigiría ordenar el gradiente completo en cada paso; en su lugar se **estima muestreando** hasta 16 384 valores y aplicando una selección parcial de costo lineal. El formato serializado codifica tramos contiguos por encima del umbral mediante desplazamientos relativos, lo que evita transmitir un índice por cada valor.

La decisión más interesante de este componente, sin embargo, no es ninguna de las dos técnicas sino **cómo se eligen entre sí**. La compresión es **adaptativa**: se calcula la versión dispersa y solo se utiliza si su tamaño resulta efectivamente menor que el de la versión densa en media precisión; en caso contrario se degrada limpiamente a esta última. Un gradiente denso, que es lo que produce una red convolucional, no se beneficia de la codificación dispersa, y el sistema lo detecta en lugar de suponerlo.

### Transporte

El transporte está definido como un *trait* asincrónico con dos operaciones, envío y recepción, pensado para ser apilado siguiendo el patrón decorador. Durante buena parte del proyecto tuvo tres capas: enmarcado de mensajes, vencimiento por tiempo y reintento. **Al cierre del trabajo quedó una sola**, la de enmarcado, y las otras dos fueron eliminadas.

Esta reversión se describe en detalle en *Riesgos materializados y lecciones aprendidas*, porque su razón es una lección más que una decisión técnica aislada. En resumen: los vencimientos enmascaraban errores reales en lugar de mitigarlos, declarando muerto a un nodo que simplemente estaba ocupado, y reintentar sobre una conexión ya muerta no aporta nada sin un protocolo de reconexión que nunca se logró construir. El sistema hereda hoy su fiabilidad de TCP, y el *trait* permanece como costura para retomar el trabajo. La consecuencia es explícita y se declara sin rodeos: **O.N.O. no tolera fallos de nodos**.

### Handles tipados

Un rasgo de diseño que atraviesa toda la capa de comunicación es el uso del sistema de tipos para hacer **irrepresentables los estados inválidos**. Cada rol tiene su propio handle con únicamente las operaciones que le corresponden. La asignación de rol a un nodo recién conectado **consume** el handle sin rol y devuelve uno de trabajador o de servidor, lo que hace imposible por construcción operar sobre un nodo al que todavía no se le asignó un papel. El mismo mecanismo sostiene la promoción en Strategy-Switch: el handle de trabajador se consume y devuelve un handle de servidor reutilizando la misma conexión, de modo que la transición no puede dejar al sistema en un estado intermedio.

La misma filosofía aparece en los valores numéricos. Los hiperparámetros que tienen rangos válidos acotados, como una probabilidad o una tasa de aprendizaje positiva, se representan mediante tipos que validan en el borde de deserialización. Un valor inválido no llega nunca al interior del sistema.

## Motor de redes neuronales

El motor de aprendizaje se implementó por completo en el marco de este trabajo, sobre una biblioteca que provee únicamente operaciones sobre arreglos numéricos n-dimensionales. La propagación hacia adelante y hacia atrás, los optimizadores, las funciones de pérdida y la representación de los tensores son propios.

### La decisión que habilita todo lo demás

La decisión de diseño más consecuente del motor es que **las capas no almacenan sus parámetros**. Los parámetros se pasan como un fragmento plano de memoria en cada llamada, y la capa los reinterpreta con la forma que le corresponde.

Vale la pena detenerse en por qué esto importa, porque su alcance excede al motor de aprendizaje. El modelo completo vive en un **único buffer plano**, sobre el cual las capas operan como una vista sin estado definida por desplazamientos y formas. Esa representación es exactamente lo que permite tres cosas que de otro modo serían imposibles: que la entrada y salida de red no requiera copias, ya que el buffer del modelo es directamente el buffer que se transmite; que el modelo pueda **partirse entre varios servidores** entregándole a cada uno un rango del buffer; y que la misma implementación del modelo funcione sin cambios tanto en un régimen donde una sola entidad posee todas las capas como en uno donde están repartidas.

Esa última propiedad se materializa en el componente que unifica los tres algoritmos, que no serializa nada: simplemente rebana uno o varios buffers planos y entrega a cada capa su ventana. Bajo All-Reduce, una sola entidad posee todas las capas y cada nodo tiene una réplica completa. Bajo Parameter Server, un vector de asignación indica a qué servidor pertenece cada capa. **El mismo modelo secuencial corre sin modificaciones en ambos regímenes**, y esa es la verdadera costura de polimorfismo entre algoritmos del sistema.

Como se explica en la sección de riesgos, esta decisión tuvo un costo que solo se hizo visible mucho después: el préstamo de memoria que habilita la ausencia de copias es el que impidió construir la capa de reconexión.

Una segunda decisión, complementaria, es que **las capas poseen sus buffers de salida y devuelven vistas hacia sí mismas**. Después del calentamiento inicial, una pasada de entrenamiento no reserva memoria dinámica.

### Componentes

Las capas se modelan como un enumerado cerrado con despacho estático y no como un *trait* con despacho dinámico, decisión coherente con el objetivo de rendimiento. Se implementaron capas densas, convolucionales, de *max-pooling*, de reordenamiento de forma y las activaciones sigmoide, tangente hiperbólica, ReLU y softmax. La capa de reordenamiento, que es de costo nulo, es el adaptador que permite convivir a las capas densas con las convolucionales, y el constructor la **inserta automáticamente** en las fronteras correspondientes, de modo que el lenguaje de configuración no necesita exponer el concepto.

La convolución merece una mención aparte porque concentró buena parte del esfuerzo del trabajo. Su implementación final está basada en la reformulación de la convolución como una **multiplicación de matrices**, lo que permite delegar el trabajo pesado en las rutinas de álgebra lineal optimizadas y aprovechar la localidad de caché. Se llegó a ella después de probar y descartar dos alternativas, la transformada rápida de Fourier y la paralelización con hilos, según se detalla en las lecciones aprendidas. La mejora resultante fue de 2,27 veces en tiempo total, con la exactitud subiendo de 97,26 % a 98,5 %.

Se implementaron dos funciones de pérdida, error cuadrático medio y entropía cruzada, ambas calculando el valor y el gradiente en una sola pasada fusionada. La entropía cruzada acota su argumento para protegerse del logaritmo de cero. Corresponde señalar una **brecha de optimización conocida**: softmax es una capa ordinaria cuyo jacobiano completo se compone con el gradiente de la entropía cruzada, en lugar de estar fusionados. Matemáticamente el resultado es equivalente, pero cuesta una pasada adicional por fila y reintroduce una división acotada que la forma fusionada existe precisamente para evitar.

Los optimizadores implementados son descenso por gradiente, descenso con momento clásico y Adam [@kingma2014adam; @sutskever2013momentum]. Las inicializaciones cubren los esquemas de Xavier-Glorot [@glorot2010understanding], Kaiming-He [@he2015delving] y LeCun, en sus variantes uniforme y normal. Los inicializadores no son generadores por tensor sino **generadores con presupuesto** que producen valores sobre demanda, lo cual es la consecuencia natural de que el modelo sea un buffer plano; existe además un mecanismo que cose pedidos que cruzan la frontera entre dos generadores, y es el que permite inicializar cada capa con su esquema propio sobre una única tira de memoria.

Una decisión sutil de ubicación merece señalarse: los **optimizadores con estado viven en la frontera de sincronización, no en la del lote local**. El bucle interno por lote usa optimizadores sin estado que replican la tasa de aprendizaje, mientras que los momentos de Adam o de momento clásico solo se aplican al agregar gradientes. La razón es que, de otro modo, en All-Reduce los buffers internos del optimizador contendrían basura al momento de la reducción, y en Parameter Server habría que transmitir el estado del optimizador por mensaje.

El modelo entrenado se exporta en formato `safetensors`, siguiendo la convención de nombres de PyTorch para facilitar la interoperabilidad. Conviene advertir que el layout interno de las capas densas es la traspuesta del que usa PyTorch, de modo que un cargador debe transponer.

## Los tres algoritmos

### Parameter Server

El sistema implementa un Parameter Server **multi-servidor con particionado del modelo**. La decisión de diseño relevante es **cómo se reparten las capas** entre los servidores, y se tomó descartando la alternativa obvia.

El particionado equitativo por cantidad de parámetros produce un problema: los parámetros de una misma capa pueden quedar a caballo de dos servidores, lo que hace que el fragmento deje de ser contiguo y fuerce una reserva de memoria. Se optó por **particionar respetando los límites de capa**, repartiendo las capas completas entre los servidores mediante un algoritmo voraz de empaquetado que asigna primero las más grandes. Junto con la asignación viaja al trabajador un vector de orden que le indica, para cada capa, a qué servidor debe pedirle su fragmento.

Esa decisión tiene una consecuencia en el reensamblado final del modelo que no es evidente: como las capas se reparten por tamaño y no por orden, concatenar ingenuamente lo que devuelve cada servidor produciría un modelo barajado. El reensamblado recorre explícitamente los desplazamientos por capa. Un error precisamente de este tipo, en el que los servidores almacenaban las capas ordenadas por tamaño en lugar de por su posición en el modelo, sobrevivió meses en el código porque todas las redes de prueba eran monótonamente decrecientes y ambos órdenes coincidían por casualidad.

Del lado del servidor, el almacenamiento y la sincronización se abstraen detrás de dos *traits* independientes, lo que convierte el régimen de operación en una **elección de configuración en tiempo de ejecución** en lugar de una decisión de compilación. El almacenamiento con bloqueo está fragmentado internamente y usa **doble buffer**: los trabajadores siguen acumulando sobre el buffer activo mientras el congelado se consume, y la actualización usa una operación atómica de comparación e intercambio que la vuelve de ejecución única, de modo que si otro hilo ya está actualizando la llamada no hace nada. El sincronizador con barrera es el que da la semántica sincrónica utilizada en los experimentos.

La barrera merece una nota, porque resuelve el problema de coordinación distribuida más costoso del proyecto. Es una **barrera de tamaño dinámico**: un trabajador que termina su entrenamiento y se retira decrementa permanentemente el contador, y si con eso la barrera se completa, la avanza él mismo antes de irse. Esa lógica está atada a la destrucción del objeto, de modo que ocurre necesariamente. Es la solución al problema canónico de **membresía estática frente a terminación dinámica**, que con un contador fijo dejaba a los supervivientes esperando indefinidamente a alguien que ya se había ido.

La segunda variante de almacenamiento, sin bloqueos y siguiendo el enfoque de *Hogwild!* [@recht2011hogwild], está implementada pero **deshabilitada**, por las razones de corrección y de validez de supuestos que se detallan en las lecciones aprendidas. No se la considera una variante operativa del sistema.

### Ring All-Reduce

La implementación sigue la formulación en anillo de Patarasuk y Yuan [@patarasuk2009bandwidth], en dos fases de $n-1$ pasos cada una. En la primera, cada nodo termina con un fragmento del gradiente reducido sobre todos los participantes; en la segunda, ese fragmento ya reducido se propaga por el anillo. El costo total es de $2(n-1)$ pasos, cada uno moviendo aproximadamente $1/n$ del modelo, lo que da el costo óptimo en ancho de banda.

Tres decisiones concretas hacen que funcione. La primera es que el **fragmentado es balanceado**: los fragmentos difieren a lo sumo en un elemento, en lugar de dejar una cola corta, de modo que ningún paso del anillo es sistemáticamente más lento que los demás. La segunda es que **el envío y la recepción de cada paso son concurrentes**. Esto último no fue una decisión de diseño previa sino la corrección de un bloqueo: hacerlos secuencialmente provocaba que todos los nodos intentaran enviar antes de recibir, llenando los buffers de los sockets sin que nadie los drenara. El sistema se colgaba silenciosamente, sin error y sin consumo de CPU, con modelos de casi dos millones de parámetros. La misma lección se aplicó a la construcción del anillo, donde aceptar la conexión entrante y establecer la saliente también se hacen de forma concurrente.

La tercera es que el gradiente reducido **se promedia dividiendo por la cantidad de trabajadores**. Puede parecer un detalle, pero su ausencia fue uno de los tres errores silenciosos que la suite de benchmarks destapó: el anillo sumaba en lugar de promediar, de modo que la tasa de aprendizaje efectiva escalaba con la cantidad de nodos y el entrenamiento divergía al agregar trabajadores.

### Strategy Switch

La implementación de Strategy-Switch [@provatas2025strategyswitch] separa la **decisión**, que vive en el orquestador, de la **ejecución**, que ocurre en los nodos. El criterio de conmutación mantiene una ventana deslizante de las últimas pérdidas y calcula su cambio relativo medio; cuando ese valor cae por debajo de un umbral, es decir, cuando la curva de pérdida **se aplana**, se dispara la transición. La ventana y el umbral son los valores indicados en el trabajo original.

Una decisión de diseño que conviene destacar es que **el plan se precomputa antes de entrenar**. La topología completa del Parameter Server, la ubicación de los servidores y el particionado de las capas se calculan en tiempo de configuración, y para cada nodo se construye por anticipado la acción que le corresponderá. Lo único dinámico es el **momento**. La razón es que calcular la topología requiere las mediciones de latencia, que ya se hicieron al conectar, y hacerlo en caliente introduciría una pausa en el entrenamiento.

La promoción de un trabajador a servidor es real y no una reasignación de etiquetas: el nodo rebana sus **parámetros ya entrenados** según los rangos recibidos y arranca como servidor sembrado con esos pesos, no con valores aleatorios. La ligadura entre cada promoción y su fragmento se hace **por dirección del servidor** y no por orden de aparición en el recorrido de la topología; hacerlo por orden fue un error real que provocaba que el fragmento cayera en el servidor equivocado.

Corresponde declarar tres limitaciones. La transición es **unidireccional**, de All-Reduce a Parameter Server, y **de un solo disparo** por sesión. Y, lo más importante para la lectura de los resultados experimentales, **en la campaña de benchmarks ejecutada el criterio de conmutación nunca llegó a dispararse**: el umbral de estabilidad, tomado con fidelidad del trabajo original, no se alcanza dentro del presupuesto de épocas evaluado. El sistema registra explícitamente esa condición en cada corrida en lugar de ocultarla. Se optó por mantener la fidelidad al paper antes que calibrar el umbral para el régimen propio, y por declarar la consecuencia.

## Plano de control

El orquestador es el plano de control del sistema. Recibe la configuración del entrenamiento, la valida, se conecta a los nodos, mide la topología de la red, deriva de allí toda la asignación de roles y especificaciones, y conduce la sesión.

### Selección de topología

Una de las líneas no previstas en el plan original y que resultó de las más interesantes es la **selección de topología a partir de mediciones**. Antes de comenzar, el orquestador realiza rondas de medición de latencia entre todos los pares de nodos y construye con ellas un grafo.

La decisión de diseño relevante es que el peso de cada arista es el **RTT máximo observado y no el promedio**, y que ambos problemas de optimización se plantean sobre el peor caso. El orden del anillo de All-Reduce se resuelve como un problema del viajante, y la ubicación de los servidores de parámetros como un problema minimax que elige los $k$ nodos que minimizan la latencia peor caso hacia los trabajadores. El criterio quedó resumido en la discusión que lo originó: *"es mejor tener 2 conexiones más o menos que una muy buena y una muy mala"*, que es exactamente la diferencia entre optimizar la suma y optimizar el máximo.

Ambos problemas se resuelven con **algoritmos exactos** y no con heurísticas: programación dinámica sobre subconjuntos para el ciclo, y enumeración con poda para la ubicación de servidores. La decisión es defendible dado el régimen al que apunta el sistema, de decenas de nodos, aunque debe señalarse que el algoritmo del ciclo degenera para tamaños grandes.

### Sesión orientada a eventos

La ejecución de un entrenamiento se modela como una **sesión orientada a eventos**. El orquestador expone un canal de eventos y mantiene una tarea por trabajador que traduce los mensajes del protocolo en eventos de dominio: pérdidas publicadas, trabajador terminado, entrenamiento completo, nodo promoviéndose. Este diseño reemplazó a un panel de control simulado y es lo que permite que las tres interfaces del sistema consuman exactamente la misma información sin duplicar lógica.

El punto de agregación es el manejo de pérdidas: se registra la última pérdida de cada trabajador, se exige que **todos los trabajadores vivos** hayan reportado, y recién entonces se alimentan los dos criterios que observan la curva, el de detención temprana y el de conmutación de estrategia. Que la condición sea sobre los trabajadores vivos y no sobre una cantidad fija es otra consecuencia del problema de membresía dinámica.

La detención temprana [@prechelt1998early] se implementó como un criterio de tolerancia: cuando tres pérdidas agregadas consecutivas se mueven por debajo de un umbral configurable, se considera que el entrenamiento convergió y se difunde la orden de detención. Es un criterio de tolerancia y no de paciencia, y la ventana está fijada en el código.

## Interfaces

El sistema expone tres formas de uso, todas sobre el mismo plano de control y **compartiendo el mismo esquema de configuración canónico**, sin duplicarlo. Esa última propiedad no fue gratuita: el esquema llegó a estar replicado en tres documentaciones distintas que divergieron entre sí, y se resolvió creando un documento de referencia único y eliminando las copias.

La **interfaz interactiva de terminal** permite configurar, lanzar y monitorear un entrenamiento en vivo. Presenta un asistente de configuración con sugerencias y validación, y un panel que muestra la evolución de la pérdida, el progreso por trabajador y una vista de topología. Esa vista de topología tiene un detalle que vale la pena mencionar porque es el único lugar donde el sistema se observa a sí mismo: las partículas que representan el flujo de datos entre nodos **se mueven al ritmo real del entrenamiento**, siguiendo una media móvil del intervalo entre reportes, y permiten ver en vivo cómo un trabajador se convierte en servidor durante una conmutación de estrategia. El entrenamiento corre en un hilo de fondo para que la interfaz nunca se bloquee.

La **biblioteca de Python** se construyó sobre PyO3. La decisión de diseño detrás de ella se tomó descartando una alternativa que era más limpia en teoría: anotar los tipos existentes del orquestador con los atributos de PyO3 detrás de una bandera de compilación, manteniendo una única fuente de verdad. Se descartó por **restricciones duras del compilador**, ya que los atributos de PyO3 no admiten tipos genéricos, las variantes complejas de enumerados requieren formas incompatibles y algunos tipos de error habrían necesitado capacidades adicionales. La solución fue construir un crate envoltorio separado, lo que mantiene el orquestador como Rust puro. El entrenamiento libera el bloqueo global del intérprete y se ejecuta en un hilo dedicado, de modo que el proceso de Python actúa efectivamente como orquestador.

El **binario de ejecución desatendida** completa el conjunto, y es el que utiliza la suite de benchmarks.

## Infraestructura y experimentación

El despliegue se realiza con una **única imagen de Docker**, coherente con el agnosticismo de rol: no hay imagen de trabajador ni imagen de servidor. La construcción es multietapa con caché de dependencias, y un conjunto de scripts genera la configuración del clúster, ajusta la resolución de nombres y levanta el entorno a partir de un único parámetro con la cantidad de nodos.

Corresponde declarar con precisión lo que esta infraestructura **no** hace, porque condiciona todos los resultados de rendimiento del informe: no fija afinidad de núcleos, no limita CPU por contenedor y no emula latencia ni ancho de banda de red. Todos los nodos comparten un host de ocho núcleos sobre loopback. Los tiempos medidos son, por lo tanto, una cota inferior del costo de comunicación.

La suite de experimentación está escrita en Python y organizada de forma declarativa en cuatro campañas que miden convergencia, velocidad de ejecución, velocidad de convergencia y escalabilidad. Persiste los resultados de forma incremental, agrega repeticiones en media y desvío, y **regenera su propia documentación** en cada corrida. Su aspecto más valioso, y el que se considera un aporte metodológico del trabajo, es que documenta explícitamente sus propios criterios de equidad y sus fuentes de ruido: justifica el presupuesto fijo de nodos, aclara qué significa una época en cada campaña, y explica por qué ciertas variantes se comparan por exactitud y no por velocidad, dado que una medición de tiempo aislada tiene un ruido cercano al 25 %.

## Estado de la implementación

El sistema compila sin advertencias y su conjunto de pruebas pasa con 87 pruebas en verde. El motor de aprendizaje propio converge a 0,983 de exactitud sobre MNIST con la red de referencia y a 0,980 con LeNet-5, frente a 0,989 y 0,990 de una implementación de PyTorch con la misma receta, lo que sitúa la implementación propia a alrededor de medio punto porcentual de un framework maduro.

En honor a la precisión, corresponde declarar las limitaciones verificadas al cierre. La desconexión de un servidor provoca la caída del trabajador, ya que ese camino no está implementado. La variante de almacenamiento sin bloqueos tiene un problema de corrección y está deshabilitada. La propagación hacia atrás de la ReLU con pendiente negativa es incorrecta, aunque el caso estándar resulta correcto y ningún modelo evaluado la utiliza. Las dos pruebas de integración existentes carecen de aserciones y funcionan como pruebas de humo. Y **el anillo de All-Reduce, que es el algoritmo más intrincado del sistema, no tiene cobertura de pruebas**: su corrección está verificada por sus resultados de convergencia y no por pruebas unitarias, lo cual es una deuda real y no una decisión.
