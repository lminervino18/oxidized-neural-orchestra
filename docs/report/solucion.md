## Solución implementada

Se construyó Oxidized Neural Orchestra (O.N.O.), un sistema distribuido de entrenamiento de modelos de aprendizaje profundo. Comprende una biblioteca de redes neuronales, una capa de comunicación con protocolo propio, un plano de control que coordina la ejecución, tres algoritmos distribuidos de entrenamiento y dos interfaces de uso.

Esta sección describe la arquitectura resultante. El énfasis está puesto en las decisiones de diseño y en las razones que las motivaron, por encima de los detalles de implementación: en un trabajo cuyo objetivo era construir una base de comparación, son esas decisiones las que determinan si la base sirve para lo que fue construida.

### Componentes generales de la arquitectura

1. **Capa de Comunicación (`comms`):** Abstrae la infraestructura de red mediante un protocolo propio optimizado. Separa el plano de control (comandos en formato JSON) del plano de datos (tensores y gradientes mediante reinterpretación de memoria sin copias).
2. **Motor de Redes Neuronales (`machine_learning`):** Provee la biblioteca de aprendizaje profundo (capas, funciones de activación, pérdidas, entrenadores y modelos), con inicializaciones de Xavier-Glorot [@glorot2010understanding] y Kaiming-He [@he2015delving] y optimizadores que incluyen el descenso por gradiente con momento [@sutskever2013momentum] y Adam [@kingma2014adam]. Su diseño organiza el modelo como un buffer de memoria plano sobre el cual operan las capas, facilitando la partición y transmisión de parámetros sin serializaciones complejas.
3. **Plano de Control y Orquestación (`orchestrator`):** Administra el ciclo de vida del entrenamiento. Realiza mediciones de latencia en la red para seleccionar topologías óptimas, coordina la ejecución orientada a eventos y gestiona criterios de detención temprana [@prechelt1998early] de los entrenamientos realizados.
4. **Nodos de Cómputo (`node`):** Arquitectura agnóstica de rol donde todos los nodos ejecutan el mismo binario base (`node`). Según la especificación asignada dinámicamente por el orquestador, un nodo opera como *Worker* o como *Parameter Server*.
5. **Estrategias de Sincronización y Distribución:** Soporta tres esquemas de distribución integrados:
  * **Parameter Server:** Sincronización multi-servidor con particionado de capas.
  * **Ring All-Reduce:** Algoritmo descentralizado con comunicación lógica de anillo entre los nodos.
  * **Strategy-Switch:** Conmutación dinámica de All-Reduce a Parameter Server asincrónico cuando la pérdida se estabiliza. Sobre una ventana de las últimas seis pérdidas registradas se promedia la variación relativa entre pérdidas consecutivas, y el cambio de estrategia se dispara apenas ese promedio cae por debajo de un umbral fijo (0,01), siguiendo el criterio de Provatas et al. [@provatas2025strategyswitch].
6. **Interfaces de Usuario:** Exposición del sistema mediante una interfaz gráfica de terminal interactiva (`orchestui`) y una librería de bindings para Python vía FFI (`orchestra-py`).

### Decisiones de diseño

**Nodo agnóstico de rol.** Todos los nodos ejecutan el mismo binario y el rol de cada uno es una especificación que el orquestador asigna en tiempo de ejecución. Cada conexión se identifica a sí misma en el handshake inicial, un intercambio `Connect`/`Accept` con su identificador y su rol declarado, de modo que el otro extremo construye el handle tipado correspondiente sin necesidad de conocer de antemano quién se está conectando. La razón de fondo es Strategy-Switch: conmutar de All-Reduce a Parameter Server en medio de un entrenamiento exige promover trabajadores a servidores sin redesplegar el clúster, y eso solo es posible si el rol no queda fijado en el despliegue. La misma propiedad deja abierta, como extensión, la participación de un mismo proceso en más de un entrenamiento.

**Modelo como buffer plano.** El modelo no es una estructura de objetos con sus pesos adentro: es un único buffer contiguo de memoria sobre el que las capas operan mediante vistas, y el objeto modelo conserva solo la metadata que describe cómo interpretarlo. La consecuencia buscada es que particionar el modelo entre servidores o transmitirlo por la red no exige copiar ni serializar los pesos; basta delimitar rangos del buffer.

**Separación del plano de control y el plano de datos.** Los comandos de coordinación viajan en JSON, legible y depurable; los tensores y gradientes viajan por reinterpretación de memoria sin copias. La división responde a que ambos planos tienen necesidades opuestas: el de control es poco frecuente y conviene poder inspeccionarlo; el de datos concentra el volumen y cada copia intermedia cuesta.

**Particionado respetando límites de capa.** Cuando Parameter Server fragmenta el modelo entre varios servidores, el reparto se realiza por capas enteras. Así, cada gradiente producido por una capa viaja por completo a un único servidor, sin partir tensores al enviar ni recomponerlos al recibir.

**Selección de topología por latencia medida.** El orquestador mide la latencia entre los nodos antes de entrenar y con esas mediciones decide el orden del anillo y la ubicación de los servidores. La topología resulta de la red real y no de la configuración, lo que evita que una asignación arbitraria introduzca diferencias de tiempo ajenas al algoritmo.

**Ubicación del optimizador: gradiente plano en los workers, estado en el servidor.** Los workers siempre computan localmente un descenso por gradiente sin estado, y son los servidores de parámetros los que aplican el optimizador con estado (momento, Adam) sobre el gradiente agregado. La razón es doble: evita transmitir el estado del optimizador por la red en cada ronda, y en All-Reduce directamente no sería aplicable, porque los buffers internos de un optimizador con estado quedarían con valores incoherentes al momento de reducir los gradientes de distintos workers.

**Actualización de parámetros sin bloqueos, detrás de un trait polimórfico.** El servidor de parámetros admite dos regímenes de actualización intercambiables en tiempo de configuración: uno con lock por fragmento (`BlockingStore`) y uno que directamente asume las *race conditions* al escribir sobre los parámetros compartidos, siguiendo a Niu et al. [@recht2011hogwild] (`WildStore`). Ambos implementan el mismo trait `Store`, de modo que la elección entre corrección estricta y velocidad queda expuesta como un parámetro (`store: blocking|wild`) y no como una bifurcación de código.

**Interfaz de Python como crate separado, no como feature-flag.** La primera aproximación anotaba los tipos existentes con atributos condicionales (`#[cfg_attr(feature="python-ffi", pyclass)]`) para mantener una única fuente de verdad. Se descartó por límites duros del compilador: `#[pyclass]` no admite tipos genéricos, algunas variantes de enum no tienen una forma compatible, y ciertos tipos de error hubieran requerido implementar `Clone`. `orchestra-py` se construyó entonces como un crate aparte que envuelve al `orchestrator`, manteniendo este último en Rust puro y liberando el GIL con `py.allow_threads` durante las llamadas bloqueantes.

Estas decisiones sostienen, en conjunto, la propiedad que da sentido al sistema como base de comparación: que entre dos corridas la única variable sea la estrategia de distribución.
