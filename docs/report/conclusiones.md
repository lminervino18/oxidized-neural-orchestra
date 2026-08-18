\newpage
# Conclusiones

El trabajo se propuso estudiar las principales estrategias de entrenamiento distribuido de modelos de aprendizaje profundo y construir un sistema que permitiera compararlas de forma controlada. Ese objetivo se cumplió.

**Sobre el producto.** Se construyó O.N.O., un sistema distribuido de entrenamiento que incluye una biblioteca de redes neuronales propia, una capa de comunicación con su propio protocolo, un plano de control y dos interfaces de uso. Sobre esa base se implementaron los tres algoritmos de referencia del área, se agregaron técnicas de reducción del tráfico de red y se instrumentó todo con una suite de experimentación reproducible.

**Sobre los resultados.** Los estudios experimentales realizados sobre MNIST muestran que, los algoritmos sincrónicos convergen con buena exactitud a costa de un mayor tiempo de entrenamiento, que *Parameter Server asincrónico* es más rápido pero a costa de exactitud y que *Strategy-Switch* logra combinar las ventajas de ambos mundos exitosamente; y que postergar la sincronización entre nodos mediante epochs offline degrada la exactitud final de forma monótona sin aportar, en ese entorno, la ganancia de tiempo que la motiva. La elección de la estrategia en un despliegue real depende también de la heterogeneidad del hardware de los nodos y de la relación entre el tamaño del modelo y el ancho de banda de la red.

**Sobre la contribución.** El trabajo no innova en algoritmos: los tres implementados están publicados. La contribución es el instrumento, y responde exactamente al problema detectado en la propuesta: la ausencia de una base común sobre la cual comparar estrategias sin que la diferencia medida mezcle el efecto del algoritmo con el de su implementación. La evidencia de que esa base funciona es el conjunto de benchmarks obtenidos.

**Sobre la formación.** El trabajo obligó a atravesar de punta a punta tres dominios que la carrera aborda por separado. El de los sistemas distribuidos, el del aprendizaje profundo, que dejó una comprensión del entrenamiento que ningún framework habría permitido adquirir, y el de la ingeniería de software, donde la lección central fue que la instrumentación no es un anexo de medición sino una herramienta de detección, según se detalla en *Riesgos materializados y lecciones aprendidas*.

Se cierra el trabajo con un sistema funcionando, con evidencia experimental que lo respalda y con las líneas que quedaron abiertas documentadas al detalle suficiente para que otro las retome.
