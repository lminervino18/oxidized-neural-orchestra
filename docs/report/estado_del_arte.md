# Estado del arte

Este capítulo aborda los enfoques de entrenamiento distribuido relevantes para este trabajo: cómo se clasifican según qué particionan entre nodos, las dos estrategias de sincronización que sirven de base a todo lo implementado, sus combinaciones, y las técnicas de reducción de tráfico y de sincronización diferida que también se incorporaron. Este relevamiento sienta las bases del problema que se plantea en el capítulo siguiente.

## Paralelismo de datos, de modelo y de pipeline

El entrenamiento distribuido surge como una extensión de las técnicas de paralelización en *HPC* (High Performance Computing). Ben-Nun y Hoefler [@bennun2019demystifying] y, de forma independiente, Dehghani y Yazdanparast [@dehghani2023survey] coinciden en clasificar la concurrencia en el entrenamiento distribuido según qué se particiona entre los nodos: paralelismo de datos, que replica los parámetros en cada nodo y particiona el conjunto de datos; paralelismo de modelo, que particiona la red entre nodos y replica el lote de entrenamiento; pipelining, que asigna capas contiguas a distintos nodos y solapa sus cómputos; e híbrido, que combina los esquemas anteriores. O.N.O. se ubica exclusivamente en el primero: cada nodo mantiene una réplica completa del modelo y procesa una partición distinta del conjunto de datos, y es sobre ese esquema que se comparan las estrategias de sincronización de este trabajo.

Los primeros enfoques de paralelismo de datos en frameworks como TensorFlow [@2021zndo4758419D] y PyTorch [@paszke2019pytorchimperativestylehighperformance], y sus implementaciones más recientes optimizadas para arquitecturas heterogéneas, reflejan la tensión constante entre escalabilidad y coherencia de los parámetros del modelo. Dehghani y Yazdanparast señalan, entre las limitaciones abiertas del área, la ausencia de benchmarks que permitan comparar el desempeño de las distintas propuestas de forma uniforme, más allá de que cada estudio suele evaluarse sobre conjuntos de datos y métricas propios [@dehghani2023survey]. Esa misma ausencia es el punto de partida del problema que este trabajo aborda.

## Parameter Server y All-Reduce

El equilibrio entre convergencia y eficiencia de comunicación ha motivado una amplia línea de investigación. En la actualidad, existen dos enfoques fundamentales que sirven como base para el desarrollo de nuevas técnicas: Parameter Server [@10.5555/2685048.2685095] y All-Reduce [@10.1145/3146347.3146350].

En el enfoque de Parameter Server, el modelo se divide en un conjunto de parámetros centralizados (no necesariamente en un único nodo), a los cuales los nodos trabajadores envían sus gradientes y desde los cuales reciben las actualizaciones, que surgen de la agregación de estas contribuciones. Este esquema admite tanto una coordinación sincrónica, en la que el servidor espera los gradientes de todos los trabajadores antes de actualizar el modelo, como una asincrónica, en la que aplica cada gradiente apenas llega; esta última mejora la escalabilidad y reduce el tiempo de entrenamiento, aunque a costa de la coherencia entre copias del modelo. El caso extremo de esa relajación es *Hogwild!* [@recht2011hogwild], que directamente abraza las *race conditions* sobre los parámetros compartidos y demuestra que, bajo hipótesis de esparsidad, la convergencia se preserva.

En contraste, en All-Reduce todos los nodos intercambian gradientes de manera directa y avanzan de forma sincrónica: en cada paso todos aplican la misma actualización, lo que preserva la coherencia entre réplicas. La formulación en anillo de Patarasuk y Yuan [@patarasuk2009bandwidth] es óptima en ancho de banda, ya que el volumen que comunica cada nodo resulta esencialmente independiente de la cantidad de participantes, y es la que popularizó Horovod [@sergeev2018horovod] en el ámbito del aprendizaje profundo. Trabajos posteriores como BytePS [@jiang2020byteps] muestran que, en clústeres heterogéneos, la elección entre ambas familias deja de ser evidente.

## Strategy-Switch

Uno de los claros casos de combinación de estos algoritmos es Strategy-Switch [@provatas2025strategyswitch], que inicia el entrenamiento de forma sincrónica sobre All-Reduce y, guiado por una regla empírica, continúa con Parameter Server asincrónico una vez que el modelo en entrenamiento se estabiliza; logrando así combinar la precisión del régimen sincrónico de *All-Reduce* con la reducción de tiempo del régimen asincrónico de *Parameter Server*.

## Reducción del volumen de comunicación

Una línea transversal a los tres enfoques es la reducción del volumen de comunicación. Aji y Heafield [@aji2017sparse] proponen transmitir únicamente los componentes de mayor magnitud del gradiente y acumular localmente el residuo, y Lin et al. [@lin2018deep] extienden la idea con compensaciones que preservan la convergencia a tasas de compresión muy altas. Una alternativa complementaria es reducir la precisión numérica de los mensajes sin alterar la precisión del cómputo. Ambas técnicas fueron implementadas en este trabajo.

## Aprendizaje federado y sincronización diferida

Finalmente, en el terreno del aprendizaje federado, McMahan et al. [@mcmahan2017federated] formalizan la idea de dejar que cada nodo entrene de forma autónoma durante varias épocas antes de sincronizar, y caracterizan el fenómeno de divergencia entre réplicas que esto induce, conocido como *client drift*. Ese mecanismo, que en O.N.O. se expone bajo el nombre de *offline epochs*, es el objeto de uno de los estudios experimentales de este informe.
