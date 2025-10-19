\newpage
# Estado del arte
El principal desafío en la convergencia del entrenamiento distribuido de modelos de aprendizaje profundo radica en que cada paso de la optimización del error depende del anterior.  
En un esquema de paralelización del entrenamiento por datos, existe el riesgo de incurrir en *overfitting* sobre las particiones si los pesos no se sincronizan con la frecuencia suficiente.  

Por otro lado, si la sincronización se realiza con demasiada frecuencia, el tiempo de comunicación entre los nodos puede volverse dominante. Este costo no es en absoluto despreciable: si la comunicación representa una fracción significativa del tiempo total de entrenamiento, la distribución del cómputo pierde sentido, ya que la ejecución paralela termina siendo más lenta que el entrenamiento secuencial.  

Este equilibrio entre **convergencia** y **eficiencia de comunicación** ha motivado una amplia línea de investigación. En la actualidad, existen dos enfoques fundamentales que sirven como base para el desarrollo de nuevas técnicas: **Parameter Server** [@10.5555/2685048.2685095] y **All-Reduce** [@10.1145/3146347.3146350].  
Uno de los casos de los claros casos de combinación de los algoritmos mencionados es **Strategy-Switch** [@article], que inicia iterando sobre All-Reduce y, guiado por una regla empírica, sigue con Parameter Server asincrónico una vez que el modelo en entrenamiento se estabiliza; logrando así mantener la precisión del entrenamiento de *All-Reduce* y la reducción significativa del tiempo total de entrenamiento de *Parameter Server asincrónico*.  

<!-- strategy-switch abstract: "...This method initiates training under the All-Reduce system and, guided by an empirical rule, transitions to asynchronous Parameter Server training once the model stabilizes. Our experimental analysis demonstrates that we can achieve comparable accuracy to All-Reduce training but with significantly accelerated training." -->

En este trabajo se estudian en profundidad ambos algoritmos base y se investigan los métodos que de ellos derivan. Y se busca a partir de dicho análisis proponer mejoras mediante la combinación de estrategias o la optimización de sus componentes de comunicación y sincronización.
