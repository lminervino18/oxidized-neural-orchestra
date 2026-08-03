\newpage
# Problema detectado y/o faltante

El entrenamiento distribuido se adopta por dos razones: reducir el tiempo de entrenamiento y superar la incapacidad de una única máquina para alojar el conjunto de datos o el modelo. Ninguna de las dos se resuelve mediante una paralelización directa, porque las iteraciones del entrenamiento no son independientes entre sí: la dirección de cada paso del descenso por gradiente depende del punto en el que se encuentra la optimización, es decir, de los pesos que dejó la iteración anterior. De ahí se sigue el compromiso central del área, ya descrito en el *Estado del arte*: sincronizar con poca frecuencia degrada la convergencia, y sincronizar con demasiada frecuencia vuelve dominante el tiempo de comunicación, hasta el punto en que la ejecución distribuida resulta más lenta que la secuencial.

Sobre ese problema general se recorta la dificultad concreta que motivó este trabajo: **comparar las estrategias de distribución entre sí de forma controlada es difícil**. La literatura las describe y reporta resultados sobre ellas, pero las implementaciones de referencia disponibles, como las de TensorFlow [@2021zndo4758419D] y PyTorch [@paszke2019pytorchimperativestylehighperformance], están optimizadas para escenarios de producción y arrastran un trabajo de ingeniería que no es uniforme entre estrategias. Detrás de sus capas de abstracción quedan ocultas exactamente las decisiones que se querría poder variar:

1. El régimen de sincronización entre nodos.
2. La ubicación del optimizador dentro del ciclo de entrenamiento.
3. El esquema de particionado de los parámetros.
4. La codificación de los mensajes que se intercambian.

Al comparar dos estrategias sobre dos implementaciones distintas, la diferencia medida mezcla el efecto del algoritmo con el de la calidad de su implementación, sin que exista forma de separar ambos efectos.

Se detectó entonces la ausencia de una base común, parametrizable y de código abierto, sobre la cual las distintas estrategias de distribución compartan la misma capa de comunicación, el mismo motor de redes neuronales y el mismo plano de control, de modo que la única variable entre dos ejecuciones sea la estrategia bajo estudio. Sin esa base, cualquier comparación entre algoritmos, y en particular la evaluación de mejoras propuestas sobre ellos, queda expuesta a un sesgo que no se puede cuantificar. Construir esa base, e instrumentarla para que produzca mediciones, es el problema concreto que este trabajo resuelve.
