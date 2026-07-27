# Solución propuesta

Se construyó Oxidized Neural Orchestra (O.N.O.), un sistema distribuido de entrenamiento de modelos de aprendizaje profundo. Comprende una biblioteca de redes neuronales, una capa de comunicación con protocolo propio, un plano de control que coordina la ejecución, tres algoritmos de distribuidos de entrenamiento y dos interfaces de uso.

Esta sección describe la arquitectura resultante. El énfasis está puesto en las decisiones de diseño y en las razones que las motivaron, por encima de los detalles de implementación: en un trabajo cuyo objetivo era construir una base de comparación, son esas decisiones las que determinan si la base sirve para lo que fue construida.

## Componentes generales de la arquitectura

1. **Capa de Comunicación (`comms`):** Abstrae la infraestructura de red mediante un protocolo propio optimizado. Separa el plano de control (comandos en formato JSON) del plano de datos (tensores y gradientes mediante reinterpretación de memoria sin copias).
2. **Motor de Redes Neuronales (`machine_learning`):** Provee la biblioteca de aprendizaje profundo (capas, funciones de activación, optimizadores, pérdidas, entrenadores y modelos). Su diseño organiza el modelo como un buffer de memoria plano sobre el cual operan las capas, facilitando la partición y transmisión de parámetros sin serializaciones complejas.
3. **Plano de Control y Orquestación (`orchestrator`):** Administra el ciclo de vida del entrenamiento. Realiza mediciones de latencia en la red para seleccionar topologías óptimas, coordina la ejecución orientada a eventos y gestiona criterios de detención temprana de los entrenamientos realizados.
4. **Nodos de Cómputo (`node`):** Arquitectura agnóstica de rol donde todos los nodos ejecutan el mismo binario base (`node`). Según la especificación asignada dinámicamente por el orquestador, un nodo opera como *Worker* o como *Parameter Server*.
5. **Estrategias de Sincronización Distribución:** Soporta tres esquemas de distribución integrados:
  * **Parameter Server:** Sincronización multi-servidor con particionado de capas.
  * **Ring All-Reduce:** Algoritmo descentralizado con comunicación lógica de anillo entre los nodos.
  * **Strategy-Switch:** Conmutación dinámica de All-Reduce a Parameter Server según la curva de pérdida.
6. **Interfaces de Usuario:** Exposición del sistema mediante una interfaz gráfica de terminal interactiva (`orchestui`) y una librería de bindings para Python vía FFI (`orchestra-py`).
