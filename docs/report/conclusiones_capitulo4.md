\newpage
## Conclusiones del capítulo

El análisis teórico y empírico de las distintas estrategias de entrenamiento distribuido permite extraer tres conclusiones principales sobre la gestión de tiempo y convergencia:

**El cuello de botella es la sincronización centralizada.** En escenarios con ancho de banda restringido, centralizar la comunicación en Parameter Server sincrónico escala de forma deficiente y aumenta el tiempo total de entrenamiento al agregar nodos luego de pasar un umbral. La topología descentralizada de Ring All-Reduce soluciona el escalado, pero mantiene la imposición del límite estricto de la mejora en performance alcanzable.

**Existe un *trade-off* directo entre asincronismo y convergencia en Parameter Server asincrónico.** Eliminar los bloqueos de sincronización en Parameter Server asincrónico reduce radicalmente los tiempos de ejecución, pero a costa de destruir por completo la convergencia del modelo debido al fuerte *client drift* en las primeras etapas del entrenamiento.

**Superioridad de la estrategia híbrida.** Strategy-Switch resultó ser la estrategia de entrenamiento distribuido más eficiente. Al comportarse de forma sincrónica durante la fase crítica del entrenamiento y lograr que los mínimos locales en los que se posicionan los workers se consensuen hasta cierto punto, logra mantener la convergencia del entrenamiento. A su vez, al conmutar al esquema asincrónico luego de ese consenso, rompe el *trade-off* que sí paga Parameter Server asincrónico. La estrategia logra la exactitud de los métodos sincrónicos, equivalentes a la ejecución serial en cuanto a convergencia, en una fracción del tiempo, valiéndose empíricamente como la mejor alternativa evaluada en esta comparativa.

La observación transversal es que el sistema cumplió su propósito: validada la corrección de las estrategias por la vía de su convergencia, las diferencias de tiempo, throughput y escalabilidad admiten atribuirse al algoritmo y no a su implementación, que era exactamente el problema que este trabajo se propuso resolver.
