\newpage
## Conclusiones del capítulo

<!--
TODO(Ale): redactar las conclusiones del capítulo pedidas por el tutor
(anotación de la pág. 32 del PDF corregido: "Faltan conclusiones de este capítulo!").
Deben cerrar el capítulo contrastando, estrategia por estrategia, el costo de
sincronización esperado con el medido sobre O.N.O., y decir de forma explícita
cuándo un resultado es bueno (la anotación de la pág. 30 sobre "Resultados
obtenidos" pide justamente eso). El párrafo que sigue, hoy titulado "Síntesis de
la validación", se movió acá desde la antigua sección de experimentación como
material de partida; conviene ampliarlo con el veredicto por estrategia
(Parameter Server se degrada a partir de N_W = 6, All-Reduce queda acotado, el
Parameter Server asincrónico rompe la convergencia, Strategy-Switch la recupera)
y con la conclusión del estudio derivado de los epochs offline.
-->

La observación transversal es que el sistema cumplió su propósito: validada la corrección de las estrategias por la vía de su convergencia, las diferencias de tiempo, throughput y escalabilidad admiten atribuirse al algoritmo y no a su implementación, que era exactamente el problema que este trabajo se propuso resolver.
