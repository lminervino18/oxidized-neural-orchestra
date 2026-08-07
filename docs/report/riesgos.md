\newpage
# Riesgos materializados y lecciones aprendidas

La propuesta identificó cuatro riesgos iniciales. Los cuatro se materializaron, uno con una forma distinta de la prevista, y aparecieron otros no contemplados. Esta sección los recorre con la evidencia del historial del proyecto y cierra con las lecciones que dejaron.

## Riesgos previstos que se materializaron

**La complejidad de la implementación distribuida en Rust.** Fue el riesgo dominante. Las garantías de *fearless concurrency* cumplieron lo que prometen, ya que ninguna condición de carrera sobre memoria compartida llegó a la rama principal, pero desplazaron la dificultad hacia la coordinación distribuida, que el compilador no puede verificar porque no es un problema de memoria sino de protocolo. La dificultad real fue sincronizar el estado de fase de cada entidad dentro del algoritmo que ejecuta, sobre todo en All-Reduce, donde cada nodo avanza por pasos que dependen de sus vecinos.

De ahí salieron los dos bloqueos más costosos. Uno de membresía: el contador de la barrera se fijaba en $N$ trabajadores al inicio y, cuando uno terminaba y se desconectaba, quedaba esperando a alguien que ya no estaba. Otro de control de flujo: el anillo de All-Reduce enviaba y recibía de forma secuencial, así que con modelos grandes todos intentaban enviar antes de recibir y el buffer del socket se llenaba sin que nadie lo drenara. Ninguno producía error: el sistema simplemente se detenía.

**La disponibilidad de hardware para simulaciones representativas.** Es la limitación transversal del trabajo, y los contenedores con límites de recursos previstos como mitigación no suplen lo que no está. La respuesta, y la decisión metodológica más valiosa del trabajo, fue instrumentar la comparación para que el lector pueda separar lo propio del algoritmo de lo que es artefacto del entorno; el mecanismo se describe en *Experimentación y validación*.

**La amplitud del alcance y la estimación de tiempos.** La mitigación prevista, priorizar un sistema base funcional y optimizar de forma incremental, se aplicó y funcionó. El costo fue que algunas líneas postergadas no llegaron a cerrarse, según se detalla más abajo.

**La dependencia de trabajos previos.** Se materializó de una forma no anticipada. Lo previsto era que la interpretación de un algoritmo publicado difiriera de lo documentado; lo que ocurrió fue más sutil: los supuestos de los papers no siempre se cumplen en el régimen propio.

## Riesgos no previstos

**Aplicar un algoritmo sin verificar la validez de sus supuestos.** El sistema implementa la actualización de parámetros libre de bloqueos siguiendo a Niu et al. [@recht2011hogwild], y una auditoría tardía mostró que la variante no era utilizable, por dos razones independientes. La primera es de corrección. Rust no permite obtener dos referencias mutables a la misma memoria, de modo que implementar la técnica del paper exigió desactivar esa comprobación de forma deliberada: los parámetros se alojan en un contenedor de mutabilidad interna y el tipo se declara compartible entre hilos a mano. Hecho eso, varios trabajadores escriben a la vez sobre los mismos parámetros, lo cual es comportamiento indefinido, y el compilador sigue optimizando bajo el supuesto de que ese solapamiento no ocurre. No es que C++ ofreciera una garantía que Rust niega, porque una carrera de datos es igual de indefinida allí; la diferencia es que en Rust hay que pedirla de forma explícita. El síntoma fue un cuelgue silencioso que solo aparecía con varios trabajadores, y las pruebas existentes no lo detectaron porque ejercitaban únicamente la variante con bloqueo. La segunda es conceptual: el argumento de convergencia del paper supone gradientes ralos, y una red convolucional densa actualiza cada parámetro en cada paso, de modo que el supuesto no se cumple en el régimen propio. La variante quedó fuera de los benchmarks y las salidas posibles se detallan en *Trabajos futuros*.

## Desvíos de alcance

**La optimización de carga en configuraciones heterogéneas no se abordó.** Estaba enunciada en la propuesta y el entorno de contenedores admitía representar nodos de capacidades distintas, de modo que no hubo un impedimento técnico: la tarea se fue postergando frente a otras prioridades y no llegó a abordarse.

**Las pruebas de integración de alto nivel fueron reemplazadas.** Las que levantaban un clúster completo desde Python se abandonaron en favor de la suite de benchmarks: sin un estándar acordado para las configuraciones que requerían, su mantenimiento superaba la confianza que aportaban.

## Lecciones aprendidas

**Conviene tener una medición de punta a punta desde temprano.** Es la lección central del trabajo. La suite de benchmarks llegó al final y con otro propósito, el de comparar estrategias, y aun así terminó destapando tres errores de corrección que llevaban meses en el código: All-Reduce sumaba los gradientes en lugar de promediarlos, con lo que la tasa de aprendizaje efectiva escalaba con la cantidad de trabajadores; los servidores de parámetros almacenaban las capas ordenadas por tamaño y no por su orden en el modelo; y al conmutar de estrategia los fragmentos se asignaban en un orden distinto del que usaba el trabajador para indexarlos. Los tres eran silenciosos: no provocaban fallos, solo degradaban la exactitud, y ninguna prueba unitaria los habría encontrado porque cada componente hacía correctamente lo suyo.

El caso del orden de capas muestra por qué sobrevivió tanto tiempo. El servidor las guardaba ordenadas de mayor a menor tamaño, cuando debía guardarlas en el orden en que aparecen en el modelo. Como todas las redes de prueba iban de más grande a más chica, con 784, 128, 64 y 10 neuronas, los dos órdenes daban lo mismo y el error quedaba tapado; solo se manifiesta con una red cuyas capas no decrecen, que ninguna prueba usaba. Un conjunto de pruebas que no varía la forma de sus datos no distingue una implementación correcta de una que acierta por casualidad.
