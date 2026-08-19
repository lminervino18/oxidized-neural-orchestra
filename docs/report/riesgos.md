\newpage
## Riesgos materializados y lecciones aprendidas

La propuesta identificó tres riesgos iniciales. Los tres se materializaron, uno con una forma distinta de la prevista, y aparecieron otros no contemplados. Esta sección los recorre con la evidencia del historial del proyecto y cierra con las lecciones que dejaron.

### Riesgos previstos que se materializaron

**La complejidad de la implementación distribuida en Rust.** Fue el riesgo dominante. Las garantías de *fearless concurrency* cumplieron lo que prometen, ya que ninguna condición de carrera sobre memoria compartida llegó a la rama principal, pero desplazaron la dificultad hacia la coordinación distribuida, que el compilador no puede verificar porque no es un problema de memoria sino de protocolo. La dificultad real fue sincronizar el estado de fase de cada entidad dentro del algoritmo que ejecuta, sobre todo en All-Reduce, donde cada nodo avanza por pasos que dependen de sus vecinos.

**La amplitud del alcance y la estimación de tiempos.** La mitigación prevista, priorizar un sistema base funcional y optimizar de forma incremental, se aplicó y funcionó. El costo fue que algunas líneas postergadas no llegaron a cerrarse, según se detalla más abajo.

**La dependencia de trabajos previos.** Se materializó de una forma no anticipada. Lo previsto era que la interpretación de un algoritmo publicado difiriera de lo documentado; lo que ocurrió fue más sutil: los supuestos de los papers no siempre se cumplen en el régimen propio.

### Riesgos no previstos

**Aplicar un algoritmo sin verificar la validez de sus supuestos.** El sistema implementa la actualización de parámetros libre de bloqueos siguiendo a Niu et al. [@recht2011hogwild], y una auditoría tardía mostró que la variante no era utilizable. Rust no permite obtener dos múltiples referencias mutables a la misma memoria por defecto, de modo que implementar la técnica del paper exigió utilizar estrategias para sortear esa limitación. En particular, se usó el patrón de *interior mutability* para alojar los parámetros y compartir su mutabilidad. El síntoma fue un cuelgue silencioso que solo aparecía con varios trabajadores, y las pruebas existentes no lo detectaron porque ejercitaban únicamente la variante con bloqueo.

### Desvíos de alcance

**La optimización de carga en configuraciones heterogéneas no se abordó.** Estaba enunciada en la propuesta y el entorno de contenedores admitía representar nodos de capacidades distintas, de modo que no hubo un impedimento técnico: la tarea se fue postergando frente a otras prioridades y no llegó a abordarse.

**Las pruebas de integración de alto nivel fueron reemplazadas.** Las que levantaban un clúster completo desde Python se abandonaron en favor de la suite de benchmarks: sin un estándar acordado para las configuraciones que requerían, su mantenimiento superaba la confianza que aportaban.

### Lecciones aprendidas

**Conviene tener una medición *end-to-end* desde temprano.** La suite de benchmarks llegó al final y con otro propósito, el de comparar estrategias, y aun así terminó destapando tres errores de corrección que llevaban meses en el código: All-Reduce sumaba los gradientes en lugar de promediarlos, con lo que la tasa de aprendizaje efectiva escalaba con la cantidad de workers; al recuperar de los servidores los pesos entrenados, el orquestador no revertía el reparto de capas que él mismo había hecho, de modo que el modelo final quedaba con sus capas desordenadas; y esto se observó recién en *Strategy-Switch* cuando al conmutar de estrategia, los fragmentos se asignaban en un orden distinto del que usaba el trabajador para indexarlos. Los tres eran silenciosos: no provocaban fallos, solo degradaban la exactitud, y ninguna prueba unitaria los habría encontrado porque cada componente hacía correctamente lo suyo.

**Lo perfecto es enemigo de lo bueno.** Varios tiempos de trabajo sobre la implementación de funcionalidades del sistema difirieron mucho con respecto a lo esperado. La razón fue la "búsqueda de la perfección", una búsqueda cuyo encuentro se vuelve inalcanzable y que es un problema común en proyectos de este estilo. En particular, el desarrollo del motor de redes neuronales ocupó mucho más tiempo del planificado, debido principalmente a intentos de optimizar el tiempo de cómputo. Conviene siempre trabajar en ítems que conformen cortes transversales de funcionalidad del producto, como sugirió Kent Beck: *"make it work, make it right, make it fast"*.
