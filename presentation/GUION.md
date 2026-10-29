# Guion de la defensa

Generado desde `index.html` con `python3 tools/export_notes.py`. No editar a mano: los textos se cambian en las notas de cada diapositiva.

Cada **→** marca que hay que avanzar una etapa (flecha derecha) antes de decir ese punto. «En pantalla» es lo único que ve el jurado; todo lo demás se dice.

## Reparto

| Quién | Diapositivas | Tiempo |
|---|---|---|
| Alejo Ordoñez | 01–07 | 10:30 |
| Marcos Bianchi Fernández | 08–13 | 11:15 |
| Lorenzo Minervino | 14–22 | 10:45 |
| [a completar] | 23 | 00:30 + demo 10:00 |
| Los tres | 24 | 02:00 |
| **Total** | | **45:00** |

## Línea de tiempo

| Desde | Hasta | # | Diapositiva | Quién | En pantalla |
|---|---|---|---|---|---|
| 00:00 | 00:45 | 01 | Portada | Alejo Ordoñez | O.N.O. · entrenamiento distribuido · Rust |
| 00:45 | 02:15 | 02 | Entrenar es caro | Alejo Ordoñez | «más datos» «más parámetros» «días de cómputo» «no entra en memoria» |
| 02:15 | 03:30 | 03 | Una máquina buena es cara | Alejo Ordoñez | «escalar hacia arriba» «$$$ · pocos la tienen» «tiene un techo» |
| 03:30 | 04:45 | 04 | Podemos usar muchas | Alejo Ordoñez | «hardware común» «escalar hacia los costados» «unidas por la red» |
| 04:45 | 07:00 | 05 | Paralelismo de modelo y de datos | Alejo Ordoñez | «modelo → se parte la red» «datos → se parte el dataset» «O.N.O. → datos» |
| 07:00 | 09:15 | 06 | Client drift | Alejo Ordoñez | «misma réplica, distintos datos» «client drift» «sincronizar = promediar» «¿cómo sincronizar?» |
| 09:15 | 10:30 | 07 | Cuatro formas de sincronizar | Alejo Ordoñez | Parameter Server · All-Reduce · PS asincrónico · Strategy-Switch · misma base |
| 10:30 | 12:45 | 08 | Parameter Server | Marcos Bianchi Fernández | «servidor central» «pull θ» «push gradiente» «promedio + optimizador» «varios servidores: por capa» |
| 12:45 | 15:15 | 09 | Ring All-Reduce | Marcos Bianchi Fernández | «sin centro» «N fragmentos» «reduce-scatter: sumar» «all-gather: copiar» «÷ N en cada worker» |
| 15:15 | 16:45 | 10 | Costo de sincronizar | Marcos Bianchi Fernández | «PS: crece con N» «cuello de botella: el servidor» «AR: nunca supera 2S/R» |
| 16:45 | 18:30 | 11 | Parameter Server asincrónico | Marcos Bianchi Fernández | «sincrónico: espera al más lento» «asincrónico: sin barrera» «θ viejo → más drift» |
| 18:30 | 20:00 | 12 | Strategy-Switch: la idea | Marcos Bianchi Fernández | «arranca con All-Reduce» «¿se estancó la pérdida?» «ventana de 6 · umbral 1 %» «pasa a PS asincrónico» |
| 20:00 | 21:45 | 13 | Strategy-Switch en O.N.O. | Marcos Bianchi Fernández | «el orchestrator mide» «Upgrade · Switch» «un worker pasa a servidor» «mismo proceso, otro rol» |
| 21:45 | 22:45 | 14 | ¿Por qué Rust? | Lorenzo Minervino | sin recolector de basura · seguridad de memoria · sin data races · async con tokio · el protocolo no lo verifica |
| 22:45 | 24:45 | 15 | La orquesta por dentro | Lorenzo Minervino | «mismo binario en cada nodo» «orchestrator: control» «el rol llega al conectarse» «datos: de nodo a nodo» |
| 24:45 | 25:45 | 16 | La TUI, en vivo | Lorenzo Minervino | «anillo de All-Reduce» «Parameter Server» «el switch, en vivo» |
| 25:45 | 27:00 | 17 | Cómo medimos | Lorenzo Minervino | «MNIST · LeNet-5» «1 contenedor = 1 nodo» «de 2 a 16 workers» «red lenta, a propósito» |
| 27:00 | 28:15 | 18 | Resultados: sincrónicos | Lorenzo Minervino | «PS empeora desde 6 workers» «AR deja de mejorar» «como predijo el modelo» |
| 28:15 | 29:00 | 19 | Resultados: asincrónicos | Lorenzo Minervino | «el tiempo sigue bajando» «¿pero aprende?» |
| 29:00 | 30:30 | 20 | Resultados: exactitud | Lorenzo Minervino | «Strategy-Switch ≈ sincrónicos» «PS asincrónico colapsa» «salvedad: WildStore» |
| 30:30 | 31:30 | 21 | El drift, medido | Lorenzo Minervino | «E épocas sin sincronizar» «E = 1 → cerca de −1 punto» «más nodos, más caída» «sin ganancia de tiempo» |
| 31:30 | 32:30 | 22 | Trabajos futuros | Lorenzo Minervino | tolerancia a fallos · Hogwild! · convolución en paralelo · datasets por flujo · más arquitecturas · compresión |
| 32:30 | 33:00 | 23 | Ahora, la demo | [a completar] | «[a completar]» «[a completar]» «[a completar]» «10 min» |
| 33:00 | 43:00 | — | Demo en vivo | [a completar] | ver `GUION_DEMO.md` |
| 43:00 | 45:00 | 24 | Gracias por venir | Los tres | «4 estrategias, 1 base» «Strategy-Switch: mejor balance» «¿Preguntas?» |

## Qué dice cada uno

### 00:00 → 00:45 · 01 · Portada

**Alejo Ordoñez** · 00:45 · en pantalla: O.N.O. · entrenamiento distribuido · Rust

> Presentarnos y decir en una frase qué hicimos.

- Saludo al jurado. Somos Alejo, Lorenzo y Marcos; nuestro tutor es Ricardo Veiga y el co-tutor, José Ignacio Alvarez-Hamelin.
- En una frase: construimos O.N.O., un sistema para entrenar redes neuronales entre varias máquinas, escrito en Rust, con el que implementamos y comparamos cuatro formas de sincronizar ese entrenamiento.
- El recorrido: por qué distribuir, cómo se sincroniza, cómo lo construimos, qué medimos y una demo.

**Pase:** «Arranquemos por el principio: ¿por qué hace falta distribuir?»

Respaldo: docs/report/caratula.md · resumen.md

### 00:45 → 02:15 · 02 · Entrenar es caro

**Alejo Ordoñez** · 01:30 · en pantalla: «más datos» «más parámetros» «días de cómputo» «no entra en memoria»

> Entrenar un modelo profundo cuesta en dos dimensiones: tiempo y memoria.

- Un modelo chico con pocos datos entrena rápido: la barra se llena enseguida.
- **→** Pero los modelos útiles crecen: más datos y más parámetros. Cada época recorre todo el dataset y actualiza todos los parámetros.
- **→** Sobre conjuntos grandes, un entrenamiento completo puede tardar días en una sola máquina.
- **→** La otra dimensión es la memoria: el dataset, el modelo y sus estados intermedios pueden no entrar en un solo dispositivo.

**Pase:** «¿La solución obvia? Comprar una máquina más grande.»

Respaldo: docs/report/problema.md (párrafo 1) · Ben-Nun y Hoefler (2019)

### 02:15 → 03:30 · 03 · Una máquina buena es cara

**Alejo Ordoñez** · 01:15 · en pantalla: «escalar hacia arriba» «$$$ · pocos la tienen» «tiene un techo»

> Escalar con un solo equipo es caro y además tiene un límite.

- La primera reacción es comprar una máquina más potente. Eso es escalar hacia arriba, o verticalmente.
- **→** Un equipo de gran porte es caro, y en la práctica quien necesita entrenar rara vez tiene uno.
- **→** Y aun pagándolo, un solo equipo tiene un techo: su memoria y su cómputo son finitos.

**Pase:** «Lo que sí suele haber son varias máquinas comunes.»

Respaldo: docs/report/problema.md (párrafo 1)

### 03:30 → 04:45 · 04 · Podemos usar muchas

**Alejo Ordoñez** · 01:15 · en pantalla: «hardware común» «escalar hacia los costados» «unidas por la red»

> Sumar máquinas modestas es más accesible que comprar una que las iguale: eso es el entrenamiento distribuido.

- Una máquina común sola no alcanza.
- **→** Pero es frecuente tener varias. Sumarlas es escalar hacia los costados, u horizontalmente, con hardware que ya existe.
- **→** Conectadas por red suman cómputo y memoria. El precio es que ahora hay que coordinarlas, y la red pasa a ser parte del problema.

**Pase:** «¿Y cómo se reparte el trabajo entre ellas? Hay dos grandes formas.»

Respaldo: docs/report/problema.md · introduccion.md

### 04:45 → 07:00 · 05 · Paralelismo de modelo y de datos

**Alejo Ordoñez** · 02:15 · en pantalla: «modelo → se parte la red» «datos → se parte el dataset» «O.N.O. → datos»

> Se puede partir el modelo o partir los datos; O.N.O. trabaja con paralelismo de datos.

- Tenemos una red y un dataset. Hay dos formas clásicas de repartirlos (Ben-Nun y Hoefler, 2019).
- **→** Paralelismo de modelo: cada máquina tiene un pedazo de la red, por ejemplo una capa. La activación pasa de una máquina a la siguiente, así que cada una espera a la anterior. Sirve cuando el modelo no entra en una sola máquina.
- **→** Paralelismo de datos: cada máquina tiene el modelo completo y una porción distinta del dataset, y todas entrenan a la vez.
- **→** O.N.O. trabaja con paralelismo de datos, que también es el esquema principal de TensorFlow y PyTorch. El paralelismo de modelo y el de pipeline quedaron fuera de alcance.

**Pase:** «Pero el paralelismo de datos tiene una trampa.»

Respaldo: docs/report/estado_del_arte.md (Paralelismo de datos, de modelo y de pipeline) · introduccion.md

### 07:00 → 09:15 · 06 · Client drift

**Alejo Ordoñez** · 02:15 · en pantalla: «misma réplica, distintos datos» «client drift» «sincronizar = promediar» «¿cómo sincronizar?»

> Con paralelismo de datos cada réplica tira hacia el mínimo de su partición; por eso hay que sincronizarlas.

- Los tres workers arrancan con la misma réplica del modelo. Las curvas de nivel son la pérdida sobre todo el dataset y la estrella es su mínimo. El dibujo es ilustrativo.
- **→** Cada worker entrena con su partición, así que su gradiente apunta al mínimo de sus datos y no al global. En O.N.O. cada worker recorre al menos una época completa sobre su partición antes de sincronizar, y en ese tiempo las réplicas se separan. Eso se llama *client drift* (McMahan et al., 2017).
- **→** La solución es sincronizar: se combinan los aportes de todos (en O.N.O., promediando gradientes) y todos siguen desde el mismo punto.
- **→** Alternando entrenar y sincronizar, el conjunto converge al mínimo global. Cuanto más se demora la sincronización, más drift hay; lo medimos al final.

**Pase:** «La pregunta, entonces, es cómo sincronizar. Hay varias respuestas.»

Respaldo: docs/report/resultados.md (PS asincrónico, client drift) · estado_del_arte.md (aprendizaje federado) · código: machine_learning/src/training/backprop_trainer.rs

### 09:15 → 10:30 · 07 · Cuatro formas de sincronizar

**Alejo Ordoñez** · 01:15 · en pantalla: Parameter Server · All-Reduce · PS asincrónico · Strategy-Switch · misma base

> Vamos a ver cuatro formas de sincronizar, y O.N.O. las implementa todas sobre la misma base.

- Parameter Server: un servidor central junta los gradientes y reparte los parámetros.
- **→** All-Reduce: no hay centro; los workers se pasan los gradientes en anillo.
- **→** Parameter Server asincrónico: como el primero, pero nadie espera a nadie.
- **→** Strategy-Switch: arranca con All-Reduce y, en un momento, cambia a Parameter Server asincrónico.
- **→** Nuestro trabajo fue implementar las cuatro sobre la misma capa de comunicación, el mismo motor de redes neuronales y el mismo plano de control. Así, al compararlas, la diferencia viene del algoritmo y no de la implementación. Con TensorFlow o PyTorch eso no se puede separar (apéndice A10).

**Pase:** Traspaso: «Marcos les va a contar cómo funciona cada una.»

Respaldo: docs/report/problema.md · introduccion.md · estado_del_arte.md

### 10:30 → 12:45 · 08 · Parameter Server

**Marcos Bianchi Fernández** · 02:15 · en pantalla: «servidor central» «pull θ» «push gradiente» «promedio + optimizador» «varios servidores: por capa»

> En Parameter Server un servidor concentra los parámetros y el estado del optimizador.

- A la izquierda, los workers, cada uno con su partición. A la derecha, el servidor, que guarda los parámetros y el optimizador con estado.
- **→** Cada worker pide los parámetros actuales (pull).
- **→** Entrena sobre su partición y acumula su gradiente.
- **→** Envía el gradiente al servidor (push). Con el sincronizador `barrier`, el servidor espera a los N gradientes.
- **→** El servidor promedia y aplica el optimizador, por ejemplo Adam, una sola vez: hay un único estado del optimizador.
- **→** Con varios servidores, el modelo se parte por capas completas y cada capa vive en un servidor. El orchestrator asigna las capas, de mayor a menor, al servidor menos cargado.

**Pase:** «Si todo pasa por el servidor, ¿qué pasa si no hay servidor? Eso es All-Reduce.»

Respaldo: docs/report/solucion.md (Parameter Server) · código: worker/src/workers/parameter_server.rs, parameter_server/src/synchronization

### 12:45 → 15:15 · 09 · Ring All-Reduce

**Marcos Bianchi Fernández** · 02:30 · en pantalla: «sin centro» «N fragmentos» «reduce-scatter: sumar» «all-gather: copiar» «÷ N en cada worker»

> En All-Reduce no hay servidor: los gradientes circulan por un anillo y todos terminan con el mismo promedio.

- Cuatro workers en un anillo lógico: cada uno recibe del anterior y envía al siguiente. Es la variante de Patarasuk y Yuan (2009).
- **→** Cada worker parte su gradiente en N fragmentos, acá cuatro. Los colores indican de qué worker viene cada aporte.
- **→** Reduce-scatter: N−1 rondas. En cada una, cada worker envía un fragmento y suma el que le llega. Al final, cada uno tiene un fragmento con la suma de todos.
- **→** All-gather: otras N−1 rondas, ahora copiando, para que los fragmentos completos den la vuelta.
- **→** Cada worker divide por N y aplica el optimizador localmente. Todos llegan al mismo resultado sin que nadie centralice.

**Pase:** «¿Cuál conviene? Depende de cuánto cuesta sincronizar.»

Respaldo: docs/report/solucion.md (Ring All-Reduce) · código: worker/src/middlewares/worker_ring.rs

### 15:15 → 16:45 · 10 · Costo de sincronizar

**Marcos Bianchi Fernández** · 01:30 · en pantalla: «PS: crece con N» «cuello de botella: el servidor» «AR: nunca supera 2S/R»

> El modelo analítico predice que, con un servidor, PS empeora al sumar workers, mientras que All-Reduce queda acotado.

- En el eje X, la cantidad de workers; en el eje Y, el tiempo de sincronización por ronda. S es el tamaño del modelo y R, el ancho de banda.
- **→** PS con un servidor: todo pasa por la placa de red del servidor, así que d_sync = 2·N·S/R. Crece linealmente, y el servidor es el cuello de botella.
- **→** Ring All-Reduce: cada worker envía 1/N del gradiente por paso, así que d_sync = 2(1 − 1/N)·S/R, que nunca supera 2S/R.
- Las curvas usan S/R ≈ 0,1 s, el valor de la campaña (LeNet-5 en f16 a 10 Mbps). La derivación completa está en el apéndice A3.

**Pase:** «¿Y si directamente no esperamos a nadie?»

Respaldo: docs/report/resultados.md (derivación de d_sync) · experimentacion.md (S, R)

### 16:45 → 18:30 · 11 · Parameter Server asincrónico

**Marcos Bianchi Fernández** · 01:45 · en pantalla: «sincrónico: espera al más lento» «asincrónico: sin barrera» «θ viejo → más drift»

> Sacar la barrera elimina la espera, pero los gradientes se calculan sobre parámetros desactualizados y el drift crece.

- Arriba, el régimen sincrónico: en cada ronda hay una barrera, el worker más lento marca el ritmo y los demás esperan (zonas rayadas).
- **→** Abajo, el asincrónico: el servidor aplica cada gradiente apenas llega, y la espera desaparece. El costo es que un worker puede enviar un gradiente calculado sobre parámetros viejos. Con particiones distintas, las réplicas tiran cada una para su lado y el client drift empeora (McMahan et al., 2017).
- Si preguntan: si llega un gradiente mientras el servidor aplica otro, se acumula en un segundo buffer y entra en la actualización siguiente; no se pierde.

**Pase:** «¿Se puede tener lo mejor de los dos? Esa es la idea de Strategy-Switch.»

Respaldo: docs/report/resultados.md (Parameter Server asincrónico) · estado_del_arte.md

### 18:30 → 20:00 · 12 · Strategy-Switch: la idea

**Marcos Bianchi Fernández** · 01:30 · en pantalla: «arranca con All-Reduce» «¿se estancó la pérdida?» «ventana de 6 · umbral 1 %» «pasa a PS asincrónico»

> Strategy-Switch usa All-Reduce mientras los workers buscan el mínimo y pasa a PS asincrónico cuando la pérdida se estabiliza.

- Lo propusieron Provatas et al. (2025). Al principio el drift es más peligroso y conviene sincronizar fuerte, por eso arranca con All-Reduce. La curva es ilustrativa.
- **→** En cada ronda se mira la variación relativa de la pérdida en una ventana de 6 rondas. Cuando el promedio baja de 1 %, la pérdida se estancó.
- **→** En ese momento se cambia a Parameter Server asincrónico para seguir bajando sin esperas.
- Diferencia con el paper: usamos la pérdida de entrenamiento máxima entre workers, no la de validación. Así no pagamos una pasada de validación por ronda y el criterio es conservador.

**Pase:** «Lo interesante es cómo se hace ese cambio con el sistema corriendo.»

Respaldo: docs/report/estado_del_arte.md (Strategy-Switch) · solucion.md · código: orchestrator/src/sessions/switch_tracker.rs

### 20:00 → 21:45 · 13 · Strategy-Switch en O.N.O.

**Marcos Bianchi Fernández** · 01:45 · en pantalla: «el orchestrator mide» «Upgrade · Switch» «un worker pasa a servidor» «mismo proceso, otro rol»

> Qué dispara el cambio, quién lo decide y cómo se transiciona, tal como está implementado.

- Fase 1: los workers entrenan en anillo y en cada ronda le reportan su pérdida al orchestrator.
- **→** El orchestrator registra la pérdida máxima de cada ronda en el SwitchTracker (ventana de 6, umbral 0,01).
- **→** Cuando se cumple el umbral, envía dos órdenes precalculadas: Upgrade a los nodos que van a ser servidores, elegidos al inicio por latencia, y Switch al resto.
- **→** El nodo promovido toma su tramo de parámetros y arranca como servidor. Los demás dejan el anillo y se conectan como workers de PS.
- **→** No se reinicia nada: es el mismo proceso con otro rol. Lo hace posible la arquitectura que les va a contar Lorenzo.
- Si preguntan: el usuario elige la estrategia, la cantidad de servidores, el sincronizador y el store en `training.json`; el sistema solo decide el momento del cambio.

**Pase:** Traspaso: «Lorenzo les va a contar cómo está construido y qué medimos.»

Respaldo: docs/report/solucion.md (Strategy-Switch) · código: orchestrator/src/sessions/event_listener.rs, orchestrator/src/configs/adapter.rs

### 21:45 → 22:45 · 14 · ¿Por qué Rust?

**Lorenzo Minervino** · 01:00 · en pantalla: sin recolector de basura · seguridad de memoria · sin data races · async con tokio · el protocolo no lo verifica

> Elegimos Rust por el control de bajo nivel con garantías de seguridad; la dificultad se movió al protocolo.

- Todo O.N.O. está escrito en Rust, incluida la biblioteca de redes neuronales.
- **→** Da control de bajo nivel, como C o C++, pero sin recolector de basura y con la seguridad de memoria verificada al compilar.
- **→** En concurrencia, el compilador rechaza las condiciones de carrera sobre memoria compartida (*fearless concurrency*): ninguna llegó a la rama principal. Las comunicaciones son async, sobre tokio.
- **→** Para ser honestos, el compilador no verifica la coordinación distribuida. Esa fue la dificultad real, sobre todo en All-Reduce.
- Si preguntan por `unsafe`: está acotado y marcado, por ejemplo en WildStore (Hogwild!) y en la codificación de mensajes.

**Pase:** «Veamos cómo está armada la orquesta por dentro.»

Respaldo: docs/report/herramientas.md (Rust) · riesgos.md (complejidad de la implementación distribuida en Rust)

### 22:45 → 24:45 · 15 · La orquesta por dentro

**Lorenzo Minervino** · 02:00 · en pantalla: «mismo binario en cada nodo» «orchestrator: control» «el rol llega al conectarse» «datos: de nodo a nodo»

> Todos los nodos ejecutan el mismo binario y reciben su rol al conectarse; eso es lo que habilita Strategy-Switch.

- Cada máquina corre el mismo binario, `node`. No hay una imagen de worker y otra de servidor.
- **→** El orchestrator es el plano de control y vive dentro de la interfaz de terminal o de la biblioteca de Python. Se conecta con cada nodo mediante un handshake Connect/Accept.
- **→** Le manda a cada nodo su especificación y su rol, worker o servidor. Cada extremo recibe un handle tipado que solo permite los mensajes de ese rol, y el compilador rechaza el resto.
- **→** Los gradientes y los parámetros viajan entre nodos, en f16 y sin copias; el orchestrator queda fuera del plano de datos.
- Si preguntan por el motor: el modelo es un único buffer contiguo, y particionarlo es delimitar rangos (apéndice A6).

**Pase:** «Y todo esto se ve en vivo desde la terminal.»

Respaldo: docs/report/solucion.md (Nodo agnóstico de rol · Separación de planos) · README.md (Architecture)

### 24:45 → 25:45 · 16 · La TUI, en vivo

**Lorenzo Minervino** · 01:00 · en pantalla: «anillo de All-Reduce» «Parameter Server» «el switch, en vivo»

> La topología que dibuja la TUI es la que construyó el orchestrator, y cambia en vivo cuando Strategy-Switch promueve un nodo.

- orchestui es la interfaz de terminal que embebe al orchestrator. Son capturas reales de una ejecución local con 4 nodos, fuera de la campaña experimental. Acá, el anillo de All-Reduce con la pérdida y la época de cada worker.
- **→** La misma vista con Parameter Server: los workers a la izquierda y el servidor a la derecha.
- **→** Strategy-Switch en el instante del cambio: el encabezado dice «switching W3 → PS» y la topología pasa de anillo a estrella.
- También está orchestra-py, una biblioteca de Python con la misma funcionalidad. La suite de benchmarks está escrita sobre ella (apéndice A9).

**Pase:** «Con esto armamos los experimentos.»

Respaldo: orchestui/src/ui/components/topology.rs · presentation/captures/raw/

### 25:45 → 27:00 · 17 · Cómo medimos

**Lorenzo Minervino** · 01:15 · en pantalla: «MNIST · LeNet-5» «1 contenedor = 1 nodo» «de 2 a 16 workers» «red lenta, a propósito»

> Fijamos todo salvo la estrategia, y forzamos una red lenta para que las diferencias se vean.

- Un clúster simulado en una sola máquina física: un contenedor de Docker por nodo, con un núcleo cada uno. Entrenamos LeNet-5 (61.706 parámetros) sobre MNIST completo, con Adam, hasta 48 épocas.
- **→** Variamos la cantidad de workers: 2, 4, 6, 8, 12 y 16. PS con un solo servidor.
- **→** Limitamos la salida de cada contenedor a 10 Mbps con Pumba. ¿Por qué? Con un modelo chico, la red real es tan rápida que no se distinguen las estrategias. Bajando R, la transmisión queda en ≈ 0,1 s por ronda. Como referencia, en una sola máquina tarda 600 s.
- Antes de medir tiempos validamos la corrección con pruebas unitarias, de integración y de aceptación en CI: la convergencia distribuida coincide con la secuencial.

**Pase:** «Primero, los tiempos de los algoritmos sincrónicos.»

Respaldo: docs/report/experimentacion.md (Pruebas automatizadas · Entorno de experimentación)

### 27:00 → 28:15 · 18 · Resultados: sincrónicos

**Lorenzo Minervino** · 01:15 · en pantalla: «PS empeora desde 6 workers» «AR deja de mejorar» «como predijo el modelo»

> Lo medido coincide con lo que predice el modelo analítico.

- El tiempo total de entrenamiento según la cantidad de workers. Es la figura del informe, con los colores invertidos para el proyector, y el eje Y está en segundos.
- **→** Parameter Server con un servidor tiene su mínimo en 6 workers y desde ahí empeora: el servidor concentra todo el tráfico.
- **→** All-Reduce no se degrada, pero deja de mejorar: el cómputo por worker baja y la sincronización se queda en 2S/R. Es lo que predecía el modelo analítico.
- Lectura aproximada, por si preguntan: AR cerca de 290 s desde 6 workers; PS cerca de 340 s en 6 y algo más de 400 s en 16.

**Pase:** «¿Y si sacamos la barrera?»

Respaldo: docs/report/resultados.md (Resultados obtenidos) · figura execution_time_per_nodes_ar_vs_ps.png

### 28:15 → 29:00 · 19 · Resultados: asincrónicos

**Lorenzo Minervino** · 00:45 · en pantalla: «el tiempo sigue bajando» «¿pero aprende?»

> En velocidad pura, los dos asincrónicos ganan.

- Con PS asincrónico y Strategy-Switch, el tiempo sigue bajando al sumar workers, a diferencia de los sincrónicos.
- **→** Pero acelerar no sirve de nada si el modelo no aprende. Hay que mirar la exactitud.

**Pase:** «Veamos la exactitud final.»

Respaldo: docs/report/resultados.md (Resultados obtenidos) · figura execution_times_async_parameter_server_and_strategy_switch.png

### 29:00 → 30:30 · 20 · Resultados: exactitud

**Lorenzo Minervino** · 01:30 · en pantalla: «Strategy-Switch ≈ sincrónicos» «PS asincrónico colapsa» «salvedad: WildStore»

> Strategy-Switch fue el mejor balance de esta campaña; la salvedad sobre WildStore es parte del resultado.

- La exactitud final sobre el conjunto de prueba de MNIST. All-Reduce, PS sincrónico y Strategy-Switch se superponen arriba.
- **→** PS asincrónico cae al sumar workers, hasta valores cercanos al azar desde 8 workers: es el client drift en acción. Strategy-Switch, en cambio, conserva la exactitud de los sincrónicos con tiempos cercanos a los del asincrónico.
- **→** La salvedad, dicha por nosotros antes de que pregunten: PS asincrónico corrió con WildStore, que hace escrituras sin protección al estilo Hogwild!. Con estos datos no podemos separar el drift del efecto de esas escrituras. Repetirlo con BlockingStore es trabajo futuro.
- Alcance: estas conclusiones valen para MNIST, LeNet-5 y este entorno.

**Pase:** «El drift también lo medimos de forma directa.»

Respaldo: docs/report/resultados.md (salvedad final) · conclusiones_capitulo4.md · riesgos.md

### 30:30 → 31:30 · 21 · El drift, medido

**Lorenzo Minervino** · 01:00 · en pantalla: «E épocas sin sincronizar» «E = 1 → cerca de −1 punto» «más nodos, más caída» «sin ganancia de tiempo»

> Demorar la sincronización es pagar drift.

- Es un estudio derivado, de la monografía de Marcos: All-Reduce con E épocas offline, es decir, E épocas extra sin sincronizar. Se corrió en una máquina de 4 núcleos con red hogareña emulada.
- **→** Con una sola época offline (la mitad de sincronizaciones) la exactitud cae cerca de un punto porcentual, y la caída crece con la cantidad de nodos: las particiones son más chicas y más sesgadas.
- **→** Y en ese entorno no hubo ganancia de tiempo que lo compense.

**Pase:** «¿Qué queda por hacer?»

Respaldo: docs/report/estudio_derivado.md

### 31:30 → 32:30 · 22 · Trabajos futuros

**Lorenzo Minervino** · 01:00 · en pantalla: tolerancia a fallos · Hogwild! · convolución en paralelo · datasets por flujo · más arquitecturas · compresión

> El sistema quedó listo para retomarse, y varias líneas ya tienen el diseño especificado.

- Las líneas abiertas, ordenadas por relevancia. Varias ya tienen el diseño especificado en el repositorio.
- **→** Tolerancia a fallos: es imprescindible para escalar, y el comportamiento por entidad ya está especificado. También hay que resolver o retirar la variante Hogwild! y repetir PS asincrónico con BlockingStore.
- **→** Paralelizar el cálculo de la convolución y leer los datasets por flujo desde disco, para los que no entran en memoria.
- **→** Ampliar el motor a otras arquitecturas y medir de forma aislada la compresión de gradientes.

**Pase:** «Y ahora, vamos a verlo funcionando.»

Respaldo: docs/report/trabajos_futuros.md

### 32:30 → 33:00 · 23 · Ahora, la demo

**[a completar]** · 00:30 · en pantalla: «[a completar]» «[a completar]» «[a completar]» «10 min»

> [a completar por el equipo]

- [a completar: qué se muestra, en qué orden, qué se ejecuta y qué resultado hay que observar]
- El guion en blanco está en presentation/GUION_DEMO.md. Material de respaldo opcional: apéndice A1 (grabación real de Strategy-Switch en la TUI) y A2 (capturas reales).

**Pase:** [a completar]

Respaldo: [a completar]

### 43:00 → 45:00 · 24 · Gracias por venir

**Los tres** · 02:00 · en pantalla: «4 estrategias, 1 base» «Strategy-Switch: mejor balance» «¿Preguntas?»

> Volver al mensaje central después de la demo y abrir las preguntas.

- Retomar en una frase lo que se vio en la demo: la misma base ejecuta las cuatro estrategias y permite compararlas.
- Las tres ideas que queremos que se lleven: la sincronización centralizada es el cuello de botella; asincronismo y convergencia están en tensión directa; y Strategy-Switch fue el mejor balance que evaluamos.
- Agradecer al jurado, al tutor y al co-tutor. El repositorio es público.
- Preguntas probables: WildStore o BlockingStore; por qué 10 Mbps; por qué la pérdida de entrenamiento y no la de validación; tolerancia a fallos; uso de herramientas de IA, declarado en el informe. Para volver a una diapositiva: tecla G (índice) o número + Enter; el apéndice empieza en la 25.

**Pase:** —

Respaldo: docs/report/conclusiones.md · anexos.md (repositorio)

## Apéndice (solo para preguntas)

### 25 · A1 · Grabación real de Strategy-Switch

> Material de respaldo opcional, por si se quiere mostrar el cambio de estrategia sin ejecutar en vivo.

- Secuencia real de pantallas (4 nodos locales, subconjunto de 10.000 muestras de MNIST, configs en presentation/captures/configs/). Mostrar el anillo, el instante «switching → PS» y la topología en estrella posterior. Si se usa, aclarar que es una grabación.

Respaldo: presentation/captures/rec/ · presentation/captures/configs/training_strategy_switch.json

### 26 · A2 · Capturas reales de la TUI

> Capturas de respaldo de las pantallas principales.

- Menú, ayuda de configuración y dashboard al finalizar una ejecución de Strategy-Switch (pérdida promedio, pérdida por worker, tabla de workers, parámetros finales y eventos).
- Ojo: en esa ejecución la tabla muestra 99–100 epochs con max_epochs = 60; ver README, sección «Diferencias detectadas».

Respaldo: presentation/captures/raw/

### 27 · A3 · Derivación de d_sync

> Respaldo para preguntas sobre el modelo analítico.

- Con N_S = N_W, Multi-PS alcanza 2S/R, el mismo límite que AR cuando N_W crece.

Respaldo: docs/report/resultados.md

### 28 · A4 · Detención temprana y disparador

> Respaldo: dos criterios distintos sobre la misma señal.

- Ambos usan la pérdida de entrenamiento máxima para evitar una pasada de evaluación por ronda y para ser conservadores.

Respaldo: docs/report/solucion.md (Plano de Control y Orquestación) · código: orchestrator/src/sessions/

### 29 · A5 · Lecciones: errores silenciosos

> Respaldo: la lección central de ingeniería del trabajo.

- Conviene tener una medición end-to-end desde temprano. Otra lección: «make it work, make it right, make it fast».

Respaldo: docs/report/riesgos.md (Lecciones aprendidas)

### 30 · A6 · Modelo como buffer plano

> Una decisión del motor de redes neuronales habilita la comunicación sin copias de todo el sistema.

- La biblioteca está construida sobre ndarray: capas densas, convolucionales, max pooling, reshape y activaciones; inicializaciones Xavier-Glorot y Kaiming-He; optimizadores detrás de un mismo trait.
- La decisión clave es que el modelo sea un buffer contiguo y las capas, vistas sobre tramos. La alternativa, un árbol de objetos con sus pesos, obligaba a serializar cada capa y, en Rust, a acoplar el motor con comms para prestar referencias mutables.
- **→** Particionar entre servidores es delimitar rangos. **→** Enviar es reinterpretar memoria (bytemuck).
- Limitación: los modelos son secuencias de capas; arquitecturas con conexiones de salto, como U-Net, no se pueden expresar hoy.

Respaldo: docs/report/solucion.md (Motor de Redes Neuronales · Modelo como buffer plano)

### 31 · A7 · Compresión con error feedback

> Separar control de datos y comprimir sin perder la información acumulada.

- Control: comandos poco frecuentes en JSON, inspeccionables. Datos: concentran el volumen y viajan por reinterpretación de memoria, en media precisión.
- **→** Solo viajan los componentes por encima de un umbral, elegido para que pase una fracción r; el umbral se estima muestreando hasta 16384 componentes con quickselect (Aji y Heafield, 2017).
- **→** Lo que no se envía queda en un residual persistente.
- **→** En la ronda siguiente el residual se suma al gradiente nuevo: un componente chico pero constante termina cruzando el umbral (error feedback, Lin et al., 2018). Si la codificación dispersa pesa más que la densa en f16, se envía la densa.
- Limitación: no se midió de forma aislada el compromiso entre tasa de compresión y convergencia.

Respaldo: docs/report/solucion.md (Separación de planos · Compresión de gradientes con error feedback)

### 32 · A8 · Topología por latencia

> Ninguna diferencia de tiempo debería venir del orden en que el usuario escribió las direcciones.

- Antes de entrenar, cada par de nodos intercambia diez pings y se toma el mayor RTT. Se mide una sola vez (limitación en redes con carga variable).
- **→** All-Reduce: el orden del anillo que minimiza la latencia total es un circuito hamiltoniano de costo mínimo, resuelto en forma exacta con programación dinámica sobre subconjuntos (Held y Karp, 1962).
- **→** Parameter Server: se eligen los k servidores que minimizan la mayor latencia worker–servidor, por backtracking. Y el optimizador con estado se aplica después de agregar: si cada worker aplicara Adam sobre su gradiente local, acumularían momentos distintos y las réplicas divergirían.

Respaldo: docs/report/solucion.md (Selección de topología · Ubicación del optimizador) · código: orchestrator/src/calculator/

### 33 · A9 · Interfaz de Python

> orchestra-py expone el mismo orchestrator para experimentar programáticamente.

- Módulo nativo vía PyO3 y maturin. El fragmento muestra la API real (funciones strategy_switch y orchestrate). Se construyó como crate separado porque anotar los tipos existentes con #[pyclass] chocaba con genéricos, variantes de enum y Clone. La suite de benchmarks está escrita sobre esta interfaz.

Respaldo: docs/report/solucion.md (Interfaz de Python como crate separado) · orchestra-py/local.py

### 34 · A10 · Por qué una base común

> Respaldo: por qué hizo falta construir una base común en lugar de comparar sobre TensorFlow o PyTorch.

- Esas implementaciones están optimizadas para producción y no aplican el mismo trabajo de bajo nivel a cada estrategia (solapamiento de cómputo y comunicación, kernels especializados). Al comparar dos estrategias sobre dos implementaciones, la diferencia mezcla algoritmo y calidad de implementación, y ese sesgo no se puede cuantificar.

Respaldo: docs/report/problema.md · estado_del_arte.md
