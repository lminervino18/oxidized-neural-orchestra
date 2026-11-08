# Guion de la defensa

Generado desde `index.html` con `python3 tools/export_notes.py`. No editar a mano: los textos se cambian en las notas de cada diapositiva.

Cada **→** marca que hay que avanzar una etapa (flecha derecha) antes de decir ese punto. «En pantalla» es lo único que ve el jurado; todo lo demás se dice. Las diapositivas A1–A8 del apéndice se cuentan mientras entrena la demo.

## Reparto

| Quién | Diapositivas | Tiempo |
|---|---|---|
| Alejo Ordoñez | 01–09 | 09:15 |
| Marcos Bianchi Fernández | 10–16 | 11:45 |
| Alejo Ordoñez | 17–19 | 02:45 |
| Lorenzo Minervino | 20–29 | 08:45 |
| Alejo Ordoñez | A1–A2 | 02:30 |
| Marcos Bianchi Fernández | A3–A4 | 02:30 |
| Lorenzo Minervino | A5–30 | 02:45 + demo 02:00 |
| Los tres | 31–32 | 01:00 |
| **Total** | | **43:15** |

## Línea de tiempo

| Desde | Hasta | # | Diapositiva | Quién | En pantalla |
|---|---|---|---|---|---|
| 00:00 | 00:30 | 01 | Portada | Alejo Ordoñez | O.N.O. · entrenamiento distribuido · Rust |
| 00:30 | 01:00 | 02 | Cronograma | Alejo Ordoñez | motivación · sincronización · arquitectura · costo de sincronización · resultados · demostración en vivo |
| 01:00 | 02:00 | 03 | Definición de modelo | Alejo Ordoñez | «entrada → salida» «parámetros internos» «entrenamiento: ajuste de parámetros» «GPT-3: 175 mil millones de parámetros» |
| 02:00 | 03:00 | 04 | Costo del entrenamiento | Alejo Ordoñez | «más datos» «más parámetros» «días de cómputo» «excede la memoria» |
| 03:00 | 04:00 | 05 | Escalado vertical | Alejo Ordoñez | «mayor capacidad en un solo equipo» «costo elevado · acceso limitado» «límite físico» |
| 04:00 | 05:00 | 06 | Escalado horizontal | Alejo Ordoñez | «hardware de uso general» «más equipos en paralelo» «interconectados por red» |
| 05:00 | 06:45 | 07 | Paralelismo de modelo y de datos | Alejo Ordoñez | «modelo: partición de la red» «datos: partición del dataset» «O.N.O.: paralelismo de datos» |
| 06:45 | 08:30 | 08 | Client drift | Alejo Ordoñez | «réplicas iguales, datos distintos» «client drift» «sincronización: promedio de réplicas» «¿cómo sincronizar?» |
| 08:30 | 09:15 | 09 | Estrategias de sincronización | Alejo Ordoñez | Parameter Server · All-Reduce · PS asincrónico · Strategy-Switch · misma base |
| 09:15 | 11:15 | 10 | Parameter Server | Marcos Bianchi Fernández | «servidor central» «pull θ» «push gradiente» «promedio + optimizador» «varios servidores: partición por capa» |
| 11:15 | 13:30 | 11 | Ring All-Reduce | Marcos Bianchi Fernández | «sin nodo central» «N fragmentos» «reduce-scatter: suma» «all-gather: copia» «÷ N en cada worker» |
| 13:30 | 15:15 | 12 | Parameter Server asincrónico | Marcos Bianchi Fernández | «sincrónico: espera al nodo más lento» «asincrónico: sin barrera» «θ desactualizado → mayor drift» |
| 15:15 | 16:45 | 13 | Strategy-Switch: principio | Marcos Bianchi Fernández | «inicio en All-Reduce» «criterio: estabilización de la pérdida» «ventana de 6 · umbral 1 %» «cambio a PS asincrónico» |
| 16:45 | 18:30 | 14 | Strategy-Switch en O.N.O. | Marcos Bianchi Fernández | «medición en el orchestrator» «Upgrade · Switch» «un worker promovido a servidor» «mismo proceso, nuevo rol» |
| 18:30 | 19:15 | 15 | Medición del drift | Marcos Bianchi Fernández | «E: epochs sin sincronización» «E = 1: ≈ −1 punto de accuracy» «mayor caída con más nodos» «sin ganancia de tiempo» |
| 19:15 | 21:00 | 16 | Arquitectura del sistema | Marcos Bianchi Fernández | «nodo agnóstico de rol» «orchestrator: control» «rol asignado al conectarse» «datos: directamente entre nodos» |
| 21:00 | 22:00 | 17 | Biblioteca de ML propia | Alejo Ordoñez | «sin PyTorch ni TensorFlow» «capas · pérdidas · optimizadores» «Adam · momentum · SGD» |
| 22:00 | 22:30 | 18 | Teorema de la convolución | Alejo Ordoñez | «convolución → producto ⊙» |
| 22:30 | 23:45 | 19 | Implementación de la convolución | Alejo Ordoñez | «FFT: sin mejora» «im2col: ventana → columna» «una única multiplicación de matrices» «trabajo futuro: según el tamaño» |
| 23:45 | 24:45 | 20 | Tiempo de un entrenamiento | Lorenzo Minervino | «el cómputo se reparte» «sincronización: todos esperan» «una ronda por epoch» «d_sync determina la escalabilidad» |
| 24:45 | 26:00 | 21 | Cálculo de d_sync | Lorenzo Minervino | «S: tamaño del modelo · R: ancho de banda» «PS: todo el tráfico por una placa» «crece con la cantidad de workers» «AR: transmisión simultánea» «acotado» |
| 26:00 | 26:45 | 22 | Costo de sincronización | Lorenzo Minervino | «PS: crecimiento lineal» «cuello de botella: el servidor» «AR: acotado por 2S/R» |
| 26:45 | 27:30 | 23 | Interfaz de Python | Lorenzo Minervino | «PyO3 + maturin» «liberación del GIL» «suite de benchmarks en Python» |
| 27:30 | 28:30 | 24 | Metodología de medición | Lorenzo Minervino | «MNIST · LeNet-5» «1 contenedor = 1 nodo» «de 2 a 16 workers» «red limitada deliberadamente» |
| 28:30 | 29:30 | 25 | Resultados: sincrónicos | Lorenzo Minervino | «PS: degradación desde 6 workers» «AR: sin mejora desde 6 workers» «consistente con el modelo» |
| 29:30 | 30:00 | 26 | Resultados: asincrónicos | Lorenzo Minervino | «el tiempo disminuye con más workers» «¿converge el modelo?» |
| 30:00 | 31:15 | 27 | Resultados: accuracy | Lorenzo Minervino | «Strategy-Switch ≈ sincrónicos» «PS asincrónico: colapso de accuracy» |
| 31:15 | 32:00 | 28 | Trabajos futuros | Lorenzo Minervino | tolerancia a fallos · convolución según el tamaño · nuevas arquitecturas |
| 32:00 | 32:30 | 29 | Demostración | Lorenzo Minervino | «3 computadoras» «Strategy-Switch · MNIST» «≈ 10 min de entrenamiento» «luego: el modelo, en el navegador» |
| 32:30 | 33:45 | A1 | A1 · Por qué una base común | Alejo Ordoñez | — |
| 33:45 | 35:00 | A2 | A2 · Modelo como buffer plano | Alejo Ordoñez | «capas = vistas» «particionar = delimitar rangos» «enviar sin copiar» |
| 35:00 | 36:15 | A3 | A3 · Compresión con error feedback | Marcos Bianchi Fernández | «control: JSON» «datos: f16, sin copias» «envío de valores sobre el umbral» «el resto se acumula en el residual» «error feedback» |
| 36:15 | 37:30 | A4 | A4 · Topología por latencia | Marcos Bianchi Fernández | «10 pings por par» «anillo: viajante, solución exacta» «servidor: menor latencia máxima» «optimizador después de agregar» |
| 37:30 | 38:30 | A5 | A5 · Elección de Rust | Lorenzo Minervino | — |
| 38:30 | 40:00 | A6 | A6 · Lecciones: errores silenciosos | Lorenzo Minervino | — |
| 40:00 | 40:15 | 30 | ¿Y la demo? | Lorenzo Minervino | — |
| 40:15 | 42:15 | — | Fin del entrenamiento y página de la demo | Lorenzo Minervino | ver `GUION_DEMO.md` |
| 42:15 | 42:45 | 31 | ¿Preguntas? | Los tres | — |
| 42:45 | 43:15 | 32 | ¡Gracias por venir! | Los tres | — |

## Qué dice cada uno

### 00:00 → 00:30 · 01 · Portada

**Alejo Ordoñez** · 00:30 · en pantalla: O.N.O. · entrenamiento distribuido · Rust

> Presentarnos y decir en una frase qué hicimos.

- Saludo al jurado. Somos Alejo, Lorenzo y Marcos; nuestro tutor es Ricardo Veiga y el co-tutor, José Ignacio Alvarez-Hamelin.
- En una frase: construimos O.N.O., un sistema para entrenar redes neuronales entre varias máquinas, escrito en Rust, con el que implementamos y comparamos cuatro formas de sincronizar ese entrenamiento.

**Pase:** «Este es el recorrido que vamos a hacer.»

Respaldo: docs/report/caratula.md · resumen.md

### 00:30 → 01:00 · 02 · Cronograma

**Alejo Ordoñez** · 00:30 · en pantalla: motivación · sincronización · arquitectura · costo de sincronización · resultados · demostración en vivo

> Mostrar el recorrido y quién cuenta cada parte.

- Primero, por qué hace falta distribuir el entrenamiento. Después Marcos cuenta cómo se sincroniza y cómo está armado el sistema, y yo vuelvo para contar el motor de redes neuronales.
- Lorenzo cuenta cuánto cuesta sincronizar, cómo medimos y qué dio.
- Cerramos con una demo en vivo en tres computadoras. Mientras entrena, contamos algunas decisiones de diseño.

**Pase:** «Antes de distribuir nada: ¿qué es un modelo?»

Respaldo: —

### 01:00 → 02:00 · 03 · Definición de modelo

**Alejo Ordoñez** · 01:00 · en pantalla: «entrada → salida» «parámetros internos» «entrenamiento: ajuste de parámetros» «GPT-3: 175 mil millones de parámetros»

> Un modelo es una función con parámetros ajustables; entrenar es encontrar esos valores a partir de ejemplos.

- Visto de afuera, un modelo es una caja negra: entra la imagen de un dígito escrito a mano y sale qué dígito es.
- **→** Por dentro es una función con muchos parámetros, como perillas. Según dónde estén las perillas, la salida cambia.
- **→** Entrenar es ajustar esas perillas con ejemplos: el modelo responde, se compara con la respuesta correcta y el error indica hacia dónde mover cada perilla. Se repite miles de veces.
- **→** Es lo mismo que hay detrás de ChatGPT, a otra escala: entra texto y sale texto. GPT-3 tiene 175 mil millones de parámetros; LeNet-5, el modelo que usamos nosotros, tiene 61.706.

**Pase:** «Y ajustar tantos parámetros cuesta.»

Respaldo: LeNet-5: docs/report/experimentacion.md · GPT-3: Brown et al. (2020), no está en la bibliografía del informe

### 02:00 → 03:00 · 04 · Costo del entrenamiento

**Alejo Ordoñez** · 01:00 · en pantalla: «más datos» «más parámetros» «días de cómputo» «excede la memoria»

> Entrenar un modelo profundo cuesta en dos dimensiones: tiempo y memoria.

- Un modelo chico con pocos datos entrena rápido: la barra se llena enseguida.
- **→** Pero los modelos útiles crecen: más datos y más parámetros. Cada epoch recorre todo el dataset y actualiza todos los parámetros.
- **→** Sobre conjuntos grandes, un entrenamiento completo puede tardar días en una sola máquina.
- **→** La otra dimensión es la memoria: el dataset, el modelo y sus estados intermedios pueden no entrar en un solo dispositivo.

**Pase:** «¿La solución obvia? Comprar una máquina más grande.»

Respaldo: docs/report/problema.md (párrafo 1) · Ben-Nun y Hoefler (2019)

### 03:00 → 04:00 · 05 · Escalado vertical

**Alejo Ordoñez** · 01:00 · en pantalla: «mayor capacidad en un solo equipo» «costo elevado · acceso limitado» «límite físico»

> Escalar con un solo equipo es caro y además tiene un límite.

- La primera reacción es comprar una máquina más potente. Eso es escalar verticalmente.
- **→** Un equipo de gran porte es caro, y en la práctica quien necesita entrenar rara vez tiene uno.
- **→** Y aun pagándolo, un solo equipo tiene un techo: su memoria y su cómputo son finitos.

**Pase:** «Lo que sí suele haber son varias máquinas comunes.»

Respaldo: docs/report/problema.md (párrafo 1)

### 04:00 → 05:00 · 06 · Escalado horizontal

**Alejo Ordoñez** · 01:00 · en pantalla: «hardware de uso general» «más equipos en paralelo» «interconectados por red»

> Sumar máquinas modestas es más accesible que comprar una que las iguale: eso es el entrenamiento distribuido.

- Una máquina común sola no alcanza.
- **→** Pero es frecuente tener varias. Sumarlas es escalar horizontalmente, con hardware que ya existe.
- **→** Conectadas por red suman cómputo y memoria. El precio es que ahora hay que coordinarlas, y la red pasa a ser parte del problema.

**Pase:** «¿Y cómo se reparte el trabajo entre ellas? Hay dos grandes formas.»

Respaldo: docs/report/problema.md · introduccion.md

### 05:00 → 06:45 · 07 · Paralelismo de modelo y de datos

**Alejo Ordoñez** · 01:45 · en pantalla: «modelo: partición de la red» «datos: partición del dataset» «O.N.O.: paralelismo de datos»

> Se puede partir el modelo o partir los datos; O.N.O. trabaja con paralelismo de datos.

- Tenemos una red y un dataset. Hay dos formas clásicas de repartirlos (Ben-Nun y Hoefler, 2019).
- **→** Paralelismo de modelo: cada máquina tiene un pedazo de la red, por ejemplo una capa. La activación pasa de una máquina a la siguiente, así que cada una espera a la anterior. Sirve cuando el modelo no entra en una sola máquina.
- **→** Paralelismo de datos: cada máquina tiene el modelo completo y una porción distinta del dataset, y todas entrenan a la vez.
- **→** O.N.O. trabaja con paralelismo de datos, que también es el esquema principal de TensorFlow y PyTorch. El paralelismo de modelo y el de pipeline quedaron fuera de alcance.

**Pase:** «Pero el paralelismo de datos tiene una trampa.»

Respaldo: docs/report/estado_del_arte.md (Paralelismo de datos, de modelo y de pipeline) · introduccion.md

### 06:45 → 08:30 · 08 · Client drift

**Alejo Ordoñez** · 01:45 · en pantalla: «réplicas iguales, datos distintos» «client drift» «sincronización: promedio de réplicas» «¿cómo sincronizar?»

> Con paralelismo de datos cada réplica tira hacia el mínimo de su partición; por eso hay que sincronizarlas.

- Los tres workers arrancan con la misma réplica del modelo. Las curvas de nivel son la pérdida sobre todo el dataset y la estrella es su mínimo. El dibujo es ilustrativo.
- **→** Cada worker entrena con su partición, así que su gradiente apunta al mínimo de sus datos y no al global. En O.N.O. cada worker recorre al menos una epoch completa sobre su partición antes de sincronizar, y en ese tiempo las réplicas se separan. Eso se llama *client drift* (McMahan et al., 2017).
- **→** La solución es sincronizar: se combinan los aportes de todos (en O.N.O., promediando gradientes) y todos siguen desde el mismo punto.
- **→** Alternando entrenar y sincronizar, el conjunto converge al mínimo global. Cuanto más se demora la sincronización, más drift hay; lo medimos al final.

**Pase:** «La pregunta, entonces, es cómo sincronizar. Hay varias respuestas.»

Respaldo: docs/report/resultados.md (PS asincrónico, client drift) · estado_del_arte.md (aprendizaje federado) · código: machine_learning/src/training/backprop_trainer.rs

### 08:30 → 09:15 · 09 · Estrategias de sincronización

**Alejo Ordoñez** · 00:45 · en pantalla: Parameter Server · All-Reduce · PS asincrónico · Strategy-Switch · misma base

> Vamos a ver cuatro formas de sincronizar, y O.N.O. las implementa todas sobre la misma base.

- Parameter Server: un servidor central junta los gradientes y reparte los parámetros.
- **→** All-Reduce: no hay centro; los workers se pasan los gradientes en anillo.
- **→** Parameter Server asincrónico: como el primero, pero nadie espera a nadie.
- **→** Strategy-Switch: arranca con All-Reduce y, en un momento, cambia a Parameter Server asincrónico.
- **→** Nuestro trabajo fue implementar las cuatro sobre la misma capa de comunicación, el mismo motor de redes neuronales y el mismo plano de control. Así, al compararlas, la diferencia viene del algoritmo y no de la implementación. Con TensorFlow o PyTorch eso no se puede separar (apéndice A1).

**Pase:** Traspaso: «Marcos les va a contar cómo funciona cada una.»

Respaldo: docs/report/problema.md · introduccion.md · estado_del_arte.md

### 09:15 → 11:15 · 10 · Parameter Server

**Marcos Bianchi Fernández** · 02:00 · en pantalla: «servidor central» «pull θ» «push gradiente» «promedio + optimizador» «varios servidores: partición por capa»

> En Parameter Server un servidor concentra los parámetros y el estado del optimizador.

- A la izquierda, los workers, cada uno con su partición. A la derecha, el servidor, que guarda los parámetros y el optimizador con estado.
- **→** Cada worker pide los parámetros actuales (pull).
- **→** Entrena sobre su partición y acumula su gradiente.
- **→** Envía el gradiente al servidor (push). Con el sincronizador `barrier`, el servidor espera a los N gradientes.
- **→** El servidor promedia y aplica el optimizador, por ejemplo Adam, una sola vez: hay un único estado del optimizador.
- **→** Con varios servidores, el modelo se parte por capas completas y cada capa vive en un servidor. El orchestrator asigna las capas, de mayor a menor, al servidor menos cargado.

**Pase:** «Si todo pasa por el servidor, ¿qué pasa si no hay servidor? Eso es All-Reduce.»

Respaldo: docs/report/solucion.md (Parameter Server) · código: worker/src/workers/parameter_server.rs, parameter_server/src/synchronization

### 11:15 → 13:30 · 11 · Ring All-Reduce

**Marcos Bianchi Fernández** · 02:15 · en pantalla: «sin nodo central» «N fragmentos» «reduce-scatter: suma» «all-gather: copia» «÷ N en cada worker»

> En All-Reduce no hay servidor: los gradientes circulan por un anillo y todos terminan con el mismo promedio.

- Cuatro workers en un anillo lógico: cada uno recibe del anterior y envía al siguiente. Es la variante de Patarasuk y Yuan (2009).
- **→** Cada worker parte su gradiente en N fragmentos, acá cuatro. Los colores indican de qué worker viene cada aporte.
- **→** Reduce-scatter, iteración 1: cada worker envía un fragmento al siguiente, y el que lo recibe lo suma al suyo.
- **→** Iteración 2: cada uno reenvía el fragmento que acaba de sumar, que ya trae dos aportes.
- **→** Iteración 3: son N−1 iteraciones. Al final, cada worker tiene un fragmento con la suma de todos.
- **→** All-gather, iteración 1: ahora no se suma, se copia. Cada worker le pasa su fragmento completo al siguiente.
- **→** Iteración 2: los fragmentos completos siguen dando la vuelta.
- **→** Iteración 3: después de otras N−1 iteraciones, todos tienen los cuatro fragmentos completos.
- **→** Cada worker divide por N y aplica el optimizador localmente. Todos llegan al mismo resultado sin que nadie centralice.

**Pase:** «Las dos esperan, en cada ronda, a que todos terminen. ¿Y si no esperamos a nadie?»

Respaldo: docs/report/solucion.md (Ring All-Reduce) · código: worker/src/middlewares/worker_ring.rs

### 13:30 → 15:15 · 12 · Parameter Server asincrónico

**Marcos Bianchi Fernández** · 01:45 · en pantalla: «sincrónico: espera al nodo más lento» «asincrónico: sin barrera» «θ desactualizado → mayor drift»

> Sacar la barrera elimina la espera, pero los gradientes se calculan sobre parámetros desactualizados y el drift crece.

- Arriba, el régimen sincrónico: en cada ronda hay una barrera, el worker más lento marca el ritmo y los demás esperan (zonas rayadas).
- **→** Abajo, el asincrónico: el servidor aplica cada gradiente apenas llega, y la espera desaparece. El costo es que un worker puede enviar un gradiente calculado sobre parámetros viejos. Con particiones distintas, las réplicas tiran cada una para su lado y el client drift empeora (McMahan et al., 2017).
- Si preguntan: si llega un gradiente mientras el servidor aplica otro, se acumula en un segundo buffer y entra en la actualización siguiente; no se pierde.

**Pase:** «¿Se puede tener lo mejor de los dos? Esa es la idea de Strategy-Switch.»

Respaldo: docs/report/resultados.md (Parameter Server asincrónico) · estado_del_arte.md

### 15:15 → 16:45 · 13 · Strategy-Switch: principio

**Marcos Bianchi Fernández** · 01:30 · en pantalla: «inicio en All-Reduce» «criterio: estabilización de la pérdida» «ventana de 6 · umbral 1 %» «cambio a PS asincrónico»

> Strategy-Switch usa All-Reduce mientras los workers buscan el mínimo y pasa a PS asincrónico cuando la pérdida se estabiliza.

- Lo propusieron Provatas et al. (2025). Al principio el drift es más peligroso y conviene sincronizar fuerte, por eso arranca con All-Reduce. La curva es ilustrativa.
- **→** En cada ronda se mira la variación relativa de la pérdida en una ventana de 6 rondas. Cuando el promedio baja de 1 %, la pérdida se estancó.
- **→** En ese momento se cambia a Parameter Server asincrónico para seguir bajando sin esperas.
- Diferencia con el paper: usamos la pérdida de entrenamiento máxima entre workers, no la de validación. Así no pagamos una pasada de validación por ronda y el criterio es conservador.

**Pase:** «Lo interesante es cómo se hace ese cambio con el sistema corriendo.»

Respaldo: docs/report/estado_del_arte.md (Strategy-Switch) · solucion.md · código: orchestrator/src/sessions/switch_tracker.rs

### 16:45 → 18:30 · 14 · Strategy-Switch en O.N.O.

**Marcos Bianchi Fernández** · 01:45 · en pantalla: «medición en el orchestrator» «Upgrade · Switch» «un worker promovido a servidor» «mismo proceso, nuevo rol»

> Qué dispara el cambio, quién lo decide y cómo se transiciona, tal como está implementado.

- Fase 1: los workers entrenan en anillo y en cada ronda le reportan su pérdida al orchestrator.
- **→** El orchestrator registra la pérdida máxima de cada ronda en el SwitchTracker (ventana de 6, umbral 0,01).
- **→** Cuando se cumple el umbral, envía dos órdenes precalculadas: Upgrade a los nodos que van a ser servidores, elegidos al inicio por latencia, y Switch al resto.
- **→** El nodo promovido toma su tramo de parámetros y arranca como servidor. Los demás dejan el anillo y se conectan como workers de PS.
- **→** No se reinicia nada: es el mismo proceso con otro rol. Lo hace posible la arquitectura, que les cuento enseguida.
- Si preguntan: el usuario elige la estrategia, la cantidad de servidores, el sincronizador y el store en `training.json`; el sistema solo decide el momento del cambio.

**Pase:** «Antes de la arquitectura: el drift, que justifica arrancar sincronizando fuerte, lo medimos aparte.»

Respaldo: docs/report/solucion.md (Strategy-Switch) · código: orchestrator/src/sessions/event_listener.rs, orchestrator/src/configs/adapter.rs

### 18:30 → 19:15 · 15 · Medición del drift

**Marcos Bianchi Fernández** · 00:45 · en pantalla: «E: epochs sin sincronización» «E = 1: ≈ −1 punto de accuracy» «mayor caída con más nodos» «sin ganancia de tiempo»

> Demorar la sincronización es pagar drift.

- Es un estudio derivado, de mi monografía: All-Reduce con E epochs offline, es decir, E epochs extra sin sincronizar. Se corrió en una máquina de 4 núcleos con red hogareña emulada.
- **→** Con una sola epoch offline (la mitad de sincronizaciones) la accuracy cae cerca de un punto porcentual, y la caída crece con la cantidad de nodos: las particiones son más chicas y más sesgadas.
- **→** Y en ese entorno no hubo ganancia de tiempo que lo compense.

**Pase:** «Volviendo al sistema: ¿cómo está armado para cambiar de estrategia sin reiniciar?»

Respaldo: docs/report/estudio_derivado.md

### 19:15 → 21:00 · 16 · Arquitectura del sistema

**Marcos Bianchi Fernández** · 01:45 · en pantalla: «nodo agnóstico de rol» «orchestrator: control» «rol asignado al conectarse» «datos: directamente entre nodos»

> Todos los nodos ejecutan el mismo programa y reciben su rol al conectarse; eso es lo que habilita Strategy-Switch.

- Todo O.N.O. está escrito en Rust (si preguntan por qué, apéndice A5). Cada máquina corre el mismo ejecutable, `node`, que no tiene un rol fijo. No hay un programa de worker y otro de servidor.
- **→** El orchestrator es el plano de control y vive dentro de la interfaz de terminal o de la biblioteca de Python. Se conecta con cada nodo mediante un handshake Connect/Accept.
- **→** Le manda a cada nodo su especificación y su rol, worker o servidor. Cada extremo recibe un handle tipado que solo permite los mensajes de ese rol, y el compilador rechaza el resto.
- **→** Los gradientes y los parámetros viajan entre nodos, en f16 y sin copias; el orchestrator queda fuera del plano de datos.
- Si preguntan por el motor: el modelo es un único buffer contiguo, y particionarlo es delimitar rangos (apéndice A2).

**Pase:** Traspaso: «Adentro de cada nodo corre una librería de redes neuronales propia. Alejo la cuenta.»

Respaldo: docs/report/solucion.md (Nodo agnóstico de rol · Separación de planos) · README.md (Architecture)

### 21:00 → 22:00 · 17 · Biblioteca de ML propia

**Alejo Ordoñez** · 01:00 · en pantalla: «sin PyTorch ni TensorFlow» «capas · pérdidas · optimizadores» «Adam · momentum · SGD»

> El motor de redes neuronales también es nuestro, y eso nos dio control total sobre la memoria y la comunicación.

- No usamos PyTorch ni TensorFlow: la librería de redes neuronales la escribimos en Rust, sobre ndarray. Así se define un modelo para MNIST: una lista de capas, parecido al Sequential de Keras.
- **→** Tiene capas densas, convolucionales, max pooling y activaciones; pérdidas MSE y entropía cruzada; y optimizadores de descenso por gradiente, con momentum y Adam, detrás de un mismo trait.
- ¿Por qué no usar una existente? Porque necesitábamos que el modelo fuera un único buffer de memoria, para mandarlo por la red sin copias (apéndice A2), y que la comparación no dependiera de un framework ajeno.

**Pase:** «La pieza más artesanal fue la convolución.»

Respaldo: machine_learning/src/models.rs · docs/report/solucion.md (Motor de Redes Neuronales)

### 22:00 → 22:30 · 18 · Teorema de la convolución

**Alejo Ordoñez** · 00:30 · en pantalla: «convolución → producto ⊙»

> La FFT se probó por este resultado clásico: convierte la convolución en un producto, a cambio de pagar las transformadas.

- Probamos FFT por este resultado: la transformada de Fourier de una convolución es el producto de Hadamard, es decir, punto a punto, de las transformadas.
- **→** Entonces se transforman la entrada y el kernel, se multiplican punto a punto y se vuelve con la inversa.
- Si preguntan por el costo: con una imagen de n×n y un kernel de k×k, la directa cuesta n² por k², y con FFT, n² por log n. En teoría gana con kernels grandes; con los kernels chicos de nuestros modelos, no compensó.
- Si preguntan: las redes no invierten el kernel, es una correlación. El teorema vale igual con un conjugado, y como el kernel se aprende, no cambia nada.

**Pase:** «Veamos cómo la implementamos.»

Respaldo: docs/report/trabajos_futuros.md (variación del modelo de cómputo de la convolución) · Mathieu et al., 2013 · commit d880f65 (conv_fft de ndarray_conv)

### 22:30 → 23:45 · 19 · Implementación de la convolución

**Alejo Ordoñez** · 01:15 · en pantalla: «FFT: sin mejora» «im2col: ventana → columna» «una única multiplicación de matrices» «trabajo futuro: según el tamaño»

> La convolución la implementamos a mano y probamos tres variantes; quedó la que la convierte en una multiplicación de matrices.

- La convolución es la operación más cara de la red. A la izquierda, una entrada de 4×4 y un kernel de 3×3.
- **→** Primero la implementamos con transformadas rápidas de Fourier. Con nuestros datasets y modelos chicos, como los de MNIST, el costo de transformar superaba la ganancia.
- **→** Lo que quedó es im2col: cada ventana donde se apoya el kernel se copia como una columna de una matriz.
- **→** Así, toda la convolución es una sola multiplicación de matrices, el kernel como fila por esa matriz, y la multiplicación de matrices está muy optimizada. Después reordenamos im2col y col2im para aprovechar mejor la caché.
- **→** Con modelos, batches y kernels más grandes ese balance podría invertirse, así que queda como trabajo futuro elegir el algoritmo según el tamaño de las matrices.
- Si preguntan: también probamos paralelizarla entre hilos y no convino, porque con modelos chicos la sobrecarga superaba el beneficio.

**Pase:** Traspaso: «Ya vimos cómo está construido. ¿Cuánto esperamos que cueste sincronizar? Lorenzo.»

Respaldo: machine_learning/src/arch/layers/conv/conv2d.rs · docs/report/trabajos_futuros.md · commits d880f65 (FFT), a60fa08 y 93ae0c8 (im2col)

### 23:45 → 24:45 · 20 · Tiempo de un entrenamiento

**Lorenzo Minervino** · 01:00 · en pantalla: «el cómputo se reparte» «sincronización: todos esperan» «una ronda por epoch» «d_sync determina la escalabilidad»

> El tiempo total tiene una parte que se reparte entre workers y otra que se paga por sincronizar; la segunda decide si conviene sumar máquinas.

- Ya vimos cómo funciona cada estrategia y cómo está construido; ahora, cuánto cuesta sincronizar. Arriba está la fórmula del informe y la vamos a armar término por término. Primero, una epoch entera en una sola máquina: T_epoch.
- **→** Con N_W workers, cada uno procesa solo su partición: el cómputo de la epoch se divide, T_epoch/N_W. Esta es la parte que mejora al sumar máquinas.
- **→** Pero al final de cada epoch hay que sincronizar, y mientras tanto nadie avanza: eso es d_sync. Esta parte no se divide.
- **→** Se repite una vez por epoch, N_epochs veces, más un costo de arranque, d_init, que se paga una sola vez. Conclusión: si d_sync crece con la cantidad de workers, llega un punto en que sumar máquinas empeora.

**Pase:** «¿Y de qué depende d_sync? De cómo viajan los datos.»

Respaldo: docs/report/resultados.md (modelo del tiempo total de entrenamiento)

### 24:45 → 26:00 · 21 · Cálculo de d_sync

**Lorenzo Minervino** · 01:15 · en pantalla: «S: tamaño del modelo · R: ancho de banda» «PS: todo el tráfico por una placa» «crece con la cantidad de workers» «AR: transmisión simultánea» «acotado»

> Contar transferencias alcanza para derivar d_sync: en PS todo pasa por la placa del servidor; en el anillo, todos transmiten en paralelo.

- S es el tamaño del modelo, lo que pesa un gradiente, y R es el ancho de banda de cada placa de red. Mandar S por una placa tarda S/R. A la izquierda, Parameter Server con un servidor; a la derecha, el anillo.
- **→** En PS, los cuatro gradientes entran por la misma placa del servidor, uno detrás del otro, y después salen los cuatro modelos actualizados: 8 veces S/R. En general, 2·N_W·S/R, que crece linealmente con los workers. El servidor es el cuello de botella.
- **→** En el anillo, cada worker manda solo un fragmento de tamaño S/N_W por paso, y todos transmiten a la vez. Son 2(N_W − 1) pasos: con 4 workers, 6 pasos de S/4R, o sea 1,5·S/R. En general, 2(1 − 1/N_W)·S/R, que nunca supera 2S/R por más workers que sumemos.
- Supuestos, si preguntan: el mismo R en todos los nodos, placas full-duplex y latencias de cola despreciables.

**Pase:** «Si lo graficamos en función de los workers…»

Respaldo: docs/report/resultados.md (derivación de d_sync)

### 26:00 → 26:45 · 22 · Costo de sincronización

**Lorenzo Minervino** · 00:45 · en pantalla: «PS: crecimiento lineal» «cuello de botella: el servidor» «AR: acotado por 2S/R»

> El modelo predice que, con un servidor, PS empeora al sumar workers, mientras que All-Reduce queda acotado.

- Las dos fórmulas, graficadas con los valores de nuestra campaña: S/R ≈ 0,1 s (LeNet-5 en f16 a 10 Mbps). En el eje X, los workers; en el Y, d_sync por ronda.
- **→** PS con un servidor crece en línea recta: con 16 workers son más de 3 segundos por ronda solo en sincronizar.
- **→** All-Reduce se aplana en 2S/R = 0,2 s. Esto es lo que predice el modelo; en un rato vamos a ver que lo medido coincide.

**Pase:** «Para medirlo, armamos los experimentos desde Python.»

Respaldo: docs/report/resultados.md (derivación de d_sync) · experimentacion.md (S, R)

### 26:45 → 27:30 · 23 · Interfaz de Python

**Lorenzo Minervino** · 00:45 · en pantalla: «PyO3 + maturin» «liberación del GIL» «suite de benchmarks en Python»

> orchestra-py expone el mismo orchestrator desde Python; sobre ella escribimos toda la suite de benchmarks.

- Además de la interfaz de terminal, hay una biblioteca de Python con la misma funcionalidad. Este fragmento es la API real: se arma un entrenamiento con Strategy-Switch y un servidor, y se lanza con `orchestrate`.
- Es un módulo nativo hecho con PyO3 y maturin, y libera el GIL mientras entrena. La suite de benchmarks de la campaña está escrita sobre esta interfaz.
- Si preguntan por qué es un crate aparte: anotar los tipos existentes con #[pyclass] chocaba con genéricos, variantes de enum y Clone.

**Pase:** «Con esto armamos los experimentos.»

Respaldo: docs/report/solucion.md (Interfaz de Python como crate separado) · orchestra-py/local.py

### 27:30 → 28:30 · 24 · Metodología de medición

**Lorenzo Minervino** · 01:00 · en pantalla: «MNIST · LeNet-5» «1 contenedor = 1 nodo» «de 2 a 16 workers» «red limitada deliberadamente»

> Fijamos todo salvo la estrategia, y forzamos una red lenta para que las diferencias se vean.

- Un clúster simulado en una sola máquina física: un contenedor de Docker por nodo, con un núcleo cada uno. Entrenamos LeNet-5 (61.706 parámetros) sobre MNIST completo, con Adam, hasta 48 epochs.
- **→** Variamos la cantidad de workers: 2, 4, 6, 8, 12 y 16. PS con un solo servidor.
- **→** Limitamos la salida de cada contenedor a 10 Mbps con Pumba. ¿Por qué? Con un modelo chico, la red real es tan rápida que no se distinguen las estrategias. Bajando R, la transmisión queda en ≈ 0,1 s por ronda. Como referencia, en una sola máquina tarda 600 s.
- Antes de medir tiempos validamos la corrección con pruebas unitarias, de integración y de aceptación en CI: la convergencia distribuida coincide con la secuencial.

**Pase:** «Primero, los tiempos de los algoritmos sincrónicos.»

Respaldo: docs/report/experimentacion.md (Pruebas automatizadas · Entorno de experimentación)

### 28:30 → 29:30 · 25 · Resultados: sincrónicos

**Lorenzo Minervino** · 01:00 · en pantalla: «PS: degradación desde 6 workers» «AR: sin mejora desde 6 workers» «consistente con el modelo»

> Lo medido coincide con lo que predice el modelo analítico.

- El tiempo total de entrenamiento según la cantidad de workers. En el eje Y, el tiempo total en segundos; en el X, la cantidad de workers.
- **→** Parameter Server con un servidor tiene su mínimo en 6 workers y desde ahí empeora: el servidor concentra todo el tráfico.
- **→** All-Reduce no se degrada, pero deja de mejorar: el cómputo por worker baja y la sincronización se queda en 2S/R. Es lo que predijo el modelo de la diapositiva 22.
- Lectura aproximada, por si preguntan: AR cerca de 290 s desde 6 workers; PS cerca de 340 s en 6 y algo más de 400 s en 16.

**Pase:** «¿Y si sacamos la barrera?»

Respaldo: docs/report/resultados.md (Resultados obtenidos) · figura execution_time_per_nodes_ar_vs_ps.png

### 29:30 → 30:00 · 26 · Resultados: asincrónicos

**Lorenzo Minervino** · 00:30 · en pantalla: «el tiempo disminuye con más workers» «¿converge el modelo?»

> En velocidad pura, los dos asincrónicos ganan.

- Con PS asincrónico y Strategy-Switch, el tiempo sigue bajando al sumar workers, a diferencia de los sincrónicos.
- **→** Pero acelerar no sirve de nada si el modelo no aprende. Hay que mirar la accuracy.

**Pase:** «Veamos la accuracy final.»

Respaldo: docs/report/resultados.md (Resultados obtenidos) · figura execution_times_async_parameter_server_and_strategy_switch.png

### 30:00 → 31:15 · 27 · Resultados: accuracy

**Lorenzo Minervino** · 01:15 · en pantalla: «Strategy-Switch ≈ sincrónicos» «PS asincrónico: colapso de accuracy»

> Strategy-Switch fue el mejor balance de esta campaña; la salvedad sobre WildStore es parte del resultado.

- La accuracy final sobre el conjunto de prueba de MNIST. All-Reduce, PS sincrónico y Strategy-Switch se superponen arriba.
- **→** PS asincrónico cae al sumar workers, hasta valores cercanos al azar desde 8 workers: es el client drift en acción. Strategy-Switch, en cambio, conserva la accuracy de los sincrónicos con tiempos cercanos a los del asincrónico.
- La salvedad, dicha por nosotros antes de que pregunten: PS asincrónico corrió con WildStore, que hace escrituras sin protección al estilo Hogwild!. Con estos datos no podemos separar el drift del efecto de esas escrituras.
- Alcance: estas conclusiones valen para MNIST, LeNet-5 y este entorno.

**Pase:** «¿Qué queda por hacer?»

Respaldo: docs/report/resultados.md (salvedad final) · conclusiones_capitulo4.md · riesgos.md

### 31:15 → 32:00 · 28 · Trabajos futuros

**Lorenzo Minervino** · 00:45 · en pantalla: tolerancia a fallos · convolución según el tamaño · nuevas arquitecturas

> El sistema quedó listo para retomarse, y varias líneas ya tienen el diseño especificado.

- Las líneas abiertas, ordenadas por relevancia. Varias ya tienen el diseño especificado en el repositorio.
- **→** Tolerancia a fallos: es imprescindible para escalar, y el comportamiento por entidad ya está especificado.
- **→** Elegir el algoritmo de convolución según el tamaño de las matrices, como contamos antes.
- **→** Ampliar el motor a otras arquitecturas.

**Pase:** «Y ahora, vamos a verlo funcionando.»

Respaldo: docs/report/trabajos_futuros.md

### 32:00 → 32:30 · 29 · Demostración

**Lorenzo Minervino** · 00:30 · en pantalla: «3 computadoras» «Strategy-Switch · MNIST» «≈ 10 min de entrenamiento» «luego: el modelo, en el navegador»

> Lanzar el entrenamiento real en tres computadoras y aprovechar los diez minutos para contar el apéndice.

- Ahora lo vamos a ver funcionando: tres computadoras, cada una con un nodo, y el orchestrator en la TUI. Lanzamos Strategy-Switch sobre MNIST.
- Tarda unos diez minutos. Mientras entrena contamos decisiones de diseño del apéndice y cada tanto miramos la TUI: el anillo, el momento del switch y la topología en estrella.
- Cuando termine, guardamos el modelo y lo cargamos en una página web para probarlo dibujando números.
- Si preguntan por qué siempre cambia de estrategia: las etiquetas de entrenamiento están suavizadas (ε = 0,1), así la pérdida tiene un piso y se estabiliza entre las epochs 14 y 16. La accuracy se mide contra las etiquetas originales.
- Los pasos exactos están en presentation/GUION_DEMO.md.

**Pase:** Traspaso: «Mientras entrena, Alejo cuenta por qué tuvimos que construir una base común.»

Respaldo: presentation/GUION_DEMO.md · demo/

### 32:30 → 33:45 · A1 · A1 · Por qué una base común

**Alejo Ordoñez** · 01:15 · en pantalla: —

> Por qué hizo falta construir una base común en lugar de comparar sobre TensorFlow o PyTorch.

- Esas implementaciones están optimizadas para producción y no aplican el mismo trabajo de bajo nivel a cada estrategia (solapamiento de cómputo y comunicación, kernels especializados). Al comparar dos estrategias sobre dos implementaciones, la diferencia mezcla algoritmo y calidad de implementación, y ese sesgo no se puede cuantificar. A eso se suman las decisiones que el framework toma por dentro y no se pueden configurar.

**Pase:** «Una de las decisiones de esa base común es cómo se guarda el modelo en memoria.»

Respaldo: docs/report/problema.md · estado_del_arte.md

### 33:45 → 35:00 · A2 · A2 · Modelo como buffer plano

**Alejo Ordoñez** · 01:15 · en pantalla: «capas = vistas» «particionar = delimitar rangos» «enviar sin copiar»

> Una decisión del motor de redes neuronales habilita la comunicación sin copias de todo el sistema.

- La biblioteca está construida sobre ndarray: capas densas, convolucionales, max pooling, reshape y activaciones; inicializaciones Xavier-Glorot y Kaiming-He; optimizadores detrás de un mismo trait.
- La decisión clave es que el modelo sea un buffer contiguo y las capas, vistas sobre tramos. La alternativa, un árbol de objetos con sus pesos, obligaba a serializar cada capa y, en Rust, a acoplar el motor con comms para prestar referencias mutables.
- **→** Particionar entre servidores es delimitar rangos. **→** Enviar es reinterpretar memoria (bytemuck).
- Limitación: los modelos son secuencias de capas; arquitecturas con conexiones de salto, como U-Net, no se pueden expresar hoy.

**Pase:** Traspaso: «Para mover menos datos por la red, Marcos cuenta la compresión.»

Respaldo: docs/report/solucion.md (Motor de Redes Neuronales · Modelo como buffer plano)

### 35:00 → 36:15 · A3 · A3 · Compresión con error feedback

**Marcos Bianchi Fernández** · 01:15 · en pantalla: «control: JSON» «datos: f16, sin copias» «envío de valores sobre el umbral» «el resto se acumula en el residual» «error feedback»

> Separar control de datos y comprimir sin perder la información acumulada.

- Control: comandos poco frecuentes en JSON, inspeccionables. Datos: concentran el volumen y viajan por reinterpretación de memoria, en media precisión.
- **→** Solo viajan los componentes por encima de un umbral, elegido para que pase una fracción r; el umbral se estima muestreando hasta 16384 componentes con quickselect (Aji y Heafield, 2017).
- **→** Lo que no se envía queda en un residual persistente.
- **→** En la ronda siguiente el residual se suma al gradiente nuevo: un componente chico pero constante termina cruzando el umbral (error feedback, Lin et al., 2018). Si la codificación dispersa pesa más que la densa en f16, se envía la densa.
- Limitación: no se midió de forma aislada el compromiso entre tasa de compresión y convergencia.

**Pase:** «Otra decisión: el orden del anillo y dónde van los servidores.»

Respaldo: docs/report/solucion.md (Separación de planos · Compresión de gradientes con error feedback)

### 36:15 → 37:30 · A4 · A4 · Topología por latencia

**Marcos Bianchi Fernández** · 01:15 · en pantalla: «10 pings por par» «anillo: viajante, solución exacta» «servidor: menor latencia máxima» «optimizador después de agregar»

> Ninguna diferencia de tiempo debería venir del orden en que el usuario escribió las direcciones.

- Antes de entrenar, cada par de nodos intercambia diez pings y se toma el mayor RTT. Se mide una sola vez (limitación en redes con carga variable).
- **→** All-Reduce: el orden del anillo que minimiza la latencia total es un circuito hamiltoniano de costo mínimo, resuelto en forma exacta con programación dinámica sobre subconjuntos (Held y Karp, 1962).
- **→** Parameter Server: se eligen los k servidores que minimizan la mayor latencia worker–servidor, por backtracking. Y el optimizador con estado se aplica después de agregar: si cada worker aplicara Adam sobre su gradiente local, acumularían momentos distintos y las réplicas divergirían.

**Pase:** Traspaso: «Lorenzo cuenta por qué elegimos Rust.»

Respaldo: docs/report/solucion.md (Selección de topología · Ubicación del optimizador) · código: orchestrator/src/calculator/

### 37:30 → 38:30 · A5 · A5 · Elección de Rust

**Lorenzo Minervino** · 01:00 · en pantalla: —

> Elegimos Rust por el control de bajo nivel con garantías de seguridad.

- Todo O.N.O. está escrito en Rust, incluida la biblioteca de redes neuronales.
- **→** Buscábamos un lenguaje de systems programming, con control de bajo nivel como C o C++, y sin garbage collector: sus pausas no se pueden predecir y frenan el cómputo y la red.
- **→** Y el compilador verifica dos cosas antes de ejecutar: la seguridad de memoria y que no haya data races sobre memoria compartida (*fearless concurrency*). Ninguno de esos errores llegó a la rama principal.
- Si preguntan por las dificultades: el compilador no verifica la coordinación distribuida. Esa fue la dificultad real, sobre todo en All-Reduce.
- Si preguntan por `unsafe`: está acotado y marcado, por ejemplo en WildStore (Hogwild!) y en la codificación de mensajes.

**Pase:** «Lo que más aprendimos en el camino.»

Respaldo: docs/report/herramientas.md (Rust) · riesgos.md (complejidad de la implementación distribuida en Rust)

### 38:30 → 40:00 · A6 · A6 · Lecciones: errores silenciosos

**Lorenzo Minervino** · 01:30 · en pantalla: —

> La lección central de ingeniería del trabajo.

- Conviene tener una medición end-to-end desde temprano. Otra lección: «make it work, make it right, make it fast».

**Pase:** «Veamos cómo terminó el entrenamiento.»

Respaldo: docs/report/riesgos.md (Lecciones aprendidas)

### 40:00 → 40:15 · 30 · ¿Y la demo?

**Lorenzo Minervino** · 00:15 · en pantalla: —

> Volver al entrenamiento que lanzamos y ver qué aprendió el modelo.

- Volvemos al entrenamiento que lanzamos. Cuando la TUI diga FINISHED, guardamos el modelo y lo cargamos en la página, que mide su accuracy y permite dibujar números.
- Los pasos exactos están en presentation/demo/INSTRUCCIONES.md.

**Pase:** «Con esto terminamos. ¿Preguntas?»

Respaldo: presentation/demo/INSTRUCCIONES.md · presentation/GUION_DEMO.md

### 42:15 → 42:45 · 31 · ¿Preguntas?

**Los tres** · 00:30 · en pantalla: —

> Abrir la ronda de preguntas.

- Volver de la página de la demo y abrir las preguntas.
- Preguntas probables: WildStore o BlockingStore; por qué 10 Mbps; por qué la pérdida de entrenamiento y no la de validación; tolerancia a fallos; uso de herramientas de IA, declarado en el informe.
- Para volver a una diapositiva: tecla G (índice) o número + Enter. El apéndice va del 33 (A1) al 41 (A9).

**Pase:** «¡Gracias por venir!»

Respaldo: docs/report/conclusiones.md

### 42:45 → 43:15 · 32 · ¡Gracias por venir!

**Los tres** · 00:30 · en pantalla: —

> Cerrar.

- Agradecer al jurado, al tutor y al co-tutor. El repositorio es público.

**Pase:** —

Respaldo: docs/report/agradecimientos.md

## Respaldo (si falla la demo o para preguntas)

### A7 · A7 · Grabación de Strategy-Switch

> Respaldo si falla la demo: el cambio de estrategia grabado en una ejecución anterior.

- Secuencia real de pantallas (4 nodos locales, subconjunto de 10.000 muestras de MNIST, configs en presentation/captures/configs/). Mostrar el anillo, el instante «switching → PS» y la topología en estrella posterior. Si se usa, aclarar que es una grabación.

Respaldo: presentation/captures/rec/ · presentation/captures/configs/training_strategy_switch.json

### A8 · A8 · Capturas de la TUI

> Capturas de respaldo de las pantallas principales.

- Menú, ayuda de configuración y dashboard al finalizar una ejecución de Strategy-Switch (pérdida promedio, pérdida por worker, tabla de workers, parámetros finales y eventos).
- Ojo: en esa ejecución la tabla muestra 99–100 epochs con max_epochs = 60; ver README, sección «Diferencias detectadas».

Respaldo: presentation/captures/raw/

### A9 · A9 · Topologías en la TUI

> La topología que dibuja la TUI es la que construyó el orchestrator, y cambia en vivo cuando Strategy-Switch promueve un nodo.

- orchestui es la interfaz de terminal que embebe al orchestrator. Son capturas reales de una ejecución local con 4 nodos, fuera de la campaña experimental. Acá, el anillo de All-Reduce con la pérdida y la epoch de cada worker.
- **→** La misma vista con Parameter Server: los workers a la izquierda y el servidor a la derecha.
- **→** Strategy-Switch en el instante del cambio: el encabezado dice «switching W3 → PS» y la topología pasa de anillo a estrella.

Respaldo: orchestui/src/ui/components/topology.rs · presentation/captures/raw/
