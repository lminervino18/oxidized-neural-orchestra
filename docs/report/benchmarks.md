\newpage
# NOTAS
- TODO, algo como: La penalización por no sincronizar frecuentemente se incrementa con la varianza del dataset. Cada nodo va a recibir una partición del dataset cuya superficie de error va a ser muy distinta a la de los demás.
- TODO: gráficos efficiency o speedup, time spent, después otro copado podría ser parameter server y gráfico de tiempo vs. cantidad de workers, con una curva base q es la cantidad con bandwidth infinita y la otra con bandwidth limitada y comparando con ambos resultados teóricos.
- TODO: quizás gráficos que muestren tiempo con workers fijos vs d_trans, o sea S/R, y por supuesto afectando únicamente a R porque no voy a tocar el modelo.

# Benchmarks
El objetivo principal de los algoritmos que se implementan en este trabajo es el de reducir el tiempo de entrenamiento de un modelo de deep learning $T_1$.  
$T_1$, a su vez, está dado por la suma de $T_\text{ser}$ y $T_\text{par}$, los tiempos de procesamiento de las porciones de trabajo serializado y paralelizable, respectivamente. Dado que $T_\text{ser}\ll T_\text{par}$, pues, para los algoritmos que se presentan, el trabajo serializado consiste en compartir el dataset y la metadata del modelo entre los nodos, y que además $T_{ser}$ no cambia entre los algoritmos, podemos despreciar $T_\text{ser}$ y hablar de $T_\text{par}$ como $T_1$.  
En un mundo perfecto, donde el costo de sincronización de los nodos es despreciable, el tiempo de entrenamiento es inversamente proporcional a la cantidad de nodos $P$:
$$
T_P = \frac{T_1}{P}.
$$
Ahora bien, el mundo no es perfecto y el costo de sincronización no es despreciable; más aún, veremos a continuación que en algunos casos puede incluso crecer con la cantidad de máquinas que deben sincronizarse.

A continuación, se realiza un análisis de la reducción del tiempo de entrenamiento esperada para cada algoritmo y se la compara con la que fue obtenida con ONO.  

Sin perder generalidad y conservando dinamismo, se opta por mostrar los resultados contra el dataset MNIST. Como veremos a continuación, un factor clave en el delay de sincronización que introducen los algoritmos de entrenamiento distribuido es el tamaño del modelo $S$ con respecto al bandwidth $R$ de la red, es decir, el delay de transmisión $d_\text{trans}$ de la red.  
Dado que el tamaño de los modelos que resultan útiles para aprender MNIST no es lo suficientemente grande para obtener diferencias observables entre los algoritmos, para tener una relación $S/R$ buena para evaluación, se elige un $R$ subrealistamente bajo para evitar usar modelos innecesariamente grandes para MNIST o cambiar el problema a uno que así los requiera.

## Parameter server sincrónico con un sólo servidor
En una configuración con un sólo server, el delay que introduce la sincronización del sistema está dado por el tiempo de comunicación de la ida de los parámetros del server hacia los workers, y de la vuelta de los gradientes de los workers hacia el server.  

Suponiendo un mismo bandwidth $R$ para todos los nodos, delays de encolado, propagación y procesamiento despreciables; y un modelo de tamaño $S$ ($S=N_\text{params} \times 2 \text{bytes}$, dado que los parámetros se serializan como flotantes de 16 bits); el tiempo que tarda la ida de los parámetros es el su envío en $N_W$ mensajes unicast $N_W \frac{S}{R}$.  
Suponiendo un ambiente homogéneo, en el que el tiempo de procesamiento de los workers es similar, todos comienzan a contestar con los respectivos gradientes en simultáneo. Sin embargo, el server sólo tiene una tarjeta de red con la cuál recibir estos gradientes, por lo que el tiempo que tarda la vuelta de los gradientes es también $N_W \frac{S}{R}$.  

El delay de sincronización de parameter server sincrónico con un sólo servidor es entonces
$$
d_\text{sync, PS} = 2 N_W \frac{S}{R}.
$$
Nótese que el delay crece con la cantidad de nodos del sistema.

Cabe destacar que generalmente uno querría que el nodo que ejecuta el parameter server ejecute también un worker, dado que de otra forma se estaría desperdiciando mucho cómputo siendo que esta entidad es I/O bounded. En cuyo caso el delay de comunicación entre el parameter server y el worker que residen en el mismo nodo se anula y por tanto
$$
d_\text{sync, PS} = 2 (N_W-1) \frac{S}{R}.
$$

## Ring all-reduce
El delay de sincronización de ring all-reduce está dado por el delay del *scatter* y por el delay del *gather*. Ambas etapas del algoritmo son de $N_W-1$ iteraciones. En ambas, la sincronización en una iteración consiste en enviar una $N_W$-ésima parte del gradiente del $i$-ésimo al $i+1$-ésimo nodo. Como despreciamos los delays de encolado, propagación y procesamiento; y como las tarjetas de red son *full-duplex* (tienen canales dedicados para leer y escribir), podemos pensar que la comunicación en anillo se realiza en paralelo.

Considerando los mismos supuestos de delays de comunicación por red que usamos para el desarrollo anterior, el delay de sincronización de all-reduce, está dado por hacer dar vuelta una $N_W$-ésima parte del gradiente dos veces. Esto es
$$
\begin{aligned}
d_\text{sync, AR} &= 2(N_W - 1) \frac{S/N_W}{R}\\
&=2(1-\frac{1}{N_W}) \frac{S}{R}.
\end{aligned}
$$

A diferencia de parameter server, el factor afectado por la cantidad de workers $\frac{1}{N_W}\rightarrow 0$ cuando $N_W\rightarrow \infty$,
$$
\lim_{N_W\rightarrow \infty} d_\text{sync, AR} = 2 \frac{S}{R}.
$$
El delay de sincronización está acotado por el doble del delay de transmición del tamaño de los parámetros del modelo.

## Parameter server sincrónico con $N_S$ servers
Como ya vimos, la mayor parte del delay de sincronización en parameter server viene del hecho de que que todos los workers tienen que comunicarse con un único server, y dado que este server puede enviar y recibir una sóla tira de parámetros/gradientes a la vez, el delay escala con $N_W$.  
Una forma de resolver este problema es incrementar la cantidad de servidores. Cada servidor tiene una porción disjunta de los parámetros del modelo y los workers solicitan parámetros y envían gradientes al server que corresponda.

En parameter server, por tratarse de comunicación de uno a muchos, el cuello de botella de la sincronización está del lado del servidor. Con un sólo servidor ya vimos que el envío de parámetros y la recepción de gradientes ambos tardan $N_W \frac{S}{R}$, que es el máximo entre el delay desde el punto de vista de un worker y desde el del server: $\max{\left(\frac{S}{R}, N_W \frac{S}{R}\right)}$. Pero con $N_S$ servers, el delay del lado del servidor se transforma en $N_W \frac{S/N_S}{R}$. Por lo tanto
$$
d_\text{sync, Multi-PS} = 2 \frac{N_W}{N_S} \frac{S}{R} \qquad (N_W\leq N_S).
$$

Vale la misma aclaración que hicimos en el caso para un solo servidor con respecto a que los nodos que ejecutan un server normalmente van a estar también ejecutando un worker, por lo que los delays de transmisión se anulan en esos nodos
$$
d_\text{sync, Multi-PS} = 2 \frac{N_W-1}{N_S} \frac{S}{R}.
$$

Notar que $\lim_{N_S\rightarrow \infty} d_\text{sync, Multi-PS} = 2 \frac{S}{R}$, que es el mismo resultado obtenido que para all-reduce.

## Parameter server asincrónico
El delay de sincronización desarollado en los puntos anteriores se da luego de cada ronda de epochs de *todos* los workers. Esto es, cada worker computa una epoch (o más, si `offline_epochs` está configurado) para luego sincronizar sus parámetros con el resto.  

Parameter server asincrónico libera al entrenamiento distribuido de esta restricción, dejando que el parameter server mueva los parámetros del modelo a medida que van llegando gradientes de los workers *sin necesidad de esperar al resto*. En este caso el costo de sincronización se elimina porque la sincronización simplemente desaparece, aunque aún se tiene que considerar el delay de transmición $2\frac{S}{R}$ de cada worker.  
El problema con esta estrategia es que la convergencia empeora mucho cuando la desincronización es grande, es decir, cuando los workers mueven demasiado los pesos en la dirección sesgada que les dicta su partición de los datos.

## Strategy switch
Strategy switch combina la convergencia de all-reduce y la velocidad de ejecución de parameter server asincrónico. La idea es encontrar un buen mínimo local con el primer algoritmo para luego seguir bajándolo por medio del segundo.  
El delay de sincronización de strategy switch depende entonces de si se realizo el *switch*:
$$
d_\text{sync, SS} =
\begin{cases}
2(1-\frac{1}{N_W}) \frac{S}{R}, & \text{antes del switch (All-reduce)}\\
-, & \text{después (Parameter server asincrónico)}.
\end{cases}
$$
TODO: definir bien d_sync para poder poner 0 en vez de - ?
