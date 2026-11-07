# Guion de la demo

Tres computadoras entrenan en vivo LeNet-5 sobre MNIST con Strategy-Switch, durante unos diez minutos. Mientras entrena se cuenta el apéndice A1–A6. Al terminar, el modelo se guarda y se carga en una página web que mide su exactitud sobre las 10.000 imágenes de prueba de MNIST y permite dibujar números con el mouse para que el modelo los adivine.

## Quién hace qué

| Diapositiva | Tiempo | Qué pasa | Quién |
|---|---|---|---|
| 29 · Ahora, la demo | 0:30 | Lanza el entrenamiento en la TUI y muestra el anillo de All-Reduce | Lorenzo |
| A1–A2 | 2:30 | Base común, buffer plano | Alejo |
| A3–A4 | 2:30 | Compresión y topología | Marcos |
| A5–A6 | 2:30 | ¿Por qué Rust?, lecciones | Lorenzo |
| 30 · ¿Y la demo? | 2:15 | Guarda el modelo y muestra la página | Lorenzo |
| 31 · ¿Preguntas? | — | Abre las preguntas | Los tres |

Entre bloque y bloque, quien opera la TUI la muestra unos segundos con la tecla `v`: primero el anillo de All-Reduce, después el momento «switching → PS» y al final la estrella de Parameter Server.

## Preparación, antes del día

### Las tres computadoras

- Ponerlas en la misma red local, idealmente con un router o switch propio y cable. La red de la facultad puede bloquear las conexiones entre máquinas.
- Anotar la IP de cada una: `ip -4 addr` en Linux, `ipconfig` en Windows.
- Abrir el puerto 40000 en el firewall de las tres. Con ufw: `sudo ufw allow 40000/tcp`.
- Usar el mismo commit del repositorio en las tres y compilar desde la raíz:
  - en las tres: `cargo build --release -p node`;
  - además, en la que va a correr la TUI: `cargo build --release -p orchestui`.

### El dataset, solo en la computadora de la TUI

El orchestrator lee MNIST y les manda a los nodos su partición, así que el dataset hace falta solo en la computadora de la TUI. La carpeta `datasets/` no está en git. Hay dos formas de conseguirlo:

- copiar `datasets/mnist/mnist_train_samples.bin` (179 MB) y `datasets/mnist/mnist_train_labels.bin` desde una computadora que ya los tenga;
- o generarlos desde la raíz del repositorio, con conexión a internet:
  ```bash
  python3 -c "import sys; sys.path.insert(0, 'benchmarks'); from issue.runner import prepare_dataset; prepare_dataset()"
  mkdir -p datasets/mnist && cp notebooks/mnist/data/mnist_*.bin datasets/mnist/
  ```

Después, una sola vez, generar las etiquetas suavizadas que usa la demo (ver «Por qué cambia siempre»):

```bash
python3 presentation/tools/smooth_labels.py datasets/mnist/mnist_train_labels.bin datasets/mnist/mnist_train_labels_smooth.bin 0.1
```

### Las configs

En `presentation/demo/configs/`:

- `model.json`: LeNet-5, 61.706 parámetros.
- `training.json`: Strategy-Switch con los tres nodos, descenso por gradiente (lr 0,2), batch 64, 47 épocas en total y semilla 42. **Hay que reemplazar las tres IPs de `addrs`** por las de las computadoras; el puerto es 40000 en las tres.
- `training.local.json`: la misma configuración con los tres nodos en una sola computadora (`127.0.0.1`), para ensayar.

### Ensayo y ajuste del tiempo

La calibración se hizo en una computadora de 8 núcleos, con cada nodo limitado a 2 núcleos para simular tres máquinas y unas 80 épocas en total: el switch llegó entre las épocas 14 y 16 (≈ 1:30) y el entrenamiento terminó entre 9:15 y 10:30. La configuración actual tiene 47 épocas en total y todavía no se midió completa. En las computadoras de la defensa los tiempos van a cambiar, así que hay que hacer un ensayo completo con las tres y ajustar `max_epochs`:

- `max_epochs` es el total de épocas del entrenamiento: al llegar el switch, el orchestrator les manda a los workers solo las que quedan, así que el contador de la TUI no pasa del 100 %.
- Medir en la TUI cuánto tarda una época después del switch y despejar: `max_epochs ≈ épocas hasta el switch + (600 s − tiempo hasta el switch) / segundos por época`.
- No bajar `max_epochs` de 20: el switch llega entre las épocas 14 y 16, y si no queda margen después, el entrenamiento termina sin cambiar de estrategia.

### Por qué cambia siempre

Strategy-Switch pasa a Parameter Server cuando la pérdida se estabiliza: la variación relativa media en una ventana de 6 épocas baja del 1 % (criterio de Provatas et al.). Con un entrenamiento sano la pérdida sigue bajando y ese momento puede no llegar en diez minutos: en las calibraciones sin suavizado no cambió en ninguna de las tres corridas de 40 épocas.

Por eso las etiquetas de entrenamiento de la demo están suavizadas (ε = 0,1; Szegedy et al., 2016): la clase correcta vale 0,91 y las demás 0,01. Así la pérdida tiene un piso mayor que cero y, cuando el modelo ya aprendió, se estabiliza entre las épocas 14 y 16. La fase de All-Reduce es sincrónica y usa la semilla de `training.json`, así que la época del switch prácticamente no depende de la velocidad de las computadoras. La exactitud de la página se mide contra las etiquetas originales. Un efecto visible: en la página, la probabilidad del dígito elegido ronda el 92 % y no llega al 100 %, porque el modelo aprendió que la clase correcta vale 0,91. Si el jurado pregunta, se explica así.

### La página

`presentation/demo/index.html` se abre con doble clic. No necesita servidor ni internet, y funciona en Chrome, Edge o Firefox actualizados. La arquitectura y las 10.000 imágenes de prueba están en `presentation/demo/data.js`. Si cambian `model.json`, hay que regenerarlo desde `presentation/`:

```bash
python3 tools/build_demo_data.py ../datasets/mnist
```

El argumento es la carpeta que tiene `mnist_test_samples.bin` y `mnist_test_labels.bin`. La página también acepta que se suelte un `model.json` junto con el `.safetensors`.

## Paso a paso en la defensa

1. **Antes de empezar la presentación**, en cada una de las tres computadoras, desde la raíz del repositorio, dejar corriendo:
   ```bash
   PORT=40000 RUST_LOG=warn ./target/release/node
   ```
2. **En la diapositiva 29**, en la computadora de la TUI, desde la raíz del repositorio:
   ```bash
   ./target/release/orchestui
   ```
   Elegir *Start Training*. Como modelo, `presentation/demo/configs/model.json`, y como entrenamiento, `presentation/demo/configs/training.json`.
3. Mostrar el dashboard y, con `v`, la topología: el anillo de All-Reduce. Avisar que en unos minutos va a cambiar de estrategia.
4. Volver a las diapositivas y contar A1 → A6. Entre bloques, mirar la TUI.
5. Cuando el encabezado diga **FINISHED**, apretar `s`, escribir `demo.safetensors` y Enter. El log confirma con «model saved to demo.safetensors».
6. Abrir la página y arrastrar `demo.safetensors`. Recorre las 10.000 imágenes de prueba y muestra la exactitud en grande.
7. Dibujar dos o tres números con el mouse. Si un dibujo sale mal, usar «una imagen del test, al azar».
8. Volver a la presentación, que quedó en ¿Y la demo?, y apretar → para ir a ¿Preguntas?

## Qué decir

- **Al lanzar:** «Son tres computadoras, cada una con un nodo, y el orchestrator en la TUI. Arranca en All-Reduce: los tres en anillo.»
- **Al ver el switch, cerca del minuto 1:30:** «Acá la pérdida se estabilizó, el orchestrator mandó Upgrade a un nodo y Switch al resto, y siguió en Parameter Server sin reiniciar nada.»
- **En la página:** «Este archivo es el que acaba de salir de las tres computadoras. Lo corremos en el navegador sobre las 10.000 imágenes de prueba de MNIST, que el modelo no vio al entrenar.»
- **Al dibujar:** «El dibujo se achica a 28×28 y se centra como las imágenes de MNIST: esa es la imagen que ve el modelo.»

## Si algo falla

- **Un nodo no conecta:** revisar la IP en `training.json`, el puerto y el firewall, y reiniciar ese nodo. En la TUI, `q` vuelve al menú para relanzar.
- **No da el tiempo o el entrenamiento se corta:** cargar en la página `presentation/demo/respaldo.safetensors`, un modelo entrenado en un ensayo, y decir que es de un ensayo. Para mostrar la TUI quedan A7 (grabación), A8 (capturas) y A9 (topologías).
- **La página no muestra nada:** `data.js` tiene que estar en la misma carpeta que `index.html`.

## Calibración realizada

Diez entrenamientos completos con `training.local.json`, uno por semilla y con `max_epochs` en 65, que entonces se contaba después del switch, en una computadora de 8 núcleos con cada nodo limitado a 2. Cambió de estrategia en los diez. La exactitud se mide sobre las 10.000 imágenes de prueba, contra las etiquetas originales.

| Semilla | Switch (época) | Switch (tiempo) | Épocas totales | Duración | Exactitud |
|---|---|---|---|---|---|
| 42 | 16 | 1:30 | 81 | 9:45 | 99,03 % |
| 1 | 15 | 1:30 | 80 | 9:45 | 98,90 % |
| 2 | 16 | 1:45 | 80 | 10:30 | 98,89 % |
| 3 | 16 | 1:30 | 80 | 9:15 | 98,92 % |
| 4 | 16 | 1:30 | 81 | 9:30 | 98,96 % |
| 5 | 16 | 1:30 | 81 | 9:30 | 98,93 % |
| 6 | 14 | 1:30 | 79 | 9:30 | 98,85 % |
| 7 | 16 | 1:30 | 80 | 9:16 | 98,84 % |
| 8 | 15 | 1:30 | 79 | 9:15 | 98,97 % |
| 9 | 16 | 1:30 | 81 | 9:45 | 98,93 % |

El tiempo del switch tiene una resolución de 15 segundos, que es cada cuánto se registró el progreso. `respaldo.safetensors` es el modelo de la semilla 42.
