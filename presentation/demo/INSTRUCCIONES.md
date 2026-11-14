# Cómo correr la demo

Tres computadoras en la misma red entrenan LeNet-5 sobre MNIST con Strategy-Switch. Al terminar, el modelo se carga en una página web que mide su exactitud y permite dibujar números.

| Quién | Qué corre |
|---|---|
| Ale | Un nodo, el orchestrator (TUI) y la página |
| Loren | Un nodo |
| Marcos | Un nodo |

Todos los comandos se corren desde la raíz del repositorio.

## Qué se usa

Todo esto hace falta solo en la computadora de Ale. El orchestrator lee el dataset y les manda su partición a los nodos.

| Qué | Archivo |
|---|---|
| Modelo (LeNet-5, 61.706 parámetros) | `presentation/demo/configs/model.json` |
| Entrenamiento (Strategy-Switch, 3 nodos, batch 64, lr 0,2) | `presentation/demo/configs/training.json` |
| Dataset, imágenes | `datasets/mnist/mnist_train_samples.bin` |
| Dataset, etiquetas suavizadas | `datasets/mnist/mnist_train_labels_smooth.bin` |
| Página | `presentation/demo/index.html` |
| Modelo de respaldo | `presentation/demo/respaldo.safetensors` |

La carpeta `datasets/` no está en git: se genera en el paso 1.

## 1. Preparación

Se hace una sola vez.

**Ale, Loren y Marcos.** Si no tienen el repositorio:

```bash
git clone https://github.com/lminervino18/oxidized-neural-orchestra.git
cd oxidized-neural-orchestra
git checkout feat/defense-presentation
cargo build --release -p node
```

Si ya lo tienen:

```bash
git fetch origin
git checkout feat/defense-presentation
git pull --rebase --autostash
cargo build --release -p node
```

Los tres tienen que estar en el mismo commit. Compárenlo con `git log -1 --oneline`.

**Solo Ale.** Hacen falta `curl` y `unzip`:

```bash
cargo build --release -p orchestui
python3 scripts/download_mnist.py
python3 presentation/tools/smooth_labels.py datasets/mnist/mnist_train_labels.bin datasets/mnist/mnist_train_labels_smooth.bin 0.1
```

## 2. Antes de cada práctica

Hay que repetirlo cada vez que cambie la red.

**Los tres.** Conéctense a la misma red y anoten su IP. `make node` la muestra al arrancar, o la pueden ver antes con:

```bash
ip -4 route get 1.1.1.1 | grep -oP 'src \K\S+'
```

Revisen el firewall con `sudo ufw status`. Si dice `inactive`, no hay que hacer nada. Si dice `active`, abran el puerto:

```bash
sudo ufw allow 40000/tcp
```

**Ale.** Abrí `presentation/demo/configs/training.json` y reemplazá la línea `addrs` por las tres IP reales:

```json
"addrs": ["IP_ALE:40000", "IP_LOREN:40000", "IP_MARCOS:40000"],
```

También va tu propia IP, nunca `127.0.0.1`: los nodos se conectan entre sí con estas direcciones. No commitees este cambio.

**Los tres.** Prueben la conexión **con los nodos apagados**. Cada uno deja esto corriendo:

```bash
python3 -m http.server 40000
```

Y prueba a los otros dos, cambiando la IP:

```bash
curl -sI http://IP_DEL_OTRO:40000 | head -1
```

Tiene que responder `HTTP/1.0 200 OK`. Al terminar, cierren el servidor con `Ctrl+C`.

**Nunca prueben el puerto de un nodo prendido.** `nc`, `curl`, `telnet` o un navegador lo cierran al instante, con `early eof` o `memory allocation failed`.

## 3. Levantar los nodos

**Ale, Loren y Marcos**, cada uno en una terminal:

```bash
make node
```

Primero muestra la dirección que va en `training.json`, por ejemplo `node address for training.json: 192.168.0.150:40000`. Después tiene que mostrar `listening at 0.0.0.0:40000`, y `orchestrator connected` cuando Ale lance el entrenamiento. La terminal queda ocupada: déjenla abierta hasta el final.

## 4. Lanzar el entrenamiento

**Ale**, en otra terminal y a pantalla completa:

```bash
./target/release/orchestui
```

1. Con **Start Training** marcado, apretá Enter.
2. En `model.json path`, escribí `presentation/demo/configs/model.json` y apretá Enter.
3. En `training.json path`, escribí `presentation/demo/configs/training.json` y apretá Enter.

El encabezado pasa de `CONNECTING` a `TRAINING`. `CONNECTING` tarda unos segundos porque reparte el dataset.

## 5. Mientras entrena

- `v` alterna entre el panel y la topología: primero se ve el anillo de All-Reduce.
- El cambio de estrategia llega cerca de la época 15. El encabezado muestra `⟳ switching W0 → PS` unos instantes y queda registrado en el panel `Events`.
- Después del cambio, el encabezado dice `workers 2/2` y `servers 1`. Es lo esperado: un nodo pasó a ser el servidor. Con `v` se ve la estrella.
- El contador de épocas no pasa de `max_epochs`, ni siquiera después del cambio. Un worker puede terminar una época antes que el otro.

## 6. Guardar el modelo

Cuando el encabezado diga `FINISHED`, **Ale** aprieta `s`, escribe `demo.safetensors` y aprieta Enter. `Events` confirma con `model saved to demo.safetensors`, y el archivo queda en la raíz del repositorio.

Si se olvidan, la TUI ya guardó una copia sola con un nombre al azar. Se ve con `ls safetensors/`.

## 7. Ver la exactitud en la página

**Ale:**

```bash
make presentation
xdg-open .
```

El primer comando abre la presentación y la página de la demo, sin internet. El segundo abre la carpeta del repositorio.

1. Arrastrá `demo.safetensors` a la página. También podés hacer clic en el recuadro y elegirlo.
2. Esperá unos 12 segundos, hasta que diga cuántas de las 10.000 imágenes clasificó bien. La exactitud aparece en grande: esperen cerca del 99 %.
3. Dibujá números con el mouse, grandes y centrados. Si sale mal, usá **una imagen del test, al azar**.

Para cargar otro modelo, recargá la página.

## 8. Repetir o cerrar

- **Otra corrida:** en la TUI, apretá `q` y después `y` para volver al menú, y seguí desde el paso 4. Los nodos no se reinician.
- **Cerrar:** apretá `q` en el menú de la TUI y `Ctrl+C` en cada nodo.

## Si algo falla

| Qué pasa | Por qué | Qué hacer |
|---|---|---|
| `Failed to Start Training` con `Connection refused` | Hay un nodo apagado, o una IP, puerto o firewall mal | Corregilo, apretá `q` y seguí desde el paso 4 |
| El encabezado dice `ERROR` y `Events`, `worker N error: early eof` | Se cayó un nodo o se cortó su red | Apretá `q`, relanzá solo ese nodo y seguí desde el paso 4. El entrenamiento arranca de cero |
| Un nodo se cierra con `early eof` o `memory allocation failed` | Algo que no es el orchestrator se conectó al puerto | Relanzalo y no pruebes el puerto con el nodo prendido |
| `Address already in use` al levantar un nodo | Ya hay otro proceso en el puerto | Buscá el PID con `ss -ltnp \| grep 40000` y cerralo con `kill PID` |
| La TUI no encuentra un archivo | La terminal no está en la raíz, o falta el dataset | Revisá con `ls datasets/mnist presentation/demo/configs` |
| No termina a tiempo | El entrenamiento tarda más de lo previsto | Cargá `presentation/demo/respaldo.safetensors` en la página y aclará que es de un ensayo |

## Ajustar la duración

Midan en la práctica cuánto tarda, porque depende de las computadoras. `max_epochs` es el total de épocas del entrenamiento, antes y después del cambio.

```text
max_epochs ≈ épocas hasta el cambio + (600 − segundos hasta el cambio) / segundos por época después del cambio
```

El resultado va en `max_epochs`, en `training.json`, para una demo de 10 minutos. No lo bajen de 20: si no queda margen después del cambio, el entrenamiento termina sin cambiar de estrategia.

## Si alguna computadora no usa Linux

Estos comandos no están probados.

| | macOS | Windows (PowerShell) |
|---|---|---|
| Ver la IP | `ipconfig getifaddr en0` | `ipconfig` |
| Levantar el nodo | `PORT=40000 RUST_LOG=warn ./target/release/node` | `$env:PORT=40000; $env:RUST_LOG="warn"; .\target\release\node.exe` |
| Abrir la página | `open presentation/demo/index.html` | `start presentation\demo\index.html` |
