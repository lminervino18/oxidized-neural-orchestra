# Presentación de la defensa

Presentación web de la defensa del Trabajo Profesional (O.N.O.). Está hecha en HTML, CSS y JavaScript, con GSAP para las animaciones y Three.js para la escena de la portada. Funciona sin conexión.

## Abrir

- `index.html`: versión editable. Se abre con doble clic en Chrome, Edge o Firefox.
- `dist/ono-defensa.html`: la misma presentación en un único archivo de 2,9 MB, con estilos, scripts, fuentes e imágenes incluidos. Es la más cómoda para llevar en un pendrive.

Ninguna de las dos versiones usa CDNs, fuentes remotas ni servicios externos. Está diseñada para 16:9: el escenario de 1920×1080 se escala a la ventana, así que se ve igual en 1920×1080 y en 1366×768.

## Navegar

| Tecla | Acción |
|---|---|
| `→` / espacio | Siguiente etapa; al terminar las etapas, siguiente diapositiva |
| `←` | Etapa anterior; al inicio, diapositiva anterior en su estado final |
| `⇧→` / `⇧←` | Salta las etapas y cambia de diapositiva |
| `R` | Repite la animación de la etapa actual |
| `P` | Pausa o reanuda las animaciones |
| `M` | Modo de movimiento reducido. Se activa solo si el sistema tiene `prefers-reduced-motion` |
| `F` | Pantalla completa |
| `N` | Abre las notas del expositor en otra ventana (hay que permitir ventanas emergentes) |
| `G` | Índice de diapositivas |
| número + `↵` | Ir a una diapositiva: 01–32 son las principales y 33–41 el apéndice (A1 = 33) |
| `?` / `H` | Ayuda |

En pantalla solo aparece el contador de diapositivas (`13 / 32`, o `A3 / A9` en el apéndice); no hay botones. Si se pausan las animaciones o se activa el movimiento reducido, al lado del contador aparece un aviso.

La ventana de notas muestra el expositor sugerido, el tiempo de la diapositiva, el plan acumulado, un reloj, la idea principal, las palabras clave que están en pantalla, el desarrollo oral, la transición y el archivo del informe que respalda el contenido. Las teclas también funcionan desde esa ventana. La URL guarda la posición actual (`#12.3` es la diapositiva 12, etapa 3).

## Estructura y tiempos

Son 32 diapositivas principales y 9 de apéndice. Siguen este hilo: cronograma, qué es un modelo, entrenar es caro, una sola máquina buena es cara, mejor usar muchas, paralelismo de modelo o de datos, el *client drift*, las cuatro formas de sincronizar y cómo funciona cada una, cuánto cuesta sincronizar (el modelo analítico), cómo está construido (arquitectura, la librería de ML hecha a mano, la convolución y la API de Python), resultados, trabajos futuros y demo. La demo entrena un modelo en vivo durante unos diez minutos y, mientras tanto, se cuenta el apéndice A1–A8; A9–A11 quedan de respaldo por si falla. Al final van «¿Preguntas?» y «¡Gracias por venir!», después del apéndice. El costo de sincronizar va justo después de los algoritmos y antes de los resultados, que confirman lo que predice. En pantalla solo hay palabras clave y animaciones; lo que se dice está en las notas. Los tiempos suman 45:00:

| Bloque | Diapositivas | Expositor sugerido | Tiempo |
|---|---|---|---|
| Portada, cronograma, qué es un modelo, por qué distribuir, *client drift* y alternativas | 01–09 | Alejo Ordoñez | 9:15 |
| Parameter Server, All-Reduce, PS asincrónico, Strategy-Switch, epochs offline (su monografía) y arquitectura | 10–16 | Marcos Bianchi Fernández | 11:45 |
| La librería de ML, la convolución y su teorema | 17–19 | Alejo Ordoñez | 2:45 |
| Costo de sincronizar, Python, cómo medimos, resultados, trabajos futuros y lanzamiento de la demo | 20–29 | Lorenzo Minervino | 8:45 |
| Apéndice mientras entrena: base común y buffer plano | A1–A2 | Alejo Ordoñez | 2:30 |
| Apéndice mientras entrena: compresión y topología | A3–A4 | Marcos Bianchi Fernández | 2:30 |
| Apéndice mientras entrena: por qué Rust y lecciones | A5–A6 | Lorenzo Minervino | 2:30 |
| ¿Y la demo?: el modelo en la página | 30 | Lorenzo Minervino | 0:15 + 2:00 |
| ¿Preguntas? y ¡Gracias por venir! | 31–32 | los tres | 1:00 |

En total, Alejo habla 14:30, Marcos 14:15 y Lorenzo 13:30.

La asignación sigue los aportes individuales del informe (`docs/report/aportes_individuales.md`): Alejo, el núcleo de aprendizaje automático y los datos; Marcos, los algoritmos, la coordinación, el protocolo y su monografía; Lorenzo, el worker, Python y la TUI, las simulaciones, los benchmarks y el análisis de resultados. Se cambia en el atributo `data-speaker` de cada diapositiva.

`GUION.md` es el guion completo: el reparto, la línea de tiempo minuto a minuto con quién habla y qué hay en pantalla, y qué tiene que decir cada uno en cada etapa. `GUION_DEMO.md` es la plantilla de la demo, que queda por completar.

## Editar

- **Contenido**: cada diapositiva es un `<section class="slide">` en `index.html`. Sus atributos son `data-title`, `data-section`, `data-speaker` y `data-time`; `data-after` suma tiempo fuera de las diapositivas (la demo) y `data-keywords` reemplaza las palabras clave del guion cuando la diapositiva no usa `.kw`. Las notas van en `<aside class="notes">`, que el público no ve: `n-idea` es el mensaje, `n-oral` es una lista de lo que se dice (cada `→` es una etapa), `n-trans` es el pase y `n-src` es el respaldo.
- **Palabras clave**: `<span class="kw">` dibuja una etiqueta de terminal; las variantes `on`, `warn`, `bad`, `ps`, `ar`, `ss` y `grad` cambian el color.
- **Revelado por etapas**: `data-anim` anima el elemento al entrar a la diapositiva. `data-step="n"` lo muestra a partir de la etapa `n`. Los títulos y las `.kw` aparecen con un efecto de decodificado, los elementos con `data-type` se tipean y las figuras con `data-wipe` aparecen con un barrido.
- **Diagramas**: `js/diagrams.js`, un constructor por diagrama (`data-diagram="ps"`, `"ring"`, `"ssflow"`, etc.). Cada uno arma un timeline de GSAP pausado con etiquetas `s0…sN`. Avanzar anima entre etiquetas y retroceder hace *seek*, así nunca quedan animaciones a medio camino.
- **Estilos**: `css/deck.css`. Los colores base salen de `orchestui/src/ui/theme.rs` (negro, `#39FF14`, `#00BE00`, `#505A50`) y del logo (`#0AF886`). Los colores de datos de los gráficos se validaron contra el fondo oscuro.
- **Motor**: `js/deck.js` (navegación, etapas, notas, pausa, movimiento reducido), `js/tui.js` (render de capturas de la TUI) y `js/scene.js` (escena de la portada).

Después de editar:

```bash
cd presentation
python3 tools/export_notes.py    # regenera GUION.md y verifica que los tiempos sumen 45:00
python3 tools/build_single.py    # regenera dist/ono-defensa.html
```

## Demo

La demo entrena LeNet-5 en vivo con Strategy-Switch en tres computadoras y después prueba el modelo en una página web. Todo el procedimiento, con comandos, tiempos y qué hacer si algo falla, está en `GUION_DEMO.md`.

- `demo/configs/`: `model.json` y los `training.json` del entrenamiento en vivo.
- `demo/index.html`: la página. Recibe el `.safetensors` que guarda la TUI, calcula la exactitud sobre las 10.000 imágenes de prueba de MNIST y clasifica números dibujados con el mouse. Corre en el navegador, sin servidor ni internet.
- `demo/data.js`: la arquitectura y el conjunto de prueba comprimido. Se regenera con `python3 tools/build_demo_data.py <carpeta con mnist_test_*.bin>`.
- `demo/respaldo.safetensors`: un modelo entrenado con la misma configuración en un ensayo, por si falla el entrenamiento en vivo.

## Capturas de la TUI

Las capturas del apéndice (A7, A8 y A9) son **buffers reales de la terminal**. Se obtuvieron el 28/10/2026 ejecutando `orchestui` dentro de tmux (190×46) con 4 procesos `node` locales y las configs de `captures/configs/`, sobre un subconjunto de 10.000 muestras de MNIST. Ninguna ejecución forma parte de la campaña experimental del informe.

- `captures/raw/*.ans`: salida de `tmux capture-pane -e -p`.
- `captures/rec/`: una captura por segundo de una ejecución completa de Strategy-Switch, en la vista de topología.
- `tools/ansi_to_capture.py` convierte todo eso en `data/captures.js`. `js/tui.js` lo dibuja celda por celda en un `<canvas>`, con los puntos braille, los bordes y los bloques trazados geométricamente, para que se vea igual en cualquier proyector.

Para regenerarlas:

```bash
cargo build --release -p node -p orchestui
bash scripts/demo_nodes.sh up                      # 4 nodos en 40000..40003
tmux new-session -d -s ono-tui -x 190 -y 46 "target/release/orchestui"
# cargar presentation/captures/configs/model.json y un training_*.json desde la TUI
tmux capture-pane -t ono-tui -e -p > presentation/captures/raw/<nombre>.ans
python3 presentation/tools/ansi_to_capture.py
```

## Resultados y figuras

Los resultados son las figuras originales del informe (`docs/report/figures/`), con los colores invertidos para proyectarlas sobre fondo negro, sin otros cambios. Los datos crudos de esas figuras no están en el repositorio, así que no se redibujaron. Los diagramas de las diapositivas 03 a 07 son conceptuales y no representan mediciones. Los gráficos que podrían confundirse con datos están identificados en pantalla:
- El de costo de sincronización (21) usa las fórmulas del capítulo de Resultados con S/R ≈ 0,1 s.
- El tiempo total (19), la derivación de d_sync (20, con N_W = 4) y la convolución (18, con una entrada de 4×4) son esquemas para explicar las fórmulas y el algoritmo, no mediciones.
- El paisaje de pérdida del *client drift* (08), la curva de pérdida de Strategy-Switch (13) y la compresión (A3) están marcados como ilustrativos.
- Las latencias de la topología (A4) están marcadas como ilustrativas.

## Diferencias detectadas entre informe, README y código

Conviene revisarlas antes de la defensa:

1. **Épocas después del switch.** Antes, con `strategy_switch`, al conmutar el worker se reconstruía con un trainer nuevo y el `max_epochs` completo. En la ejecución local de las capturas, con `max_epochs = 60`, los workers terminaron con 99–100 épocas. Ahora el orchestrator le manda solo las épocas que quedan (`orchestrator/src/sessions/event_listener.rs`), así que `max_epochs` es el tope de todo el entrenamiento. La campaña del informe corrió con el comportamiento anterior: hay que verificar si las corridas de Strategy-Switch tenían detención temprana, porque si no, hicieron más épocas que las otras estrategias.
2. **Disparador del switch.** El `README.md` del repo dice que se conmuta cuando «gradients converge», y la TUI muestra «all-reduce → will switch to PS when gradients converge». El código (`orchestrator/src/sessions/switch_tracker.rs`) y el informe usan la variación relativa de la **pérdida** de entrenamiento máxima (ventana 6, umbral 0,01).
3. **Régimen después del switch.** El informe habla de «Parameter Server asincrónico». En el código, el `synchronizer` y el `store` posteriores al cambio son configurables (`barrier` o `non_blocking`). Las notas de la diapositiva 14 lo dicen así.
4. **Unidades.** Las figuras de tiempo del informe no indican la unidad en el eje Y. La presentación usa segundos, en línea con T₁ = 600 s.

## Verificación realizada

Probado con Chrome headless (puppeteer-core) sobre `index.html` y sobre `dist/ono-defensa.html`:
- Las 41 diapositivas, etapa por etapa hacia adelante y entrando hacia atrás en su estado final, en 1920×1080 y 1366×768: sin errores de consola, sin pedidos de red externos y sin elementos fuera del escenario.
- Con mouse o teclado no queda ningún botón en pantalla: el pie muestra solo el contador.
- Controles: flechas, ⇧+flechas, `Home` y `End`, `?` y `Esc`, `G` e índice, número + Enter, `P`, `M`, `R`, ventana de notas `N` y restauración por URL.
- En pantallas táctiles aparece una barra con `‹ ›` (animación) y `« »` (diapositiva). Tocar a la derecha o a la izquierda avanza o retrocede una animación, y deslizar el dedo cambia de diapositiva. La barra va abajo y, en un celular en horizontal, a los costados.
- WebGL se probó con SwiftShader. Si no hay WebGL, la portada se muestra sin la escena y sin errores.

No se pudo verificar en un proyector real ni con la GPU de la computadora de la defensa.

## Dependencias incluidas

- GSAP 3.12.5 (`vendor/`), [licencia estándar de GSAP](https://gsap.com/standard-license).
- Three.js r149 (`vendor/`, MIT, `vendor/LICENSE-three.txt`).
- JetBrains Mono, en un subconjunto generado de la versión 2.304, e IBM Plex Sans (`fonts/`, SIL OFL 1.1, licencias en `fonts/`).
