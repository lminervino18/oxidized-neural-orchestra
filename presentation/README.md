# Presentación de la defensa

Presentación web de la defensa del Trabajo Profesional (O.N.O.). Está hecha en HTML, CSS y JavaScript, con GSAP para las animaciones y Three.js para la escena de la portada. Funciona sin conexión.

## Abrir

- `index.html`: versión editable. Se abre con doble clic en Chrome, Edge o Firefox.
- `dist/ono-defensa.html`: la misma presentación en un único archivo de 2,6 MB, con estilos, scripts, fuentes e imágenes incluidos. Es la más cómoda para llevar en un pendrive.

Ninguna de las dos versiones usa CDNs, fuentes remotas ni servicios externos. Está diseñada para 16:9: el escenario de 1920×1080 se escala a la ventana, así que se ve igual en 1920×1080 y en 1366×768.

## Navegar

| Tecla | Acción |
|---|---|
| `→` / espacio | Siguiente etapa; al terminar las etapas, siguiente diapositiva |
| `←` | Etapa anterior; al inicio, diapositiva anterior en su estado final |
| `⇧→` / `⇧←` | Salta las etapas y cambia de diapositiva |
| `R` | Repite la animación de la etapa actual (también el botón «↻ repetir») |
| `P` | Pausa o reanuda las animaciones |
| `M` | Modo de movimiento reducido. Se activa solo si el sistema tiene `prefers-reduced-motion` |
| `F` | Pantalla completa |
| `N` | Abre las notas del expositor en otra ventana (hay que permitir ventanas emergentes) |
| `G` | Índice de diapositivas |
| número + `↵` | Ir a una diapositiva; el apéndice empieza en la 25 |
| `?` / `H` | Ayuda |

La ventana de notas muestra el expositor sugerido, el tiempo de la diapositiva, el plan acumulado, un reloj, la idea principal, las palabras clave que están en pantalla, el desarrollo oral, la transición y el archivo del informe que respalda el contenido. Las teclas también funcionan desde esa ventana. La URL guarda la posición actual (`#12.3` es la diapositiva 12, etapa 3).

## Estructura y tiempos

Son 24 diapositivas principales y 10 de apéndice para las preguntas. Siguen este hilo: entrenar es caro, una sola máquina buena es cara, mejor usar muchas, paralelismo de modelo o de datos, el *client drift*, las cuatro formas de sincronizar, cómo funciona cada una, por qué Rust, cómo está construido, resultados, trabajos futuros, demo y cierre. En pantalla solo hay palabras clave y animaciones; lo que se dice está en las notas. Los tiempos suman 45:00:

| Bloque | Diapositivas | Expositor sugerido | Tiempo |
|---|---|---|---|
| Por qué distribuir, *client drift* y alternativas | 01–07 | Alejo Ordoñez | 10:30 |
| Parameter Server, All-Reduce, PS asincrónico y Strategy-Switch | 08–13 | Marcos Bianchi Fernández | 11:15 |
| Rust, arquitectura, TUI, resultados y trabajos futuros | 14–22 | Lorenzo Minervino | 10:45 |
| Demo | 23 y la demo en vivo | a definir | 0:30 + 10:00 |
| Cierre | 24 | los tres | 2:00 |

La asignación es una sugerencia que sigue los aportes individuales del informe. Se cambia en el atributo `data-speaker` de cada diapositiva.

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

## Capturas de la TUI

Las capturas de la diapositiva 16 y del apéndice son **buffers reales de la terminal**. Se obtuvieron el 28/10/2026 ejecutando `orchestui` dentro de tmux (190×46) con 4 procesos `node` locales y las configs de `captures/configs/`, sobre un subconjunto de 10.000 muestras de MNIST. Ninguna ejecución forma parte de la campaña experimental del informe.

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

Los resultados son las figuras originales del informe (`docs/report/figures/`), con los colores invertidos para proyectarlas sobre fondo negro, sin otros cambios. Los datos crudos de esas figuras no están en el repositorio, así que no se redibujaron. Los diagramas de las diapositivas 02 a 05 son conceptuales y no representan mediciones. Los gráficos que podrían confundirse con datos están identificados en pantalla:
- El de costo de sincronización (10) usa las fórmulas del capítulo de Resultados con S/R ≈ 0,1 s.
- El paisaje de pérdida del *client drift* (06), la curva de pérdida de Strategy-Switch (12) y la compresión (A7) están marcados como ilustrativos.
- Las latencias de la topología (A8) están marcadas como ilustrativas.

## Diferencias detectadas entre informe, README y código

Conviene revisarlas antes de la defensa:

1. **Épocas después del switch.** Con `strategy_switch`, al conmutar el worker se reconstruye con un trainer nuevo y el `max_epochs` completo (`node/src/router.rs`, `Run::Switch`; `orchestrator/src/configs/adapter.rs`, `adapt_trainer`). En la ejecución local de las capturas, con `max_epochs = 60`, los workers terminaron con 99–100 épocas. El código es el mismo desde antes de la campaña (julio de 2026). Hay que verificar si las corridas de Strategy-Switch del informe tenían detención temprana, porque si no, hicieron más épocas que las otras estrategias.
2. **Disparador del switch.** El `README.md` del repo dice que se conmuta cuando «gradients converge», y la TUI muestra «all-reduce → will switch to PS when gradients converge». El código (`orchestrator/src/sessions/switch_tracker.rs`) y el informe usan la variación relativa de la **pérdida** de entrenamiento máxima (ventana 6, umbral 0,01).
3. **Régimen después del switch.** El informe habla de «Parameter Server asincrónico». En el código, el `synchronizer` y el `store` posteriores al cambio son configurables (`barrier` o `non_blocking`). Las notas de la diapositiva 13 lo dicen así.
4. **Unidades.** Las figuras de tiempo del informe no indican la unidad en el eje Y. La presentación usa segundos, en línea con T₁ = 600 s.

## Verificación realizada

Probado con Chrome headless (puppeteer-core) sobre `index.html`, y la versión de archivo único con la misma prueba:
- Recorrido de las 34 diapositivas hacia adelante, etapa por etapa, en 1920×1080 y 1366×768, y recorrido completo hacia atrás: sin errores de consola, sin pedidos de red externos y sin elementos fuera del escenario ni sobre el pie.
- Al volver a una diapositiva queda una sola diapositiva activa y ningún elemento de etapa visible queda oculto.
- Controles: flechas, ⇧+flechas, botones, `?`, `G` e índice, `P`, `M`, `R`, número + Enter, ventana de notas `N` (incluidas las teclas desde ella), restauración por URL, API de pantalla completa, `prefers-reduced-motion` y reproductor de la grabación.
- WebGL se probó con SwiftShader. Si no hay WebGL, la portada se muestra sin la escena y sin errores.

No se pudo verificar en un proyector real ni con la GPU de la computadora de la defensa.

## Dependencias incluidas

- GSAP 3.12.5 (`vendor/`), [licencia estándar de GSAP](https://gsap.com/standard-license).
- Three.js r149 (`vendor/`, MIT, `vendor/LICENSE-three.txt`).
- JetBrains Mono, en un subconjunto generado de la versión 2.304, e IBM Plex Sans (`fonts/`, SIL OFL 1.1, licencias en `fonts/`).
