# Guion de la demo

Pendiente de definir por el equipo. En la agenda hay **10 minutos** reservados entre la diapositiva 25 («Ahora, la demo») y la 26 (cierre). Al definirlo, actualizar también la lista *[a completar]* de la diapositiva 25 y sus notas en `index.html`.

## Qué mostrar y en qué orden

| Minuto | Qué se muestra | Quién |
|---|---|---|
| [a completar] | [a completar] | [a completar] |

## Qué tiene que estar preparado

- [a completar]

## Qué ejecutar

- [a completar]

## Qué resultado observar

- [a completar]

## Relación con el aporte de la tesis

- [a completar]

## Si la demo falla

- [a completar]

Hay material opcional que ya existe: el apéndice A1 de la presentación (grabación real de una ejecución de Strategy-Switch en la TUI, identificada como grabación) y el A2 (capturas reales).

## Referencia útil

Datos del repositorio, para tener a mano:

- Compilar: `cargo build --release -p node -p orchestui`.
- Nodos locales: `bash scripts/demo_nodes.sh up|down|status` levanta 4 nodos en los puertos 40000–40003.
- TUI: `cargo run -p orchestui --release`; pide las rutas de `model.json` y `training.json`.
- Teclas de la TUI: `v` alterna dashboard y topología, `←/→` cambian de worker, `x` detiene, `s` guarda el modelo al terminar y `q` vuelve al menú (pide confirmación con `y`).
- Configs usadas para las capturas: `presentation/captures/configs/`. Cada corrida dura entre 1 y 2,5 minutos en una notebook.
