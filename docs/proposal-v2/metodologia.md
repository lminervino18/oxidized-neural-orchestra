\newpage
# Metodología
## Gestión y roles
Se establece un compromiso por parte de cada estudiante para dedicar un total de 500 horas al desarrollo del trabajo profesional. Esto representa, en promedio, 15 horas semanales por persona a la ejecución de las tareas asignadas. Este compromiso se mantendrá a lo largo de 32 semanas (dos cuatrimestres). Además, se tiene previsto llevar a cabo encuentros periódicos en formato virtual entre los miembros del equipo y los tutores, cada semana. El propósito de estas reuniones es informar el avance y desarrollo del proyecto en curso. Asimismo, se abordarán aspectos como la definición de prioridades en las labores a realizar y la planificación requerida para la próxima etapa del proceso.

En cuanto a los roles, el Ing. Ricardo A. Veiga cumple la función de tutor y el Dr. Ing. J. Ignacio Alvarez-Hamelin la de co-tutor, orientando el desarrollo y revisando periódicamente el avance. Los tres estudiantes realizan todas las etapas de análisis, implementación y pruebas del proyecto.

## Proceso de desarrollo
El método de desarrollo es incremental: el trabajo se organiza en iteraciones con entregables intermedios y una cadencia de prácticas de seguimiento y mejora continua. Las tareas de desarrollo, prueba y despliegue que se puedan automatizar estarán automatizadas, y todo el código y los artefactos que requieran versionarse se gestionan con una herramienta de control de versiones.

<!--
[COMPLETAR] Para cumplir las pautas del reglamento de la asignatura, detallar (con las
herramientas concretas que el equipo ya usa o va a usar):
- Herramienta de versionado de código y de artefactos (por ejemplo, Git/GitHub).
- Automatización de desarrollo, prueba y despliegue (CI, Docker).
- Herramienta de calidad profesional para gestionar el proceso, seguimiento de tickets y bugs.
- Artefactos de gestión: alcance, riesgos, métricas de calidad del producto, criticidad de
  bugs, indicadores de tiempos y costos, minutas.
- Criterios de aceptación de entregas y pruebas de aceptación de las funcionalidades.
- Indicadores de calidad de proceso definidos y cómo se miden.
-->

## Riesgos iniciales
Se identifican los siguientes riesgos iniciales, junto con las medidas previstas para mitigarlos:

- **Complejidad de la implementación distribuida en Rust.** La coordinación entre nodos y el manejo de la concurrencia son intrínsecamente complejos y propensos a errores sutiles, como bloqueos o condiciones de carrera. Como mitigación, el desarrollo es incremental y se apoya en las garantías de *fearless-concurrency* del lenguaje y en una cobertura de pruebas unitarias y de integración que acompaña cada avance.
- **Disponibilidad de hardware para simulaciones representativas.** Reproducir un entorno distribuido realista requiere de múltiples máquinas, algo no siempre disponible. Para mitigarlo, las simulaciones se ejecutan sobre contenedores Docker con límites de recursos por nodo, lo que permite emular configuraciones heterogéneas de forma controlada; se asume que los resultados así obtenidos son una aproximación y no un reemplazo exacto de un despliegue sobre máquinas físicas separadas.
- **Dependencia de trabajos previos.** Parte del trabajo se apoya en algoritmos y resultados publicados, como *Strategy-Switch*, cuya interpretación e implementación pueden diferir de lo documentado. La mitigación consiste en implementar versiones de referencia de los algoritmos base y validarlas antes de construir mejoras sobre ellas.
- **Amplitud del alcance y estimación de tiempos.** El trabajo abarca tanto el sistema base como varias líneas de optimización (comunicación, sincronización y carga en configuraciones heterogéneas), lo que dificulta la estimación del esfuerzo. Para mitigarlo, se prioriza primero un sistema base funcional junto con las implementaciones de referencia, y las optimizaciones se abordan de forma incremental según el avance, con reuniones semanales de seguimiento para reajustar prioridades.
