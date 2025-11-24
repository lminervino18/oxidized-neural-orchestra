\newpage
# Planificación
## Gestión
Se establece un compromiso por parte de cada estudiante para dedicar un total de 500 horas al desarrollo del trabajo profesional. Esto representa, en promedio 15 horas semanales por persona a la ejecución de las tareas asignadas. Este compromiso se mantendrá a lo largo de 32 semanas (dos cuatrimestres). Además, se tiene previsto llevar a cabo encuentros periódicos en formato virtual entre los miembros del equipo y los tutores, cada semana. El propósito de estas reuniones es informar el avance y desarrollo del proyecto en curso. Asimismo, se abordarán aspectos como la definición de prioridades en las labores a realizar y la planificación requerida para la próxima etapa del proceso.

## Tareas
Las principales tareas para llevar a cabo el desarrollo de este trabajo son:

1. Leer y analizar los trabajos previos, actuales y que surjan sobre el entrenamiento distribuido de modelos de aprendizaje profundo para entender lo mejor posible el panorama.
2. Investigar implementaciones existentes de este tipo de sistemas para conocer las mejores abstracciones y desarrollar el mejor sistema posible.
3. Desarrollar un sistema distribuido de entrenamiento de modelos de aprendizaje profundo en Rust, que permita la ejecución parametrizada para abarcar la mayor cantidad de usos posibles.
4. Implementar los distintos algoritmos que se utilicen para la ejecución del entrenamiento distribuido, como All-Reduce, Parameter-Server y Strategy-Switch.
5. Implementar una interfaz funcional externa para poder usar el sistema a desarrollar desde Python mediante una biblioteca propia.
6. Simular la ejecución sobre distintas configuraciones del sistema distribuido y tomar medidas.
7. Analizar los datos obtenidos de la comparación de los algoritmos en (5) para sacar conclusiones sobre los algoritmos en distintos casos de uso y datasets.
8. Realizar un informe detallado de la evolución del trabajo y los resultados obtenidos.

## Carga horaria
| Tarea                                                                          | Duración (hs) | Responsable |
|--------------------------------------------------------------------------------|---------------|-------------|
| 1. Revisión bibliográfica sobre entrenamiento distribuido en Deep Learning     | 90            | A, L y M    |
| 2. Estudio de implementaciones existentes (TensorFlow, PyTorch, Horovod, etc.) | 90            | A, L y M    |
| 3. Desarrollo del sistema distribuido en Rust                                  | 510           | A, L y M    |
| 4. Implementación de algoritmos de entrenamiento distribuido                   | 300           | A, L y M    |
| 5. Creación de la interfaz externa en Python                                   | 120           | A, L y M    |
| 6. Simulación sobre distintas configuraciones del sistema distribuido          | 120           | A, L y M    |
| 7. Análisis de datos comparativos usando Python                                | 150           | A, L y M    |
| 8. Informe detallado de evolución y resultados                                 | 120           | A, L y M    |
| **Total**                                                                      | **1500**      | A, L y M    |

