# medical_chatbot_AI

### Objetivo del Trabajo

Desarrollar un chatbot que ayude a los pacientes a identificar la especialidad médica adecuada para tratar sus síntomas, mejorando la precisión en la derivación a especialistas y reduciendo los retrasos en el diagnóstico y tratamiento. Este proyecto aborda tareas clave de clasificación y predicción para asistir en la selección de especialistas y recomendar médicos expertos.

### Nombre de los Alumnos Participantes
1. Andrea Katherina Tapia Pescoran - U202120058
2. Renato Guillermo Vivas Alejandro - U202021644

### Breve Descripción del Dataset
El conjunto de datos utilizado en este proyecto proviene de la plataforma Hugging Face y tiene las siguientes características:
| Nombre  | Descripción |
| ------------- | ------------- |
| Description  | La pregunta que resume una consulta por paciente  |
| Patient  | Consulta detallada del paciente  |
| Doctor  | Respuestas a las consultas del paciente  |

Link: [hugging face dataset](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)

### Conclusiones
En conclusión, gracias al curso de Aplicaciones de Data Science se aprendió lo necasario para contar con un modelo como el _Fine-tuned LLaMA_ que demuestra ser adecuado(ver models/test_models/evaluation_results.csv) para implementar un chatbot de asistencia médica, gracias a su rendimiento superior en métricas clave como precisión y recall. Estas métricas son especialmente críticas en este ámbito, ya que con una mayor precisión, el modelo puede reducir la tasa de respuestas incorrectas, asegurando que las sugerencias y respuestas proporcionadas sean relevantes. Su recall del 87.95% garantiza que el modelo no pase por alto casos importantes. Por último, el equilibrio entre estas métricas, refuerza su capacidad para manejar consultas complejas con un gran grado de precisión y recall.

