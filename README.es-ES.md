

# Detección de Fraude Financiero con IA

Para una comprensión más profunda, desde la motivación hasta la recopilación de datos, el entrenamiento del modelo y los resultados finales, puede consultar los blogs del proyecto aquí: https://www.amitkedia.com/project/67ce1a818013ee818192b171
## Vista General del Proyecto

**Introducción:** Este proyecto utiliza aprendizaje automático, aprendizaje profundo y Modelos de Lenguaje Grande (LLM) para detectar fraude financiero. Se basa en un conjunto de datos exhaustivo derivado de presentaciones financieras a la Comisión de Bolsa y Valores de los Estados Unidos (SEC), con el objetivo de comparar y mejorar los modelos de IA en la identificación de actividades financieras fraudulentas.

**Objetivo:** El objetivo es fomentar una plataforma colaborativa donde científicos de datos e investigadores puedan desarrollar, probar y mejorar modelos de IA para la detección de fraude financiero.

## Descripción del Conjunto de Datos

**Fuente:** El conjunto de datos incluye presentaciones financieras de 170 empresas, divididas por igual entre aquellas involucradas en actividades fraudulentas y no fraudulentas.

**Estructura:** Cada entrada del conjunto de datos contiene detalles como la Central Index Key (CIK), año de presentación, nombre de la empresa y un indicador categórico de fraude.

**Conjunto de Datos Final:** Finalmente, el conjunto de datos está disponible en Kaggle, revísalo [aquí..](https://www.kaggle.com/datasets/amitkedia/financial-statement-fraud-data/)

## Preprocesamiento de Datos

Los pasos de preprocesamiento implican la limpieza de texto, la tokenización y la transformación de datos en formatos legibles por máquina, garantizando un entrenamiento del modelo equilibrado y justo.

## Implementación del Modelo

El proyecto abarca una variedad de modelos, incluyendo Logistic Regression, SVM, Random Forest, XGBoost, ANN, HAN, GPT-2 y FinBERT, seleccionados por sus capacidades de NLP y su potencial en la detección de fraude.

## Cómo Reproducir

**Código Fuente:** El código completo para la extracción de datos, preprocesamiento, entrenamiento y evaluación del modelo está disponible en este repositorio.

**Entorno:** Se proporciona un archivo `requirements.txt` para configurar un entorno consistente.

**Documentación:** Cada script está documentado con instrucciones claras en el `README.md`, guiando a través de la configuración del entorno, la ejecución del script y la interpretación de los resultados.

## Directrices de Contribución

**Primeros Pasos:**
- Realiza un fork del repositorio.
- Configura tu entorno con `requirements.txt`.
- Familiarízate con el código y el conjunto de datos.

**Contribuir:**
- Agrega o mejora modelos, o refine los métodos de preprocesamiento.
- Asegúrate de que tu código esté documentado y se alinee con el estilo del proyecto.
- Envía pull requests con una descripción detallada de los cambios.

**Informar Problemas:**
- Utiliza GitHub Issues para reportar errores, solicitar características o participar en discusiones.
- Proporcione descripciones detalladas de los errores y los pasos para reproducirlos.

**Comunidad:**
- Participa en discusiones, comparte resultados y haz preguntas.
- Cumple con las directrices de la comunidad para mantener un entorno colaborativo.

## Licencia

Este proyecto es de código abierto, disponible bajo la Licencia MIT.

## Agradecimientos

Gracias a todos los colaboradores y miembros de la comunidad por su valiosa participación e ideas para avanzar en la IA aplicada a la detección de fraude financiero.
