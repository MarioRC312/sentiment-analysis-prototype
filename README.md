# Análisis de Sentimientos (Prototipo)

Este es un proyecto **de prueba** en Python donde implemento un clasificador de sentimientos usando **Naive Bayes + TF-IDF**.  
Es mi primer acercamiento al **Procesamiento de Lenguaje Natural (NLP)** y el objetivo es **aprender** cómo funcionan los modelos de texto.


## Tecnologías usadas
- Python 3.x
- [scikit-learn](https://scikit-learn.org/)
- pandas


## Estructura del proyecto
dataset.csv # Dataset pequeño de frases positivas/negativas
main.py # Código principal
requirements.txt # Dependencias
README.md # Este archivo

## Dataset
Actualmente el dataset es muy pequeño (unas pocas decenas de frases).  
Por este motivo, la **precisión del modelo es baja**.  
Sin embargo, la arquitectura Naive Bayes + TF-IDF es una técnica muy común que **funciona mucho mejor cuando se entrena con miles de ejemplos**.

Ejemplo de datos:
```csv
texto,sentimiento
"Me encanta programar en Python",positivo
"Odio los atascos de tráfico",negativo
"Hoy es un día maravilloso",positivo
"Este producto es terrible",negativo
```

## Cómo ejecutar el proyecto

Clona este repositorio:

git clone https://github.com/TU-USUARIO/sentiment-analysis-prototype.git
cd sentiment-analysis-prototype

Instala las dependencias:

pip install -r requirements.txt

Ejecuta el script:

python main.py

## Notas

Este es solo un prototipo de aprendizaje.

Con un dataset más grande, los resultados mejorarían considerablemente.

Futuras mejoras: usar datasets reales como IMDb o Twitter Sentiment Analysis.

## Autor

Proyecto creado por Mario Ruiz Castillo.
Primera práctica con modelos de clasificación de texto
