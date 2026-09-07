# ML1_ExamenAplicado_Aguila_Kevin
Examen final práctico de Machine Learning I - Predicción de valores inmobiliarios en California.
# Examen Final Machine Learning I 🚀

## Defensa en Video
[](https://drive.google.com/file/d/1hmfF9fL9WjIn_boI5kWps1LIsw7FmWEZ/view?usp=drive_link)

**Autor:** Kevin Andrés Águila Urrutia

## Descripción del Proyecto
Este repositorio contiene el examen final práctico de la asignatura Machine Learning I. El objetivo central fue implementar un pipeline completo de Machine Learning (abarcando aprendizaje supervisado y no supervisado) para predecir el valor de propiedades inmobiliarias.

* **Dataset:** California Housing Dataset
* **Fuente original:** Scikit-learn (`fetch_california_housing`)
* **Variable Objetivo:** `MedHouseVal` (Valor medio en cientos de miles de dólares)
* **Tipo de Tarea:** Regresión Múltiple

## Metodología y Hallazgos Principales

### 1. Análisis Exploratorio y Preprocesamiento
* Se comprobó la ausencia de valores nulos, pero se detectaron outliers severos en variables demográficas y arquitectónicas, los cuales fueron tratados mediante el método de Rango Intercuartílico (IQR = 1.5).
* El análisis de correlación lineal determinó que el **Ingreso Medio (`MedInc`)** es el predictor individual más fuerte.
* Se implementó un pipeline automatizado (`ColumnTransformer` y `StandardScaler`) asegurando una estricta división de datos (Train/Test) para evitar el *data leakage*.

### 2. Aprendizaje No Supervisado (PCA y Clustering)
* **Reducción de dimensionalidad:** Se aplicó PCA, logrando retener más del 84% de la varianza original utilizando únicamente 5 componentes principales.
* **Segmentación (K-Means):** Utilizando el método de Silhouette, se validó un óptimo de $K=2$. El modelo logró perfilar y separar el estado de California de forma puramente geográfica (Norte vs. Sur) sin el uso de etiquetas previas.

### 3. Aprendizaje Supervisado y Selección de Modelo
Se contrastó el rendimiento de un modelo lineal penalizado (Ridge) frente a un modelo de ensamble (Random Forest Regressor), optimizando los hiperparámetros de ambos mediante `GridSearchCV` (CV=5).

* **Mejor Modelo:** Random Forest Regressor.
* **Rendimiento:** Superó ampliamente al modelo lineal, logrando un $R^2$ cercano a 0.80 en el conjunto de prueba, justificando su mayor costo computacional (aprox. 13 minutos de entrenamiento).
* **Análisis de Errores:** Al revisar los residuales absolutos más altos, se detectó que el dataset original contiene censura de datos (tope artificial). Las propiedades con valor real superior a $500,000 USD fueron truncadas a un valor fijo de `5.0`, generando falsos residuos altos cuando el algoritmo predice precios realistas por encima de ese techo.

## Reproducibilidad del Entorno
Para clonar este repositorio y reproducir el entorno virtual con todas las dependencias exactas, ejecuta los siguientes comandos en tu terminal:

```bash
---
*Nota de Probidad Académica: Se declara explícitamente que, durante el desarrollo de este examen y la redacción de sus conclusiones, se utilizaron herramientas de Inteligencia Artificial generativa (Gemini) como asistencia para la corrección de estilo, estructuración del código y apoyo en la redacción del guion de presentación.*

# Clonar el repositorio
git clone [https://github.com/kevinaguila97/ML1_ExamenAplicado_Aguila_Kevin.git](https://github.com/kevinaguila97/ML1_ExamenAplicado_Aguila_Kevin.git)

# Entrar a la carpeta
cd ML1_ExamenAplicado_Aguila_Kevin

# Instalar los requerimientos
pip install -r requirements.txt
