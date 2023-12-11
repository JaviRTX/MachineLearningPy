# Machine Learning (Python)

## Tabla de contenidos

- [Machine Learning (Python)](#machine-learning-python)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [**Instalación**](#instalación)
    - [Windows](#windows)
      - [**Paso 1: Descargar Python**](#paso-1-descargar-python)
      - [**Paso 2: Instalación de Python**](#paso-2-instalación-de-python)
      - [**Paso 3: Verificar la Instalación**](#paso-3-verificar-la-instalación)
      - [**Paso 4: Instalar Paquetes para Machine Learning**](#paso-4-instalar-paquetes-para-machine-learning)
    - [**Linux (Debian/Ubuntu)**](#linux-debianubuntu)
      - [**Paso 1: Actualizar el Sistema**](#paso-1-actualizar-el-sistema)
      - [**Paso 2: Instalación de Python**](#paso-2-instalación-de-python-1)
      - [**Paso 3: Instalar pip**](#paso-3-instalar-pip)
      - [**Paso 4: Verificar la Instalación de pip**](#paso-4-verificar-la-instalación-de-pip)
      - [**Paso 5: Instalar Paquetes para Machine Learning**](#paso-5-instalar-paquetes-para-machine-learning)
  - [Parte I: Fundamentos](#parte-i-fundamentos)
    - [Introducción al Machine Learning](#introducción-al-machine-learning)
      - [Definición y Aplicaciones](#definición-y-aplicaciones)
      - [Aplicaciones Prácticas](#aplicaciones-prácticas)
    - [Tipos de Aprendizaje](#tipos-de-aprendizaje)
      - [Aprendizaje Supervisado](#aprendizaje-supervisado)
        - [Caso Práctico](#caso-práctico)
      - [Aprendizaje No Supervisado](#aprendizaje-no-supervisado)
        - [Caso Práctico](#caso-práctico-1)
      - [Aprendizaje por Refuerzo](#aprendizaje-por-refuerzo)
        - [Caso Práctico](#caso-práctico-2)
    - [Fundamentos de Python para Machine Learning](#fundamentos-de-python-para-machine-learning)
  - [Parte II: Aprendizaje Supervisado](#parte-ii-aprendizaje-supervisado)
    - [Regresión](#regresión)
    - [Clasificación](#clasificación)
  - [Parte III: Aprendizaje No Supervisado](#parte-iii-aprendizaje-no-supervisado)
    - [Clustering](#clustering)
    - [Reducción de Dimensionalidad](#reducción-de-dimensionalidad)
  - [Parte IV: Aprendizaje por Refuerzo](#parte-iv-aprendizaje-por-refuerzo)
    - [Conceptos Básicos de Aprendizaje por Refuerzo](#conceptos-básicos-de-aprendizaje-por-refuerzo)
    - [Aplicaciones Avanzadas](#aplicaciones-avanzadas)
  - [Parte V: Técnicas Avanzadas](#parte-v-técnicas-avanzadas)
    - [Redes Neuronales y Deep Learning](#redes-neuronales-y-deep-learning)
    - [Natural Language Processing (NLP) con Python](#natural-language-processing-nlp-con-python)
  - [Parte VI: Herramientas y Mejores Prácticas](#parte-vi-herramientas-y-mejores-prácticas)
    - [Evaluación y Ajuste de Modelos](#evaluación-y-ajuste-de-modelos)
    - [Despliegue de Modelos de Machine Learning](#despliegue-de-modelos-de-machine-learning)
  - [Parte VII: Estudios de Caso y Proyectos](#parte-vii-estudios-de-caso-y-proyectos)
    - [Proyectos de Machine Learning](#proyectos-de-machine-learning)
    - [Ética y Consideraciones Legales en Machine Learning](#ética-y-consideraciones-legales-en-machine-learning)
  - [Bibliografía](#bibliografía)
      - [Kaggle](#kaggle)
      - [W3Schools](#w3schools)
      - [Machinelearningmastery](#machinelearningmastery)
      - [Scipy](#scipy)
      - [FreeCodeCamp](#freecodecamp)
      - [ChatGPT](#chatgpt)
      - [SIIM](#siim)

## **Instalación**

### Windows

#### **Paso 1: Descargar Python**

- Visita la página oficial de Python: [python.org](https://www.python.org/).
- Haz clic en **Downloads** y selecciona la versión para Windows.
- Descarga el instalador ejecutable (`.exe`) para Windows.

#### **Paso 2: Instalación de Python**

- Ejecuta el instalador descargado.
- Asegúrate de marcar la opción **"Add Python to PATH"** al inicio de la instalación.
- Haz clic en **"Install Now"**.

#### **Paso 3: Verificar la Instalación**

- Abre el **Command Prompt (CMD)** y escribe `python --version`.
- Deberías ver la versión de Python instalada.

#### **Paso 4: Instalar Paquetes para Machine Learning**

- Utiliza el siguiente comando para instalar paquetes comunes de ML:

  ```
  pip install numpy scipy matplotlib scikit-learn jupyter
  ```

### **Linux (Debian/Ubuntu)**

#### **Paso 1: Actualizar el Sistema**

- Abre la terminal.
- Escribe `sudo apt update` y luego `sudo apt upgrade`.

#### **Paso 2: Instalación de Python**

- La mayoría de las distribuciones de Linux vienen con Python preinstalado. Para verificar, escribe `python3 --version`.
- Si no está instalado, usa `sudo apt install python3`.

#### **Paso 3: Instalar pip**

- Usa el comando `sudo apt install python3-pip`.

#### **Paso 4: Verificar la Instalación de pip**

- Escribe `pip3 --version` para verificar la instalación.

#### **Paso 5: Instalar Paquetes para Machine Learning**

- Utiliza el siguiente comando para instalar paquetes comunes de ML:

  ```
  pip3 install numpy scipy matplotlib scikit-learn jupyter
  ```

## Parte I: Fundamentos

[Tabla de contenidos](#tabla-de-contenidos)

### Introducción al Machine Learning

El Machine Learning (ML) es una rama fascinante y en constante evolución de la Inteligencia Artificial (IA) que se centra en el desarrollo de algoritmos y modelos que permiten a las computadoras aprender y actuar sin estar explícitamente programadas para tareas específicas. Esta capacidad de aprendizaje automático se logra mediante la identificación de patrones en los datos y la aplicación de estos conocimientos para realizar predicciones o tomar decisiones.

#### Definición y Aplicaciones

Machine Learning se define como la capacidad de un sistema para aprender y mejorar a partir de la experiencia sin ser programado explícitamente. Esta área combina elementos de estadística, matemáticas, programación y otras disciplinas para crear modelos que puedan procesar grandes conjuntos de datos y realizar tareas como clasificación, predicción, y reconocimiento de patrones.

#### Aplicaciones Prácticas

- **Salud**: Diagnóstico de enfermedades, análisis de imágenes médicas.
- **Finanzas**: Detección de fraudes, asesoramiento financiero automatizado.
- **Retail**: Personalización de recomendaciones de productos, análisis de comportamiento del consumidor.
- **Automatización y Vehículos Autónomos**: Vehículos que se conducen solos, robots para tareas domésticas.

![Ejemplo de Aplicación de ML en Salud](Img/SIIM-ISIC%20Melanoma%20Classification.png)

### Tipos de Aprendizaje

El Machine Learning se clasifica generalmente en tres tipos principales, basados en la naturaleza de la "señal" o "retroalimentación" disponible para el sistema de aprendizaje:

#### Aprendizaje Supervisado

El Aprendizaje Supervisado ocurre cuando el modelo se entrena en un conjunto de datos etiquetado. Esto significa que cada ejemplo en el conjunto de datos de entrenamiento está emparejado con la respuesta correcta (la etiqueta). El modelo aprende a predecir la salida a partir de las entradas durante el entrenamiento.

##### Caso Práctico

- **Predicción de Precios de Viviendas**: Utilizando datos históricos de ventas de casas, un modelo puede aprender a predecir precios futuros basándose en características como el tamaño, la ubicación y el número de habitaciones.

#### Aprendizaje No Supervisado

El Aprendizaje No Supervisado se utiliza cuando no hay datos etiquetados disponibles. El sistema intenta aprender patrones y estructuras a partir de los datos sin ninguna guía explícita.

##### Caso Práctico

- **Segmentación de Clientes en Marketing**: Agrupar clientes en diferentes categorías basadas en sus comportamientos y preferencias de compra, sin información previa sobre los grupos.

#### Aprendizaje por Refuerzo

El Aprendizaje por Refuerzo es un tipo de ML donde un agente aprende a tomar decisiones optimizando una recompensa a través de la prueba y error. No se le dan datos de entrada y salida, sino que debe descubrir por sí mismo cuáles acciones producen las mayores recompensas.

##### Caso Práctico

- **Juegos**: Algoritmos que aprenden a jugar y mejorar en juegos complejos como el tres en raya o el ajedrez, a través de la competencia contra sí mismos o contrincantes humanos.

![Ejemplo de Aprendizaje por Refuerzo en Juegos](https://rubenlopezg.files.wordpress.com/2015/05/direct_reward1.png)

### Fundamentos de Python para Machine Learning

- Introducción a Python
- Bibliotecas Esenciales: NumPy, Pandas
- Visualización de Datos: Matplotlib, Seaborn

## Parte II: Aprendizaje Supervisado

### Regresión

- Regresión Lineal
- Regresión Polinómica
- Regresión con Árboles de Decisión

### Clasificación

- K-Nearest Neighbors (KNN)
- Máquinas de Vectores de Soporte (SVM)
- Árboles de Decisión y Bosques Aleatorios

## Parte III: Aprendizaje No Supervisado

### Clustering

- K-Means
- Clustering Jerárquico
- Clustering DBSCAN

### Reducción de Dimensionalidad

- Análisis de Componentes Principales (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)

## Parte IV: Aprendizaje por Refuerzo

### Conceptos Básicos de Aprendizaje por Refuerzo

- El Problema del Bandido Multibrazo
- Algoritmos de Valor y Política

### Aplicaciones Avanzadas

- Q-Learning
- Deep Q-Networks (DQN)

## Parte V: Técnicas Avanzadas

### Redes Neuronales y Deep Learning

- Perceptrones y Redes Neuronales Artificiales
- Redes Neuronales Convolucionales (CNN)
- Redes Neuronales Recurrentes (RNN)

### Natural Language Processing (NLP) con Python

- Procesamiento del Lenguaje Natural
- Modelado de Temas
- Análisis de Sentimientos

## Parte VI: Herramientas y Mejores Prácticas

### Evaluación y Ajuste de Modelos

- Validación Cruzada
- Ajuste de Hiperparámetros
- Métricas de Evaluación

### Despliegue de Modelos de Machine Learning

- Introducción al Despliegue de Modelos
- Uso de Flask para APIs de Modelos de ML
- Consideraciones de Escalabilidad y Rendimiento

## Parte VII: Estudios de Caso y Proyectos

### Proyectos de Machine Learning

- Detección de Fraude
- Recomendaciones de Productos
- Reconocimiento de Imágenes y Voz

### Ética y Consideraciones Legales en Machine Learning

- Sesgos en los Datos y Modelos
- Privacidad y Seguridad de Datos
- Regulaciones y Cumplimiento Legal

## Bibliografía

#### [Kaggle](https://www.kaggle.com/learn)

#### [W3Schools](https://www.w3schools.com/python/python_ml_getting_started.asp)

#### [Machinelearningmastery](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)

#### [Scipy](https://scipy.org/)

#### [FreeCodeCamp](https://www.freecodecamp.org/espanol/learn/machine-learning-with-python/)

#### [ChatGPT](https://chat.openai.com/)

#### [SIIM](https://siim.org/)
