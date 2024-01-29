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
    - [**Google Colab (Sin Instalación)** Enlace](#google-colab-sin-instalación-enlace)
  - [Instalación de Bibliotecas](#instalación-de-bibliotecas)
    - [Bibliotecas Básicas](#bibliotecas-básicas)
    - [Machine Learning y Procesamiento de Datos](#machine-learning-y-procesamiento-de-datos)
    - [Procesamiento del Lenguaje Natural](#procesamiento-del-lenguaje-natural)
    - [Deep Learning](#deep-learning)
    - [Trabajo con APIs y Web](#trabajo-con-apis-y-web)
    - [Otras Bibliotecas Útiles](#otras-bibliotecas-útiles)
    - [Verificación de la Instalación](#verificación-de-la-instalación)
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
      - [Introducción a Python](#introducción-a-python)
      - [Ventajas para el ML:](#ventajas-para-el-ml)
    - [Bibliotecas Esenciales: NumPy, Pandas](#bibliotecas-esenciales-numpy-pandas)
      - [NumPy](#numpy)
        - [Ejemplo de Uso:](#ejemplo-de-uso)
      - [Pandas](#pandas)
        - [Ejemplo de Uso:](#ejemplo-de-uso-1)
    - [Visualización de Datos: Matplotlib, Seaborn](#visualización-de-datos-matplotlib-seaborn)
      - [Matplotlib](#matplotlib)
        - [Ejemplo de Gráfico:](#ejemplo-de-gráfico)
      - [Seaborn](#seaborn)
        - [Ejemplo de Gráfico:](#ejemplo-de-gráfico-1)
  - [Parte II: Aprendizaje Supervisado](#parte-ii-aprendizaje-supervisado)
    - [Regresión](#regresión)
    - [Clasificación](#clasificación)
      - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
      - [Máquinas de Vectores de Soporte (SVM)](#máquinas-de-vectores-de-soporte-svm)
      - [Árboles de Decisión y Bosques Aleatorios](#árboles-de-decisión-y-bosques-aleatorios)
        - [Árboles de Decisión](#árboles-de-decisión)
        - [Bosques Aleatorios](#bosques-aleatorios)
  - [Parte III: Aprendizaje No Supervisado](#parte-iii-aprendizaje-no-supervisado)
    - [Clustering](#clustering)
      - [K-Means](#k-means)
        - [Características del K-Means:](#características-del-k-means)
        - [Ejemplo de Código en Python:](#ejemplo-de-código-en-python)
        - [Resultados y Aplicaciones:](#resultados-y-aplicaciones)
      - [Clustering Jerárquico](#clustering-jerárquico)
        - [Características del Clustering Jerárquico:](#características-del-clustering-jerárquico)
      - [Ejemplo de Código en Python:](#ejemplo-de-código-en-python-1)
        - [Aplicaciones del Clustering Jerárquico:](#aplicaciones-del-clustering-jerárquico)
      - [Clustering DBSCAN](#clustering-dbscan)
        - [Características del DBSCAN:](#características-del-dbscan)
        - [Ejemplo de Código en Python:](#ejemplo-de-código-en-python-2)
        - [Aplicaciones del DBSCAN:](#aplicaciones-del-dbscan)
    - [Reducción de Dimensionalidad](#reducción-de-dimensionalidad)
      - [Análisis de Componentes Principales (PCA)](#análisis-de-componentes-principales-pca)
        - [Ejemplo de Código en Python:](#ejemplo-de-código-en-python-3)
      - [t-Distributed Stochastic Neighbor Embedding (t-SNE)](#t-distributed-stochastic-neighbor-embedding-t-sne)
        - [Ejemplo de Código en Python:](#ejemplo-de-código-en-python-4)
  - [Parte IV: Aprendizaje por Refuerzo](#parte-iv-aprendizaje-por-refuerzo)
    - [Conceptos Básicos de Aprendizaje por Refuerzo](#conceptos-básicos-de-aprendizaje-por-refuerzo)
      - [El Problema del Bandido Multibrazo](#el-problema-del-bandido-multibrazo)
      - [Algoritmos de Valor y Política](#algoritmos-de-valor-y-política)
    - [Aplicaciones Avanzadas](#aplicaciones-avanzadas)
      - [Q-Learning](#q-learning)
        - [Ejemplo de Código en Python:](#ejemplo-de-código-en-python-5)
      - [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
        - [Características de DQN:](#características-de-dqn)
        - [Ejemplo de Código en Python para DQN:](#ejemplo-de-código-en-python-para-dqn)
  - [Parte V: Técnicas Avanzadas](#parte-v-técnicas-avanzadas)
    - [Redes Neuronales y Deep Learning](#redes-neuronales-y-deep-learning)
      - [Perceptrones y Redes Neuronales Artificiales](#perceptrones-y-redes-neuronales-artificiales)
        - [Ejemplo de Código en Python para una ANN:](#ejemplo-de-código-en-python-para-una-ann)
      - [Redes Neuronales Convolucionales (CNN)](#redes-neuronales-convolucionales-cnn)
        - [Ejemplo de Código en Python para una CNN:](#ejemplo-de-código-en-python-para-una-cnn)
      - [Redes Neuronales Recurrentes (RNN)](#redes-neuronales-recurrentes-rnn)
        - [Ejemplo de Código en Python para una RNN:](#ejemplo-de-código-en-python-para-una-rnn)
    - [Natural Language Processing (NLP) con Python](#natural-language-processing-nlp-con-python)
      - [Procesamiento del Lenguaje Natural](#procesamiento-del-lenguaje-natural-1)
        - [Ejemplo de Código en Python para Tokenización:](#ejemplo-de-código-en-python-para-tokenización)
      - [Modelado de Temas](#modelado-de-temas)
        - [Ejemplo de Código en Python para LDA:](#ejemplo-de-código-en-python-para-lda)
      - [Análisis de Sentimientos](#análisis-de-sentimientos)
  - [Parte VI: Herramientas y Mejores Prácticas](#parte-vi-herramientas-y-mejores-prácticas)
    - [Evaluación y Ajuste de Modelos](#evaluación-y-ajuste-de-modelos)
      - [Validación Cruzada](#validación-cruzada)
        - [Ejemplo de Código en Python para Validación Cruzada:](#ejemplo-de-código-en-python-para-validación-cruzada)
      - [Ajuste de Hiperparámetros](#ajuste-de-hiperparámetros)
        - [Ejemplo de Código en Python para Ajuste de Hiperparámetros:](#ejemplo-de-código-en-python-para-ajuste-de-hiperparámetros)
      - [Métricas de Evaluación](#métricas-de-evaluación)
        - [Ejemplo de Métricas para Clasificación:](#ejemplo-de-métricas-para-clasificación)
    - [Despliegue de Modelos de Machine Learning](#despliegue-de-modelos-de-machine-learning)
      - [Introducción al Despliegue de Modelos](#introducción-al-despliegue-de-modelos)
      - [Uso de Flask para APIs de Modelos de ML](#uso-de-flask-para-apis-de-modelos-de-ml)
      - [Consideraciones de Escalabilidad y Rendimiento](#consideraciones-de-escalabilidad-y-rendimiento)
  - [Parte VII: Estudios de Caso y Proyectos](#parte-vii-estudios-de-caso-y-proyectos)
    - [Proyectos de Machine Learning](#proyectos-de-machine-learning)
      - [Detección de Fraude](#detección-de-fraude)
        - [Ejemplo de Código en Python para Detección de Fraude:](#ejemplo-de-código-en-python-para-detección-de-fraude)
      - [Recomendaciones de Productos](#recomendaciones-de-productos)
        - [Ejemplo de Código en Python para Recomendaciones de Productos:](#ejemplo-de-código-en-python-para-recomendaciones-de-productos)
      - [Reconocimiento de Imágenes y Voz](#reconocimiento-de-imágenes-y-voz)
        - [Ejemplo de Código en Python para Reconocimiento de Imágenes:](#ejemplo-de-código-en-python-para-reconocimiento-de-imágenes)
        - [Ejemplo de Código en Python para Reconocimiento de Voz:](#ejemplo-de-código-en-python-para-reconocimiento-de-voz)
    - [Ética y Consideraciones Legales en Machine Learning](#ética-y-consideraciones-legales-en-machine-learning)
      - [Sesgos en los Datos y Modelos](#sesgos-en-los-datos-y-modelos)
        - [Ejemplo de lo que NO se debe hacer:](#ejemplo-de-lo-que-no-se-debe-hacer)
      - [Privacidad y Seguridad de Datos](#privacidad-y-seguridad-de-datos)
        - [Buenas Prácticas:](#buenas-prácticas)
      - [Regulaciones y Cumplimiento Legal](#regulaciones-y-cumplimiento-legal)
        - [Consideraciones Clave:](#consideraciones-clave)
  - [Ejercicios](#ejercicios)
    - [Parte I: Fundamentos](#parte-i-fundamentos-1)
    - [Parte II: Aprendizaje Supervisado (Completa y Revisada)](#parte-ii-aprendizaje-supervisado-completa-y-revisada)
    - [Parte III: Aprendizaje No Supervisado](#parte-iii-aprendizaje-no-supervisado-1)
    - [Parte IV: Aprendizaje por Refuerzo](#parte-iv-aprendizaje-por-refuerzo-1)
    - [Parte V: Técnicas Avanzadas](#parte-v-técnicas-avanzadas-1)
    - [Parte VI: Herramientas y Mejores Prácticas](#parte-vi-herramientas-y-mejores-prácticas-1)
    - [Parte VII: Estudios de Caso y Proyectos](#parte-vii-estudios-de-caso-y-proyectos-1)
  - [Sistema CRUD para Machine Learning en Python con MySQL](#sistema-crud-para-machine-learning-en-python-con-mysql)
    - [Paso 1: Configurar el Entorno](#paso-1-configurar-el-entorno)
      - [Instalar MySQL y Python](#instalar-mysql-y-python)
    - [Paso 2: Implementación CRUD (Código)](#paso-2-implementación-crud-código)
  - [Bibliografía](#bibliografía)
      - [Kaggle](#kaggle)
      - [W3Schools](#w3schools)
      - [Machinelearningmastery](#machinelearningmastery)
      - [Scipy](#scipy)
      - [FreeCodeCamp](#freecodecamp)
      - [ChatGPT](#chatgpt)
      - [SIIM](#siim)
      - [Documentación BeautifulSoup (bs4)](#documentación-beautifulsoup-bs4)
      - [Documentación collections](#documentación-collections)
      - [Documentación Flask](#documentación-flask)
      - [Documentación Gym](#documentación-gym)
      - [Documentación joblib](#documentación-joblib)
      - [Documentación keras API](#documentación-keras-api)
      - [Documentación Matplotlib](#documentación-matplotlib)
      - [Documentación MySQL Connector/Python (mysql.connector)](#documentación-mysql-connectorpython-mysqlconnector)
      - [Documentación NLTK](#documentación-nltk)
      - [Documentación NumPy](#documentación-numpy)
      - [Documentación pandas](#documentación-pandas)
      - [Documentación Random](#documentación-random)
      - [Documentación requests](#documentación-requests)
      - [Documentación SciPy](#documentación-scipy)
      - [Documentación seaborn](#documentación-seaborn)
      - [Documentación scikit-learn (sklearn)](#documentación-scikit-learn-sklearn)
      - [Documentación Surprise](#documentación-surprise)
      - [Documentación TensorFlow](#documentación-tensorflow)
      - [Documentación TextBlob](#documentación-textblob)
      - [IMAGENET](#imagenet)
      - [VoxForge](#voxforge)
      - [Google Colab](#google-colab)

## **Instalación**

[Tabla de contenidos](#tabla-de-contenidos)

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

### **Google Colab (Sin Instalación)** [Enlace](https://colab.google/)
Es necesario crear una cuenta de google.

## Instalación de Bibliotecas

Puedes instalar las siguientes bibliotecas utilizando pip, el gestor de paquetes de Python. Abre tu terminal o línea de comandos e introduce los siguientes comandos:

### Bibliotecas Básicas
**Para facilitar el aprendizaje se aporta documentación del contenido en la bibliografía y en los distintos apartados**

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```
Estas bibliotecas son esenciales para el análisis de datos y la visualización.

- [Documentación numpy](https://numpy.org/doc/)
- [Documentación pandas](https://pandas.pydata.org/docs/)
- [Documentación matplotlib](https://matplotlib.org/stable/index.html)
- [Documentación seaborn](https://seaborn.pydata.org/)
  


### Machine Learning y Procesamiento de Datos

```bash
pip install scikit-learn
pip install scipy
pip install beautifulsoup4
```
Scikit-learn y SciPy son fundamentales para algoritmos de Machine Learning y operaciones matemáticas.

- [Documentación scikit-learn](https://scikit-learn.org/stable/user_guide.html)
- [Documentación scipy](https://docs.scipy.org/doc/scipy/)


### Procesamiento del Lenguaje Natural

```bash
pip install nltk
pip install textblob
```

NLTK y TextBlob son útiles para tareas de procesamiento del lenguaje natural.

- [Documentación nltk](https://www.nltk.org/)
- [Documentación textblob](https://textblob.readthedocs.io/en/dev/)

### Deep Learning

```bash
pip install tensorflow
pip install keras
```

TensorFlow es una biblioteca poderosa para la creación de modelos de Deep Learning.

- [Documentación tensorflow](https://www.tensorflow.org/api_docs)
- [Guía de instalación de TF en caso de errores](https://www.tensorflow.org/install/pip?hl=es#linux)
- [En caso de no poder instalar TF usar Google Colab](https://colab.research.google.com/)

### Trabajo con APIs y Web

```bash
pip install flask
pip install joblib
pip install requests
```

Flask es un micro framework web, y joblib es útil para guardar y cargar modelos.

- [Documentación flask](https://flask-es.readthedocs.io/)
- [Documentación joblib](https://joblib.readthedocs.io/en/stable/)

### Otras Bibliotecas Útiles

```bash
pip install collections-extended # Extiende las colecciones integradas en Python
pip install surprise
pip install mysql-connector-python
pip install gym
```

- [Documentación collections-extended](https://collections-extended.lenzm.net/)
  
### Verificación de la Instalación

Para verificar que las bibliotecas se han instalado correctamente, puedes importarlas en un intérprete de Python o un Jupyter Notebook:

```python
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn
import sklearn
import scipy
import nltk
import textblob
import tensorflow
import flask
import joblib
import collections
```

Si no se producen errores, las bibliotecas están instaladas correctamente y listas para usar.

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

[Ejemplo de Aplicación de ML en Salud](https://www.kaggle.com/c/siim-isic-melanoma-classification/)

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

[Ejemplo de Aprendizaje por Refuerzo en Juegos](https://medium.com/deelvin-machine-learning/how-to-play-google-chrome-dino-game-using-reinforcement-learning-d5b99a5d7e04)

[Tabla de contenidos](#tabla-de-contenidos)

### Fundamentos de Python para Machine Learning

Python se ha convertido en uno de los lenguajes de programación más populares en el campo del Machine Learning y la Ciencia de Datos. Su simplicidad, legibilidad y una amplia gama de bibliotecas hacen de Python una herramienta esencial para cualquier profesional del ML.

#### Introducción a Python

Python es un lenguaje de programación de alto nivel, interpretado, con un enfoque en la simplicidad y la legibilidad del código. Es ampliamente utilizado por su eficiencia y su extensa librería estándar, además de la gran comunidad que lo respalda.

#### Ventajas para el ML:

- **Sintaxis Clara y Concisa**: Facilita la escritura y lectura de código complejo.
- **Gran Comunidad**: Amplio soporte y recursos de aprendizaje.
- **Portabilidad y Extensibilidad**: Compatible con diversas plataformas y lenguajes.

### Bibliotecas Esenciales: NumPy, Pandas

#### NumPy

NumPy es una biblioteca fundamental para la computación científica en Python. Proporciona soporte para arrays y matrices grandes y multidimensionales, junto con una colección de funciones matemáticas para operar en estos arrays.

##### Ejemplo de Uso:

```python
import numpy as np

# Crear un array NumPy
array = np.array([1, 2, 3, 4, 5])
print(array)
```

#### Pandas

Pandas es una biblioteca que proporciona estructuras de datos y herramientas de análisis de datos de alto rendimiento y fácil de usar. Es ideal para trabajar con datos tabulares o heterogéneos.

##### Ejemplo de Uso:

```python
import pandas as pd

# Crear un DataFrame
data = {'Name': ['John', 'Anna'], 'Age': [28, 22]}
df = pd.DataFrame(data)
print(df)
```

### Visualización de Datos: Matplotlib, Seaborn

La visualización es una parte crucial en el análisis de datos y el Machine Learning, ya que permite comprender mejor los datos y los resultados de los modelos.

#### Matplotlib

Matplotlib es una biblioteca de gráficos 2D en Python que permite crear figuras y gráficos de alta calidad.

##### Ejemplo de Gráfico:

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel('Algunos números')
plt.show()
```

#### Seaborn

Seaborn es una biblioteca de visualización de datos en Python basada en Matplotlib. Ofrece una interfaz de alto nivel para dibujar gráficos estadísticos atractivos y informativos.

##### Ejemplo de Gráfico:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Datos de ejemplo
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips)

plt.show() 
```

## Parte II: Aprendizaje Supervisado

[Tabla de contenidos](#tabla-de-contenidos)

En el aprendizaje supervisado, los algoritmos aprenden de un conjunto de datos etiquetado, buscando predecir la salida para nuevas entradas basándose en el conocimiento adquirido.

### Regresión

La regresión en aprendizaje supervisado implica predecir valores continuos. Es crucial en muchos campos como la economía, la biología, y la ingeniería.

- Regresión Lineal
  
  La Regresión Lineal es fundamental para entender cómo las variables independientes están relacionadas con la variable dependiente.

  Ejemplo: 
  ```python
  import numpy as np
  from sklearn.linear_model import LinearRegression


  # Datos de entrenamiento
  X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
  y = np.dot(X, np.array([1, 2])) + 3

  # Crear y entrenar el modelo
  modelo = LinearRegression().fit(X, y)

  # Predecir nuevos valores
  print(modelo.predict(np.array([[3, 5]])))
  ```

- Regresión Polinómica
  
  La Regresión Polinómica es útil cuando la relación entre las variables independientes y dependientes no es lineal.

  Ejemplo:

  ```python 
  import numpy as np
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.linear_model import LinearRegression

  # Datos de entrenamiento
  X = np.array([2, 3, 4]).reshape(-1, 1)
  y = np.array([3, 5, 7])

  # Transformar datos a una forma polinómica
  poly = PolynomialFeatures(degree=2)
  X_poly = poly.fit_transform(X)

  # Crear y entrenar el modelo
  modelo = LinearRegression().fit(X_poly, y)

  # Predecir nuevos valores
  print(modelo.predict(poly.fit_transform(np.array([5]).reshape(-1, 1))))
  ```

- Regresión con Árboles de Decisión
  
  Los Árboles de Decisión son versátiles y pueden usarse tanto para clasificación como para regresión.

  ```python
  from sklearn.tree import DecisionTreeRegressor

  # Datos de entrenamiento
  X = [[0, 0], [2, 2]]
  y = [0.5, 2.5]

  # Crear y entrenar el modelo
  modelo = DecisionTreeRegressor().fit(X, y)

  # Predecir nuevos valores
  print(modelo.predict([[1, 1]]))
  ```


En cada uno de estos ejemplos, el código en Python ilustra cómo se pueden implementar y entrenar modelos de regresión, utilizando bibliotecas como sklearn, que es estándar en la industria del Machine Learning.

### Clasificación
La clasificación es una tarea fundamental en el aprendizaje supervisado, donde el objetivo es predecir etiquetas discretas (categorías) para nuevas instancias basándose en el aprendizaje realizado a partir de un conjunto de datos etiquetado.
#### K-Nearest Neighbors (KNN)

El algoritmo K-Nearest Neighbors (KNN) clasifica una nueva instancia basándose en la mayoría de votos de sus 'k' vecinos más cercanos.

```python
from sklearn.neighbors import KNeighborsClassifier

# Datos de entrenamiento
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 1]

# Crear y entrenar el modelo
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X, y)

# Predecir una nueva instancia
print(knn.predict([[1, 1]]))
```

#### Máquinas de Vectores de Soporte (SVM)
Las Máquinas de Vectores de Soporte (SVM) son un conjunto de algoritmos de aprendizaje supervisado utilizados para clasificación y regresión, destacando por su eficacia en espacios de alta dimensión.
```python
from sklearn.svm import SVC

# Datos de entrenamiento
X = [[0, 0], [1, 1]]
y = [0, 1]

# Crear y entrenar el modelo
svc = SVC()
svc.fit(X, y)

# Predecir una nueva instancia
print(svc.predict([[2., 2.]]))
```

#### Árboles de Decisión y Bosques Aleatorios
##### Árboles de Decisión
Un árbol de decisión es un modelo de predicción utilizado en el ámbito del aprendizaje supervisado. Representa una serie de decisiones basadas en una secuencia de preguntas que pueden conducir a una conclusión o clasificación específica. Empieza en la raíz del árbol y se divide en varias ramas, cada una representando una de las posibles respuestas a la pregunta del nodo. Este proceso se repite en cada nodo subsiguiente hasta llegar a un nodo hoja.

```python
from sklearn.tree import DecisionTreeClassifier

# Datos de entrenamiento
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 1]

# Crear y entrenar el modelo
tree = DecisionTreeClassifier()
tree.fit(X, y)

# Predecir una nueva instancia
print(tree.predict([[1, 1]]))
```

##### Bosques Aleatorios
Un bosque aleatorio es un conjunto (ensemble) de árboles de decisión, generalmente entrenados con el método de "bagging". La idea es mejorar la precisión predictiva y controlar el sobreajuste.
```python
from sklearn.ensemble import RandomForestClassifier

# Datos de entrenamiento
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 1]

# Crear y entrenar el modelo
forest = RandomForestClassifier(n_estimators=10)
forest.fit(X, y)

# Predecir una nueva instancia
print(forest.predict([[1, 1]]))
```
Ambos, los árboles de decisión y los bosques aleatorios, son herramientas poderosas en machine learning. Los árboles de decisión son útiles por su simplicidad y facilidad de interpretación, mientras que los bosques aleatorios ofrecen una mayor precisión y robustez, especialmente en conjuntos de datos grandes y complejos.

En esta sección, hemos cubierto tres métodos populares de clasificación en el aprendizaje supervisado, cada uno con su propio enfoque y ventajas. Los ejemplos de código proporcionan una base práctica para implementar estos algoritmos utilizando la biblioteca sklearn de Python, permitiendo una comprensión más profunda de cómo se pueden aplicar en problemas reales de clasificación.

## Parte III: Aprendizaje No Supervisado

[Tabla de contenidos](#tabla-de-contenidos)

El aprendizaje no supervisado es una técnica de machine learning en la que los modelos se entrenan usando un conjunto de datos sin etiquetas. La idea es explorar la estructura subyacente de los datos para extraer patrones significativos o insights. A diferencia del aprendizaje supervisado, no se utilizan respuestas o etiquetas correctas para guiar el proceso de aprendizaje. El algoritmo intenta organizar los datos de manera que se revelen patrones o características intrínsecas.

- Auto-organización: Los algoritmos de aprendizaje no supervisado deben ser capaces de identificar patrones y estructuras por sí mismos.
- Exploración de Datos: Es ideal para explorar la estructura de los datos cuando no se conocen las categorías o grupos previamente.
- Flexibilidad: Puede adaptarse a una amplia variedad de datos y no está limitado por la necesidad de datos etiquetados.

### Clustering
El Clustering es una técnica esencial en el aprendizaje no supervisado, que busca agrupar datos similares en conjuntos o clusters. 

#### K-Means

K-Means es uno de los algoritmos de clustering más populares y sencillos. Busca dividir un conjunto de observaciones en 'k' grupos, minimizando la varianza dentro de cada grupo.

##### Características del K-Means:

- Asignación de Cluster: Cada punto de datos se asigna al cluster más cercano, basado en la distancia euclidiana.
- Centroides: Cada cluster tiene un centroide, que es un punto virtual representando el centro del cluster.
- Iterativo: El algoritmo alterna entre asignar puntos a los clusters y actualizar los centroides hasta que se alcanza la convergencia.

##### Ejemplo de Código en Python:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Crear y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Predecir los clusters para los datos
clusters = kmeans.predict(X)

# Graficar los datos y los centroides
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=169, linewidths=3, color='r', zorder=10)
plt.title('Ejemplo de Clustering con K-Means')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

##### Resultados y Aplicaciones:

- Visualización: En este ejemplo, los datos son agrupados en dos clusters distintos, lo cual se puede visualizar en el gráfico generado.
- Aplicaciones: K-Means es ampliamente utilizado en segmentación de mercado, organización de computadoras en redes, clasificación de documentos, y en muchas otras áreas donde se requiere una agrupación eficiente de conjuntos de datos.

El algoritmo K-Means en Python, especialmente con la ayuda de **sklearn**, es una herramienta poderosa y fácil de usar para realizar tareas de clustering. Su simplicidad y eficacia lo hacen ideal para una amplia gama de aplicaciones en el campo del aprendizaje no supervisado.

#### Clustering Jerárquico

El Clustering Jerárquico es una técnica que busca construir una jerarquía de clusters. A diferencia de K-Means, no requiere especificar el número de clusters de antemano, y resulta en una estructura de árbol o dendrograma.

##### Características del Clustering Jerárquico:

- Métodos de Enlace: Determina cómo se miden las distancias entre clusters. Los métodos comunes incluyen enlace simple, completo y promedio.
- Dendrograma: Un árbol que muestra la disposición de los clusters formados en cada etapa.
- Corte del Dendrograma: Al cortar el dendrograma en un nivel específico, se pueden obtener un número deseado de clusters.

#### Ejemplo de Código en Python:

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# Datos de ejemplo
X = np.array([[5, 3], [10, 15], [15, 12], [24, 10], [30, 30],
              [85, 70], [71, 80], [60, 78], [55, 52], [80, 91]])

# Realizar el clustering jerárquico
linked = linkage(X, 'single')

# Graficar el dendrograma
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.show()
```
##### Aplicaciones del Clustering Jerárquico:
- Análisis de Datos Genéticos: Agrupar genes o muestras con perfiles de expresión genética similares.
- Segmentación del Mercado: Agrupar clientes con comportamientos o preferencias similares.
- Organización de Información: Como en bibliotecas o sistemas de información para agrupar recursos similares.

El Clustering Jerárquico es una herramienta poderosa en Machine Learning para descubrir relaciones inherentes en los datos, especialmente útil cuando la estructura de los clusters es compleja o cuando no se conoce el número de clusters a priori.

#### Clustering DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) es un algoritmo de clustering basado en la densidad, especialmente efectivo para identificar clusters de formas arbitrarias y manejar puntos de ruido.

##### Características del DBSCAN:

- **Basado en Densidad**: Define clusters como áreas de alta densidad separadas por áreas de baja densidad.
- **Puntos Núcleo, Frontera y Ruido**: Clasifica los puntos en núcleo, frontera o ruido, según la densidad.
- **Parámetros `eps` y `min_samples`**: `eps` define el radio de búsqueda de vecinos; `min_samples` es el número mínimo de puntos para formar un cluster.

##### Ejemplo de Código en Python:

```python
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# Crear y ajustar el modelo DBSCAN
dbscan = DBSCAN(eps=3, min_samples=2).fit(X)

# Etiquetas de los clusters
labels = dbscan.labels_

# Identificar puntos únicos (ruido)
unique_labels = set(labels)

# Colores para cada cluster
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# Graficar los puntos con colores por cluster
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Color negro usado para ruido
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Ejemplo de Clustering con DBSCAN')
plt.show()
```

##### Aplicaciones del DBSCAN:
- Detección de Anomalías: Ideal para identificar comportamientos atípicos en diversas aplicaciones, como fraude en tarjetas de crédito o actividades inusuales en la vigilancia.

- Segmentación Espacial: Útil en la identificación de regiones de alta densidad en mapas, como en estudios geográficos o urbanísticos.
  
- Agrupación en Bioinformática: Utilizado en el análisis de datos de expresión genética o en la categorización de tipos de proteínas.
  
- DBSCAN es particularmente valioso para tratar con datos complejos y ruidosos donde otros métodos de clustering pueden no ser efectivos, ofreciendo una forma robusta de identificar patrones y agrupaciones basadas en la densidad.
  
### Reducción de Dimensionalidad

La reducción de dimensionalidad es una técnica crucial en el aprendizaje no supervisado, que busca simplificar los datos sin perder información importante. Esto se hace reduciendo el número de variables aleatorias bajo consideración.

#### Análisis de Componentes Principales (PCA)

PCA es un método estadístico que transforma los datos a un nuevo sistema de coordenadas, reduciendo la dimensionalidad del espacio de características, manteniendo la mayor varianza posible.

##### Ejemplo de Código en Python:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
X = np.random.rand(100, 5)

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Graficar los componentes principales
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('Ejemplo de PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
```

#### t-Distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE es una técnica para la visualización de datos de alta dimensión, reduciendo los datos a dos o tres dimensiones de manera que datos similares estén cerca.
##### Ejemplo de Código en Python:
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
X = np.random.rand(100, 5)

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# Graficar los datos reducidos
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title('Ejemplo de t-SNE')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.show()
```

Estas técnicas de reducción de dimensionalidad, PCA y t-SNE, son herramientas fundamentales para simplificar y visualizar datos complejos, facilitando su análisis y comprensión.

## Parte IV: Aprendizaje por Refuerzo

[Tabla de contenidos](#tabla-de-contenidos)

El Aprendizaje por Refuerzo es un enfoque del Machine Learning donde un agente aprende a tomar decisiones para maximizar alguna noción de recompensa acumulativa a través de la interacción con un entorno.

### Conceptos Básicos de Aprendizaje por Refuerzo

#### El Problema del Bandido Multibrazo

Es un problema clásico que ejemplifica el dilema entre exploración y explotación. Un agente elige entre varias opciones, cada una con una recompensa desconocida, intentando maximizar la recompensa total.

#### Algoritmos de Valor y Política

- **Algoritmos de Valor**: Estos algoritmos buscan estimar cuánto valor (recompensa total a largo plazo) se puede obtener tomando ciertas acciones en ciertos estados.
- **Algoritmos de Política**: Aprenden directamente la política de acción que un agente debe tomar en un estado dado.

### Aplicaciones Avanzadas

#### Q-Learning

  Q-Learning es un algoritmo de aprendizaje por refuerzo basado en valores. No requiere un modelo del entorno y puede manejar problemas con transiciones estocásticas y recompensas.

##### Ejemplo de Código en Python:

```python
import numpy as np
import random

# Suposiciones sobre el entorno y las acciones
espacio_de_estados = 10  
espacio_de_acciones = 4  

# Inicializar la tabla Q
Q = np.zeros([espacio_de_estados, espacio_de_acciones])

# Parámetros del algoritmo
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.6  # Factor de descuento
total_episodios = 10000  # Total de episodios para entrenar

# Funciones adicionales necesarias
def resetear_entorno():
    # Devuelve un estado inicial aleatorio
    return random.randint(0, espacio_de_estados - 1)

def elegir_accion(estado, Q):
    # Ejemplo: elegir una acción aleatoriamente
    return random.randint(0, espacio_de_acciones - 1)

def tomar_accion(accion):
    # Devuelve un nuevo estado, recompensa y si es el estado final
    nuevo_estado = random.randint(0, espacio_de_estados - 1)
    recompensa = random.random()  
    final = nuevo_estado == espacio_de_estados - 1  
    return nuevo_estado, recompensa, final

def actualizar_Q(Q, estado, accion, recompensa, nuevo_estado, alpha, gamma):
    # Fórmula de actualización de Q-Learning
    mejor_prediccion = np.max(Q[nuevo_estado])
    Q_actual = Q[estado, accion]
    Q[estado, accion] = Q_actual + alpha * (recompensa + gamma * mejor_prediccion - Q_actual)
    return Q[estado, accion]

# Proceso de aprendizaje
for episodio in range(total_episodios):
    estado = resetear_entorno()
    final = False

    while not final:
        accion = elegir_accion(estado, Q)
        nuevo_estado, recompensa, final = tomar_accion(accion)
        Q[estado, accion] = actualizar_Q(Q, estado, accion, recompensa, nuevo_estado, alpha, gamma)
        estado = nuevo_estado

# Mostrar la tabla Q final
print("Tabla Q aprendida:")
print(Q)
```

#### Deep Q-Networks (DQN)

DQN es una técnica avanzada de aprendizaje por refuerzo que combina redes neuronales con Q-Learning para trabajar en entornos de alta complejidad.

##### Características de DQN:
- Redes Neuronales Profundas: Utiliza redes neuronales para aproximar la función Q.
- Experience Replay: Almacena las experiencias del agente para romper la correlación en la secuencia de observaciones.
- Target Networks: Redes neuronales separadas para estabilizar el proceso de aprendizaje.

##### Ejemplo de Código en Python para DQN:

Este código proporciona una implementación básica de DQN, utilizando TensorFlow para construir y entrenar una red neuronal que aproxima la función Q en un entorno de aprendizaje por refuerzo. El algoritmo utiliza Experience Replay y una política epsilon-greedy para la selección de acciones.

```python
import random
import numpy as np
import tensorflow as tf
from collections import deque

# Define el número de estados y acciones según tu entorno
numero_de_estados = 4  
numero_de_acciones = 2  

# Definir el modelo de red neuronal para DQN
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(numero_de_estados,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(numero_de_acciones, activation='linear')
])

# Compilar el modelo
modelo.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Memory buffer para Experience Replay
memory_buffer = deque(maxlen=2000)

# Parámetros adicionales
total_episodios = 100  
epsilon = 1.0  # Exploración inicial
epsilon_min = 0.01  # Mínima exploración
epsilon_decay = 0.995  # Tasa de decaimiento de exploración
gamma = 0.95  # Factor de descuento
batch_size = 32  # Tamaño del batch para entrenamiento

# Funciones adicionales
def resetear_entorno():
    # Devuelve un estado inicial aleatorio
    return np.random.rand(numero_de_estados)

def paso_del_entorno(accion):
    # Devuelve un nuevo estado, recompensa y si es el estado final
    nuevo_estado = np.random.rand(numero_de_estados)
    recompensa = np.random.random()
    final = np.random.choice([True, False])
    return nuevo_estado, recompensa, final, None

# Proceso de aprendizaje
for episodio in range(total_episodios):
    estado = resetear_entorno()  # Inicializar el estado del entorno
    final = False
    recompensa_total = 0  # Para registrar la recompensa total obtenida en el episodio

    while not final:
        # Elegir acción con política epsilon-greedy
        if np.random.rand() <= epsilon:
            accion = np.random.choice(numero_de_acciones)
        else:
            accion = np.argmax(modelo.predict(estado.reshape(1, -1))[0])

        # Tomar acción y observar el resultado
        nuevo_estado, recompensa, final, _ = paso_del_entorno(accion)
        
        # Almacenar la experiencia
        memory_buffer.append((estado, accion, recompensa, nuevo_estado, final))

        # Entrenar el modelo con un minibatch del memory buffer
        if len(memory_buffer) > batch_size:
            minibatch = random.sample(memory_buffer, batch_size)
            for estado_b, accion_b, recompensa_b, nuevo_estado_b, final_b in minibatch:
                target = recompensa_b
                if not final_b:
                    target = (recompensa_b + gamma * np.max(modelo.predict(nuevo_estado_b.reshape(1, -1))[0]))
                target_f = modelo.predict(estado_b.reshape(1, -1))
                target_f[0][accion_b] = target
                modelo.fit(estado_b.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        estado = nuevo_estado  # Actualizar el estado
        recompensa_total += recompensa  # Acumular la recompensa obtenida

    # Ajustar epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Imprimir el progreso cada 10 episodios
    if episodio % 10 == 0:  
        print(f"Episodio: {episodio}, Recompensa Total: {recompensa_total}, Epsilon: {epsilon}")
```

El Aprendizaje por Refuerzo, con técnicas como Q-Learning y DQN, es fundamental para problemas donde la toma de decisiones es secuencial y el entorno puede ser complejo y desconocido.

## Parte V: Técnicas Avanzadas

[Tabla de contenidos](#tabla-de-contenidos)

El Deep Learning, una subárea del Machine Learning, utiliza redes neuronales con muchas capas (de ahí el término "profundo") para aprender de los datos. Estas técnicas son especialmente potentes en tareas como el procesamiento de imágenes, lenguaje natural y secuencias temporales.

### Redes Neuronales y Deep Learning

#### Perceptrones y Redes Neuronales Artificiales

- **Perceptrones**: Son la forma más simple de una red neuronal artificial, basada en un modelo matemático para el aprendizaje supervisado. Consisten en una sola neurona con pesos ajustables.
- **Redes Neuronales Artificiales (ANN)**: Compuestas por capas de perceptrones, las ANN pueden aprender tareas complejas mediante la combinación de muchas funciones simples.

##### Ejemplo de Código en Python para una ANN:

```python
import tensorflow as tf
import numpy as np

# Datos de ejemplo (ficticios)
datos_entrada = np.random.random((100, 8))  # 100 muestras, 8 características
datos_salida = np.random.randint(2, size=(100, 1))  # 100 muestras, 1 etiqueta binaria

# Definir un modelo secuencial
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(12, input_dim=8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(datos_entrada, datos_salida, epochs=10, batch_size=10)

# Evaluar el modelo
evaluacion = modelo.evaluate(datos_entrada, datos_salida)
print(f"Evaluación del modelo: {evaluacion}")
```
#### Redes Neuronales Convolucionales (CNN)

Las CNN son un tipo de redes neuronales profundas utilizadas principalmente para el procesamiento de imágenes, donde pueden reconocer patrones espaciales y temporales.
##### Ejemplo de Código en Python para una CNN:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import numpy as np

# Supongamos que tenemos datos de entrenamiento y prueba (datos ficticios para este ejemplo)
x_train = np.random.random((100, 28, 28, 1))  # 100 imágenes, 28x28 píxeles, 1 canal (blanco y negro)
y_train = np.random.randint(0, 10, (100,))  # 100 etiquetas para las imágenes, 10 clases
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)  # Convertir a formato categórico

x_test = np.random.random((20, 28, 28, 1))  # 20 imágenes para prueba
y_test = np.random.randint(0, 10, (20,))  # 20 etiquetas para las imágenes de prueba
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)  # Convertir a formato categórico

# Crear modelo CNN
modelo_cnn = Sequential()
modelo_cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
modelo_cnn.add(Conv2D(32, kernel_size=3, activation='relu'))
modelo_cnn.add(Flatten())
modelo_cnn.add(Dense(10, activation='softmax'))

# Compilar el modelo
modelo_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo_cnn.fit(x_train, y_train, epochs=5, batch_size=10)

# Evaluar el modelo
evaluacion = modelo_cnn.evaluate(x_test, y_test)
print(f"Evaluación del modelo: {evaluacion}")
```

#### Redes Neuronales Recurrentes (RNN)
Las RNN son utilizadas para trabajar con secuencias de datos, como el lenguaje hablado o escrito, ya que tienen "memoria" de entradas anteriores.

##### Ejemplo de Código en Python para una RNN:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import numpy as np

# Datos ficticios de ejemplo
# Supongamos que tenemos 100 secuencias de 3 pasos de tiempo con 1 característica
x_train = np.random.random((100, 3, 1))
y_train = np.random.random((100, 1))

x_test = np.random.random((20, 3, 1))
y_test = np.random.random((20, 1))

# Crear modelo RNN
modelo_rnn = Sequential()
modelo_rnn.add(SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(3, 1)))
modelo_rnn.add(SimpleRNN(50, activation='relu'))
modelo_rnn.add(Dense(1))

# Compilar el modelo
modelo_rnn.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
modelo_rnn.fit(x_train, y_train, epochs=10, batch_size=10)

# Evaluar el modelo
evaluacion = modelo_rnn.evaluate(x_test, y_test)
print(f"Evaluación del modelo: {evaluacion}")
```

Estas técnicas avanzadas de Deep Learning permiten a los modelos aprender y realizar tareas que serían imposibles o muy difíciles para los algoritmos de Machine Learning tradicionales.

### Natural Language Processing (NLP) con Python

El Procesamiento del Lenguaje Natural (NLP) es una rama del Machine Learning y la Inteligencia Artificial que se centra en la interacción entre computadoras y lenguaje humano. Python, con sus numerosas bibliotecas y frameworks, es una herramienta excelente para el trabajo en NLP.

#### Procesamiento del Lenguaje Natural

El NLP implica una serie de técnicas para permitir a las computadoras entender y procesar el lenguaje humano, desde el texto hasta el habla.

##### Ejemplo de Código en Python para Tokenización:

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

texto = "¡Bienvenido al mundo del NLP con Python!"
palabras = word_tokenize(texto)
print(palabras)
```

#### Modelado de Temas

El modelado de temas es un enfoque de NLP para descubrir temas abstractos dentro de un conjunto de documentos, comúnmente utilizado en la clasificación y organización de grandes volúmenes de texto.

##### Ejemplo de Código en Python para LDA:
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Datos de ejemplo
textos = ["Texto sobre política.", "Texto sobre economía.", "Texto sobre deportes."]

# Vectorización del texto
vectorizador = CountVectorizer()
X = vectorizador.fit_transform(textos)

# Aplicar LDA
lda = LatentDirichletAllocation(n_components=3, random_state=0)
lda.fit(X)

# Mostrar los temas
palabras = vectorizador.get_feature_names_out()

# Mostrar las 5 palabras más relevantes por tema
for indice_tema, tema in enumerate(lda.components_):
    print(f"Tema {indice_tema}:")
    print(" ".join([palabras[i] for i in tema.argsort()[:-4:-1]]))  
```

#### Análisis de Sentimientos
El análisis de sentimientos es una técnica de NLP utilizada para determinar la actitud o emoción del hablante o escritor respecto a un tema particular.
```python
from textblob import TextBlob
# Solo funciona con texto en inglés
texto = "I love Python, it is the best thing"
blob = TextBlob(texto)

# Obtener el sentimiento del texto
sentimiento = blob.sentiment.polarity
print("Sentimiento:", sentimiento)
```

El NLP es un campo de rápido crecimiento en la ciencia de datos y la inteligencia artificial, y Python ofrece un ecosistema robusto y versátil para su exploración y aplicación práctica.

## Parte VI: Herramientas y Mejores Prácticas

[Tabla de contenidos](#tabla-de-contenidos)

Esta sección aborda herramientas y estrategias fundamentales para la evaluación y optimización de modelos de Machine Learning, asegurando su efectividad y confiabilidad.

### Evaluación y Ajuste de Modelos

#### Validación Cruzada

La Validación Cruzada es una técnica para evaluar la generalización de un modelo. Implica dividir el conjunto de datos en varias partes, utilizando cada parte para validar el modelo entrenado en el resto.

##### Ejemplo de Código en Python para Validación Cruzada:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# Crear modelo
modelo = RandomForestClassifier()

# Realizar validación cruzada
scores = cross_val_score(modelo, X, y, cv=5)
print("Precisión de cada pliegue:", scores)
print("Precisión promedio:", scores.mean())
```

#### Ajuste de Hiperparámetros
El ajuste de hiperparámetros implica encontrar la combinación de parámetros que produce el mejor rendimiento del modelo.

##### Ejemplo de Código en Python para Ajuste de Hiperparámetros:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

# Datos ficticios aleatorios
# X: datos de entrada, y: etiquetas 
X = np.random.rand(100, 5)  # 100 muestras, 5 características por muestra
y = np.random.randint(2, size=100)  # 100 etiquetas binarias

# Parámetros a ajustar
parametros = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

# Crear modelo
svc = SVC()

# Ajuste de hiperparámetros
clf = GridSearchCV(svc, parametros)
clf.fit(X, y)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros:", clf.best_params_)
```

#### Métricas de Evaluación
Las métricas de evaluación son cruciales para entender el rendimiento de un modelo. Dependiendo del tipo de tarea (clasificación, regresión, etc.), estas métricas pueden variar.

##### Ejemplo de Métricas para Clasificación:
- Precisión: Proporción de predicciones correctas.
- Recall: Capacidad del modelo para encontrar todas las instancias relevantes.
- F1-Score: Media armónica de la precisión y el recall.
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Datos ficticios aleatorios
X = np.random.rand(100, 5)  # 100 muestras, 5 características por muestra
y = np.random.randint(2, size=100)  # 100 etiquetas binarias

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80% entrenamiento 20% prueba

# Entrenar y predecir con un modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Calcular métricas
reporte = classification_report(y_test, y_pred)
print(reporte)
```

Estas herramientas y técnicas son fundamentales para el desarrollo, evaluación y ajuste de modelos de Machine Learning, asegurando su rendimiento y aplicabilidad en problemas del mundo real.

### Despliegue de Modelos de Machine Learning

Una vez que un modelo de Machine Learning ha sido entrenado y evaluado, el siguiente paso es ponerlo en producción, es decir, hacer que esté disponible para ser utilizado en aplicaciones reales. Esta sección explora cómo desplegar modelos de ML.

#### Introducción al Despliegue de Modelos

El despliegue de un modelo implica integrarlo en una aplicación existente o en un sistema de producción. El objetivo es que el modelo pueda recibir datos de entrada y proporcionar predicciones o resultados en un entorno real y en tiempo real.

#### Uso de Flask para APIs de Modelos de ML

Flask es un micro framework web en Python que es frecuentemente utilizado para crear APIs que permiten interactuar con modelos de ML. A través de Flask, se puede exponer un modelo como un servicio web que puede recibir datos y devolver predicciones.

#### Consideraciones de Escalabilidad y Rendimiento

Al desplegar modelos de ML, es crucial considerar su escalabilidad y rendimiento. Esto implica asegurar que el modelo pueda manejar una gran cantidad de solicitudes sin degradar su velocidad o precisión. Se deben considerar aspectos como la optimización del modelo, el uso eficiente de recursos y la posibilidad de escalar horizontalmente. El despliegue efectivo de modelos de ML es un paso crucial para llevar las soluciones de Machine Learning desde el laboratorio hasta aplicaciones del mundo real, impactando directamente en usuarios y negocios.

## Parte VII: Estudios de Caso y Proyectos

[Tabla de contenidos](#tabla-de-contenidos)

Esta sección se centra en la aplicación práctica de las técnicas de Machine Learning en diversos escenarios y proyectos.

### Proyectos de Machine Learning

#### Detección de Fraude

La detección de fraude es un campo crucial en el sector financiero, donde el Machine Learning puede identificar transacciones sospechosas.

##### Ejemplo de Código en Python para Detección de Fraude:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
predicciones = modelo.predict(X_test)

# Mostrar las primeras 5 predicciones
print("Primeras 5 predicciones:", predicciones[:5])
```

#### Recomendaciones de Productos

Los sistemas de recomendación utilizan el aprendizaje automático para sugerir productos a los usuarios.

##### Ejemplo de Código en Python para Recomendaciones de Productos:

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Datos ficticios de tarjetas gráficas NVIDIA
# Columnas: Memoria (GB), Velocidad del Reloj Base (MHz), Núcleos CUDA, Ancho de Banda de Memoria (GB/s), TDP (Consumo en Vatios)
tarjetas_graficas = np.array([
    [8, 1500, 2304, 448, 180],   # GTX 1080
    [11, 1350, 3584, 616, 250],  # RTX 2080 Ti
    [24, 1400, 10496, 936, 350], # RTX 3090
    [10, 1607, 2944, 484, 215],  # GTX 1080 Ti
    [8, 1410, 2176, 448, 175]    # RTX 2060
])

# Nombres de las tarjetas gráficas para referencia
nombres_tarjetas = ['GTX 1080', 'RTX 2080 Ti', 'RTX 3090', 'GTX 1080 Ti', 'RTX 2060']

# Preferencias del usuario (ejemplo): [Memoria, Velocidad del Reloj, Núcleos CUDA, Ancho de Banda, TDP]
preferencias_usuario = np.array([[10, 1500, 3000, 500, 200]])

# Utilizar Nearest Neighbors para encontrar la mejor opción
modelo_nn = NearestNeighbors(n_neighbors=1)
modelo_nn.fit(tarjetas_graficas)
distancia, indice = modelo_nn.kneighbors(preferencias_usuario)

# Recomendar una tarjeta gráfica
tarjeta_recomendada = nombres_tarjetas[indice[0][0]]
print(f"La tarjeta grafica recomendada es: {tarjeta_recomendada}")
```

#### Reconocimiento de Imágenes y Voz

El reconocimiento de imágenes y voz son aplicaciones populares del Machine Learning, con tecnologías como CNN y RNN.

##### Ejemplo de Código en Python para Reconocimiento de Imágenes:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Cargar el modelo MobileNetV2 preentrenado
modelo = MobileNetV2(weights='imagenet')

def predecir_imagen(ruta_imagen):
    # Cargar y preprocesar la imagen
    img = image.load_img(ruta_imagen, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Hacer la predicción
    prediccion = modelo.predict(img_array)

    # Decodificar las predicciones
    decoded = decode_predictions(prediccion, top=1)[0][0]
    return decoded

# Probar la función con una nueva imagen
ruta_imagen = '/content/malik-color1.jpg'  # Cambiar imagen
resultado = predecir_imagen(ruta_imagen)
print(f"Predicción: {resultado[1]}, Probabilidad: {resultado[2]}")
```

##### Ejemplo de Código en Python para Reconocimiento de Voz:

```python
import requests
from bs4 import BeautifulSoup

def descargar_archivo(url, ruta_destino):
    respuesta = requests.get(url, stream=True)
    with open(ruta_destino, 'wb') as archivo:
        for chunk in respuesta.iter_content(chunk_size=128):
            archivo.write(chunk)

def obtener_enlaces_descarga(url_base):
    pagina = requests.get(url_base)
    soup = BeautifulSoup(pagina.content, 'html.parser')
    
    # Aquí deberías analizar la página para encontrar los enlaces de descarga
    # Esto dependerá de la estructura HTML del sitio web de VoxForge
    enlaces = []
    return enlaces

# Conjuntos de datos demasiado grandes
url_base = 'https://repository.voxforge1.org/downloads/es/Trunk/Audio/Original/48kHz_16bit/CarlosEspinoAngulo-20140525-vip.tgz'

# Obtener enlaces de descarga
enlaces = obtener_enlaces_descarga(url_base)

# Descargar cada archivo
for enlace in enlaces:
    descargar_archivo(enlace, 'C:/laragon/www/python/test.m4a')

# Debido al tamaño del conjunto de datos, este código es solo un ejemplo y no ha sido probado.
```

Estos ejemplos de proyectos de Machine Learning ilustran cómo las diversas técnicas pueden ser aplicadas para resolver problemas reales en diferentes dominios.

### Ética y Consideraciones Legales en Machine Learning

El uso responsable del Machine Learning implica considerar aspectos éticos y legales críticos, como los sesgos en los datos y modelos, la privacidad y seguridad de datos, y el cumplimiento de regulaciones legales.

#### Sesgos en los Datos y Modelos

El sesgo en Machine Learning puede provenir de datos sesgados o prácticas de modelado inadecuadas, lo que puede conducir a resultados injustos o discriminatorios.

##### Ejemplo de lo que NO se debe hacer:

Supongamos que estamos construyendo un modelo de contratación de personal y utilizamos un conjunto de datos históricos que contienen sesgos de género.
Esto podría ser un ejemplo de un enfoque sesgado: Usar variables como género o edad para predecir la idoneidad de un candidato.

**Este tipo de prácticas no solo son éticamente cuestionables, sino que también pueden ser ilegales.**


#### Privacidad y Seguridad de Datos

La protección de los datos personales y sensibles es esencial, especialmente en el contexto de la creciente regulación global.

##### Buenas Prácticas:

- Implementar técnicas robustas de anonimización y encriptación de datos.
- Asegurar el cumplimiento de normativas como el GDPR para la protección de datos.

#### Regulaciones y Cumplimiento Legal

Es importante estar informado sobre las leyes y regulaciones aplicables, especialmente en sectores regulados como la salud y las finanzas.

##### Consideraciones Clave:

- Realizar auditorías de cumplimiento legal y de privacidad de datos.
- Ser transparente sobre el uso y procesamiento de los datos.

Estas consideraciones éticas y legales son fundamentales para garantizar que el desarrollo y aplicación del Machine Learning sean responsables y sostenibles.

## Ejercicios

[Tabla de contenidos](#tabla-de-contenidos)

[Para facilitar la realización de los ejercicios, se recomienda usar Google Colab, debido a cualquier error que pueda surgir con librerias o tensorflow](https://colab.google/)

### Parte I: Fundamentos

**Introducción al Machine Learning**

- Definición y Aplicaciones

  **Ejercicio 1:** Investiga tres aplicaciones del Machine Learning en la vida real y describe cómo se utilizan. 

  <details>
  <summary>Solución</summary>

  1. **Reconocimiento de Voz:** Utilizado en asistentes virtuales como Siri o Google Assistant, el ML permite a estos sistemas entender y responder a comandos de voz.
  2. **Recomendaciones de Productos:** Sitios web como Amazon utilizan ML para analizar el comportamiento de compra y ofrecer recomendaciones personalizadas.
  3. **Detección de Fraude:** En el sector bancario, los sistemas de ML analizan patrones de transacciones para identificar y prevenir actividades fraudulentas.

  </details>

  **Ejercicio 2:** Escribe un breve párrafo sobre cómo el Machine Learning puede mejorar un área de tu interés personal o profesional.

  <details>
  <summary>Solución</summary>

  *Esta solución será subjetiva dependiendo del área de interés del usuario.*

  </details>

**Tipos de Aprendizaje**

- Aprendizaje Supervisado

  **Ejercicio 3:** Utiliza un conjunto de datos simple para realizar una regresión lineal con Python. Utiliza `sklearn` para entrenar un modelo con datos de prueba y muestra la línea de regresión.

  <details>
  <summary>Solución</summary>

  ```python
  from sklearn.linear_model import LinearRegression
  import matplotlib.pyplot as plt
  import numpy as np

  # Datos de ejemplo
  X = np.array([[1], [2], [3], [4]])
  y = np.array([2, 4, 6, 8])

  # Entrenar el modelo
  modelo = LinearRegression()
  modelo.fit(X, y)

  # Predecir y graficar
  X_pred = np.array([[0], [5]])
  y_pred = modelo.predict(X_pred)

  plt.scatter(X, y)
  plt.plot(X_pred, y_pred, color='red')
  plt.show()
  ```

  </details>

- Aprendizaje No Supervisado

  **Ejercicio 4:** Realiza un ejercicio de agrupamiento con el algoritmo K-Means en Python utilizando `sklearn`. Usa datos generados aleatoriamente y muestra los grupos resultantes en un gráfico.

  <details>
  <summary>Solución</summary>

  ```python
  from sklearn.cluster import KMeans
  import matplotlib.pyplot as plt
  import numpy as np

  # Generar datos
  X = np.random.rand(100, 2)

  # Aplicar K-Means
  kmeans = KMeans(n_clusters=3)
  kmeans.fit(X)
  y_kmeans = kmeans.predict(X)

  plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
  plt.show()
  ```

  </details>

- Aprendizaje por Refuerzo

  **Ejercicio 5:** Investiga un entorno simple de OpenAI Gym y escribe un código en Python que inicie el entorno e imprima su estado inicial. (Pista: pip install gym)

  <details>
  <summary>Solución</summary>

  ```python
  import gym

  # Crear e iniciar el entorno
  env = gym.make('CartPole-v1')
  env.reset()

  print("Estado inicial:", env.state)
  ```

  </details>


**Bibliotecas Esenciales: NumPy, Pandas**

- NumPy

  **Ejercicio 7:** Crea un array de NumPy de 2x3 con números aleatorios y calcula la media de los valores.

  <details>
  <summary>Solución</summary>

  ```python
  import numpy as np

  arr = np.random.rand(2, 3)
  media = np.mean(arr)
  print("Media del array:", media)
  ```

  </details>

- Pandas

  **Ejercicio 8:** Lee un archivo CSV utilizando Pandas y muestra las primeras cinco filas.

  <details>
  <summary>Solución</summary>

  ```python
  import pandas as pd

  # Suponiendo que existe un archivo 'datos.csv'
  df = pd.read_csv('datos.csv')
  print(df.head())
  ```

  </details>

**Visualización de Datos: Matplotlib, Seaborn**

- Matplotlib

  **Ejercicio 9:** Crea un gráfico de barras utilizando Matplotlib para visualizar la cantidad de estudiantes en diferentes cursos.

  <details>
  <summary>Solución</summary>

  ```python
  import matplotlib.pyplot as plt

  cursos = ['Curso A', 'Curso B', 'Curso C']
  estudiantes = [30, 45, 22]

  plt.bar(cursos, estudiantes)
  plt.xlabel('Cursos')
  plt.ylabel('Número de Estudiantes')
  plt.title('Estudiantes por Curso')
  plt.show()
  ```

  </details>

- Seaborn

  **Ejercicio 10:** Utiliza Seaborn para crear un gráfico de dispersión con un conjunto de datos de tu elección.

  <details>
  <summary>Solución</summary>

  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  import numpy as np

  # Datos de ejemplo
  x = np.random.rand(50)
  y = np.random.rand(50)

  sns.scatterplot(x=x, y=y)
  plt.show()
  ```

  </details>


### Parte II: Aprendizaje Supervisado (Completa y Revisada)

**Regresión**

**Ejercicio 1:** Utiliza el conjunto de datos de Boston Housing para realizar una regresión lineal múltiple con Python. Debes realizar los siguientes pasos:
1. Cargar el conjunto de datos.
2. Dividir los datos en un conjunto de entrenamiento y otro de prueba.
3. Entrenar un modelo de regresión lineal con el conjunto de entrenamiento.
4. Evaluar el rendimiento del modelo con el conjunto de prueba, utilizando el coeficiente R².

<details>
<summary>Solución</summary>

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Cargar datos
boston = load_boston()
X = boston.data
y = boston.target

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Coeficiente R²:", r2)
```

</details>

**Clasificación**

- **K-Nearest Neighbors (KNN)**

  **Ejercicio 2:** Implementa un clasificador KNN utilizando el conjunto de datos Iris. Realiza lo siguiente:
  1. Cargar el conjunto de datos Iris.
  2. Dividir los datos en conjunto de entrenamiento y de prueba.
  3. Entrenar un clasificador KNN con el conjunto de entrenamiento.
  4. Evaluar la precisión del clasificador con el conjunto de prueba.

  <details>
  <summary>Solución</summary>

  ```python
  from sklearn.datasets import load_iris
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score

  # Cargar datos
  iris = load_iris()
  X = iris.data
  y = iris.target

  # Dividir los datos
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Entrenar el modelo
  modelo_knn = KNeighborsClassifier(n_neighbors=3)
  modelo_knn.fit(X_train, y_train)

  # Evaluar el modelo
  y_pred = modelo_knn.predict(X_test)
  precision = accuracy_score(y_test, y_pred)
  print("Precisión del clasificador KNN:", precision)
  ```

  </details>


- **Máquinas de Vectores de Soporte (SVM)**
  - **Ejercicio 3:** Implementa un clasificador SVM utilizando el conjunto de datos Iris. Debes:
    1. Utilizar el mismo conjunto de datos dividido (Iris) del ejercicio anterior.
    2. Entrenar un clasificador SVM con el conjunto de entrenamiento.
    3. Evaluar la precisión del clasificador con el conjunto de prueba.

    <details>
    <summary>Solución</summary>

    ```python
    from sklearn import datasets
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # Cargar el conjunto de datos Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Dividir los datos en conjuntos de entrenamiento y prueba
    # Esto se debe hacer solo si no se ha hecho en el ejercicio anterior
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo SVM
    modelo_svm = SVC()
    modelo_svm.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred_svm = modelo_svm.predict(X_test)
    precision_svm = accuracy_score(y_test, y_pred_svm)
    print("Precisión del clasificador SVM:", precision_svm)
    ```

    </details>

- **Árboles de Decisión y Bosques Aleatorios**
  - **Árboles de Decisión**
    - **Ejercicio 4:** Implementa un árbol de decisión para clasificar el conjunto de datos Iris. Realiza lo siguiente:
      1. Utilizar el mismo conjunto de datos dividido (Iris) de los ejercicios anteriores.
      2. Entrenar un árbol de decisión con el conjunto de entrenamiento.
      3. Evaluar la precisión del árbol de decisión con el conjunto de prueba.

      <details>
      <summary>Solución</summary>

      ```python
      from sklearn import datasets
      from sklearn.model_selection import train_test_split
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.metrics import accuracy_score

      # Cargar el conjunto de datos Iris
      iris = datasets.load_iris()
      X = iris.data
      y = iris.target

      # Dividir los datos en conjuntos de entrenamiento y prueba
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      # Entrenar el modelo de árbol de decisión
      modelo_arbol = DecisionTreeClassifier()
      modelo_arbol.fit(X_train, y_train)

      # Evaluar el modelo
      y_pred_arbol = modelo_arbol.predict(X_test)
      precision_arbol = accuracy_score(y_test, y_pred_arbol)
      print("Precisión del árbol de decisión:", precision_arbol)
      ```

      </details>

  - **Bosques Aleatorios**
    - **Ejercicio 5:** Implementa un modelo de bosque aleatorio para el conjunto de datos Iris. Debes:
      1. Utilizar el mismo conjunto de datos dividido (Iris) de los ejercicios anteriores.
      2. Entrenar un modelo de bosque aleatorio con el conjunto de entrenamiento.
      3. Evaluar la precisión del modelo con el conjunto de prueba.

      <details>
      <summary>Solución</summary>

      ```python
      from sklearn import datasets
      from sklearn.model_selection import train_test_split
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.metrics import accuracy_score

      # Cargar el conjunto de datos Iris
      iris = datasets.load_iris()
      X = iris.data
      y = iris.target

      # Dividir los datos en conjuntos de entrenamiento y prueba
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      # Entrenar el modelo de bosque aleatorio
      modelo_bosque = RandomForestClassifier()
      modelo_bosque.fit(X_train, y_train)

      # Evaluar el modelo
      y_pred_bosque = modelo_bosque.predict(X_test)
      precision_bosque = accuracy_score(y_test, y_pred_bosque)
      print("Precisión del bosque aleatorio:", precision_bosque)
      ```

      </details>

### Parte III: Aprendizaje No Supervisado

**Clustering**

- **K-Means**

  - **Características del K-Means:**

  - **Ejemplo de Código en Python:**

    **Ejercicio 1:** Implementa el algoritmo K-Means en Python utilizando `sklearn` con un conjunto de datos generado aleatoriamente. Determina el número óptimo de clusters.

    <details>
    <summary>Solución</summary>

    ```python
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np

    # Generar datos
    X = np.random.rand(100, 2)

    # Determinar el número óptimo de clusters
    inercias = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)

    plt.plot(range(1, 10), inercias, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.show()
    ```

    </details>

  - **Resultados y Aplicaciones:**

- **Clustering Jerárquico**

  - **Características del Clustering Jerárquico:**

  - **Ejemplo de Código en Python:**

    **Ejercicio 2:** Realiza un clustering jerárquico en Python con `scipy`. Utiliza el método de enlace 'ward' y visualiza el dendrograma.

    <details>
    <summary>Solución</summary>

    ```python
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    import numpy as np

    # Generar datos
    X = np.random.rand(50, 2)

    # Clustering jerárquico
    Z = linkage(X, 'ward')

    # Dendrograma
    dendrogram(Z)
    plt.show()
    ```

    </details>

  - **Aplicaciones del Clustering Jerárquico:**

- **Clustering DBSCAN**

  - **Características del DBSCAN:**

  - **Ejemplo de Código en Python:**

    **Ejercicio 3:** Implementa el algoritmo DBSCAN en Python usando `sklearn`. Prueba con diferentes valores de `eps` y `min_samples`.

    <details>
    <summary>Solución</summary>

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs

    # Generar datos de ejemplo
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Valores de 'eps' y 'min_samples' para probar
    valores_eps = [0.1, 0.3, 0.5, 0.7]
    valores_min_samples = [5, 10, 15]

    # Crear subplots
    fig, axs = plt.subplots(len(valores_eps), len(valores_min_samples), figsize=(15, 10))

    # Aplicar DBSCAN con diferentes valores de 'eps' y 'min_samples'
    for i, eps in enumerate(valores_eps):
        for j, min_samples in enumerate(valores_min_samples):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X)
            axs[i, j].scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
            axs[i, j].set_title(f'eps={eps}, min_samples={min_samples}')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()
    ```

    </details>

  - **Aplicaciones del DBSCAN:**

**Reducción de Dimensionalidad**

- **Análisis de Componentes Principales (PCA)**

  - **Ejemplo de Código en Python:**

    **Ejercicio 4:** Utiliza PCA en Python para reducir la dimensionalidad de un conjunto de datos y visualiza el resultado.

    <details>
    <summary>Solución</summary>

    ```python
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # Cargar el conjunto de datos Iris
    iris = load_iris()
    X = iris.data

    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Visualizar los resultados
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('PCA del Conjunto de Datos Iris')
    plt.show()
    ```

    </details>

- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**

  - **Ejemplo de Código en Python:**

    **Ejercicio 5:** Implementa t-SNE en Python para visualizar un conjunto de datos de alta dimensionalidad.

    <details>
    <summary>Solución</summary>

    ```python
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    # Cargar el conjunto de datos Iris
    iris = load_iris()
    X = iris.data

    # Aplicar t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    # Visualizar los resultados
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('t-SNE del Conjunto de Datos Iris')
    plt.show()
    ```

    </details>

### Parte IV: Aprendizaje por Refuerzo

**Conceptos Básicos de Aprendizaje por Refuerzo**

- **El Problema del Bandido Multibrazo**
  
  **Ejercicio 1:** Implementa una simulación del problema del bandido multibrazo en Python. Utiliza un enfoque simple como el método ε-greedy para encontrar la mejor palanca.

  <details>
  <summary>Solución</summary>

  ```python
  import numpy as np

  # Número de brazos del bandido
  n_brazos = 10
  np.random.seed(42)
  q_verdaderos = np.random.randn(n_brazos)  # Recompensas verdaderas

  # Inicialización
  q_estimados = np.zeros(n_brazos)
  n_intentos = np.zeros(n_brazos)
  epsilon = 0.1
  recompensas = []

  for _ in range(1000):
      if np.random.rand() < epsilon:
          accion = np.random.choice(n_brazos)  # Exploración
      else:
          accion = np.argmax(q_estimados)  # Explotación

      # Recompensa y actualización
      recompensa = q_verdaderos[accion] + np.random.randn()
      n_intentos[accion] += 1
      q_estimados[accion] += (recompensa - q_estimados[accion]) / n_intentos[accion]
      recompensas.append(recompensa)
  mejor_brazo = np.argmax(q_estimados)
  print("Recompensas acumuladas:", np.sum(recompensas))
  print("Mejor brazo:", mejor_brazo)
  print("Estimación de recompensa del mejor brazo:", q_estimados[mejor_brazo])
  ```

  </details>

- **Algoritmos de Valor y Política**

  **Ejercicio 2:** Investiga y describe brevemente la diferencia entre algoritmos de valor y algoritmos de política en el contexto del aprendizaje por refuerzo.

  <details>
  <summary>Solución</summary>

  *La respuesta dependerá de la investigación realizada por el usuario.*

  </details>

**Aplicaciones Avanzadas**

- **Q-Learning**

  - **Ejemplo de Código en Python:**

    **Ejercicio 3:** Implementa un algoritmo básico de Q-learning en Python para resolver un entorno simple de OpenAI Gym, como 'FrozenLake-v1'.

    <details>
    <summary>Solución</summary>

    ```python
    import gym
    import numpy as np

    env = gym.make('FrozenLake-v1')
    n_estados = env.observation_space.n
    n_acciones = env.action_space.n

    Q = np.zeros([n_estados, n_acciones])
    lr = 0.8
    gamma = 0.95
    num_episodios = 2000

    for i in range(num_episodios):
        estado = env.reset()
        done = False

        while not done:
            accion = np.argmax(Q[estado,:] + np.random.randn(1, n_acciones) * (1. / (i + 1)))
            estado_nuevo, recompensa, done, _ = env.step(accion)
            Q[estado,accion] = Q[estado,accion] + lr * (recompensa + gamma * np.max(Q[estado_nuevo,:]) - Q[estado,accion])
            estado = estado_nuevo

    print("Tabla Q aprendida:")
    print(Q)
    ```

    </details>

- **Deep Q-Networks (DQN)**

  - **Características de DQN:**

  - **Ejemplo de Código en Python para DQN:**

    **Ejercicio 4:** Investiga y describe brevemente las características clave de los Deep Q-Networks (DQN) y su importancia en el aprendizaje por refuerzo.

    <details>
    <summary>Solución</summary>

    *La respuesta dependerá de la investigación realizada por el usuario.*

    </details>

### Parte V: Técnicas Avanzadas

**Redes Neuronales y Deep Learning**

- **Perceptrones y Redes Neuronales Artificiales**
  
  - **Ejemplo de Código en Python para una ANN:**

    **Ejercicio 1:** Implementa una red neuronal artificial simple en Python utilizando Keras para clasificar el conjunto de datos de dígitos MNIST.

    <details>
    <summary>Solución</summary>

    ```python
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical

    # Cargar datos
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocesamiento
    X_train = X_train.reshape(60000, 784).astype('float32') / 255
    X_test = X_test.reshape(10000, 784).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Construir el modelo
    modelo = Sequential()
    modelo.add(Dense(512, activation='relu', input_shape=(784,)))
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Entrenar el modelo
    modelo.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(X_test, y_test))

    # Evaluar el modelo
    puntuacion = modelo.evaluate(X_test, y_test, verbose=0)
    print("Precisión:", puntuacion[1])
    ```

    </details>

- **Redes Neuronales Convolucionales (CNN)**
  
  - **Ejemplo de Código en Python para una CNN:**

    **Ejercicio 2:** Crea una red neuronal convolucional en Python usando Keras para clasificar imágenes del conjunto de datos CIFAR-10.

    <details>
    <summary>Solución</summary>

    ```python
    from keras.datasets import cifar10
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.utils import to_categorical

    # Cargar datos
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocesamiento
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Construir el modelo
    modelo_cnn = Sequential()
    modelo_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    modelo_cnn.add(MaxPooling2D((2, 2)))
    modelo_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    modelo_cnn.add(MaxPooling2D((2, 2)))
    modelo_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    modelo_cnn.add(Flatten())
    modelo_cnn.add(Dense(64, activation='relu'))
    modelo_cnn.add(Dense(10, activation='softmax'))

    modelo_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    modelo_cnn.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Evaluar el modelo
    puntuacion_cnn = modelo_cnn.evaluate(X_test, y_test, verbose=0)
    print("Precisión CNN:", puntuacion_cnn[1])
    ```

    </details>

- **Redes Neuronales Recurrentes (RNN)**
  
  - **Ejemplo de Código en Python para una RNN:**

    **Ejercicio 3:** Construye y entrena una red neuronal recurrente en Python con Keras para predecir la próxima palabra en una secuencia de texto.

    <details>
    <summary>Solución</summary>
    
    ```python
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical

    # Cargar datos
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocesamiento
    X_train = X_train.reshape(60000, 784).astype('float32') / 255
    X_test = X_test.reshape(10000, 784).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Construir el modelo
    modelo = Sequential()
    modelo.add(Dense(512, activation='relu', input_shape=(784,)))
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Entrenar el modelo
    modelo.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(X_test, y_test))

    # Evaluar el modelo
    puntuacion = modelo.evaluate(X_test, y_test, verbose=0)
    print("Precisión:", puntuacion[1])
    ```

    </details>

**Natural Language Processing (NLP) con Python**

- **Procesamiento del Lenguaje Natural**
  
  - **Ejemplo de Código en Python para Tokenización:**

    **Ejercicio 4:** Utiliza NLTK en Python para realizar la tokenización de un texto en inglés.

    <details>
    <summary>Solución</summary>

    ```python
    import nltk
    from nltk.tokenize import word_tokenize

    nltk.download('punkt')
    texto = "Me alegra mucho programar en Python"

    tokens = word_tokenize(texto)
    print("Tokens:", tokens)
    ```

    </details>

### Parte VI: Herramientas y Mejores Prácticas

**Evaluación y Ajuste de Modelos**

- **Validación Cruzada**

  - **Ejemplo de Código en Python para Validación Cruzada:**

    **Ejercicio 1:** Implementa la validación cruzada en Python para evaluar la efectividad de un modelo de clasificación.

    <details>
    <summary>Solución</summary>

    ```python
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    # Cargar datos
    iris = load_iris()
    X, y = iris.data, iris.target

    # Crear el modelo
    modelo = RandomForestClassifier()

    # Validación cruzada
    puntuaciones = cross_val_score(modelo, X, y, cv=5)
    print("Precisión de cada fold:", puntuaciones)
    print("Precisión media:", puntuaciones.mean())
    ```

    </details>

- **Ajuste de Hiperparámetros**

  - **Ejemplo de Código en Python para Ajuste de Hiperparámetros:**

    **Ejercicio 2:** Utiliza `GridSearchCV` para encontrar los mejores hiperparámetros de un modelo.

    <details>
    <summary>Solución</summary>

    ```python
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification

    # Generar un conjunto de datos sintético
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

    # Crear el modelo SVC y definir una cuadrícula de parámetros para la búsqueda
    modelo = SVC()
    parametros = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

    # Configurar GridSearchCV
    buscador = GridSearchCV(modelo, parametros)

    # Ejecutar la búsqueda de parámetros
    buscador.fit(X, y)

    # Mostrar los mejores parámetros encontrados
    print("Mejores parámetros:", buscador.best_params_)
    ```

    </details>

- **Métricas de Evaluación**

  - **Ejemplo de Métricas para Clasificación:**

    **Ejercicio 3:** Calcula varias métricas de evaluación para un modelo de clasificación.

    <details>
    <summary>Solución</summary>

    ```python
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    # Generar datos sintéticos
    n_muestras = 1000
    n_caracteristicas = 10
    n_dimensiones = 10

    # Datos tridimensionales aleatorios
    X = np.random.rand(n_muestras, n_caracteristicas, n_dimensiones)

    # Etiquetas binarias aleatorias
    y = np.random.randint(0, 2, size=n_muestras)

    # Aplanar X para convertirlo a dos dimensiones
    X_2d = X.reshape(n_muestras, n_caracteristicas * n_dimensiones)

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3)

    # Entrenar el modelo
    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Métricas
    print("Precisión:", accuracy_score(y_test, y_pred))
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    ```

    </details>

### Parte VII: Estudios de Caso y Proyectos

**Proyectos de Machine Learning**

- **Detección de Fraude**

  - **Ejemplo de Código en Python para Detección de Fraude:**

    **Ejercicio 1:** Implementa un modelo básico para detectar transacciones fraudulentas en un conjunto de datos financiero 
    (Este ejercicio es conceptual, requiere de un .csv)

    <details>
    <summary>Solución</summary>

    ```python
    # Este ejercicio es conceptual, requiere de un .csv

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Supongamos que 'data' es un DataFrame que contiene nuestro conjunto de datos
    data = pd.read_csv('transacciones.csv')
    X = data.drop('etiqueta_fraude', axis=1)
    y = data['etiqueta_fraude']

    # Dividir el conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Crear y entrenar el modelo
    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)

    # Evaluar el modelo
    precision = modelo.score(X_test, y_test)
    print("Precisión:", precision)
    ```

    </details>

- **Recomendaciones de Productos**

  - **Ejemplo de Código en Python para Recomendaciones de Productos:**

    **Ejercicio 2:** Construye un sistema de recomendación simple utilizando el filtrado colaborativo.

    <details>
    <summary>Solución</summary>

    ```python
    # Ejercicio conceptual.

    # Importar bibliotecas necesarias
    from surprise import Dataset, Reader, KNNBasic
    from surprise.model_selection import cross_validate

    # Cargar los datos
    datos = Dataset.load_from_df(df[['usuario_id', 'producto_id', 'calificación']], Reader())

    # Crear y evaluar el modelo de filtrado colaborativo
    modelo = KNNBasic()
    resultados = cross_validate(modelo, datos, measures=['RMSE'], cv=3)
    ```

    </details>

- **Reconocimiento de Imágenes y Voz**

  - **Ejemplo de Código en Python para Reconocimiento de Imágenes:**

    **Ejercicio 3:** Utiliza una red neuronal convolucional para clasificar imágenes.

    <details>
    <summary>Solución</summary>
    
    ```python
    from keras.datasets import cifar10
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from keras.utils import to_categorical

    # Cargar datos
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocesamiento
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Construir el modelo CNN
    modelo = Sequential()
    modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Conv2D(64, (3, 3), activation='relu'))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dense(10, activation='softmax'))

    # Compilar el modelo
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    modelo.fit(X_train, y_train, epochs=10, batch_size=64)

    # Evaluar el modelo
    evaluacion = modelo.evaluate(X_test, y_test)
    print(f'Precisión: {evaluacion[1]}')
    ```
    </details>

  - **Ejemplo de Código en Python para Reconocimiento de Voz:**

    **Ejercicio 4:** Crea un modelo básico para reconocer comandos de voz o palabras clave.

    <details>
    <summary>Solución</summary>

    ```python
    import numpy as np
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.utils import to_categorical

    # Generar datos sintéticos para la demostración
    # 1000 muestras, cada una con 40 características (simulando MFCCs)
    X = np.random.rand(1000, 40)
    y = np.random.randint(2, size=(1000, 1))  # Etiquetas binarias

    # Convertir etiquetas a formato categórico
    y = to_categorical(y)

    # Redefinir X para tener una dimensión adicional (esperada por LSTM)
    X = np.expand_dims(X, -1)

    # Crear el modelo RNN
    modelo_rnn = Sequential()
    modelo_rnn.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))  # 50 unidades LSTM
    modelo_rnn.add(Dense(2, activation='softmax'))  # 2 clases

    # Compilar el modelo
    modelo_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenar el modelo con datos sintéticos
    modelo_rnn.fit(X, y, epochs=5, batch_size=32)
    print("Para que el modelo aprenda, se deberian de utilizar datos no aleatorios")
    ```

    </details>

## Sistema CRUD para Machine Learning en Python con MySQL

[Tabla de contenidos](#tabla-de-contenidos)

### Paso 1: Configurar el Entorno

#### Instalar MySQL y Python

- Instale MySQL en su sistema y un cliente de MySQL como MySQL Workbench.
  
  ```sql
  DROP DATABASE IF EXISTS machinelearningdb;
  CREATE DATABASE machinelearningdb;
  USE machinelearningdb;

  CREATE TABLE clientes (
      id INT AUTO_INCREMENT PRIMARY KEY,
      nombre VARCHAR(255),
      edad INT,
      ingresos FLOAT
  );

  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Ana', 30, 40000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Juan', 25, 35000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Lucía', 40, 50000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Carlos', 22, 28000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Sofía', 35, 45000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Luis', 45, 55000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Marta', 28, 32000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('David', 33, 48000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Elena', 50, 60000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Mario', 38, 52000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Laura', 29, 37000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Pedro', 41, 53000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Inés', 31, 41000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Diego', 39, 51000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Raquel', 34, 47000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Jorge', 27, 33000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Carmen', 36, 49000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Óscar', 42, 56000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('María', 31, 41000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('José', 32, 42000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Sara', 33, 43000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Luis', 34, 44000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Daniel', 35, 45000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Carolina', 36, 46000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Miguel', 37, 47000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Isabel', 38, 48000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Pablo', 39, 49000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Lucas', 40, 50000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Eva', 41, 51000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Cristian', 42, 52000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Lorena', 43, 53000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Andrés', 44, 54000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Clara', 45, 55000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Hugo', 46, 56000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Valentina', 47, 57000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Javier', 48, 58000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Adriana', 49, 59000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Roberto', 50, 60000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Marina', 31, 41000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Antonio', 32, 42000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Ana', 33, 43000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Pedro', 34, 44000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Laura', 35, 45000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Fernando', 36, 46000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Rosa', 37, 47000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Alejandro', 38, 48000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Natalia', 39, 49000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Santiago', 40, 50000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Marcela', 41, 51000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Víctor', 42, 52000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Camila', 43, 53000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Gustavo', 44, 54000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Isabella', 45, 55000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Raúl', 46, 56000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Paula', 47, 57000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Felipe', 48, 58000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Catalina', 49, 59000);
  INSERT INTO clientes (nombre, edad, ingresos) VALUES ('Joaquín', 50, 60000);
  ```

- Instale las bibliotecas necesarias en Python: `mysql-connector-python`, `numpy`, `pandas` y `scikit-learn`.

  ```bash
  pip install mysql-connector-python numpy pandas scikit-learn
  ```

### Paso 2: Implementación CRUD (Código)

  ```python
  import mysql.connector
  from mysql.connector import Error
  import pandas as pd
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error

  # Función para conectar a la base de datos
  def conectar_db():
      try:
          conn = mysql.connector.connect(
              host="localhost",
              user="root",
              password="root",
              database="machinelearningdb"
          )
          if conn.is_connected():
              return conn
      except Error as e:
          print("Error al conectar a MySQL", e)

  # Funciones CRUD

  def insertar_cliente():
      nombre = input("Ingrese el nombre del cliente: ")
      edad = int(input("Ingrese la edad del cliente: "))
      ingresos = float(input("Ingrese los ingresos del cliente: "))
      try:
          conn = conectar_db()
          cursor = conn.cursor()
          query = "INSERT INTO clientes (nombre, edad, ingresos) VALUES (%s, %s, %s)"
          cursor.execute(query, (nombre, edad, ingresos))
          conn.commit()
          print("Cliente insertado con éxito.")
      except Error as e:
          print("Error al insertar cliente", e)
      finally:
          if conn.is_connected():
              cursor.close()
              conn.close()

  def leer_clientes():
      try:
          conn = conectar_db()
          cursor = conn.cursor()
          query = "SELECT * FROM clientes"
          cursor.execute(query)
          result = cursor.fetchall()
          return pd.DataFrame(result, columns=['id', 'nombre', 'edad', 'ingresos'])
      except Error as e:
          print("Error al leer clientes", e)
      finally:
          if conn.is_connected():
              cursor.close()
              conn.close()

  def actualizar_cliente():
      id_cliente = int(input("Ingrese el ID del cliente a actualizar: "))
      nombre = input("Ingrese el nuevo nombre del cliente: ")
      edad = int(input("Ingrese la nueva edad del cliente: "))
      ingresos = float(input("Ingrese los nuevos ingresos del cliente: "))
      try:
          conn = conectar_db()
          cursor = conn.cursor()
          query = "UPDATE Clientes SET nombre = %s, edad = %s, ingresos = %s WHERE id = %s"
          cursor.execute(query, (nombre, edad, ingresos, id_cliente))
          conn.commit()
          print("Cliente actualizado con éxito.")
      except Error as e:
          print("Error al actualizar cliente", e)
      finally:
          if conn.is_connected():
              cursor.close()
              conn.close()

  def eliminar_cliente():
      id_cliente = int(input("Ingrese el ID del cliente a eliminar: "))
      try:
          conn = conectar_db()
          cursor = conn.cursor()
          query = "DELETE FROM clientes WHERE id = %s"
          cursor.execute(query, (id_cliente,))
          conn.commit()
          print("Cliente eliminado con éxito.")
      except Error as e:
          print("Error al eliminar cliente", e)
      finally:
          if conn.is_connected():
              cursor.close()
              conn.close()

  # Funciones para manejar el modelo de Machine Learning
  def entrenar_modelo():
      datos = leer_clientes()
      X = datos[['edad']]  # Característica: Edad
      y = datos['ingresos']  # Etiqueta: Ingresos
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
      modelo = LinearRegression()
      modelo.fit(X_train, y_train)
      predicciones = modelo.predict(X_test)
      mse = mean_squared_error(y_test, predicciones)
      return modelo, mse

  modelo_global = None  # Variable global para almacenar el modelo entrenado

  # Función para hacer una predicción
  def hacer_prediccion():
      global modelo_global
      if modelo_global is not None:
          try:
              edad_para_prediccion = int(input("Ingrese la edad para la predicción: "))
              # Asegurarse de que los datos para la predicción tengan el mismo formato y nombres de características
              datos_prediccion = pd.DataFrame([[edad_para_prediccion]], columns=['edad'])
              prediccion_ingreso = modelo_global.predict(datos_prediccion)
              print(f"Ingresos predichos para una persona de {edad_para_prediccion} años: {prediccion_ingreso[0]}")
          except ValueError:
              print("Por favor, ingrese un número válido.")
      else:
          print("Primero necesita entrenar el modelo.")

  # Menú interactivo para operaciones CRUD
  def menu_crud():
      global modelo_global
      while True:
          print("\nOperaciones CRUD:")
          print("1. Insertar Cliente")
          print("2. Mostrar Clientes")
          print("3. Actualizar Cliente")
          print("4. Eliminar Cliente")
          print("5. Entrenar Modelo")
          print("6. Hacer una Predicción")
          print("7. Salir")
          opcion = input("Seleccione una opción: ")

          try:
              opcion = int(opcion)
          except ValueError:
              print("Por favor, ingrese un número válido.")
              continue

          if opcion == 1:
              insertar_cliente()
          elif opcion == 2:
              print(leer_clientes())
          elif opcion == 3:
              actualizar_cliente()
          elif opcion == 4:
              eliminar_cliente()
          elif opcion == 5:
              modelo_global, mse = entrenar_modelo()
              print(f"Modelo entrenado. Error cuadrático medio: {mse}")
          elif opcion == 6:
              hacer_prediccion()
          elif opcion == 7:
              break
          else:
              print("Opción no válida.")

  menu_crud()
  ```

## Bibliografía

[Tabla de contenidos](#tabla-de-contenidos)

#### [Kaggle](https://www.kaggle.com/learn)

#### [W3Schools](https://www.w3schools.com/python/python_ml_getting_started.asp)

#### [Machinelearningmastery](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)

#### [Scipy](https://scipy.org/)

#### [FreeCodeCamp](https://www.freecodecamp.org/espanol/learn/machine-learning-with-python/)

#### [ChatGPT](https://chat.openai.com/)

#### [SIIM](https://siim.org/)

#### [Documentación BeautifulSoup (bs4)](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

#### [Documentación collections](https://docs.python.org/3/library/collections.html)

#### [Documentación Flask](https://flask.palletsprojects.com/)

#### [Documentación Gym](https://www.gymlibrary.dev/content/tutorials/)

#### [Documentación joblib](https://joblib.readthedocs.io/)

#### [Documentación keras API](https://keras.io/)

#### [Documentación Matplotlib](https://matplotlib.org/stable/contents.html)

#### [Documentación MySQL Connector/Python (mysql.connector)](https://dev.mysql.com/doc/connector-python/en/)

#### [Documentación NLTK](https://www.nltk.org/)

#### [Documentación NumPy](https://numpy.org/doc/)

#### [Documentación pandas](https://pandas.pydata.org/docs/)

#### [Documentación Random](https://docs.python.org/3/library/random.html)

#### [Documentación requests](https://docs.python-requests.org/en/latest/)

#### [Documentación SciPy](https://docs.scipy.org/doc/scipy/reference/)

#### [Documentación seaborn](https://seaborn.pydata.org/)

#### [Documentación scikit-learn (sklearn)](https://scikit-learn.org/stable/user_guide.html)

#### [Documentación Surprise](https://surprise.readthedocs.io/en/stable/)

#### [Documentación TensorFlow](https://www.tensorflow.org/learn)

#### [Documentación TextBlob](https://textblob.readthedocs.io/)

#### [IMAGENET](https://www.image-net.org/index.php)

#### [VoxForge](https://www.voxforge.org/)

#### [Google Colab](https://colab.google/)