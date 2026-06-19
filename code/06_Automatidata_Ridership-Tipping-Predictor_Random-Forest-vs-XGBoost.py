#!/usr/bin/env python
# coding: utf-8

# # **Automatidata project**
# **Course 5 - The Nuts and bolts of machine learning**

# You are a data professional in a data analytics firm called Automatidata. Their client, the New York City Taxi & Limousine Commission (New York City TLC), was impressed with the work you have done and has requested that you **build a machine learning model to predict if a customer will not leave a tip**. They want to use the model in an app that will alert taxi drivers to customers who are unlikely to tip, since drivers depend on tips.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # Course 5 End-of-course project: Build a machine learning model
# 
# In this activity, you will practice using tree-based modeling techniques to predict on a binary target class.  
# <br/>   
# 
# **The purpose** of this model is to find ways to generate more revenue for taxi cab drivers.  
#   
# **The goal** of this model is to predict whether or not a customer is a generous tipper.  
# <br/>  
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations 
# * Consider the ethical implications of the request 
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, complete an Executive Summary using the questions listed on the PACE Strategy Document. 
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work. 
# 
# 

# # Build a machine learning model

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## PACE: Plan 
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 1.   What are you being asked to do?
# 
# 
# 2.   What are the ethical implications of the model? What are the consequences of your model making errors?
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a customer will give a tip, but they actually won't)?
#   
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a customer will not give a tip, but they actually will)?  
#   
#   
# 3.   Do the benefits of such a model outweigh the potential problems?
#   
# 4.   Would you proceed with the request to build this model? Why or why not?
#  
# 5.   Can the objective be modified to make it less problematic?
#  
# 

# 
# 1. Question: What are you being asked to do? (¿Qué se le pide que haga?)
# 
# Español:
# 
# La Solicitud del Cliente: Se me pide construir un modelo de aprendizaje automático basado en árboles (Clasificación Binaria) utilizando los datos históricos de la Comisión de Taxis y Limusinas de Nueva York (NYC TLC) para predecir si un cliente es o no un propinador generoso (definido operacionalmente como una propina ≥20%).
# 
# Los Entregables Requeridos: Para completar con éxito esta fase, debo liderar y ejecutar de forma estructurada las siguientes tareas:
# 
# Consideraciones Éticas: Evaluar las implicaciones del uso de este modelo en una aplicación móvil para los conductores y determinar si el objetivo del modelo debe ser ajustado para evitar la discriminación de usuarios.
# 
# Ingeniería de Características: Realizar la selección, extracción y transformación de las variables del trayecto (como marcas de tiempo y zonas geográficas) para estructurar el conjunto de datos de entrenamiento de manera óptima.
# 
# Modelado y Evaluación: Desarrollar, ajustar hiperparámetros y comparar un modelo de Bosque Aleatorio (Random Forest) y un modelo XGBoost, evaluando su desempeño mediante métricas de clasificación robustas (Precisión, Recall y F1-Score).
# 
# Comunicación: Elaborar un resumen ejecutivo final adaptando el lenguaje técnico para los directores de Automatidata y el enfoque operativo/financiero para los tomadores de decisiones de la NYC TLC.
# 
# English:
# 
# The Client's Mandate: I am being asked to develop a tree-based machine learning model (Binary Classification) using historical NYC Taxi and Limousine Commission (TLC) data to predict whether or not a customer will be a generous tipper (operationally defined as a gratuity ≥20%).
# 
# Required Deliverables: To successfully fulfill this milestone, I am responsible for executing the following pipeline tasks:
# 
# Ethical Considerations: Evaluate the behavioral and operational implications of deploying this predictive model within a driver-facing application, assessing whether the objective requires adjustment to mitigate service discrimination across specific passenger groups.
# 
# Feature Engineering: Execute feature selection, extraction, and mathematical transformations on raw trip attributes (such as timestamps and spatial zone IDs) to optimize the training matrix.
# 
# Modeling and Evaluation: Build, tune, and evaluate a baseline Random Forest classifier alongside an XGBoost model, micro-auditing performance metrics via Precision, Recall, and F1-Score.
# 
# Communication: Author a final executive summary, tailoring technical validation parameters for Automatidata's data directors and business/operational metrics for the NYC TLC leadership.
# 
# 2. Question 1: What are the ethical implications of the model? (¿Cuáles son las implicaciones éticas del modelo?)
# 
# 
# Español:
# 
# Riesgo de Discriminación y Sesgo Geográfico: Si el modelo utiliza variables como las zonas de inicio (pickup_zone) o destino (dropoff_zone) para predecir si un cliente es propenso a no dejar propina, una aplicación móvil que alerte a los conductores podría provocar un sesgo sistémico. Los taxistas podrían comenzar a rechazar viajes o a evitar sistemáticamente ciertos vecindarios de menores ingresos, afectando gravemente la equidad del servicio público de transporte y marginando a comunidades específicas de Nueva York.
# 
# Sesgo de Datos por el Método de Pago: Dado que los datos solo registran de manera confiable las propinas electrónicas (tarjetas de crédito), el modelo etiquetará erróneamente a los usuarios que pagan en efectivo como "no propinadores". Al implementar esto en una aplicación, se penalizaría injustamente a los pasajeros que prefieren o dependen del efectivo (como la población no bancarizada), lo cual representa un dilema ético importante de exclusión financiera.
# 
# English:
# 
# Risk of Service Discrimination and Geographic Bias: If the model utilizes features like pickup_zone or dropoff_zone to predict that a customer is unlikely to tip, a driver-facing alert application could cause systematic redlining. Drivers might start refusing dispatches or completely avoiding lower-income neighborhoods, severely undermining the equity of public transportation and marginalizing specific NYC communities.
# 
# Data Bias from Payment Methods: Since the pipeline only reliably logs credit card gratuities, the model will incorrectly classify cash-paying passengers as "non-tippers." Deploying this in a live app would unfairly penalize riders who prefer or rely on cash transactions (such as unbanked populations), creating an ethical dilemma around financial and structural exclusion.
# 
# 3. Question: Do the benefits of such a model outweigh the potential problems?.¿Superan los beneficios de este modelo los problemas potenciales?.
# 
# Español:
# 
# Conclusión: En su concepción original (crear una aplicación de alertas en tiempo real para evadir pasajeros), no, los beneficios no superan los problemas potenciales.
# 
# Justificación: Aunque el beneficio de maximizar los ingresos de los conductores es legítimo y crucial para su retención, implementarlo como un sistema de filtrado directo en la calle generaría problemas operativos y éticos inaceptables para una agencia reguladora como la NYC TLC. El costo social de marginar vecindarios enteros (sesgo geográfico) y discriminar a pasajeros que pagan en efectivo (población no bancarizada) debido a los errores inherentes del modelo (falsos negativos) destruiría la equidad y confiabilidad del servicio público de transporte.
# 
# Punto de Equilibrio Estratégico: Sin embargo, el modelo sí es extremadamente valioso si cambiamos el enfoque operativo. En lugar de usarlo para rechazar clientes en una app, la TLC puede utilizar los hallazgos de forma agregada (mediante la importancia de las características) para identificar qué condiciones logísticas de los viajes promueven mejores propinas. Esto permitiría diseñar incentivos, optimizar zonas de recogida o sugerir dinámicas de tarifas que beneficien de manera general a la flota sin discriminar a ningún usuario.
# 
# English:
# 
# Conclusion: In its original deployment concept (generating real-time alerts to bypass specific passengers), no, the benefits do not outweigh the potential problems.
# 
# Justification: While maximizing driver revenue is a legitimate and crucial retention objective, implementing this model as a direct roadside filtering mechanism introduces unacceptable operational and ethical liabilities for a regulatory body like the NYC TLC. The systemic cost of redlining entire neighborhoods (geographic bias) and discriminating against cash-reliant riders (unbanked populations) due to unavoidable model errors (false negatives) would severely disrupt the equity and public trust of the transit framework.
# 
# Strategic Alternative: However, the model becomes immensely valuable if we adjust its operational scope. Instead of deploying it to filter dispatches in a live app, the TLC can utilize the model's insights aggregately (via feature importance rankings) to identify which structural trip conditions naturally foster higher gratuities. This allows management to design strategic incentives, optimize physical pickup zones, or adjust regulatory fare configurations that uplift the entire driver ecosystem without restricting user access to public transport.
# 
# 4. Question: Would you proceed with the request to build this model? Why or why not?. ¿Procedería con la solicitud de construir este modelo? ¿Por qué sí o por qué no?.
# 
# Español:
# 
# Decisión: Sí, procedería con la construcción del modelo, pero con una condición obligatoria: modificar drásticamente el enfoque de su implementación y uso final.
# 
# Justificación de la construcción: Desarrollar el modelo es metodológicamente correcto y de alto valor para Automatidata y la NYC TLC. Nos permite descubrir los factores subyacentes que maximizan las ganancias de los conductores (propósito original de la solicitud) a través del análisis de la importancia de las características (feature importance).
# 
# Restricción del uso final (Por qué NO usarlo en una app de alertas): No permitiría que el modelo se utilizara en la aplicación propuesta para alertar a los conductores en tiempo real y rechazar clientes que "probablemente no darán propina". Hacerlo rompería el mandato de equidad del servicio público, validaría falsos negativos discriminatorios y penalizaría injustamente a los usuarios que pagan en efectivo.
# 
# Propuesta de Redirección Estratégica: Procedo con el entrenamiento del Bosque Aleatorio y XGBoost para entregar un modelo de diagnóstico institucional. El modelo servirá para que el equipo directivo de la TLC (Juliana Soto y Titus Nelson) identifique las dinámicas operativas óptimas (por ejemplo, qué horas, tarifas base o combinaciones de zonas geográficas generan mejores ingresos) y diseñe políticas públicas, incentivos de flota o sugerencias en la interfaz de pago que aumenten las propinas de manera generalizada y justa.
# 
# English:
# 
# Decision: Yes, I would proceed with building the model, but under one mandatory condition: drastically shifting the scope of its deployment and end-use application.
# 
# Justification for building: Developing the model is methodologically sound and highly valuable for both Automatidata and the NYC TLC. It allows us to uncover the underlying operational variables that maximize driver compensation (the original business purpose) through robust feature importance mapping.
# 
# Deployment Restriction (Why it should NOT power a live alert app): I would strictly advise against deploying this model within a live, driver-facing application designed to filter out riders predicted as "unlikely to tip." Operating the model this way would violate public transit equity mandates, institutionalize discriminatory false negatives, and unfairly penalize cash-reliant passengers.
# 
# Strategic Pivot Proposal: I will proceed with training the Random Forest and XGBoost classifiers to deliver an institutional diagnostic tool. The model will empower TLC leadership (Juliana Soto and Titus Nelson) to isolate high-yield operational dynamics (such as optimal scheduling windows, base fare impacts, or macro-zone structures) to engineer policy incentives and payment interface upgrades that systematically uplift driver gratuities across the board.
# 
# 5. Question: Can the objective be modified to make it less problematic?. ¿Se puede modificar el objetivo para que sea menos problemático?.
# 
# Español:
# 
# Modificación del Objetivo: Sí, el objetivo puede y debe ser modificado. En lugar de construir un modelo para predecir qué clientes particulares no dejarán propina (lo que fomenta la exclusión de usuarios en tiempo real), el objetivo debe redefinirse para predecir qué condiciones operativas del viaje promueven que un cliente sea un propinador generoso (≥20%).
# 
# Cambios en la Implementación:
# 
# De Filtro Individual a Guía Estratégica: En lugar de alimentar una aplicación que le diga al taxista "no recojas a este pasajero", el modelo se utilizará para generar mapas de calor y recomendaciones de turnos basadas en datos agregados. Por ejemplo: "Los conductores que operan en la Zona X durante los días jueves de 7:00 PM a 10:00 PM tienen un 40% más de probabilidad de recibir propinas generosas".
# 
# Mitigación del Sesgo de Pago: Al cambiar el enfoque hacia la identificación de factores óptimos en viajes con tarjeta de crédito, se acepta metodológicamente que el modelo describe el comportamiento de transacciones electrónicas, utilizándolo para incentivar de forma positiva las zonas de alta demanda en lugar de penalizar el uso de efectivo en la calle.
# 
# Impacto del Cambio: Esta modificación mantiene intacto el beneficio económico para los conductores (ayudándoles a posicionarse estratégicamente en los mejores horarios y lugares para maximizar sus ingresos) pero elimina por completo el riesgo de discriminación, rechazo de viajes y pérdida de equidad en el servicio de transporte público.
# 
# English:
# 
# Objective Modification: Yes, the objective can and should be modified. Instead of engineering a model to predict which individual passengers will not leave a tip (which actively incentivizes real-time user exclusion), the objective must be reframed to predict which operational trip conditions foster a generous tipping behavior (≥20%).
# 
# Implementation Shifts:
# 
# From Individual Filtering to Strategic Dispatch: Instead of powering an application that alerts a driver to "avoid this specific passenger," the model will generate aggregated heatmaps and shift scheduling recommendations. For example: "Drivers operating within Zone X on Thursdays between 7:00 PM and 10:00 PM exhibit a 40% higher probability of receiving generous gratuities."
# 
# Mitigating Payment Bias: By pivoting toward the identification of optimal drivers' environments using credit card transactions, we methodologically accept that the model describes electronic transaction dynamics, leveraging it to positively guide drivers rather than penalize cash users on the street.
# 
# Impact of the Pivot: This modification keeps the financial benefit for the drivers entirely intact—empowering them to position themselves strategically in high-yield hours and locations—while completely eliminating the risks of redlining, trip cancellations, and structural inequity across the public transport framework.
# 

# Suppose you were to modify the modeling objective so, instead of predicting people who won't tip at all, you predicted people who are particularly generous&mdash;those who will tip 20% or more? Consider the following questions:
# 
# 1.  What features do you need to make this prediction?
# 
# 2.  What would be the target variable?  
# 
# 3.  What metric should you use to evaluate your model? Do you have enough information to decide this now?
# 

# 1. Question 1: What features do you need to make this prediction?.¿Qué características necesitas para hacer esta predicción?.
# 
# Español:
# 
# Para alimentar nuestra matriz de características (X), utilizaremos variables que capturen de forma óptima el contexto temporal, geográfico y financiero del viaje, evitando cualquier fuga de datos (data leakage):
# 
# Atributos Temporales: hora_del_dia, dia_de_la_semana y un flag binario para hora_pico (rush hour), extraídos mediante ingeniería de características a partir de las marcas de tiempo brutas (timestamps).
# 
# Atributos Geográficos y de Trayecto: distancia_viaje (en millas), duracion_viaje (calculada en minutos) y los identificadores de ubicación PULocationID (origen) y DOLocationID (destino) codificados categóricamente.
# 
# Atributos Financieros: fare_amount (tarifa base), tolls_amount (peajes) y surcharges (recargos operativos).
# 
# Nota Crítica de Filtrado: Se mantendrá la columna VendorID y RatecodeID, pero se aislará y eliminará payment_type tras filtrar el dataset para incluir únicamente transacciones con tarjeta de crédito, mitigando así el sesgo de registro del efectivo.
# 
# English:
# 
# To construct our feature matrix (X), we will utilize attributes that optimally capture the temporal, spatial, and financial context of the trip while strictly avoiding data leakage:
# 
# Temporal Attributes: hour_of_day, day_of_week, and a binary rush_hour flag, engineered directly from the raw pickup and dropoff timestamps.
# 
# Spatial and Journey Metrics: trip_distance (in miles), trip_duration (computed in minutes), alongside the categorical location indices PULocationID (pickup) and DOLocationID (dropoff).
# 
# Financial Elements: fare_amount (base fare), tolls_amount (tolls), and operational surcharges.
# 
# Critical Filtering Note: We will retain VendorID and RatecodeID, but payment_type will be dropped immediately after filtering the dataset to hold strictly credit card transactions, thereby mitigating the systemic cash logging bias.
# 
# 2. Question 2: What would be the target variable?. ¿Cuál sería la variable objetivo?
# 
# Español:
# 
# La variable objetivo (y) será una variable binaria (0 o 1) calculada dinámicamente a partir del comportamiento real del usuario. Primero, derivaremos el porcentaje de la propina dividiendo tip_amount entre el costo del viaje (excluyendo la propia propina). Luego, aplicaremos la siguiente regla umbral:
# 
# 1 (Clase Positiva): El pasajero dejó una propina generosa, equivalente al 20% o más (≥0.20).
# 
# 0 (Clase Negativa): El pasajero dejó una propina inferior al 20% (<0.20).
# 
# English:
# 
# The target variable (y) will be a binary classification flag (0 or 1) dynamically engineered from the customer's transaction. First, we will derive the exact tipping percentage by dividing tip_amount by the baseline trip cost (excluding the gratuity itself). We will then map this into a threshold rule:
# 
# 1 (Positive Class): The passenger was a generous tipper, leaving a gratuity equal to or greater than 20% (≥0.20).
# 
# 0 (Negative Class): The passenger left a gratuity below 20% (<0.20).
# 
# 3. Question 2: What would be the target variable?. ¿Cuál sería la variable objetivo?.
# 
# Español:
# 
# La variable objetivo (y) será una variable binaria (0 o 1) calculada dinámicamente a partir del comportamiento real del usuario. Primero, derivaremos el porcentaje de la propina dividiendo tip_amount entre el costo del viaje (excluyendo la propia propina). Luego, aplicaremos la siguiente regla umbral:
# 
# 1 (Clase Positiva): El pasajero dejó una propina generosa, equivalente al 20% o más (≥0.20).
# 
# 0 (Clase Negativa): El pasajero dejó una propina inferior al 20% (<0.20).
# 
# English:
# 
# The target variable (y) will be a binary classification flag (0 or 1) dynamically engineered from the customer's transaction. First, we will derive the exact tipping percentage by dividing tip_amount by the baseline trip cost (excluding the gratuity itself). We will then map this into a threshold rule:
# 
# 1 (Positive Class): The passenger was a generous tipper, leaving a gratuity equal to or greater than 20% (≥0.20).
# 
# 0 (Negative Class): The passenger left a gratuity below 20% (<0.20).

# 
# **_Complete the following steps to begin:_**

# ### **Task 1. Imports and data loading**
# 
# Import packages and libraries needed to build and evaluate random forest and XGBoost classification models.

# In[1]:


# Import packages and libraries
### YOUR CODE HERE ###
# ==============================================================================
# Task 1: Consolidated Imports and Core Dependencies
# ==============================================================================

# Baseline data manipulation and analysis
import numpy as np
import pandas as pd

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns  # Agregado por buena práctica para gráficos del EDA / Added for EDA styling

# Model serialization and persistence
import pickle

# Tree-based Machine Learning models & native diagnostics
from sklearn.ensemble import RandomForestClassifier  # Requerido para el modelo base / Required for baseline
from xgboost import XGBClassifier
from xgboost import plot_importance

# Preprocessing and cross-validated data budgeting
from sklearn.model_selection import train_test_split, GridSearchCV

# Full evaluation metrics toolkit
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    RocCurveDisplay
)

# Notebook visualization settings
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)

# Suppress minor deprecation warnings for cleaner logs
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# RUN THIS CELL TO SEE ALL COLUMNS 
# This lets us see all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)


# Begin by reading in the data. There are two dataframes: one containing the original data, the other containing the mean durations, mean distances, and predicted fares from the previous course's project called nyc_preds_means.csv.
# 
# **Note:** `Pandas` reads in the dataset as `df0`, now inspect the first five rows. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.
# 
# Español:
# 
# Esta celda carga y prepara los dos componentes estructurales de nuestros datos:
# 
# df0: Contiene el conjunto de datos original de los viajes en taxi amarillo de 2017 de la NYC TLC con los registros transaccionales brutos (raw data).
# 
# nyc_preds_means: Contiene las métricas de ingeniería de características complejas que calculaste en el proyecto del curso anterior (duraciones promedio, distancias promedio y las tarifas predichas mediante Regresión Lineal Múltiple).
# 
# El siguiente paso metodológico consiste en inspeccionar las primeras filas (df0.head()) y posteriormente fusionar (merge) ambos DataFrames para consolidar todas las variables independientes (X) en una sola matriz de entrenamiento.
# 
# English:
# 
# This cell loads and structures the two fundamental components of our modeling pipeline:
# 
# df0: Holds the raw historical transactional dataset for the 2017 Yellow Taxi trip logs compiled by the NYC TLC.
# 
# nyc_preds_means: Holds the high-signal engineered features computed during the previous course's milestone project (including mean trip durations, mean distances, and Multiple Linear Regression predicted fares).
# 
# The next methodological step requires auditing the initial rows using df0.head() and subsequently merging both DataFrames to consolidate our final predictor space (X) within a unified training matrix.

# In[3]:


# RUN THE CELL BELOW TO IMPORT YOUR DATA. 

# Load dataset into dataframe
df0 = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')

# Import predicted fares and mean distance and duration from previous course
nyc_preds_means = pd.read_csv('nyc_preds_means.csv')


# Inspect the first few rows of `df0`.
# 

# In[4]:


# Inspect the first few rows of df0
### YOUR CODE HERE ###
df0.head()


# Inspect the first few rows of `nyc_preds_means`.

# In[5]:


# Inspect the first few rows of `nyc_preds_means`
### YOUR CODE HERE ###
nyc_preds_means.head()


# Task: Auditing Pre-engineered Features / Auditoría de Características Pre-diseñadas
# 
# Español:
# La inspección visual de nyc_preds_means confirma que disponemos de tres variables continuas de alta calidad para cada viaje:
# 
# mean_duration: La duración promedio estimada para viajes con perfiles similares.
# 
# mean_distance: La distancia promedio calculada para esa ruta específica.
# 
# predicted_fare: El valor de la tarifa base predicho matemáticamente por el modelo de Regresión Lineal Múltiple del curso anterior.
# 
# El paso lógico inmediato es unir (merge) este DataFrame con el original (df0) utilizando los índices de las filas para consolidar nuestro espacio de características antes de iniciar la limpieza.
# 
# English:
# 
# The visual audit of nyc_preds_means confirms that we have three high-signal continuous features mapped for each trip row:
# 
# mean_duration: The estimated mean duration for trips sharing identical routing profiles.
# 
# mean_distance: The calculated mean distance for that specific spatial trajectory.
# 
# predicted_fare: The baseline fare amount mathematically predicted by the previous course's Multiple Linear Regression model.
# 
# The immediate logical step is to merge this DataFrame with the baseline transaction data (df0) along their matching row indices to consolidate our feature space before initiating the preprocessing workflow.

# #### Join the two dataframes
# 
# Join the two dataframes using a method of your choice.
# 
# 1. Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# Para unir ambos DataFrames de la manera más limpia y eficiente, utilizamos el método .join(). Dado que ambos conjuntos de datos (df0 y nyc_preds_means) comparten exactamente la misma alineación y orden en sus filas, este método realiza una unión basada en sus índices de forma predeterminada. El resultado se almacena en una nueva variable llamada df, la cual ahora contiene tanto los datos transaccionales brutos como las características de promedios y predicciones del modelo anterior.
# 
# English:
# 
# To combine both DataFrames in the cleanest and most efficient manner, we utilize the .join() method. Since both datasets (df0 and nyc_preds_means) share the exact same row alignment and order, this method performs an index-based merge by default. The output is stored in a new variable named df, which now consolidates both the raw transactional records and the pre-engineered mean and predicted features from the previous course.

# In[6]:


# Merge datasets
### YOUR CODE HERE ###
# Merge datasets along their indices
df = df0.join(nyc_preds_means)

# Optional: Inspect the first few rows to verify the merge
df.head()


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## PACE: **Analyze**
# 
# Consider the questions in your PACE Strategy Documentto reflect on the Analyze stage.

# ### **Task 2. Feature engineering**
# 
# You have already prepared much of this data and performed exploratory data analysis (EDA) in previous courses. 
# 
# Call `info()` on the new combined dataframe.

# In[7]:


#==> ENTER YOUR CODE HERE
# Call info() on the new combined dataframe
df.info()


# Part 2: Structural Data Audit
# 
# Task: Analyzing df.info() Output / Análisis de la Salida de df.info()
# 
# Español:
# 
# Al revisar la estructura del DataFrame combinado, identificamos los siguientes puntos de atención para las siguientes celdas de ingeniería de características:
# 
# Marcas de Tiempo como Texto: Las columnas tpep_pickup_datetime y tpep_dropoff_datetime están registradas como tipo object (texto). Debemos convertirlas formalmente a datetime64 usando pd.to_datetime() para poder extraer la duración del viaje, las horas del día y los días de la semana.
# 
# Variables Categóricas Numéricas: Columnas como VendorID, RatecodeID, PULocationID, DOLocationID y payment_type están almacenadas como enteros (int64). Dado que representan categorías y no cantidades continuas, debemos asegurarnos de tratarlas adecuadamente (por ejemplo, mediante One-Hot Encoding) antes de entrenar, y usar payment_type estrictamente para filtrar por tarjeta de crédito (1).
# 
# English:
# 
# An analysis of the combined DataFrame structure highlights critical checkpoints for our feature engineering pipeline:
# 
# String-Formatted Timestamps: The columns tpep_pickup_datetime and tpep_dropoff_datetime are currently stored as object (string) data types. We must formally convert them to datetime64 via pd.to_datetime() to successfully extract trip duration, hour of the day, and day-of-week attributes.
# 
# Numerical Categorical Features: Features such as VendorID, RatecodeID, PULocationID, DOLocationID, and payment_type are stored as integers (int64). Since they represent discrete categories rather than continuous dimensions, we must ensure proper preprocessing (e.g., One-Hot Encoding) before modeling, and use payment_type exclusively to isolate credit card transactions (1).

# You know from your EDA that customers who pay cash generally have a tip amount of $0. To meet the modeling objective, you'll need to sample the data to select only the customers who pay with credit card. 
# 
# Copy `df0` and assign the result to a variable called `df1`. Then, use a Boolean mask to filter `df1` so it contains only customers who paid with credit card.

# In[8]:


# Subset the data to isolate only customers who paid by credit card
#==> ENTER YOUR CODE HERE
# Copy df and assign the result to df1
df1 = df.copy()

# Use a Boolean mask to filter df1 to isolate credit card payments (payment_type == 1)
df1 = df1[df1['payment_type'] == 1]


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# df1 = df.copy(): Creamos una copia explícita en memoria del DataFrame combinado (df) y la asignamos a df1. Esto es una buena práctica de ingeniería de software para evitar el aviso de advertencia SettingWithCopyWarning de Pandas cuando realicemos modificaciones posteriores.
# 
# Filtrado por Máscara Booleana: Tal como identificamos en la fase de análisis ético y exploratorio, en el conjunto de datos de la TLC de Nueva York el valor 1 en la columna payment_type equivale a pagos con tarjeta de crédito. Al aplicar df1['payment_type'] == 1, aislamos los únicos registros donde las propinas se registran de forma electrónica y confiable, eliminando el sesgo masivo del efectivo (0).
# 
# English:
# 
# df1 = df.copy(): We generate an explicit deep copy of the consolidated DataFrame (df) in memory and assign it to df1. This follows software engineering best practices to eliminate Pandas' SettingWithCopyWarning during downstream modifications.
# 
# Boolean Mask Filtering: As established during our ethical and exploratory analysis stages, a value of 1 within the payment_type column represents credit card transactions in the NYC TLC data schema. Applying df1['payment_type'] == 1 isolates the only rows where gratuities are reliably tracked via electronic logs, successfully mitigating the systemic cash reporting bias (0).

# ##### **Target**
# 
# Notice that there isn't a column that indicates tip percent, which is what you need to create the target variable. You'll have to engineer it. 
# 
# Add a `tip_percent` column to the dataframe by performing the following calculation:  
# <br/>  
# 
# 
# $$tip\ percent = \frac{tip\ amount}{total\ amount - tip\ amount}$$  
# 
# Round the result to three places beyond the decimal. **This is an important step.** It affects how many customers are labeled as generous tippers. In fact, without performing this step, approximately 1,800 people who do tip ≥ 20% would be labeled as not generous. 
# 
# To understand why, you must consider how floats work. Computers make their calculations using floating-point arithmetic (hence the word "float"). Floating-point arithmetic is a system that allows computers to express both very large numbers and very small numbers with a high degree of precision, encoded in binary. However, precision is limited by the number of bits used to represent a number, which is generally 32 or 64, depending on the capabilities of your operating system. 
# 
# This comes with limitations in that sometimes calculations that should result in clean, precise values end up being encoded as very long decimals. Take, for example, the following calculation:
# 
# Variable Objetivo (Target)
# 
# Observe que no existe una columna que indique el porcentaje de propina, el cual es necesario para crear la variable objetivo. Tendrá que diseñarla mediante ingeniería de características.
# 
# Añada una columna llamada tip_percent al DataFrame realizando el siguiente cálculo:
# 
# $$tip\ percent = \frac{tip\ amount}{total\ amount - tip\ amount}$$  
# 
# Redondee el resultado a tres decimales. Este es un paso importante. Afecta a cuántos clientes se etiquetarán como propinadores generosos. De hecho, si no se realiza este paso, aproximadamente 1,800 personas que sí dan una propina ≥20% serían etiquetadas como no generosas.
# 
# Para entender por qué, debe considerar cómo funcionan los números de punto flotante (floats). Las computadoras realizan sus cálculos utilizando la aritmética de punto flotante. La aritmética de punto flotante es un sistema que permite a las computadoras expresar tanto números muy grandes como números muy pequeños con un alto grado de precisión, codificados en sistema binario. Sin embargo, la precisión está limitada por el número de bits utilizados para representar un número, que generalmente es 32 o 64, dependiendo de las capacidades de su sistema operativo.
# 
# Esto conlleva limitaciones, ya que a veces cálculos que deberían dar como resultado valores limpios y precisos terminan codificándose como decimales extremadamente largos. Tome, por ejemplo, el siguiente cálculo:

# In[9]:


# Run this cell
1.1 + 2.2


# Notice the three that is 16 places to the right of the decimal. As a consequence, if you were to then have a step in your code that identifies values ≤ 3.3, this would not be included in the result. Therefore, whenever you perform a calculation to compute a number that is then used to make an important decision or filtration, round the number. How many degrees of precision you round to is your decision, which should be based on your use case. 
# 
# Refer to this [guide for more information related to floating-point arithmetic](https://floating-point-gui.de/formats/fp/).  

# In[10]:


# Create tip % col
#==> ENTER YOUR CODE HERE
# Create tip_percent column using the mathematical formula and round to 3 decimal places
df1['tip_percent'] = (df1['tip_amount'] / (df1['total_amount'] - df1['tip_amount'])).round(3)


# Español:
# Esta línea de código realiza una operación vectorizada en Pandas. Divide el monto de la propina (tip_amount) entre el costo neto del viaje (calculado al restar la propina del monto total: total_amount - tip_amount). Finalmente, se encadena el método .round(3) para mitigar los errores de precisión binaria del procesador, asegurando que los registros con un 20% real no queden fuera debido a truncamientos de punto flotante en la siguiente fase de etiquetado binario.
# 
# English:
# This line executes a vectorized operation in Pandas. It divides tip_amount by the net cost of the trip (computed by isolating the gratuity from the total ticket: total_amount - tip_amount). Finally, the .round(3) method is chained to neutralize floating-point processor arithmetic limitations, guaranteeing that records hitting a true 20% mark are not dropped due to binary truncation errors before entering the target labeling phase.

# Now create another column called `generous`. This will be the target variable. The column should be a binary indicator of whether or not a customer tipped ≥ 20% (0=no, 1=yes).
# 
# 1. Begin by making the `generous` column a copy of the `tip_percent` column.
# 2. Reassign the column by converting it to Boolean (True/False).
# 3. Reassign the column by converting Boolean to binary (1/0).

# In[11]:


# Create 'generous' col (target)
#==> ENTER YOUR CODE HERE
# Step 1: Make 'generous' column a copy of the 'tip_percent' column
df1['generous'] = df1['tip_percent']

# Step 2: Reassign the column by converting it to Boolean (True/False) based on the threshold >= 0.20
df1['generous'] = df1['generous'] >= 0.20

# Step 3: Reassign the column by converting Boolean to binary (1/0) using astype(int)
df1['generous'] = df1['generous'].astype(int)


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# Este bloque de código establece de forma robusta la variable objetivo (y) para nuestros modelos basados en árboles:
# 
# Copia: Duplica la columna tip_percent dentro de una nueva columna llamada generous.
# 
# Conversión Booleana: Evalúa la condición lógica ≥0.20. Gracias al redondeo a tres decimales que hicimos en el paso anterior, los flotantes se evalúan con absoluta precisión matemática, transformando los valores en True o False.
# 
# Conversión Binaria: El método .astype(int) mapea los valores lógicos al formato numérico estándar de Machine Learning, donde True se convierte en 1 (Clase Positiva: Propinador Generoso) y False se convierte en 0 (Clase Negativa: No Generoso).
# 
# English:
# 
# 
# This cell structures the target variable (y) for our ensemble pipeline following a strict three-step protocol:
# 
# Copy: Duplicates the newly engineered tip_percent into a dedicated column named generous.
# 
# Boolean Mapping: Evaluates the logical condition ≥0.20. Due to our previous three-decimal rounding step, floating-point numbers are filtered with exact precision, casting the array into True or False.
# 
# Binary Conversion: The .astype(int) method maps logical states into a standard Machine Learning numeric matrix, translating True into 1 (Positive Class: Generous Tipper) and False into 0 (Negative Class: Non-Generous).

# <details>
#   <summary><h5>HINT</h5></summary>
# 
# To convert from Boolean to binary, use `.astype(int)` on the column.
# </details>

# #### Create day column

# Next, you're going to be working with the pickup and dropoff columns.
# 
# Convert the `tpep_pickup_datetime` and `tpep_dropoff_datetime` columns to datetime.

# In[12]:


# Convert pickup and dropoff cols to datetime
#==> ENTER YOUR CODE HERE

# Convert pickup and dropoff columns to datetime using pd.to_datetime()
df1['tpep_pickup_datetime'] = pd.to_datetime(df1['tpep_pickup_datetime'])
df1['tpep_dropoff_datetime'] = pd.to_datetime(df1['tpep_dropoff_datetime'])


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# En la salida inicial de df.info(), pudimos notar que las marcas de tiempo se importaron originalmente como cadenas de texto (object). Mediante la función pd.to_datetime(), transformamos estas columnas al tipo de datos especializado datetime64[ns]. Este paso de preprocesamiento es crucial porque activa los descriptores de tiempo de Pandas (.dt), permitiéndonos extraer en los siguientes pasos el día de la semana (.dt.day_name()) o la hora del día, los cuales actuarán como características predictivas de alto valor para nuestros árboles.
# 
# English:
# 
# In our initial df.info() diagnostics, we observed that timestamps were natively loaded as plain text strings (object). By deploying the pd.to_datetime() framework, we formally transform these records into specialized datetime64[ns] data structures. This preprocessing step is critical because it activates Pandas' native temporal accessors (.dt), allowing us to cleanly extract downstream features like day names (.dt.day_name()) or hourly windows, which serve as high-signal predictors for our ensemble models.

# Create a `day` column that contains only the day of the week when each passenger was picked up. Then, convert the values to lowercase.

# In[13]:


# Create a 'day' col
#==> ENTER YOUR CODE HERE
# Create 'day' column with the lowercase day name of the pickup datetime
df1['day'] = df1['tpep_pickup_datetime'].dt.day_name().str.lower()


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# Este código realiza dos operaciones en cadena utilizando los métodos vectorizados de Pandas:
# 
# .dt.day_name(): Utiliza el accesor de tiempo (.dt) para extraer el nombre del día de la semana (por ejemplo, 'Monday', 'Tuesday') correspondiente a la fecha de recogida del pasajero (tpep_pickup_datetime).
# 
# .str.lower(): Convierte inmediatamente las cadenas de texto resultantes a minúsculas ('monday', 'tuesday'), cumpliendo con la normalización solicitada por el laboratorio para estandarizar los datos categóricos.
# 
# English:
# 
# This line executes a chained operational sequence leveraging Pandas' vectorized string and temporal methods:
# 
# .dt.day_name(): Utilizes the datetime accessor (.dt) to extract the explicit weekday name (e.g., 'Monday', 'Tuesday') corresponding to the customer's initial tpep_pickup_datetime log.
# 
# .str.lower(): Instantly converts the resulting text arrays into lowercase format ('monday', 'tuesday'), fulfilling the laboratory's formatting mandate to standardize categorical features.

# 
# <details>
#   <summary><h5>HINT</h5></summary>
# 
# To convert to day name, use `dt.day_name()` on the column.
# </details>

# #### Create time of day columns

# Next, engineer four new columns that represent time of day bins. Each column should contain binary values (0=no, 1=yes) that indicate whether a trip began (picked up) during the following times:
# 
# `am_rush` = [06:00&ndash;10:00)  
# `daytime` = [10:00&ndash;16:00)  
# `pm_rush` = [16:00&ndash;20:00)  
# `nighttime` = [20:00&ndash;06:00)  
# 
# To do this, first create the four columns. For now, each new column should be identical and contain the same information: the hour (only) from the `tpep_pickup_datetime` column.

# In[14]:


# Create 'am_rush' col
#==> ENTER YOUR CODE HERE
df1['am_rush'] = df1['tpep_pickup_datetime'].dt.hour
# Create 'daytime' col
#==> ENTER YOUR CODE HERE
df1['daytime'] = df1['tpep_pickup_datetime'].dt.hour
# Create 'pm_rush' col
#==> ENTER YOUR CODE HERE
df1['pm_rush'] = df1['tpep_pickup_datetime'].dt.hour
# Create 'nighttime' col
#==> ENTER YOUR CODE HERE
# Extract the hour (only) from tpep_pickup_datetime and assign it to all four new columns
df1['nighttime'] = df1['tpep_pickup_datetime'].dt.hour


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# De acuerdo con las instrucciones de la rúbrica, el primer paso para construir los contenedores temporales (time of day bins) consiste en inicializar las cuatro variables (am_rush, daytime, pm_rush, nighttime) con un valor base idéntico: la hora entera del viaje. Utilizamos el accesor .dt.hour sobre la columna tpep_pickup_datetime para extraer un entero entre 0 y 23. En la siguiente celda del laboratorio, utilizaremos funciones condicionales o máscaras lógicas para transformar estas horas en los valores binarios (1 o 0) requeridos para los rangos de hora pico y nocturna.
# 
# English:
# 
# Following the rubric's specific protocol, the initial phase to engineer our time-of-day bins requires initializing all four target features (am_rush, daytime, pm_rush, nighttime) with an identical baseline value: the raw numerical hour of the trip. We deploy the .dt.hour accessor on the tpep_pickup_datetime column to extract an integer ranging from 0 to 23. In the downstream notebook cells, we will apply conditional logic or Boolean mapping to convert these hours into the mandatory binary flags (1 or 0) corresponding to each specific operational window.

# You'll need to write four functions to convert each new column to binary (0/1). Begin with `am_rush`. Complete the function so if the hour is between [06:00–10:00), it returns 1, otherwise, it returns 0.

# In[15]:


# Define 'am_rush()' conversion function [06:00–10:00)
    #==> ENTER YOUR CODE HERE
    # Define 'am_rush()' conversion function [06:00–10:00)
def am_rush(hour):
    """
    Returns 1 if the hour is between 6 (inclusive) and 10 (exclusive), 
    otherwise returns 0.
    """
    if 6 <= hour < 10:
        return 1
    else:
        return 0


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# La notación matemática [06:00–10:00) indica un intervalo cerrado a la izquierda (incluye las 6:00) y abierto a la derecha (excluye las 10:00). La función implementa esta lógica exacta mediante la condición 6 <= hour < 10. Si la hora entera extraída cae en este rango (es decir, horas 6, 7, 8 o 9), la función devuelve un 1 lógico; de lo contrario, devuelve un 0. En el siguiente paso del laboratorio, aplicarás esta función utilizando .apply() sobre tu columna.
# 
# English:
# 
# The mathematical interval notation [06:00–10:00) denotes a range that is left-closed (inclusive of 6:00) and right-open (exclusive of 10:00). The function enforces this exact operational scope using the conditional expression 6 <= hour < 10. If the integer hour falls within this bracket (meaning hours 6, 7, 8, or 9), the function yields a logical 1; otherwise, it drops a 0. In the downstream notebook step, you will map this logic across your target column using .apply().

# Now, apply the `am_rush()` function to the `am_rush` series to perform the conversion. Print the first five values of the column to make sure it did what you expected it to do.
# 
# **Note:** Be careful! If you run this cell twice, the function will be reapplied and the values will all be changed to 0.

# In[16]:


# Apply 'am_rush' function to the 'am_rush' series
#==> ENTER YOUR CODE HERE
# Apply 'am_rush' function to the 'am_rush' series using .apply()
df1['am_rush'] = df1['am_rush'].apply(am_rush)

# Print the first five values of the column to verify the operation
df1['am_rush'].head()


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# df1['am_rush'].apply(am_rush): El método .apply() de Pandas toma nuestra función personalizada am_rush y la ejecuta fila por fila sobre los enteros de la columna. Transforma las horas (por ejemplo, hora 7 en 1, hora 14 en 0).
# 
# ¡Advertencia de ejecución única! Como indica la nota del laboratorio, si ejecutas esta celda por segunda vez, la columna ya no contendrá horas (valores del 0 al 23), sino valores binarios (0 y 1). Al volver a pasarle un 0 o un 1 a la función, la condición 6 <= hour < 10 fallará siempre, convirtiendo absolutamente toda la columna en 0. Si te equivocas y la corres dos veces, simplemente reejecuta la celda anterior donde inicializamos las cuatro columnas con dt.hour.
# 
# English:
# 
# df1['am_rush'].apply(am_rush): Pandas' .apply() method maps our custom am_rush function row-by-row across the column's integer values. It seamlessly transforms baseline hour integers (e.g., mapping hour 7 into 1, and hour 14 into 0).
# 
# Idempotency Warning! As the laboratory note cautions, running this cell a second time will corrupt the feature array. This happens because the column no longer holds raw hour indices (0–23) but binary flags (0 and 1). Passing a 0 or 1 back into the function causes the conditional check 6 <= hour < 10 to fail universally, resetting the entire series to 0. If you accidental double-run this cell, simply re-execute the previous step where we initialized the columns via dt.hour.

# Write functions to convert the three remaining columns and apply them to their respective series.

# In[17]:


# Define 'daytime()' conversion function [10:00–16:00)
#==> ENTER YOUR CODE HERE
# Define 'daytime()' conversion function [10:00–16:00)
def daytime(hour):
    if 10 <= hour < 16:
        return 1
    else:
        return 0


# In[18]:


# Apply 'daytime()' function to the 'daytime' series
#==> ENTER YOUR CODE HERE
df1['daytime'] = df1['daytime'].apply(daytime)



# In[19]:


# Define 'pm_rush()' conversion function [16:00–20:00)
#==> ENTER YOUR CODE HERE
# Define 'pm_rush()' conversion function [16:00–20:00)
def pm_rush(hour):
    if 16 <= hour < 20:
        return 1
    else:
        return 0


# In[20]:


# Apply 'pm_rush()' function to the 'pm_rush' series
#==> ENTER YOUR CODE HERE
df1['pm_rush'] = df1['pm_rush'].apply(pm_rush)


# In[21]:


# Define 'nighttime()' conversion function [20:00–06:00)
#==> ENTER YOUR CODE HERE
# Define 'nighttime()' conversion function [20:00–06:00)
def nighttime(hour):
    if hour >= 20 or hour < 6:
        return 1
    else:
        return 0


# In[22]:


# Apply 'nighttime' function to the 'nighttime' series
#==> ENTER YOUR CODE HERE
df1['nighttime'] = df1['nighttime'].apply(nighttime)


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# Este conjunto de celdas finaliza la discretización horaria del dataset. Al aplicar cada función lógica sobre su respectiva columna usando .apply(), transformamos la hora entera en un indicador binario (1 o 0). Note cómo en nighttime se utiliza un operador lógico condicional compuesto (or), garantizando que tanto las horas tardías de la noche (≥20) como las horas de la madrugada (<6) queden correctamente representadas en una sola dimensión predictiva para los modelos de ensamble.
# 
# English:
# 
# This suite of cells finalizes the structural hourly discretization of our dataset. By mapping each logical function onto its respective column using .apply(), we successfully transform raw hours into binary flags (1 or 0). Note how nighttime deploys a compound conditional logic statement (or), guaranteeing that both late evening hours (≥20) and early morning hours (<6) are precisely accounted for within a single predictive dimension tailored for our ensemble tree frameworks.

# #### Create `month` column

# Now, create a `month` column that contains only the abbreviated name of the month when each passenger was picked up, then convert the result to lowercase.

# <details>
#   <summary><h5>HINT</h5></summary>
# 
# Refer to the [strftime cheatsheet](https://strftime.org/) for help.
# </details>

# In[23]:


# Create 'month' col
#==> ENTER YOUR CODE HERE
# Create a 'month' column containing the lowercase abbreviated month name
df1['month'] = df1['tpep_pickup_datetime'].dt.strftime('%b').str.lower()


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# Para extraer el nombre abreviado del mes, utilizamos el método .dt.strftime('%b') de Pandas. La directiva %b toma el componente de fecha (tpep_pickup_datetime) y lo convierte a la abreviación estándar de tres letras en inglés (por ejemplo, 'Jan', 'Feb', 'Mar'). Posteriormente, encadenamos el método .str.lower() para transformar el texto completamente a minúsculas ('jan', 'feb', 'mar'), cumpliendo rigurosamente con los estándares de formato categórico exigidos por el laboratorio.
# 
# English:
# 
# To extract the short-form name of the month, we deploy Pandas' native .dt.strftime('%b') method. The %b token converts the initial timestamp (tpep_pickup_datetime) into a standard three-letter English abbreviation (e.g., 'Jan', 'Feb', 'Mar'). We then chain the .str.lower() string method to transform the textual array into full lowercase format ('jan', 'feb', 'mar'), strictly adhering to the categorical formatting standards dictated by the laboratory guidelines.

# Examine the first five rows of your dataframe.

# In[24]:


#==> ENTER YOUR CODE HERE
# Examine the first five rows of the modified dataframe
df1.head()


# #### Drop columns
# 
# Drop redundant and irrelevant columns as well as those that would not be available when the model is deployed. This includes information like payment type, trip distance, tip amount, tip percentage, total amount, toll amount, etc. The target variable (`generous`) must remain in the data because it will get isolated as the `y` data for modeling.

# In[25]:


# Drop columns
#==> ENTER YOUR CODE HERE
# Drop redundant, irrelevant, or data-leaking columns
cols_to_drop = [
    'Unnamed: 0', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
    'payment_type', 'trip_distance', 'tip_amount', 'tip_percent',
    'total_amount', 'tolls_amount', 'fare_amount', 'extra', 
    'mta_tax', 'improvement_surcharge'
]

df2 = df1.drop(columns=cols_to_drop)

# Verify the remaining columns
df2.info()


# Explicación del paso / Pipeline Step Explanation
# Español:
# 
# * Fuga de Datos (Data Leakage): 
# 
# Eliminar columnas como tip_amount, tip_percent y total_amount es estrictamente obligatorio. Dado que estas variables se usaron para construir directamente la variable objetivo (generous), si las dejamos en la matriz X, el modelo tendría un "trampa" matemática y su rendimiento colapsaría en producción al no contar con esos datos de antemano.
# 
# * Disponibilidad en el Despliegue: 
# 
# Métricas como la distancia o los peajes brutos del viaje no se conocen con precisión antes de que el viaje termine. Por ello, se eliminan y se reemplazan por las variables predictoras de promedios históricos (mean_duration, mean_distance, predicted_fare) que uniste al inicio.
# 
# English:
# 
# * Data Leakage Mitigation: 
# 
# Dropping columns like tip_amount, tip_percent, and total_amount is structurally mandatory. Because these attributes were directly utilized to manufacture our target label (generous), retaining them inside the feature space X would cause extreme data leakage, resulting in an artificially perfect model that fails critically upon deployment.
# 
# * Deployment Availability Constraints: 
# 
# Real-time metrics such as actual trip distance or exact toll amounts are fundamentally unknown before a journey concludes. Thus, they are removed from the feature matrix and safely substituted by the historical group aggregates (mean_duration, mean_distance, predicted_fare) engineered in the previous milestone project.

# #### Variable encoding

# Many of the columns are categorical and will need to be dummied (converted to binary). Some of these columns are numeric, but they actually encode categorical information, such as `RatecodeID` and the pickup and dropoff locations. To make these columns recognizable to the `get_dummies()` function as categorical variables, you'll first need to convert them to `type(str)`. 
# 
# 1. Define a variable called `cols_to_str`, which is a list of the numeric columns that contain categorical information and must be converted to string: `RatecodeID`, `PULocationID`, `DOLocationID`.
# 2. Write a for loop that converts each column in `cols_to_str` to string.
# 

# In[26]:


# 1. Define list of cols to convert to string
#==> ENTER YOUR CODE HERE
# 1. Define list of cols to convert to string
cols_to_str = ['RatecodeID', 'PULocationID', 'DOLocationID']


# 2. Convert each column to string
#==> ENTER YOUR CODE HERE
# 2. Convert each column to string
for col in cols_to_str:
    df2[col] = df2[col].astype(str)


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# Por defecto, Pandas interpreta las columnas que contienen números enteros como variables cuantitativas continuas o discretas. Si pasáramos PULocationID (ID de ubicación de recogida) de forma directa a la función pd.get_dummies(), Pandas no le aplicaría la codificación One-Hot, ya que asumiría erróneamente que el identificador 100 es cuantitativamente mayor o diferente al identificador 50. Al usar .astype(str) dentro del bucle for, forzamos al sistema a tratar estos números como etiquetas de texto puro, asegurando que la binarización posterior se ejecute correctamente.
# 
# English:
# 
# By default, Pandas interprets feature columns containing integer streams as continuous or discrete quantitative dimensions. If we directly feed numerical indices like PULocationID into the pd.get_dummies() operator, Pandas will fail to apply One-Hot Encoding because it mistakenly assumes location ID 100 holds a greater quantitative weight than ID 50. By applying .astype(str) inside the for loop, we force the architecture to treat these values as categorical text tokens, ensuring the downstream binarization maps accurately.

# 
# <details>
#   <summary><h5>HINT</h5></summary>
# 
# To convert to string, use `astype(str)` on the column.
# </details>

# Now convert all the categorical columns to binary.
# 
# 1. Call `get_dummies()` on the dataframe and assign the results back to a new dataframe called `df2`.
# 

# In[27]:


# Convert categoricals to binary
#==> ENTER YOUR CODE HERE
# Convert categoricals to binary using get_dummies and assign back to df2
df2 = pd.get_dummies(df2, drop_first=True)


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# La función pd.get_dummies() de Pandas toma automáticamente todas las columnas identificadas como categóricas (las cadenas de texto y objetos que preparamos en los pasos anteriores, como day, month, VendorID, RatecodeID, PULocationID y DOLocationID) y las expande en columnas binarias independientes (1 o 0). El argumento drop_first=True es un parámetro crítico de optimización: elimina la primera columna dummy de cada categoría para prevenir la multicolinealidad perfecta (también conocida como la trampa de la variable dummy), lo cual mantiene nuestra matriz numéricamente estable y eficiente para los algoritmos basados en árboles.
# 
# English:
# 
# The pd.get_dummies() function automatically sweeps across our feature space, detecting all object and string-formatted categories (such as day, month, VendorID, RatecodeID, PULocationID, and DOLocationID) and expanding them into independent binary streams (1 or 0). The drop_first=True argument is a critical structural constraint: it drops the initial dummy column for each category to systematically prevent perfect multicollinearity (commonly known as the dummy variable trap), keeping our design matrix numerically stable and highly optimized for tree-based ensemble processing.

# ##### Evaluation metric
# 
# Before modeling, you must decide on an evaluation metric. 
# 
# 1. Examine the class balance of your target variable. 

# In[28]:


# Get class balance of 'generous' col
#==> ENTER YOUR CODE HERE
# Get class balance of 'generous' col using value_counts()
df2['generous'].value_counts(normalize=True)


# Reflexión PACE: Elección de la Métrica / PACE Reflection: Metric Selection
# 
# Español:
# 
# Aunque las clases están balanceadas y el Accuracy simple sería un indicador válido del rendimiento global, mantendremos el enfoque estratégico en el F1-Score. Debido a que Automatidata busca optimizar la retención de conductores y las finanzas de la flota, el F1-Score garantiza que controlemos armónicamente tanto los falsos positivos (evitando enviar conductores a zonas sin propina real) como los falsos negativos (no perdiendo oportunidades de alta rentabilidad).
# 
# English:
# 
# Even though the classes are balanced and baseline Accuracy would serve as a robust indicator of global performance, we will maintain our strategic focus on the F1-Score. Because Automatidata aims to optimize driver retention and fleet economics simultaneously, the F1-Score guarantees that we harmoniously penalize both false positives (preventing dispatching drivers to dead zones) and false negatives (avoiding missed premium revenue opportunities).

# A little over half of the customers in this dataset were "generous" (tipped ≥ 20%). The dataset is very nearly balanced.
# 
# To determine a metric, consider the cost of both kinds of model error:
# * False positives (the model predicts a tip ≥ 20%, but the customer does not give one)
# * False negatives (the model predicts a tip < 20%, but the customer gives more)
# 
# False positives are worse for cab drivers, because they would pick up a customer expecting a good tip and then not receive one, frustrating the driver.
# 
# False negatives are worse for customers, because a cab driver would likely pick up a different customer who was predicted to tip more&mdash;even when the original customer would have tipped generously.
# 
# **The stakes are relatively even. You want to help taxi drivers make more money, but you don't want this to anger customers. Your metric should weigh both precision and recall equally. Which metric is this?**

# El F1-Score es la media armónica entre la precisión y el recall. Al ponderar ambas métricas con el mismo peso, se convierte en la herramienta matemática perfecta para este escenario, ya que penaliza con la misma severidad tanto los falsos positivos (que afectan al conductor) como los falsos negativos (que afectan al cliente).
# 
# The metric that weighs both precision and recall equally is the F1-score. Since the cost of a false positive (frustrating the taxi driver) and a false negative (turning away a generous customer) are both high and relatively even, maximizing the harmonic mean of precision and recall ensures a balanced optimization for both stakeholders. Furthermore, since the dataset is nearly balanced, the F1-score provides a robust indicator of the model's predictive power across both classes.
# 

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## PACE: **Construct**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 3. Modeling**

# ##### **Split the data**
# 
# Now you're ready to model. The only remaining step is to split the data into features/target variable and training/testing data. 
# 
# 1. Define a variable `y` that isolates the target variable (`generous`).
# 2. Define a variable `X` that isolates the features.
# 3. Split the data into training and testing sets. Put 20% of the samples into the test set, stratify the data, and set the random state.

# In[29]:


# Isolate target variable (y)
y = df2['generous']

# Isolate the features (X)
X = df2.drop(columns=['generous'])

# Split into train and test sets (20% test, stratified, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# X = df2.drop(columns=['generous']): Creamos la matriz de características excluyendo únicamente la variable objetivo, asegurando que el modelo no tenga acceso directo a las etiquetas durante el entrenamiento.
# 
# stratify=y: Dado que la distribución original es aproximadamente 52.6% y 47.3%, la estratificación fuerza a train_test_split a mantener exactamente esa misma proporción tanto en el subconjunto de entrenamiento como en el de prueba. Esto evita sesgos de selección aleatoria.
# 
# random_state=42: Fija la semilla del generador aleatorio. Esto garantiza la reproducibilidad matemática; cada vez que ejecutes esta celda, la partición de las filas será idéntica.
# 
# English:
# 
# X = df2.drop(columns=['generous']): Extracts the design feature matrix by isolating the target label, ensuring the mathematical models have no direct access to the ground-truth outcomes during fitting.
# 
# stratify=y: Since our target class balance sits around 52.6% and 47.3%, stratification forces the split mechanics to mathematically preserve this exact ratio across both training and validation subsets, eliminating random sampling bias.
# 
# random_state=42: Locks the pseudo-random generator seed. This ensures strict experimental reproducibility, guaranteeing identical row assignment across successive matrix executions.

# ##### **Random forest**
# 
# Begin with using `GridSearchCV` to tune a random forest model.
# 
# 1. Instantiate the random forest classifier `rf` and set the random state.
# 
# 2. Create a dictionary `cv_params` of any of the following hyperparameters and their corresponding values to tune. The more you tune, the better your model will fit the data, but the longer it will take. 
#  - `max_depth`  
#  - `max_features`  
#  - `max_samples` 
#  - `min_samples_leaf`  
#  - `min_samples_split`
#  - `n_estimators`  
# 
# 3. Define a set `scoring` of scoring metrics for GridSearch to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `rf1`. Pass to it as arguments:
#  - estimator=`rf`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of you cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit=_`)
# 
# 
# **Note:** `refit` should be set to `'f1'`.<font/>
# </details>
#  
# 

# In[30]:


# 1. Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
# Optimized grid for reliable performance and safe execution time
cv_params = {
    'max_depth': [None, 10, 20],
    'max_features': [1.0, 'sqrt'],
    'max_samples': [0.7],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 200]
}

# 3. Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
rf1 = GridSearchCV(
    estimator=rf,
    param_grid=cv_params,
    scoring=scoring,
    cv=4,
    refit='f1'
)


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# cv_params: Diseñamos una malla (grid) estratégica. Al usar 'max_samples': [0.7], implementamos bagging seleccionando solo el 70% de las filas para cada árbol de decisión, lo cual acelera el proceso y reduce el sobreajuste (overfitting).
# 
# cv=4: Divide el conjunto de entrenamiento en 4 pliegues (folds) independientes. El modelo se entrena en 3 y se valida en el restante de forma rotativa.
# 
# refit='f1': Esta instrucción es fundamental. Le ordena a GridSearchCV que, tras evaluar todas las combinaciones posibles de la rejilla midiendo las cuatro métricas indicadas en scoring, seleccione automáticamente el conjunto de hiperparámetros que haya maximizado el F1-Score, volviendo a entrenar el modelo definitivo con esos valores óptimos sobre la totalidad de los datos de entrenamiento.
# 
# English:
# 
# cv_params: We structure a strategic search grid. By applying 'max_samples': [0.7], we execute row-level bagging using 70% of the training matrix per tree structure, boosting processing throughput while limiting structural overfitting.
# 
# cv=4: Splits the training subset into 4 independent cross-validation folds. The algorithm fits iteratively across 3 sections and benchmarks its generalization power on the remaining slice.
# 
# refit='f1': This command dictates the optimization goal. It explicitly directs GridSearchCV to cross-examine every operational permutation against the four target evaluation metrics, automatically isolating and saving the unique hyperparameter configuration that maximizes the F1-Score.

# Now fit the model to the training data. Note that, depending on how many options you include in your search grid and the number of cross-validation folds you select, this could take a very long time&mdash;even hours. If you use 4-fold validation and include only one possible value for each hyperparameter and grow 300 trees to full depth, it should take about 5 minutes. If you add another value for GridSearch to check for, say, `min_samples_split` (so all hyperparameters now have 1 value except for `min_samples_split`, which has 2 possibilities), it would double the time to ~10 minutes. Each additional parameter would approximately double the time. 

# In[31]:


#==> ENTER YOUR CODE HERE
# Fit the GridSearchCV object to the training data
rf1.fit(X_train, y_train)


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# Al ejecutar rf1.fit(X_train, y_train), Pandas y Scikit-Learn interactúan para iniciar la fase más pesada computacionalmente de nuestro pipeline de datos. El objeto GridSearchCV tomará el espacio de búsqueda que definimos en la celda anterior (cv_params) y entrenará iterativamente múltiples combinaciones de bosques aleatorios a través de los 4 pliegues de validación cruzada. Dado que la grilla contiene varias combinaciones, la celda mostrará un indicador de ejecución (un asterisco [*] en Jupyter) mientras procesa los cálculos en segundo plano. Una vez que termine, guardará internamente el modelo con el mejor desempeño basado en el F1-Score.
# 
# English:
# 
# By executing rf1.fit(X_train, y_train), Pandas and Scikit-Learn interface to launch the most computationally intensive phase of our data pipeline. The GridSearchCV object will systematically sweep across the parameter space defined in our previous search dictionary (cv_params), iteratively training random forest architectures across the 4 cross-validation folds. Since the search grid evaluates multiple permutations, the cell will display a busy indicator (the [*] asterisk in Jupyter) while processing. Upon completion, it will internally retain the top-performing estimator configured around the optimal F1-Score.

# <details>
#   <summary><h5>HINT</h5></summary>
# 
# If you get a warning that a metric is 0 due to no predicted samples, think about how many features you're sampling with `max_features`. How many features are in the dataset? How many are likely predictive enough to give good predictions within the number of splits you've allowed (determined by the `max_depth` hyperparameter)? Consider increasing `max_features`.
# 
# </details>

# If you want, use `pickle` to save your models and read them back in. This can be particularly helpful when performing a search over many possible hyperparameter values.

# In[32]:


import pickle 

# Define a path to the folder where you want to save the model
path = '/home/jovyan/work/'


# In[33]:


def write_pickle(path, model_object, save_name:str):
    '''
    save_name is a string.
    '''
    with open(path + save_name + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)


# In[34]:


def read_pickle(path, saved_model_name:str):
    '''
    saved_model_name is a string.
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

        return model


# 
# 
# Examine the best average score across all the validation folds. 

# In[35]:


# Examine best score
#==> ENTER YOUR CODE HERE
# Examine the best F1 score from the Grid Search
rf1.best_score_


# Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# rf1.best_score_: Entrega un número entre 0 y 1. Este valor refleja la media armónica óptima entre precisión y recall lograda durante la validación cruzada.
# 
# rf1.best_params_: Devuelve un diccionario de Python. Es de gran utilidad para entender el comportamiento de regularización de tu modelo (por ejemplo, si prefirió un árbol profundo o uno más restringido para evitar el sobreajuste).
# 
# English:
# 
# rf1.best_score_: Outputs a continuous value between 0 and 1. This metrics reflects the optimal cross-validated harmonic mean between precision and recall captured during the grid search.
# 
# rf1.best_params_: Returns a Python dictionary. It is highly valuable for inspecting the regularization behavior of your ensemble (e.g., assessing whether the framework gravitated towards full-depth architectures or tighter structural constraints to combat overfitting).

# Examine the best combination of hyperparameters.

# In[36]:


#==> ENTER YOUR CODE HERE
# Examine the best combination of hyperparameters found by GridSearchCV
rf1.best_params_


# 1. max_depth: 10 & min_samples_leaf: 2 (Operational Stability / Estabilidad Operativa)
# 
# English: By capping the decision depth and forcing a minimum size for leaves, the model is restricted from making hyper-specific, reactionary rules based on outliers. For Automatidata, this means the algorithm won't micro-manage driver dispatch based on a single fluke ride (e.g., a random huge tip during a rare event). Instead, it captures broad, reliable market trends, ensuring stable and predictable revenue recommendations for the fleet.
# 
# Español: Al limitar la profundidad y exigir un mínimo de muestras por hoja, evitamos que el modelo cree reglas reactivas o hipersensibles basadas en casos aislados. Para Automatidata, esto significa que el algoritmo no tomará decisiones logísticas basadas en anomalías (como una propina gigante fortuita). En su lugar, captura tendencias de mercado sólidas y repetibles, garantizando recomendaciones de ganancias estables y predecibles para la flota de conductores.
# 
# 2. n_estimators: 200 (Risk Diversification / Diversificación del Riesgo)
# 
# English: Growing 200 trees acts as an internal board of directors making a consensus decision. In a high-stakes deployment where a false positive frustrates a driver (sending them somewhere expecting a tip that never arrives) and a false negative loses a high-value customer, relying on a large ensemble minimizes operational volatility. The high number of estimators stabilizes the prediction pipeline against the noisy, chaotic nature of NYC traffic.
# 
# Español: Cultivar 200 árboles actúa como un comité ejecutivo que toma decisiones por consenso. En un despliegue operativo donde un falso positivo frustra al conductor (enviándolo a una zona esperando una gran propina que no ocurrirá) y un falso negativo ahuyenta a un cliente generoso, depender de un ensamble grande minimiza la volatilidad. Este volumen de estimadores estabiliza el pipeline frente a la naturaleza caótica y ruidosa del tránsito de Nueva York.
# 
# 3. max_samples: 0.7 & max_features: 'sqrt' (Strategic Adaptability / Adaptabilidad Estratégica)
# 
# English: Forcing individual trees to look at only 70% of historical data and a subset of features avoids over-reliance on dominant factors like baseline fares. This structural constraint pushes the model to discover hidden, highly valuable predictive signals within temporal bins (am_rush, nighttime) and pickup/dropoff combinations. From a business lens, this builds a diversified intelligence profile that adapts much better to seasonal shifts, changing economic landscapes, or unexpected mobility patterns in the city.
# 
# Español: Forzar a los árboles individuales a examinar solo el 70% de los datos históricos y un subconjunto de características evita que el modelo dependa ciegamente de factores obvios como la tarifa base. Esta restricción obliga al algoritmo a descubrir patrones predictivos ocultos de alto valor dentro de las franjas horarias (am_rush, nighttime) y las zonas geográficas. Desde una perspectiva de negocio, esto genera una inteligencia diversificada que se adapta mucho mejor a cambios estacionales, fluctuaciones económicas o nuevos patrones de movilidad urbana.
# 
# 🎯 Executive Summary for Stakeholders / Resumen Ejecutivo para Tomadores de Decisión
# 
# English: > "The grid search successfully selected a highly regularized, risk-averse model architecture. Rather than developing an aggressive algorithm that chases short-term anomalies, the framework optimized a balanced consensus strategy. This guarantees that Automatidata's deployment will protect driver morale by reducing false-positive tip expectations, while simultaneously safeguarding customer retention through stable, generalized demand forecasting."
# 
# Español: > "La búsqueda en malla seleccionó con éxito una arquitectura de modelo altamente regularizada y adversa al riesgo. En lugar de desarrollar un algoritmo agresivo que persiga anomalías de corto plazo, el marco optimizó una estrategia de consenso equilibrada. Esto garantiza que el despliegue de Automatidata protegerá la motivación de los conductores al reducir falsas expectativas de propinas (falsos positivos), salvaguardando simultáneamente la retención de clientes mediante proyecciones de demanda estables y generalizables."

# Use the `make_results()` function to output all of the scores of your model. Note that it accepts three arguments. 

# <details>
#   <summary><h5>HINT</h5></summary>
# 
# To learn more about how this function accesses the cross-validation results, refer to the [`GridSearchCV` scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV) for the `cv_results_` attribute.
# 
# </details>

# In[37]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
    model_name (string): what you want the model to be called in the output table
    model_object: a fit GridSearchCV object
    metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                 'recall': 'mean_test_recall',
                 'f1': 'mean_test_f1',
                 'accuracy': 'mean_test_accuracy',
                 }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                        'precision': [precision],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy],
                        },
                       )

    return table


# Call `make_results()` on the GridSearch object.

# In[38]:


#==> ENTER YOUR CODE HERE
# Call the function to output the formatted table of results
rf_cv_results = make_results('Random Forest CV', rf1, 'f1')
rf_cv_results


# 🎯 High Recall (0.8151) vs. Moderate Precision (0.6931)
# English: The model is highly effective at identifying generous tippers, successfully catching over 81.5% of them (Recall). From a commercial standpoint, this means taxi drivers will rarely miss out on a premium, high-tipping opportunity because the system failed to flag it. However, the lower Precision (69.3%) means that out of all rides predicted as "generous," about 30.7% will turn out to be normal tips. While this might occasionally cause minor driver frustration, it heavily prioritizes maximizing total revenue capture across the fleet.
# 
# Español: El modelo es sumamente eficaz para identificar a los clientes generosos, logrando capturar a más del 81.5% de ellos (Recall). Comercialmente, esto significa que los taxistas rara vez perderán una oportunidad de alta rentabilidad debido a un error del sistema. Por otro lado, la Precisión (69.3%) indica que de cada 100 viajes etiquetados como generosos, unos 30 resultarán en una propina normal. Aunque esto puede generar una fricción menor en las expectativas del conductor, prioriza agresivamente que la flota capture la mayor cantidad de dinero disponible en el mercado.

# Your results should produce an acceptable model across the board. Typically scores of 0.65 or better are considered acceptable, but this is always dependent on your use case. Optional: try to improve the scores. It's worth trying, especially to practice searching over different hyperparameters.
# 
# <details>
#   <summary><h5>HINT</h5></summary>
# 
# For example, if the available values for `min_samples_split` were [2, 3, 4] and GridSearch identified the best value as 4, consider trying [4, 5, 6] this time.
# </details>

# Use your model to predict on the test data. Assign the results to a variable called `rf_preds`.

# <details>
#   <summary><h5>HINT</h5></summary>
#     
# You cannot call `predict()` on the GridSearchCV object directly. You must call it on the `best_estimator_`.
# </details>

# For this project, you will use several models to predict on the test data. Remember that this decision comes with a trade-off. What is the benefit of this? What is the drawback?

# Benefit (Beneficio): 
# 
# * English: Testing multiple final models (like Random Forest and XGBoost) on the test data allows for a direct, objective performance comparison on a completely unseen dataset. This ensures that the champion model selected for deployment is truly the most robust and adaptive to real-world deployment conditions.
# 
# * Español: Evaluar múltiples modelos finales en el conjunto de prueba permite una comparación directa y objetiva sobre datos que nunca antes han visto. Esto garantiza que el modelo campeón seleccionado para producción sea realmente el más robusto ante las condiciones del mundo real.
# 
# Drawback (Inconveniente):
# 
# * English: The test set is meant to act as a final "blind" vault to estimate true generalization error. By continuously using it to benchmark and select between multiple models, we risk introducing subtle data leakage and selection bias. The chosen model might perform well on this specific test set due to chance, slightly inflating our performance expectations for production.
# 
# 
# * Español: El conjunto de prueba debe actuar como una bóveda "a ciegas" para medir el error de generalización real. Al usarlo repetidamente para comparar y elegir entre varios modelos, corremos el riesgo de introducir un sesgo de selección (data leakage). El modelo ganador podría sobresalir en este set de prueba específico por azar, inflando ligeramente las expectativas de rendimiento real.

# In[40]:


# Get scores on test data
#==> ENTER YOUR CODE HERE
# 1. Primero, generamos las predicciones obligatorias sobre los datos de prueba
rf_preds = rf1.best_estimator_.predict(X_test)

# 2. Ahora, importamos la métrica y ejecutamos el reporte de rendimiento
from sklearn.metrics import classification_report

print("Random Forest - Test Data Performance Report:")
print(classification_report(y_test, rf_preds, target_names=['not generous', 'generous']))


# 📋 Desglose de Métricas para la Clase generous
# 
# Recall (0.82 / 82%): 
# * Business Mean: El modelo captura exitosamente al 82% de los pasajeros que dejarán buenas propinas. Operativamente, es un gran incentivo para los conductores: si hay un cliente generoso en la calle, el sistema lo identificará correctamente la gran mayoría de las veces, maximizando la captura de ingresos premium para la flota.
# 
# Recall (0.82 / 82%):
# 
# * Business Meaning: The model successfully captures 82% of all passengers who are naturally inclined to leave a generous tip (≥20%). From a fleet management perspective, this is a massive win for driver retention. It guarantees that whenever a high-yielding revenue opportunity is out on the streets of New York, the dispatch system will correctly flag it the vast majority of the time, maximizing the drivers' earning potential.
# 
# Precision (0.68 / 68%): 
# * Business Mean: De cada 100 alertas que el modelo le mande a los taxistas indicando "viaje con alta propina probable", 68 resultarán ser efectivamente generosas. El 32% restante serán propinas estándar. Aunque existe ese margen de falsas expectativas para el conductor, el beneficio económico total compensa con creces las alertas erróneas.
# 
# Precision (0.68 / 68%):
# 
# * Business Meaning: Out of 100 trips that the algorithm flags as "highly likely to be generous," 68 will result in an actual premium tip, while 32 will turn out to be standard tips. While this 32% margin of error creates a slight risk of minor driver frustration (expecting a high tip that does not materialize), it strikes an excellent commercial balance. It ensures the system aggressively prioritizes revenue capture over missing out on good fares.
# 
# Accuracy General (0.71 / 71%):
# * Business Mean: El sistema acierta en la clasificación general (tanto para determinar quién da propina como quién no) en 7 de cada 10 viajes. Para un problema con tanto comportamiento humano e impredecible como lo son las propinas en Nueva York, un 71% de exactitud global es un resultado sumamente sólido y comercialmente viable.
# 
# Overall Accuracy (0.71 / 71%):
# 
# Business Meaning: The system correctly predicts the tipping behavior (both generous and non-generous outcomes) in roughly 7 out of 10 trips. Considering how erratic, emotional, and noisy human tipping behavior can be, a 71% global accuracy rate represents a highly robust, commercially viable baseline for Automatidata's deployment.

# Use the below `get_test_scores()` function you will use to output the scores of the model on the test data.

# In[41]:


def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
    model_name (string): Your choice: how the model will be named in the output table
    preds: numpy array of test predictions
    y_test_data: numpy array of y_test data

    Out:
    table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                        'precision': [precision],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy]
                        })

    return table


# 1. Use the `get_test_scores()` function to generate the scores on the test data. Assign the results to `rf_test_scores`.
# 2. Call `rf_test_scores` to output the results.

# ###### RF test results

# In[42]:


# Get scores on test data
rf_test_scores = get_test_scores('Random Forest Test', rf_preds, y_test)

# Call rf_test_scores to output the results
rf_test_scores


# 📋 Comparison Analysis / Análisis Comparativo
# 
# English: The test results are remarkably stable and compare exceptionally well to the validation results. The validation F1-score was 0.7492, and when evaluated on the unseen test data, it reached 0.7447 (a microscopic drop of only ≈0.0045). This minimal variation proves that the model possesses excellent generalization capabilities and that the hyperparameter tuning via GridSearchCV successfully prevented overfitting.
# 
# Español: Los resultados de la prueba son notablemente estables y se comparan excepcionalmente bien con los de la validación. El F1-score de validación fue de 0.7492, y al evaluarse con los datos de prueba, alcanzó 0.7447 (una caída microscópica de solo ≈0.0045). Esta variación mínima demuestra que el modelo posee una excelente capacidad de generalización y que el ajuste de hiperparámetros mediante GridSearchCV previno con éxito el sobreajuste.

# **Question:** How do your test results compare to your validation results?

# English Option 
# "The test results compare exceptionally well with the validation results. The validation F1-score was approximately 0.7492, while the final test F1-score is 0.7448. This minimal decrease of less than 0.005 indicates that the model generalizes remarkably well to completely unseen data. Furthermore, the high recall (0.817) and solid precision (0.684) remain consistent, confirming that the random forest is structurally stable and ready for operational deployment."
# 
# Spanish
# 
# "Los resultados de la prueba se comparan de manera excepcional con los de la validación. El F1-score de validación fue de aproximadamente 0.7492, mientras que el F1-score final en el set de prueba es de 0.7448. Esta disminución mínima de menos de 0.005 indica que el modelo generaliza notablemente bien ante datos completamente nuevos. Además, la alta exhaustividad (0.817) y la precisión sólida (0.684) se mantienen consistentes, confirmando que el bosque aleatorio es estructuralmente estable y está listo para su despliegue operativo."

# ##### **XGBoost**
# 
#  Try to improve your scores using an XGBoost model.
# 
# 1. Instantiate the XGBoost classifier `xgb` and set `objective='binary:logistic'`. Also set the random state.
# 
# 2. Create a dictionary `cv_params` of the following hyperparameters and their corresponding values to tune:
#  - `max_depth`
#  - `min_child_weight`
#  - `learning_rate`
#  - `n_estimators`
# 
# 3. Define a set `scoring` of scoring metrics for grid search to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `xgb1`. Pass to it as arguments:
#  - estimator=`xgb`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit='f1'`)
#  
#  XGBoost
# Try to improve your scores using an XGBoost model.
# 
# Instantiate the XGBoost classifier xgb and set objective='binary:logistic'. Also set the random state.
# 
# Create a dictionary cv_params of the following hyperparameters and their corresponding values to tune:
# 
# max_depth
# min_child_weight
# learning_rate
# n_estimators
# Define a set scoring of scoring metrics for grid search to capture (precision, recall, F1 score, and accuracy).
# 
# Instantiate the GridSearchCV object xgb1. Pass to it as arguments:
# 
# estimator=xgb
# param_grid=cv_params
# scoring=scoring
# cv: define the number of cross-validation folds you want (cv=_)
# refit: indicate which evaluation metric you want to use to select the model (refit='f1')
# 
# XGBoost
# Intente mejorar sus puntuaciones utilizando un modelo XGBoost.
# 
# Instancie el clasificador XGBoost xgb y configure el parámetro objective='binary:logistic'. También configure el estado aleatorio (random state).
# 
# Cree un diccionario cv_params con los siguientes hiperparámetros y sus valores correspondientes para ajustar:
# 
# max_depth
# 
# min_child_weight
# 
# learning_rate
# 
# n_estimators
# 
# Defina un conjunto (set o lista) scoring de métricas de puntuación para que la búsqueda en malla las capture (precision, recall, f1 y accuracy).
# 
# Instancie el objeto GridSearchCV llamado xgb1. Pásale como argumentos:
# 
# estimator=xgb
# 
# param_grid=cv_params
# 
# scoring=scoring
# 
# cv: defina el número de pliegues de validación cruzada que desea (cv=__)
# 
# refit: indique qué métrica de evaluación desea utilizar para seleccionar el mejor modelo (refit='f1').
# 
# 

# In[43]:


# 1. Instantiate the XGBoost classifier
#==> ENTER YOUR CODE HERE
xgb = XGBClassifier(objective='binary:logistic', random_state=42)
# 2. Create a dictionary of hyperparameters to tune
#==> ENTER YOUR CODE HERE
cv_params = {
    'max_depth': [4, 6, 8],
    'min_child_weight': [1, 2, 3],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 150]
}

# 3. Define a list of scoring metrics to capture
#==> ENTER YOUR CODE HERE
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
#==> ENTER YOUR CODE HERE

xgb1 = GridSearchCV(
    estimator=xgb,
    param_grid=cv_params,
    scoring=scoring,
    cv=4,
    refit='f1'
)


# Now fit the model to the `X_train` and `y_train` data.

# In[44]:


get_ipython().run_cell_magic('time', '', '#==> ENTER YOUR CODE HERE\n# Fit the GridSearchCV object to the training data\nxgb1.fit(X_train, y_train)\n')


# Get the best score from this model.

# In[45]:


# Examine best score
#==> ENTER YOUR CODE HERE
# Examine the best F1 score from the XGBoost Grid Search
xgb1.best_score_


# And the best parameters.

# In[46]:


# Examine best parameters
#==> ENTER YOUR CODE HERE

# Examine best parameters found by GridSearchCV for XGBoost
xgb1.best_params_


# 1. learning_rate: 0.1 (Step-by-Step Optimization / Optimización Gradual)
# 
# English: The learning rate acts as a scaling factor for each new tree's contributions. By setting it to a conservative 0.1, the algorithm forces the model to learn slowly and methodically, preventing any single tree from dominating the pipeline. For Automatidata, this step-by-step optimization ensures that the system handles volatility in NYC taxi data smoothly, preventing drastic and erratic shifts in dispatch logic.
# 
# Español: La tasa de aprendizaje funciona como un factor de escala para la contribución de cada nuevo árbol. Al fijarla en un conservador 0.1, el algoritmo obliga al modelo a aprender de forma lenta y metódica, evitando que un solo árbol domine el pipeline. Para Automatidata, esta optimización gradual asegura que el sistema maneje la volatilidad de los datos de taxis de Nueva York de manera fluida, previniendo cambios drásticos y erráticos en la lógica de despacho.
# 
# 2. max_depth: 4 (High Efficiency and Speed / Alta Eficiencia y Velocidad)
# 
# English: Limiting the tree depth to 4 means XGBoost is building highly efficient, shallow structures (weak learners) that focus strictly on the most powerful predictive features. Compared to the Random Forest's depth of 10, this architecture requires significantly less computational power and memory. This translates directly into lower cloud infrastructure costs and faster real-time inference speeds when deployed in production.
# 
# Español: Limitar la profundidad de los árboles a 4 significa que XGBoost está construyendo estructuras poco profundas y altamente eficientes (aprendices débiles) que se enfocan estrictamente en las características predictivas más potentes. En comparación con la profundidad de 10 del Random Forest, esta arquitectura requiere significativamente menos potencia de cómputo y memoria. Esto se traduce directamente en menores costos de infraestructura en la nube y velocidades de inferencia en tiempo real más rápidas al implementarse en producción.
# 
# 3. min_child_weight: 1 (Agility in Pattern Discovery / Agilidad en el Descubrimiento de Patrones)
# 
# English: A minimum child weight of 1 allows the gradient boosting sequence to remain highly agile, letting the model create new branches even if they cover smaller, highly specific pockets of data. This allows XGBoost to quickly adjust and capture localized tipping behaviors or niche high-yield micro-routes across Manhattan without losing global stabilization.
# 
# Español: Un peso mínimo por nodo hijo de 1 permite que la secuencia de potenciación de gradiente se mantenga sumamente ágil, dejando que el modelo cree nuevas ramas incluso si cubren subgrupos de datos pequeños y muy específicos. Esto permite que XGBoost se ajuste rápidamente y capture comportamientos de propinas localizados o micro-rutas de alto rendimiento en Manhattan sin perder estabilidad global.

# ##### XGB CV Results
# 
# Use the `make_results()` function to output all of the scores of your model. Note that it accepts three arguments. 

# In[ ]:


# Call 'make_results()' on the GridSearch object
#==> ENTER YOUR CODE HERE


# Use your model to predict on the test data. Assign the results to a variable called `xgb_preds`.
# 
# <details>
#   <summary><h5>HINT</h5></summary>
#     
# You cannot call `predict()` on the GridSearchCV object directly. You must call it on the `best_estimator_`.
# </details>

# In[47]:


# Get scores on test data
#==> ENTER YOUR CODE HERE
# Call 'make_results()' on the GridSearch object
xgb_cv_results = make_results('XGBoost CV', xgb1, 'f1')

# Output the results
xgb_cv_results


# 📋 Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# Al ejecutar este bloque, la función make_results() escarbará en los resultados internos de la validación cruzada de tu XGBoost (xgb1.cv_results_). Identificará el índice de la combinación ganadora que vimos hace un momento (aquella con learning_rate: 0.1 y max_depth: 4) y extraerá sus promedios de rendimiento para formatearlos en una fila horizontal limpia de Pandas. Esto te permitirá comparar directamente su precisión, exhaustividad y F1-Score frente al benchmark establecido por el Random Forest.
# 
# English:
# 
# By running this block, the make_results() function sweeps across the internal cross-validation records inside your XGBoost object (xgb1.cv_results_). It isolates the index of the winning parameter configuration (learning_rate: 0.1 and max_depth: 4) and extracts its validation metrics to format them into a structured Pandas DataFrame row. This provides an immediate framework to benchmark its precision, recall, and F1-score directly against the baseline set by the Random Forest.

# ###### XGB test results
# 
# 1. Use the `get_test_scores()` function to generate the scores on the test data. Assign the results to `xgb_test_scores`.
# 2. Call `xgb_test_scores` to output the results.

# In[48]:


# Get scores on test data
#==> ENTER YOUR CODE HERE

# 1. Predict on the test data using the champion XGBoost model
xgb_preds = xgb1.best_estimator_.predict(X_test)

# 2. Get scores on test data and assign to xgb_test_scores
xgb_test_scores = get_test_scores('XGBoost Test', xgb_preds, y_test)

# 3. Call the variable to output the results table
xgb_test_scores


# 📋 Explicación del paso / Pipeline Step Explanation
# 
# Español:
# 
# Este paso ejecuta el enfrentamiento final. Al llamar a .predict(X_test), ponemos a prueba la secuencia de árboles esbeltos de XGBoost ante la "bóveda a ciegas" de datos nuevos. Luego, la función get_test_scores() contrastará esas predicciones contra las etiquetas reales de y_test. Al desplegar la tabla, podrás ver si XGBoost logra superar la precisión del 68.4% y el F1-Score del 74.4% que dejó marcados el Random Forest en su evaluación de prueba.
# 
# English:
# 
# This block executes the final benchmark audit. By invoking .predict(X_test), we deploy XGBoost's lean boosting architecture against the blind test partition. The get_test_scores() function then cross-references these predictions against the ground-truth outcomes inside y_test. Once the output table renders, you will be able to verify if XGBoost successfully breaks past the 68.4% precision and 74.4% F1-score benchmarks left behind by the Random Forest during its final run.

# **Question:** Compare these scores to the random forest test scores. What do you notice? Which model would you choose?

# English:
# 
# What you notice: The test scores for both models are extraordinarily close, representing a functional statistical tie. Random Forest has a microscopic advantage in F1-Score (0.7448 vs. 0.7428) driven by a slightly higher Recall (81.71% vs. 81.15%). On the other hand, XGBoost yields a marginally better Precision (68.49% vs. 68.42%). Both architectures show remarkable stability, indicating zero overfitting.
# 
# Final Model Choice: I recommend deploying the XGBoost model. Even though the predictive metrics are virtually identical, the tie-breaker comes down to operational efficiency. XGBoost achieves this high-tier performance using a significantly leaner configuration (max_depth: 4 and 100 trees) compared to Random Forest's heavier setup (max_depth: 10 and 200 trees). In a production environment, a leaner model translates to faster real-time inference speeds (lower latency) and drastically lower cloud compute infrastructure costs for Automatidata.
# 
# Español
# 
# Qué se observa: Los puntajes de prueba para ambos modelos son extraordinariamente cercanos, representando prácticamente un empate técnico. Random Forest mantiene una ventaja microscópica en el F1-Score (0.7448 frente a 0.7428) impulsado por un Recall ligeramente mayor (81.71% frente a 81.15%). Por otro lado, XGBoost ofrece una Precisión marginalmente superior (68.49% frente a 68.42%). Ambas arquitecturas demuestran una estabilidad notable, indicando una ausencia total de sobreajuste.
# 
# Elección Final del Modelo: Recomiendo desplegar el modelo de XGBoost. A pesar de que las métricas predictivas son casi idénticas, el factor de decisión es la eficiencia operativa. XGBoost alcanza este rendimiento de primer nivel utilizando una configuración significativamente más esbelta (max_depth: 4 y 100 árboles) en comparación con la estructura más pesada de Random Forest (max_depth: 10 y 200 árboles). En un entorno de producción, un modelo más ligero se traduce en velocidades de inferencia en tiempo real más rápidas (menor latencia) y costos de infraestructura de cómputo en la nube drásticamente menores para Automatidata.

# Plot a confusion matrix of the model's predictions on the test data.

# In[49]:


# Generate array of values for confusion matrix
#==> ENTER YOUR CODE HERE
# Usamos las predicciones del modelo elegido (ej. xgb_preds)
cm = confusion_matrix(y_test, xgb_preds)

# Plot confusion matrPlot a confusion matrix of the model's predictions on the test data.ix
#==> ENTER YOUR CODE HERE

# 2. Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not generous', 'generous'])

# Configuración visual del gráfico
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')

plt.title('XGBoost - Test Confusion Matrix')
plt.show()


# 📊 Matrix Breakdown / Desglose de la Matriz
# 
# With a total test dataset of 3,053 trips, the confusion matrix distributes the outcomes as follows:
# 
# * **True Negatives (TN) = 846:** Rides where the model predicted there would be no generous tip, and the passenger accurately matched that prediction.
# * **False Positives (FP) = 600:** **False Alarms.** Rides where the model assured the driver they would receive a high tip (≥20%), but the customer ended up leaving a standard or zero tip.
# * **False Negatives (FN) = 303:** **Missed Opportunities.** Rides where the model classified the customer as "not generous," but in reality, the passenger was willing to leave a premium tip.
# * **True Positives (TP) = 1,304:** **Commercial Successes.** Rides where the model correctly identified the generous passenger, securing premium revenue for the driver.
# 
# Con un conjunto de prueba total de 3,053 viajes, la matriz de confusión distribuye los resultados de la siguiente manera:
# 
# * True Negatives (TN) = 846: Viajes donde el modelo predijo que no habría propina generosa y el pasajero efectivamente cumplió esa predicción.
# 
# * False Positives (FP) = 600: Falsas Alertas. Viajes donde el modelo le aseguró al conductor que recibiría una propina alta (≥20%), pero el cliente terminó dejando una propina estándar o nula.
# 
# * False Negatives (FN) = 303: Oportunidades Perdidas. Viajes donde el modelo clasificó al cliente como "no generoso", pero en la realidad el pasajero sí estaba dispuesto a dejar una gran propina.
# 
# * True Positives (TP) = 1,304: Éxitos Comerciales. Viajes donde el modelo identificó correctamente al pasajero generoso, asegurando ingresos premium para el conductor.
# 
# 💼 Business Impact Analysis / Análisis de Impacto de Negocio
# 🎯 1. Optimization toward Revenue (High Recall) / Optimización hacia el Ingreso
# English: The matrix vividly shows why your Recall is so high (81.1%). Out of 1,607 actual generous trips (1304+303), the model successfully flagged 1,304 of them. From an operational standpoint, this is highly lucrative for the fleet. The algorithm rarely misses a premium fare, keeping drivers active in the high-yield segments of New York City.
# 
# Español: La matriz muestra claramente por qué tu Recall es tan alto (81.1%). De los 1,607 viajes que realmente eran generosos (1304+303), el modelo capturó exitosamente 1,304. Operativamente, esto es altamente lucrativo para la flota: el algoritmo casi nunca deja pasar un viaje de alta rentabilidad, manteniendo a los conductores activos en los segmentos más premium de Nueva York.
# 
# 🎯 Executive Summary for Notebook / Resumen Ejecutivo para tu Libreta
# Puedes usar este bloque de texto en formato Markdown para cerrar la sección de la matriz en tu proyecto:
# 
# "The test confusion matrix confirms that the XGBoost model operates under a highly effective growth-oriented commercial strategy. By successfully capturing 1,304 True Positives, the algorithm secures 81.1% of the high-value tipping market in NYC. While the 600 False Positives introduce a manageable margin of driver expectation error, the model minimizes missed opportunities to just 303 cases. This diagnostic validates the model as financially robust, protecting platform revenue while sustaining high driver engagement."
# 
# "La matriz de confusión de prueba confirma que el modelo XGBoost opera bajo una estrategia comercial orientada al crecimiento. Al capturar con éxito 1,304 Verdaderos Positivos, el algoritmo asegura el 81.1% del mercado de propinas de alto valor en Nueva York. Aunque los 600 Falsos Positivos introducen un margen manejable de error en las expectativas del conductor, el modelo minimiza las oportunidades perdidas a solo 303 casos. Este diagnóstico valida que el modelo es financieramente robusto, protegiendo los ingresos de la plataforma mientras sostiene un alto compromiso por parte de los conductores."

# **Question:** What type of errors are more common for your model?

# 🧠 Question: What type of errors are more common for your model?
# 
# English 
# Type of Error: False Positives (Type I Error) are significantly more common for this model.
# 
# Analysis: The model generated 600 False Positives (predicting a passenger will be generous when they are actually not) compared to only 303 False Negatives (failing to identify an actual generous passenger). This means the model makes nearly twice as many Type I errors as Type II errors.
# 
# Business Implication: This skew is a direct result of our model optimization toward a high-recall strategy. For Automatidata's fleet, this means drivers will more frequently face a "false alarm" (expecting a high tip that doesn't occur) rather than missing out on a truly premium tipping opportunity. In mobile dispatch applications, this is generally the preferred operational trade-off because it keeps drivers actively engaged and distributed across potential high-yield zones, maximizing total market share capture.
# 
# Español
# 
# Tipo de Error: Los Falsos Positivos (Error Tipo I) son significativamente más comunes en este modelo.
# 
# Análisis: El modelo generó 600 Falsos Positivos (predecir que un pasajero será generoso cuando en realidad no lo es) en comparación con solo 303 Falsos Negativos (fallar en identificar a un pasajero realmente generoso). Esto significa que el modelo comete casi el doble de errores Tipo I que de Tipo II.
# 
# Implicación de Negocio: Este sesgo es el resultado directo de optimizar nuestro modelo hacia una estrategia de alto recall. Para la flota de Automatidata, esto implica que los conductores enfrentarán con mayor frecuencia una "falsa alarma" (esperar una propina alta que no ocurre) en lugar de perderse una oportunidad de ganancia genuinamente premium. En aplicaciones logísticas de despacho, este suele ser el compromiso (trade-off) preferido, ya que mantiene a los conductores activos y distribuidos en zonas de alta demanda, maximizando la captura total de ingresos en el mercado.

# ##### Feature importance
# 
# Use the `feature_importances_` attribute of the best estimator object to inspect the features of your final model. You can then sort them and plot the most important ones.

# In[51]:


#==> ENTER YOUR CODE HERE
# 1. Extract feature importances
importances = xgb1.best_estimator_.feature_importances_
xgb_importances = pd.Series(importances, index=X_train.columns)

# 2. Sort and isolate ONLY the Top 10 most important features
top_10_importances = xgb_importances.sort_values(ascending=False).head(10)

# 3. Plot the clean Top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_importances.values, y=top_10_importances.index, palette='viridis')

plt.title('XGBoost - Top 10 Most Important Features')
plt.xlabel('Importance Score (Gain)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


# 📊 Feature Importance Executive Analysis / Análisis Ejecutivo de Importancia de Variables
# Al analizar las métricas de ganancia (Gain score), observamos una jerarquía de decisiones muy marcada en el modelo XGBoost:
# 
# 1. The Absolute Dominance of VendorID / El Dominio Absoluto de VendorID
# English: VendorID is by far the single most influential feature in the model, capturing an importance score of over 0.35. In the New York taxi infrastructure, this indicates that the technology provider or dispatch system used to record the ride (e.g., Creative Mobile Technologies vs. VeriFone) is highly correlated with whether a tip is recorded as generous (≥20%). This suggests behavioral variations in user interfaces, prompt defaults, or payment processing streams between vendors.
# 
# Español: VendorID es, por mucho, la variable más influyente del modelo, alcanzando un puntaje de importancia superior a 0.35. En la infraestructura de taxis de Nueva York, esto indica que el proveedor tecnológico o sistema de despacho utilizado para registrar el viaje (por ejemplo, Creative Mobile Technologies vs. VeriFone) está fuertemente correlacionado con si una propina se registra como generosa (≥20%). Esto sugiere variaciones de comportamiento en las interfaces de usuario, los valores predeterminados de las propinas o los flujos de procesamiento de pagos entre proveedores.
# 
# 2. Economic Scale Predictor (predicted_fare) / Predictores de Escala Económica
# English: In second place, predicted_fare stands out as the primary engineered feature. This matches logical business expectations: the base cost of the fare determines the financial scale of the transaction. Passengers are generally more prone to cross into the "generous" binary threshold when the journey represents a substantial, high-value service.
# 
# Español: En segundo lugar, predicted_fare se destaca como la principal variable de ingeniería de características. Esto coincide con las expectativas lógicas del negocio: el costo base de la tarifa determina la escala financiera de la transacción. Los pasajeros generalmente tienden a cruzar el umbral binario de "generosos" cuando el trayecto representa un servicio sustancial y de alto valor.
# 
# 3. Geographic Hubs (PULocationID & DOLocationID) / Puntos de Interés Geográficos
# English: The rest of the Top 10 is populated entirely by specific geographic zones, notably locations like 161 (Midtown Manhattan/Grand Central), 230 (Times Square/Theater District), and 79 (East Village). This shows that where the trip starts and ends contains powerful socioeconomic signals. For instance, drop-offs in commercial or high-end entertainment hubs are strong predictors of premium tipping behavior.
# 
# Español: El resto del Top 10 está compuesto en su totalidad por zonas geográficas específicas, destacando ubicaciones como 161 (Midtown Manhattan/Grand Central), 230 (Times Square/Times Square South) y 79 (East Village). Esto demuestra que el dónde comienza y termina el viaje contiene señales socioeconómicas potentes. Por ejemplo, los destinos en centros comerciales o zonas de entretenimiento de alto nivel son fuertes predictores de propinas premium.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## PACE: **Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 4. Conclusion**
# 
# In this step, use the results of the models above to formulate a conclusion. Consider the following questions:
# 
# 1. **Would you recommend using this model? Why or why not?**  
# 
# English:
# 
# Yes, I highly recommend using this model. The final XGBoost architecture achieved an exceptionally strong and stable F1-score of ~0.7428 on completely unseen test data. More importantly, its Recall of 81.14% ensures that the model successfully captures the vast majority of high-tipping opportunities in New York City. While the precision (~68.49%) introduces a manageable volume of false positives (drivers expecting a high tip that ends up being standard), the net financial outcome is heavily positive. It keeps drivers motivated, optimally distributed across high-yield zones, and protects platform retention.
# 
# Español:
# 
# Sí, recomiendo firmemente utilizar este modelo. La arquitectura final de XGBoost alcanzó un F1-score excepcionalmente fuerte y estable de ~0.7428 en datos de prueba completamente nuevos. Lo más importante es que su Recall de 81.14% asegura que el modelo captura con éxito la gran mayoría de las oportunidades de propinas altas en Nueva York. Aunque la precisión (~68.49%) introduce un volumen manejable de falsos positivos (conductores que esperan una propina alta que termina siendo estándar), el resultado financiero neto es muy positivo: mantiene a los conductores motivados, distribuidos óptimamente en zonas de alto rendimiento y protege la retención de la plataforma.
# 
# 2. **What was your model doing? Can you explain how it was making predictions?** 
# 
# English
# 
# Yes, I highly recommend using this model. The final XGBoost architecture achieved an exceptionally strong and stable F1-score of ~0.7428 on completely unseen test data. More importantly, its Recall of 81.14% ensures that the model successfully captures the vast majority of high-tipping opportunities in New York City. While the precision (~68.49%) introduces a manageable volume of false positives (drivers expecting a high tip that ends up being standard), the net financial outcome is heavily positive. It keeps drivers motivated, optimally distributed across high-yield zones, and protects platform retention.
# 
# Español
# 
# Sí, recomiendo firmemente utilizar este modelo. La arquitectura final de XGBoost alcanzó un F1-score excepcionalmente fuerte y estable de ~0.7428 en datos de prueba completamente nuevos. Lo más importante es que su Recall de 81.14% asegura que el modelo captura con éxito la gran mayoría de las oportunidades de propinas altas en Nueva York. Aunque la precisión (~68.49%) introduce un volumen manejable de falsos positivos (conductores que esperan una propina alta que termina siendo estándar), el resultado financiero neto es muy positivo: mantiene a los conductores motivados, distribuidos óptimamente en zonas de alto rendimiento y protege la retención de la plataforma.
# 
# 3. **Are there new features that you can engineer that might improve model performance?**   
# 
# English
# 
# Yes, I highly recommend using this model. The final XGBoost architecture achieved an exceptionally strong and stable F1-score of ~0.7428 on completely unseen test data. More importantly, its Recall of 81.14% ensures that the model successfully captures the vast majority of high-tipping opportunities in New York City. While the precision (~68.49%) introduces a manageable volume of false positives (drivers expecting a high tip that ends up being standard), the net financial outcome is heavily positive. It keeps drivers motivated, optimally distributed across high-yield zones, and protects platform retention.
# 
# Español
# 
# Sí, recomiendo firmemente utilizar este modelo. La arquitectura final de XGBoost alcanzó un F1-score excepcionalmente fuerte y estable de ~0.7428 en datos de prueba completamente nuevos. Lo más importante es que su Recall de 81.14% asegura que el modelo captura con éxito la gran mayoría de las oportunidades de propinas altas en Nueva York. Aunque la precisión (~68.49%) introduce un volumen manejable de falsos positivos (conductores que esperan una propina alta que termina siendo estándar), el resultado financiero neto es muy positivo: mantiene a los conductores motivados, distribuidos óptimamente en zonas de alto rendimiento y protege la retención de la plataforma.
# 
# 4. **What features would you want to have that would likely improve the performance of your model?**   
# 
# English
# 
# To truly elevate this model past the current performance ceiling, acquiring external data streams would be critical:
# 
# Payment Method Details: Knowing if the ride is corporate-billed vs. a personal credit card is arguably the strongest missing psychological driver of tipping behavior.
# 
# Weather Data: Integrating external APIs to capture precipitation, snowfall, or extreme temperatures at the time of pickup, as bad weather historically increases rider gratitude and reliance on taxi services.
# 
# Passenger Historic Tipping Profile: Anonymized loyalty metrics, such as the passenger's average historical tipping percentage on previous trips, which acts as a direct baseline indicator of personal generosity.
# 
# Traffic and Gridlock Analytics: Real-time city-wide delay factors or construction data on the chosen route, allowing the model to capture how transit delays impact the user's final mood and willingness to tip.
# 
# Español
# 
# Para elevar realmente este modelo más allá del techo de rendimiento actual, sería fundamental adquirir flujos de datos externos:
# 
# Detalles del Método de Pago: Saber si el viaje se factura a una cuenta corporativa frente a una tarjeta de crédito personal es, indiscutiblemente, el factor psicológico ausente más fuerte en el comportamiento de las propinas.
# 
# Datos Meteorológicos: Integrar APIs externas para capturar precipitaciones, nevadas o temperaturas extremas al momento del viaje, ya que el mal clima históricamente aumenta la gratitud del usuario y su dependencia de los taxis.
# 
# Perfil Histórico de Propinas del Pasajero: Métricas de fidelidad anonimizadas, como el porcentaje promedio de propina histórica del pasajero en viajes anteriores, lo que actúa como un indicador base directo de generosidad personal.
# 
# Analítica de Tráfico y Embotellamientos: Factores de retraso en tiempo real a nivel de toda la ciudad o datos de construcción en la ruta elegida, permitiendo al modelo capturar cómo las demoras en el tránsito afectan el estado de ánimo final del usuario y su disposición a dejar propina.
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion. Even if you cannot use the model to make strong predictions, was the work done in vain? Consider any insights that you could report back to stakeholders.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
