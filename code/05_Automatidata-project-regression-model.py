#!/usr/bin/env python
# coding: utf-8

# # **Automatidata project**
# **Course 4 - Regression Analysis: Simplify complex data relationships**

# The data consulting firm Automatidata has recently hired you as the newest member of their data analytics team. Their newest client, the NYC Taxi and Limousine Commission (New York City TLC), wants the Automatidata team to build a multiple linear regression model to predict taxi fares using existing data that was collected over the course of a year. The team is getting closer to completing the project, having completed an initial plan of action, initial Python coding work, EDA, and A/B testing.
# 
# The Automatidata team has reviewed the results of the A/B testing. Now it’s time to work on predicting the taxi fare amounts. You’ve impressed your Automatidata colleagues with your hard work and attention to detail. The data team believes that you are ready to build the regression model and update the client New York City TLC about your progress.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # Course 4 End-of-course project: Build a multiple linear regression model
# 
# In this activity, you will build a multiple linear regression model. As you've learned, multiple linear regression helps you estimate the linear relationship between one continuous dependent variable and two or more independent variables. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed. 
# 
# Completing this activity will help you practice planning out and buidling a multiple linear regression model based on a specific business need. The structure of this activity is designed to emulate the proposals you will likely be assigned in your career as a data professional. Completing this activity will help prepare you for those career moments.
# <br/>
# 
# **The purpose** of this project is to demostrate knowledge of EDA and a multiple linear regression model
# 
# **The goal** is to build a multiple linear regression model and evaluate the model
# <br/>
# *This activity has three parts:*
# 
# **Part 1:** EDA & Checking Model Assumptions
# ES:
# 1. Validar los supuestos del modelo: Identificar visualmente si existe una relación lineal entre las variables predictoras (distancia, tiempo) y la variable objetivo (fare_amount), y revisar la presencia de multicolinealidad entre variables independientes.
# 
# 2. Limpieza y detección de anomalías: Localizar y filtrar datos corruptos o imposibles, tales como tarifas negativas (fare_amount <= 0) o distancias de viaje iguales a cero que tengan costos asociados altos.
# 
# 3. Análisis de distribución y Outliers: Observar la distribución de los residuos y detectar valores atípicos extremas que puedan jalar o sesgar la línea de mejor ajuste bajo el método de Mínimos Cuadrados Ordinarios (OLS).
# 
# EN:
# 1. Validate model assumptions: Visually identify whether a linear relationship exists between the predictor variables (distance, duration) and the target variable (fare_amount), and check for multicollinearity among independent features.
# 
# 2. Data cleaning and anomaly detection: Locate and filter corrupt or impossible data points, such as negative taxi fares (fare_amount <= 0) or trip distances of zero that feature high cost overheads.
# 
# 3. Distribution and Outlier Analysis: Inspect the distributions and isolate extreme outliers that could heavily skew or bias the line of best fit under the Ordinary Least Squares (OLS) estimation method.
# 
# **Part 2:** Model Building and evaluation
# * What resources do you find yourself using as you complete this stage?
# ES:
# 1. Librerías Core de Python: scikit-learn (módulos LinearRegression, train_test_split y métricas de evaluación) y statsmodels para obtener los resúmenes estadísticos detallados (valores P y estadístico F).
# 
# 2. Librerías Gráficas: seaborn y matplotlib para graficar los diagramas de dispersión de residuos y los gráficos Q-Q de normalidad.
# 
# 3. Documentación de Soporte: El diccionario de datos de la NYC TLC para entender restricciones operativas (como tarifas base fijas o recargos nocturnos)
# 
# EN: 
# 1. Core Python Libraries: scikit-learn (specifically LinearRegression, train_test_split, and evaluation metrics packages) and statsmodels to extract detailed statistical summaries (P-values and F-statistic scores).
# 
# 2. Visualization Tools: seaborn and matplotlib to plot residual scatter plots and Q-Q normality charts.
# 
# 3. Support Documentation: The official NYC TLC data dictionary to comprehend operational rules (such as flat base fares or overnight surcharges).
# **Part 3:** Interpreting Model Results
# 
# * What key insights emerged from your model(s)?
# ES:
# 1. Predictores Dominantes: La distancia del viaje (trip_distance) y la duración estimada son las variables con mayor peso estadístico y significancia (valores P < 0.05).
# 2. Capacidad Explicativa (R^2): El coeficiente de determinación nos indica qué porcentaje de la variabilidad del costo del taxi es explicado de manera efectiva por nuestro conjunto de variables explicativas combinadas.
# 3. Comportamiento de Residuos: El análisis muestra que el modelo funciona de manera excelente para viajes estándar, pero pierde precisión en tarifas fijas especiales (como los viajes regulados hacia el aeropuerto JFK).
# 
# EN:  
# 1. Dominant Predictors: Trip distance (trip_distance) and estimated travel duration hold the highest statistical weight and significance (P-values <0.05).
# 
# 2. Explanatory Power (R2): The coefficient of determination isolates the exact percentage of taxi fare variance that is effectively captured and explained by our combined explanatory features array.
# 
# 3. Residual Behavior: Diagnostics indicate the model performs exceptionally well for standard metered trips, but loses accuracy when encountering special flat-rate structures (such as mandated trips to JFK Airport).
# 
#    
# * What business recommendations do you propose based on the models built?
# ES:
# 
# 1. Segmentación de Tarifas Fijas: Recomiendo crear un modelo separado o una regla de excepción lógica en la aplicación para viajes con tarifas planas fijas (ej. trayectos hacia aeropuertos), ya que la regresión lineal tiende a sobreestimar o subestimar estos casos especiales.
# 
# 2. Incorporación de Variables Temporales: Integrar variables que capturen el tráfico en tiempo real o recargos por hora punta para mejorar el rendimiento del modelo durante los horarios de alta congestión en Manhattan.
# 
# 3. Despliegue de la Herramienta de Estimación: El modelo es lo suficientemente robusto en sus métricas globales (MAE bajo) para ser integrado en la interfaz de usuario de la app cliente, permitiendo estimaciones transparentes que mejoren la confianza del pasajero antes de abordar.
# 
# EN:
# 
# 1. Flat-Rate Segmentation: I recommend isolating flat-rate journeys (e.g., airport transfers) into a separate logical branch or dedicated sub-model within the app, as standard linear regression tends to miscalculate these non-linear pricing constraints.
# 
# 2. Incorporate Temporal Features: Integrate real-time traffic density indexes or peak-hour binary flags to enhance model performance during severe rush-hour congestion windows across Manhattan.
# 
# 3. Deploy the Estimation Tool: The model is structurally robust regarding its baseline evaluation metrics (low MAE), making it fully viable for integration into the client application interface to provide transparent pre-trip fare estimates, thus boosting passenger confidence.
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.

# # Build a multiple linear regression model

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## PACE: **Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 

# ### Task 1. Imports and loading
# Import the packages that you've learned are needed for building linear regression models.

# In[1]:


# Imports
# Packages for numerics + dataframes
### YOUR CODE HERE ###
import pandas as pd
import numpy as np

# Packages for visualization
### YOUR CODE HERE ###
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for date conversions for calculating trip durations
### YOUR CODE HERE ###
import datetime as dt

# Packages for OLS, MLR, confusion matrix
### YOUR CODE HERE ###
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# **Note:** `Pandas` is used to load the NYC TLC dataset. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe 
df0=pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv") 
# Inspect the first 10 rows / Inspeccionar las primeras 10 filas
df0.head(10)


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## PACE: **Analyze**
# 
# In this stage, consider the following question where applicable to complete your code response:
# 
# * What are some purposes of EDA before constructing a multiple linear regression model?
# 
# 1. Checking Model Assumptions / Comprobación de los supuestos del modelo
# 
# ES:
# * La regresión lineal clásica por Mínimos Cuadrados Ordinarios (OLS) depende estrictamente de supuestos matemáticos. El EDA te permite validar visual y estadísticamente estos criterios antes de entrenar el algoritmo:
# 
# * Linealidad: Mediante gráficos de dispersión (scatter plots), puedes verificar si existe una relación en línea recta entre cada variable independiente (X) y la variable dependiente (Y). Si la relación es curva, sabrás que necesitas aplicar transformaciones polinómicas o logarítmicas.
# 
# * No Multicolinealidad: Al generar una matriz de correlación o calcular el Factor de Inflación de la Varianza (VIF), identificas si tus variables independientes están altamente correlacionadas entre sí. Esto es vital para evitar coeficientes inestables.
# 
# EN:
# 
# * Ordinary Least Squares (OLS) regression relies heavily on structural mathematical assumptions. EDA allows you to visually and statistically validate these boundaries before fitting your model:
# 
# * Linearity: By plotting scatter plots, you check whether a straight-line relationship exists between each independent feature (X) and your continuous target (Y). If a relationship is non-linear, you immediately know you must apply transformations (e.g., log or polynomial terms).
# 
# * No Multicollinearity: Generating a correlation matrix or calculating the Variance Inflation Factor (VIF) helps detect if independent variables are highly correlated with each other, preventing unstable coefficient estimations and misleading p-values.
# 
# 2. Detection and Handling of Outliers / Detección y tratamiento de valores atípicos
# ES:
# 
# * La regresión lineal es extremadamente sensible a los valores atípicos (outliers), ya que el método OLS intenta minimizar la suma de los errores al cuadrado. Un solo dato extremo puede "jalar" la línea de regresión, sesgando los coeficientes independientes de todo el modelo. Usar diagramas de caja (box plots) ayuda a aislar estas anomalías para decidir si deben ser eliminadas, recortadas o imputadas.
# EN: 
# 
# * Linear regression models are highly sensitive to extreme anomalies because OLS minimizes squared residuals. A single severe outlier can single-handedly "pull" the regression line toward it, heavily biasing the model's coefficients. Utilizing box plots during EDA helps flag these points so you can strategically determine whether to drop, winsorize, or investigate them.
# 
# 3. Understanding Distributions and Feature Scaling / Comprensión de distribuciones e ingeniería de variables
# 
# ES:
# * Aunque la regresión lineal no exige que las variables independientes (X) sean normales, revisar sus distribuciones mediante histogramas ayuda a identificar asimetrías severas (skewness). Además, revisar las escalas de tus datos te indica si necesitas estandarizar o normalizar variables continuas antes de evaluar la magnitud de los coeficientes, o si debes aplicar One-Hot Encoding a las columnas categóricas del dataset.
# 
# EN:
# * While linear regression does not mandate that predictor variables (X) follow a perfect normal distribution, inspecting histograms helps identify severe skewness that might affect model variance. Furthermore, observing the raw metric ranges alerts you if feature scaling (standardization/normalization) is necessary to interpret relative coefficient weights, or if categorical text features require conversion via One-Hot Encoding.
# 
# 4. Identifying Missing Data Patterns / Identificación de patrones de datos faltantes
# 
# ES:
# 
# * Ejecutar código exploratorio para contabilizar valores nulos (NaN) te permite diseñar una estrategia limpia de preprocesamiento. Si eliminas registros a ciegas sin hacer EDA, podrías introducir sesgos sistemáticos si los datos faltan debido a un patrón específico (Faltante no Aleatorio), afectando la representatividad de la muestra con la que entrenas tu regresión.
# 
# EN:
# 
# * Running exploratory code to map missing values (NaN) protects your training pipeline. Blindly dropping rows without doing EDA can accidentally introduce systematic bias if data is Missing Not At Random (MNAR), which dramatically limits how well your multiple regression model generalizes to the broader population.

# ==> ENTER YOUR RESPONSE HERE 

# ### Task 2a. Explore data with EDA
# 
# Analyze and discover data, looking for correlations, missing data, outliers, and duplicates.

# Start with `.shape` and `.info()`.

# In[3]:


# Start with `.shape` and `.info()`
### YOUR CODE HERE ###

# Check the dimensions of the dataset (rows, columns)
# Comprobar las dimensiones del conjunto de datos (filas, columnas)
print("=== DATASET SHAPE ===")
print(df0.shape)

# Display a summary of columns, data types, and non-null counts
# Mostrar un resumen de las columnas, tipos de datos y recuentos de valores no nulos
print("\n=== DATASET INFO ===")
df0.info()


# Check for missing data and duplicates using `.isna()` and `.drop_duplicates()`.

# In[4]:


# Check for missing data and duplicates using .isna() and .drop_duplicates()
### YOUR CODE HERE ###

# Check for missing data (NaN) by summing up true values per column
# Comprobar datos faltantes (NaN) sumando los valores verdaderos por columna
print("=== MISSING VALUES PER COLUMN ===")
print(df0.isna().sum())

# Check for duplicate rows in the dataframe
# Comprobar si existen filas duplicadas en el dataframe
print("\n=== DUPLICATE ROWS COUNT ===")
print(df0.duplicated().sum())


# Use `.describe()`.

# In[5]:


# Use .describe()
### YOUR CODE HERE ###
# Generar estadísticas descriptivas para todas las columnas numéricas
df0.describe()


# ### Task 2b. Convert pickup & dropoff columns to datetime
# 

# In[6]:


# Check the format of the data
### YOUR CODE HERE ###
# 1. Check the format of the data (data types before conversion)
# Comprobar el formato de los datos (tipos de datos antes de la conversión)
print("=== DATA TYPES BEFORE CONVERSION ===")
print(df0[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].dtypes)
print("\n=== SAMPLE DATA BEFORE CONVERSION ===")
print(df0[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].head(3))


# In[7]:


# Convert datetime columns to datetime
### YOUR CODE HERE ###

# 2. Convert datetime columns to datetime objects using pandas
# Convertir columnas de fecha a objetos datetime usando pandas
df0['tpep_pickup_datetime'] = pd.to_datetime(df0['tpep_pickup_datetime'])
df0['tpep_dropoff_datetime'] = pd.to_datetime(df0['tpep_dropoff_datetime'])

# 3. Verify the change in data types
# Verificar el cambio en los tipos de datos
print("\n=== DATA TYPES AFTER CONVERSION ===")
print(df0[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].dtypes)


# ### Task 2c. Create duration column

# Create a new column called `duration` that represents the total number of minutes that each taxi ride took.

# In[8]:


# Create `duration` column
### YOUR CODE HERE ###
# ==============================================================================
# Task 2c. Create duration column / Crear columna de duración
# ==============================================================================

# Create 'duration' column representing the total number of minutes per ride
# Crear la columna 'duration' que representa el número total de minutos por viaje
df0['duration'] = (df0['tpep_dropoff_datetime'] - df0['tpep_pickup_datetime']).dt.total_seconds() / 60

# Inspect the first few rows of the updated columns to verify the calculation
# Inspeccionar las primeras filas de las columnas actualizadas para verificar el cálculo
df0[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']].head()


# ### Outliers
# 
# Call `df.info()` to inspect the columns and decide which ones to check for outliers.

# In[9]:


### YOUR CODE HERE ###
# ==============================================================================
# Outliers Inspection / Inspección de Valores Atípicos
# ==============================================================================

# Call .info() to inspect columns and data types
# Llamar a .info() para inspeccionar las columnas y los tipos de datos
df0.info()


# Keeping in mind that many of the features will not be used to fit your model, the most important columns to check for outliers are likely to be:
# * `trip_distance`
# * `fare_amount`
# * `duration`
# 
# 

# ### Task 2d. Box plots
# 
# Plot a box plot for each feature: `trip_distance`, `fare_amount`, `duration`.

# In[10]:


### YOUR CODE HERE ###
# ==============================================================================
# Task 2d. Box plots / Diagramas de Caja para Outliers
# ==============================================================================

# Set up a 1-row, 3-column subplots figure layout
# Configurar un lienzo de subgráficos de 1 fila y 3 columnas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Box plot for trip_distance
# Diagrama de caja para la distancia del viaje
sns.boxplot(data=df0, x='trip_distance', ax=axes[0], color='skyblue')
axes[0].set_title('Trip Distance - Outliers Check')
axes[0].set_xlabel('Distance (miles)')

# 2. Box plot for fare_amount
# Diagrama de caja para el importe de la tarifa (Target)
sns.boxplot(data=df0, x='fare_amount', ax=axes[1], color='lightgreen')
axes[1].set_title('Fare Amount - Outliers Check')
axes[1].set_xlabel('Fare Amount ($)')

# 3. Box plot for duration
# Diagrama de caja para la duración calculada del viaje
sns.boxplot(data=df0, x='duration', ax=axes[2], color='salmon')
axes[2].set_title('Duration - Outliers Check')
axes[2].set_xlabel('Duration (minutes)')

# Adjust spacing between plots dynamically
# Ajustar el espacio entre gráficos de forma dinámica
plt.tight_layout()
plt.show()


# **Questions:** 
# 1. Which variable(s) contains outliers? 
# 
# 2. Are the values in the `trip_distance` column unbelievable?
# 
# 3. What about the lower end? Do distances, fares, and durations of 0 (or negative values) make sense?

# 1.- Las tres variables que examinamos contienen valores atípicos (outliers), pero la naturaleza de estos datos es distinta:
# 
# * duration (Contiene outliers que son anomalías del sistema):
# El gráfico muestra puntos extremos que van mucho más allá de un viaje normal, destacando un registro que supera los 1,400 minutos (~24 horas). Esto es claramente un error de registro o un fallo del taxímetro que se quedó encendido.
# 
# * fare_amount (Contiene outliers que son una mezcla de anomalías y valores reales):
# Muestra puntos que se extienden hacia la derecha llegando casi a los $1,000 dólares, lo cual suele ser un error de digitación o un cargo de peaje corrupto.
# Por el extremo izquierdo, tiene valores negativos o iguales a cero, lo cual representa un error en el sistema de cobro.
# También tiene valores altos legítimos que corresponden a viajes de larga distancia.
# 
# * trip_distance (Contiene outliers que son valores reales extremos):
# Aunque la gran mayoría de los viajes son de menos de 5 millas, hay puntos aislados que llegan a las 30 o 40 millas. Estos son valores atípicos desde el punto de vista estadístico, pero son físicamente reales (por ejemplo, viajes largos a los extremos de Long Island o aeropuertos lejanos).
# 
# EN:
# All three variables check positive for outliers, though their underlying structural nature differs considerably:
# 
# * duration (Contains outliers that are system anomalies):
# The box plot maps extreme points stretching far past normal operational boundaries, single-handedly highlighted by an entry exceeding 1,400 minutes (~24 hours). This serves as a textbook logging bug or an unclosed meter ledger.
# 
# * fare_amount (Contains outliers that are a mixture of anomalies and real values):
# It exhibits extreme maximum points extending close to $1,000, which point to clerical typing entry errors or corrupted toll data updates.
# On the lower boundary, it lists negative and zero-dollar entries, representing definitive transaction logging errors.
# It also hosts valid, high-dollar figures tied directly to long-distance hauling.
# 
# * trip_distance (Contains outliers that are extreme but real values):
# While the heavy distribution cluster stays locked under 5 miles, isolated marker dots bridge out to the 30–40 mile zone. These are statistical outliers due to variance, but they represent real-world commuter behavior (e.g., long-distance airport transfers or outer-borough trips).
# 
# ES:
# 2. No, los valores en la columna trip_distance no son inverosímiles (unbelievable), pero sí son extremos desde el punto de vista estadístico.
# 
#  * Por qué son creíbles: El valor máximo observado en los datos ronda las 33-40 millas. Físicamente, un viaje desde el centro de Manhattan hasta los aeropuertos periféricos (como el JFK) o hacia los límites exteriores de los condados de Nueva York (como Staten Island o el norte del Bronx) puede alcanzar fácilmente esa distancia. Por lo tanto, representan comportamientos de viaje reales y legalmente posibles para un taxi amarillo.
# 
#  * Por qué se ven como outliers: La gran mayoría de los viajes en Nueva York ocurren dentro de Manhattan, donde las distancias promedio son muy cortas (generalmente menos de 2 a 5 millas). Cuando graficas estos datos, los viajes largos hacia los aeropuertos o fuera de la ciudad aparecen como puntos aislados (outliers) debido a la enorme diferencia con el viaje promedio en la ciudad, pero no son errores de registro.
#  
#  EN
# 2. No, the values in the trip_distance column are not unbelievable, though they qualify as extreme statistical outliers.
# 
#  * Why they are plausible: The maximum values recorded in the dataset hover around 33–40 miles. Geographically, a non-stop journey from midtown Manhattan out to peripheral transportation hubs (like JFK International Airport) or the outermost borders of the NYC boroughs (such as deep Staten Island or the northern Bronx) can easily clock this mileage. Therefore, these points represent genuine, legally permissible consumer trips.
# 
#   * Why they appear as outliers: The vast majority of NYC yellow cab operations happen as short commutes within the dense grid of Manhattan, keeping the median trip length very low (typically under 2 to 5 miles). When plotted, long-haul airport runs or cross-borough transits appear as isolated data points due to the massive variance from the city baseline, but they are operationally real, not system bugs.
#   
# ES:
# 3. No, los valores negativos no tienen sentido y representan errores claros. Sin embargo, los valores de cero dependientes del contexto pueden tener una explicación operativa, aunque deben ser tratados.
# 
#   * Valores Negativos (Inverosímiles): * Las distancias de viaje, tarifas o duraciones negativas son físicamente imposibles.
# 
# En el caso de fare_amount, las tarifas negativas ocurren en los sistemas de la TLC debido a disputas de cobro, cancelaciones mal procesadas o errores de contracargo. Como violan la lógica del negocio y rompen los supuestos de una regresión lineal, deben ser eliminadas del dataset.
# 
#   * Valores de Cero (0):
# 
# trip_distance de 0.0: Puede ser real si un pasajero abordó el taxi pero canceló el viaje inmediatamente antes de avanzar, o si el conductor cobró una tarifa fija de inicio sin que el taxímetro registrara movimiento. Si la distancia es 0 pero la tarifa es alta, el dato confunde al modelo lineal.
# 
# duration de 0.0: Ocurre si la hora de recogida (pickup) y entrega (dropoff) registran exactamente el mismo segundo. Esto delata un viaje cancelado al instante o un error de sincronización del GPS.
# 
#   * fare_amount de 0.0: Un viaje de costo cero (o tarifas base extremadamente bajas como $0.01) representa viajes de cortesía, pruebas de sistema o errores de registro, ya que legalmente un taxi de Nueva York inicia con una tarifa base obligatoria (bajada de bandera de $2.50).
#   
# EN:
# No, negative values make no logical sense and represent clear data corruption. However, zero-valued entries are context-dependent and may have operational explanations, though they still require cleaning.
# 
#  * Negative Values (Implausible): * Negative trip distances, fares, or durations are physically impossible.
# 
# In the case of fare_amount, negative metrics occur within the TLC database due to payment disputes, poorly logged trip cancellations, or credit card chargeback overrides. Because they violate fundamental business logic and break linear regression assumptions, they must be purged from the dataset.
# 
#   * Zero Values (0):
# 
# trip_distance of 0.0: This can be operationally real if a passenger entered the cab but cancelled the ride immediately before the vehicle moved, or if a flat rate was applied manually. However, if distance is 0 while the fare is high, it severely misleads a linear regression model.
# 
#   * duration of 0.0: This happens when the pickup and dropoff timestamps are identical down to the exact second. This flags an instantaneously aborted ride or a tracking hardware synchronization glitch.
# 
#   * fare_amount of 0.0: A zero-dollar fare (or nominal values like $0.01) points to promotional waivers, internal system     testing, or accounting errors, since NYC yellow cabs strictly mandate an initial base flag-drop rate (historically starting at $2.50). 

# ### Task 2e. Imputations

# #### `trip_distance` outliers
# 
# You know from the summary statistics that there are trip distances of 0. Are these reflective of erroneous data, or are they very short trips that get rounded down?
# Los datos reflejan un error o anomalía en el registro, no viajes muy cortos que fueron redondeados hacia abajo.
# 
# Cómo lo sabemos: Al ordenar los valores y eliminar los duplicados, la consola demostró que el dataset registra distancias con una precisión decimal extremadamente alta (vimos valores únicos sucesivos como 0.01, 0.02, 0.03 millas, etc.).
# 
# Conclusión: Dado que el sistema tiene la capacidad técnica de registrar fracciones de milla tan pequeñas, no existe un proceso automático de "redondeo hacia abajo" que convierta viajes cortos en 0.0. Por lo tanto, una distancia de exactamente 0.0 millas representa datos erróneos, fallos en el sensor de geolocalización del taxímetro, o viajes cancelados inmediatamente antes de que el vehículo se moviera físicamente.
# 
# English
# The data points reflect recording errors or system anomalies, not short trips that were rounded down.
# 
# How we know this: By sorting the unique values, the console output proved that the dataset captures distance metrics with a highly granular decimal precision (we observed consecutive unique steps like 0.01, 0.02, 0.03 miles, etc.).
# 
# Conclusion: Since the logging software possesses the technical capability to isolate fractions of a mile this small, there is no background process rounding short distances down to zero. Therefore, a trip distance recorded as exactly 0.0 miles represents erroneous data, a tracking hardware/GPS malfunction, or an instantly aborted ride where the vehicle physically never moved.
# To check, sort the column values, eliminate duplicates, and inspect the least 10 values. Are they rounded values or precise values?

# In[11]:


# Are trip distances of 0 bad data or very short trips rounded down?
### YOUR CODE HERE ###
# ==============================================================================
# Task 2e. Imputations - Trip Distance Outliers
# Imputaciones - Valores Atípicos en Distancia de Viaje
# ==============================================================================

# Sort values, eliminate duplicates, and inspect the lowest 10 unique distances
# Ordenar valores, eliminar duplicados e inspeccionar las 10 distancias únicas más bajas
df0['trip_distance'].sort_values().drop_duplicates().head(10)


# The distances are captured with a high degree of precision. However, it might be possible for trips to have distances of zero if a passenger summoned a taxi and then changed their mind. Besides, are there enough zero values in the data to pose a problem?
# 
# Calculate the count of rides where the `trip_distance` is zero.

# In[12]:


### YOUR CODE HERE ###
# ==============================================================================
# Calculate the count of rides with 0 trip distance
# Calcular el recuento de viajes con distancia cero
# ==============================================================================

# Sum the number of True booleans where trip_distance equals 0.0
# Sumar el número de booleanos Verdaderos donde trip_distance es igual a 0.0
zero_distance_count = (df0['trip_distance'] == 0.0).sum()

print("=== COUNT OF ZERO-DISTANCE RIDES ===")
print(f"Number of rides with 0 distance: {zero_distance_count}")

# Calculate the percentage of the dataset these rows represent
# Calcular el porcentaje del dataset que representan estas filas
percentage = (zero_distance_count / len(df0)) * 100
print(f"Percentage of total data: {percentage:.2f}%")


# #### `fare_amount` outliers

# In[13]:


### YOUR CODE HERE ###
# ==============================================================================
# Inspect fare_amount for trips with a distance of 0.0
# Inspeccionar fare_amount para viajes con una distancia de 0.0
# ==============================================================================

# Calculate descriptive statistics for fare_amount when trip_distance is 0
# Calcular estadísticas descriptivas para fare_amount cuando trip_distance es 0
df0[df0['trip_distance'] == 0.0]['fare_amount'].describe()


# **Question:** What do you notice about the values in the `fare_amount` column?
# Español:
# La Mediana (50% = $3.00) y el Primer Cuartil (25% = $2.50): Esto confirma que una gran parte de estos viajes de 0 millas se quedaron estancados en el valor de la "bajada de bandera" (initial flag-drop) de la NYC TLC o un costo mínimo por abordaje. Es el comportamiento típico de un pasajero que se sube al taxi y cancela el viaje de inmediato.
# 
# El Valor Mínimo (min = -$2.50): Representa un cobro de penalización revertido o un contracargo por cancelación. Físicamente, una tarifa negativa no existe para el negocio habitual, confirmando un error de registro transaccional.
# 
# El Valor Máximo (max = $450.00): Una tarifa de $450 dólares por recorrer 0 millas es una anomalía total del sistema. Esto delata un fallo grave en el hardware del taxímetro (que no sumó la distancia real del viaje pero sí cobró el dinero), un fraude, o la aplicación manual de una tarifa plana especial mal cargada.
# 
# English:
# The Median (50% = $3.00) and First Quartile (25% = $2.50): This confirms that a large portion of these 0-mile trips remained locked at the NYC TLC "initial flag-drop" rate or a minimal entry charge. This is a classic pattern for a passenger who boards the cab and immediately changes their mind.
# 
# The Minimum Value (min = -$2.50): This represents a reversed flag-drop charge or a cancellation chargeback. Physically, negative pricing does not exist in standard business operations, confirming a transactional logging error.
# 
# The Maximum Value (max = $450.00): A $450 fare for traveling 0 miles is a complete system anomaly. This points to a severe meter hardware failure (where the device failed to log physical distance but processed the dollar ledger), fraud, or an incorrectly processed flat-rate manual override.
# 
# Impute values less than $0 with `0`.

# In[14]:


# Impute values less than $0 with 0
### YOUR CODE HERE ###
# ==============================================================================
# Impute fare_amount values less than $0 with 0
# Imputar valores de fare_amount menores que $0 con 0
# ==============================================================================

# Replace any negative fare with 0.0
# Reemplazar cualquier tarifa negativa con 0.0
df0.loc[df0['fare_amount'] < 0, 'fare_amount'] = 0.0

# Verify that the minimum value is now 0.0
# Verificar que el valor mínimo ahora sea 0.0
print("=== VERIFICATION: NEW FARE_AMOUNT MINIMUM ===")
print(df0['fare_amount'].describe()[['min', 'max']])


# Now impute the maximum value as `Q3 + (6 * IQR)`.

# In[15]:


### YOUR CODE HERE ###
   # ==============================================================================
# Task 2f (Continued): Custom Function for Outlier Imputation
# Función personalizada para la imputación de valores atípicos
# ==============================================================================

def impute_outliers(column_list, iqr_factor=6):
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.
    '''
    for col in column_list:
        # Reassign minimum to zero
        # Reasignar valores negativos o menores a cero a exactamente 0.0
        df0.loc[df0[col] < 0, col] = 0.0

        # Calculate upper threshold using Q1, Q3, and IQR
        # Calcular los cuartiles Q1 (25%), Q3 (75%) y el IQR (Rango Intercuartílico)
        q1 = df0[col].quantile(0.25)
        q3 = df0[col].quantile(0.75)
        iqr = q3 - q1
        
        # Calculate upper threshold / Calcular el umbral superior: Q3 + (6 * IQR)
        upper_threshold = q3 + (iqr_factor * iqr)
        print(f"--- Column: {col} ---")
        print(f"Calculated Upper Threshold: {upper_threshold:.2f}")

        # Reassign values > threshold to threshold
        # Reasignar valores mayores que el umbral al valor del umbral superior
        df0.loc[df0[col] > upper_threshold, col] = upper_threshold
        print(f"Maximum value after imputation: {df0[col].max():.2f}\n")

# Execute the function on the three target columns
# Ejecutar la función en las tres columnas críticas del análisis
impute_outliers(['trip_distance', 'fare_amount', 'duration'], iqr_factor=6)


# Mirando los números con ojos de científico de datos, la transformación es sumamente interesante:
# 
# trip_distance: Se acotó a un máximo de 15.48 millas. Esto elimina viajes larguísimos que salían de la jurisdicción normal y que habrían sesgado la pendiente.
# 
# fare_amount: El preocupante máximo de 999.99 ahora quedó truncado de forma segura en $62.50 dólares. Curiosamente, este umbral quedó muy cerca de la tarifa plana histórica que se cobra para viajes desde/hacia el aeropuerto JFK, lo que le da un excelente sentido de negocio.
# 
# duration: La anomalía masiva de 1,400 minutos (24 horas) desapareció, y el viaje más largo permitido ahora es de 88.78 minutos (1 hora y 28 minutos). Esto es un límite físico perfecto para el tráfico en hora punta de Nueva York.

# #### `duration` outliers
# 

# In[16]:


# Call .describe() for duration outliers
### YOUR CODE HERE ###
# ==============================================================================
# Call .describe() for duration outliers verification
# Llamar a .info() / .describe() para verificar los outliers de duración
# ==============================================================================

# Generate the statistical summary for the 'duration' column after imputation
# Generar el resumen estadístico para la columna 'duration' después de la imputación
df0['duration'].describe()


# El resumen estadístico de duration quedó impecable:
# 
# count: Mantiene las 22,699 filas intactas.
# 
# min: Se consolidó en 0.000000.
# 
# max: Quedó perfectamente acotado en 88.783333 minutos.
# 
# Con esto, la fase de imputación y tratamiento de valores atípicos de la Task 2 ha finalizado con éxito. El conjunto de datos está estadísticamente saneado para cumplir los supuestos de un modelo lineal.

# The `duration` column has problematic values at both the lower and upper extremities.
# 
# * **Low values:** There should be no values that represent negative time. Impute all negative durations with `0`.
# 
# * **High values:** Impute high values the same way you imputed the high-end outliers for fares: `Q3 + (6 * IQR)`.

# In[17]:


# Impute a 0 for any negative values
### YOUR CODE HERE ###
# 1. Impute a 0 for any negative values
# Imputar un 0 para cualquier valor de duración que sea negativo
df0.loc[df0['duration'] < 0, 'duration'] = 0.0


# In[18]:


# Impute the high outliers
### YOUR CODE HERE ###
# 2. Impute the high outliers using Q3 + (6 * IQR)
# Imputar los valores atípicos altos calculando el umbral superior
q1_dur = df0['duration'].quantile(0.25)
q3_dur = df0['duration'].quantile(0.75)
iqr_dur = q3_dur - q1_dur

# Calculate upper threshold / Calcular el umbral superior
upper_threshold_dur = q3_dur + (6 * iqr_dur)

# Reassign values > threshold to threshold
# Reasignar valores mayores que el umbral al valor del umbral superior
df0.loc[df0['duration'] > upper_threshold_dur, 'duration'] = upper_threshold_dur


# 3. Final verification of the distribution
# Verificación final de los estadísticos límites
print("=== VERIFICATION: NEW DURATION MIN & MAX ===")
print(df0['duration'].describe()[['min', 'max']])


# ### Task 3a. Feature engineering

# #### Create `mean_distance` column
# 
# When deployed, the model will not know the duration of a trip until after the trip occurs, so you cannot train a model that uses this feature. However, you can use the statistics of trips you *do* know to generalize about ones you do not know.
# 
# In this step, create a column called `mean_distance` that captures the mean distance for each group of trips that share pickup and dropoff points.
# 
# For example, if your data were:
# 
# |Trip|Start|End|Distance|
# |--: |:---:|:-:|    |
# | 1  | A   | B | 1  |
# | 2  | C   | D | 2  |
# | 3  | A   | B |1.5 |
# | 4  | D   | C | 3  |
# 
# The results should be:
# ```
# A -> B: 1.25 miles
# C -> D: 2 miles
# D -> C: 3 miles
# ```
# 
# Notice that C -> D is not the same as D -> C. All trips that share a unique pair of start and end points get grouped and averaged.
# 
# Then, a new column `mean_distance` will be added where the value at each row is the average for all trips with those pickup and dropoff locations:
# 
# |Trip|Start|End|Distance|mean_distance|
# |--: |:---:|:-:|  :--   |:--   |
# | 1  | A   | B | 1      | 1.25 |
# | 2  | C   | D | 2      | 2    |
# | 3  | A   | B |1.5     | 1.25 |
# | 4  | D   | C | 3      | 3    |
# 
# 
# Begin by creating a helper column called `pickup_dropoff`, which contains the unique combination of pickup and dropoff location IDs for each row.
# 
# One way to do this is to convert the pickup and dropoff location IDs to strings and join them, separated by a space. The space is to ensure that, for example, a trip with pickup/dropoff points of 12 & 151 gets encoded differently than a trip with points 121 & 51.
# 
# So, the new column would look like this:
# 
# |Trip|Start|End|pickup_dropoff|
# |--: |:---:|:-:|  :--         |
# | 1  | A   | B | 'A B'        |
# | 2  | C   | D | 'C D'        |
# | 3  | A   | B | 'A B'        |
# | 4  | D   | C | 'D C'        |
# 
# 
# 

# In[19]:


# Create `pickup_dropoff` column
### YOUR CODE HERE ###
# ==============================================================================
# Create 'pickup_dropoff' column by joining location IDs as strings
# ==============================================================================
df0['pickup_dropoff'] = df0['PULocationID'].astype(str) + ' ' + df0['DOLocationID'].astype(str)


# Now, use a `groupby()` statement to group each row by the new `pickup_dropoff` column, compute the mean, and capture the values only in the `trip_distance` column. Assign the results to a variable named `grouped`.

# In[20]:


### YOUR CODE HERE ###
# ==============================================================================
# Group by the new route column and compute the mean for trip_distance
# ==============================================================================
grouped = df0.groupby('pickup_dropoff')['trip_distance'].mean()


# `grouped` is an object of the `DataFrame` class.
# 
# 1. Convert it to a dictionary using the [`to_dict()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html) method. Assign the results to a variable called `grouped_dict`. This will result in a dictionary with a key of `trip_distance` whose values are another dictionary. The inner dictionary's keys are pickup/dropoff points and its values are mean distances. This is the information you want.
# 
# ```
# Example:
# grouped_dict = {'trip_distance': {'A B': 1.25, 'C D': 2, 'D C': 3}
# ```
# 
# 2. Reassign the `grouped_dict` dictionary so it contains only the inner dictionary. In other words, get rid of `trip_distance` as a key, so:
# 
# ```
# Example:
# grouped_dict = {'A B': 1.25, 'C D': 2, 'D C': 3}
#  ```

# In[21]:


# 1. Convert `grouped` to a dictionary
### YOUR CODE HERE ###
# ==============================================================================
# Task 3a (Continued) - Convert grouped to dictionary and extract inner dict
# Convertir grouped a diccionario y extraer el diccionario interno
# ==============================================================================

# 1. Convert `grouped` to a dictionary
# Convertir el objeto DataFrame/Series `grouped` a un diccionario usando .to_dict()
grouped_dict = grouped.to_dict()

# 2. Reassign to only contain the inner dictionary
# Reasignar para eliminar la llave externa 'trip_distance' y quedarse solo con el diccionario interno
# Nota: Si 'grouped' se calculó como un DataFrame, tendrá la estructura anidada; 
# si era una Serie, se accede directamente de forma segura mediante la llave del nombre de la columna.
if 'trip_distance' in grouped_dict:
    grouped_dict = grouped_dict['trip_distance']

# Confirm that it worked by printing a few items from the dictionary
# Confirmar que funcionó imprimiendo los primeros elementos del diccionario corregido
print("=== GROUPED_DICT CONVERSION VERIFICATION ===")
print({k: grouped_dict[k] for k in list(grouped_dict.keys())[:5]})


# 1. Create a `mean_distance` column that is a copy of the `pickup_dropoff` helper column.
# 
# 2. Use the [`map()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html#pandas-series-map) method on the `mean_distance` series. Pass `grouped_dict` as its argument. Reassign the result back to the `mean_distance` series.
# </br></br>
# When you pass a dictionary to the `Series.map()` method, it will replace the data in the series where that data matches the dictionary's keys. The values that get imputed are the values of the dictionary.
# 
# ```
# Example:
# df['mean_distance']
# ```
# 
# |mean_distance |
# |  :-:         |
# | 'A B'        |
# | 'C D'        |
# | 'A B'        |
# | 'D C'        |
# | 'E F'        |
# 
# ```
# grouped_dict = {'A B': 1.25, 'C D': 2, 'D C': 3}
# df['mean_distance`] = df['mean_distance'].map(grouped_dict)
# df['mean_distance']
# ```
# 
# |mean_distance |
# |  :-:         |
# | 1.25         |
# | 2            |
# | 1.25         |
# | 3            |
# | NaN          |
# 
# When used this way, the `map()` `Series` method is very similar to `replace()`, however, note that `map()` will impute `NaN` for any values in the series that do not have a corresponding key in the mapping dictionary, so be careful.
# # --------------------------------------------------------------------------------------------------------
# 
# 1. El Dilema del Modelo (Data Leakage / Fuga de Datos)
# El texto empieza con una advertencia crucial de Ciencia de Datos:
# 
# "When deployed, the model will not know the duration of a trip until after the trip occurs, so you cannot train a model that uses this feature."
# 
# Explicación: Si tú estás en la calle y pides un taxi (o un Uber), la aplicación necesita estimar y cobrarte la tarifa antes de que el auto se mueva. En ese momento, la aplicación sabe dónde estás (pickup) y a dónde vas (dropoff), pero no sabe cuántos minutos exactos va a demorar el viaje (la duración real solo se sabe al bajarse).
# Si entrenaras tu modelo usando la columna duration que calculamos en la tarea anterior, tu modelo sería excelente en el laboratorio, pero inútil en la vida real porque no tendría ese dato al momento de cotizar. Usar datos del futuro para predecir el presente se conoce como Data Leakage.
# 
# 2. La Solución: Generalizar con Estadísticas HistóricasComo no podemos usar la duración real, el texto te dice que uses el pasado para predecir el futuro:"However, you can use the statistics of trips you do know to generalize about ones you do not know."La solución consiste en calcular la distancia promedio (mean_distance) que suele haber entre un punto A y un punto B basándote en todo tu histórico de datos.Si el modelo ve que vas desde el punto 12 al punto 151, en lugar de mirar el viaje individual actual, mirará el promedio histórico de todos los taxistas que han hecho esa misma ruta exacta.3. La Regla de Negocio: El sentido del viaje importaEl texto te advierte algo muy lógico sobre las ciudades:"Notice that C -> D is not the same as D -> C."Explicación: Ir desde el punto C al punto D no es lo mismo que ir al revés (de D a C). En Nueva York (y en cualquier ciudad), las calles tienen sentidos unidireccionales (calles de una sola vía), autopistas con retornos lejanos o congestión en una sola pista. Por lo tanto, el promedio de la ruta C ---> D debe calcularse de manera completamente independiente a la ruta D ---> C.
# 
# 4. La Estrategia de Código: Crear la columna llave (pickup_dropoff)
# Para poder agrupar los viajes por "rutas únicas", necesitas que Python entienda que la combinación de un punto de origen y uno de destino forman una sola entidad. El texto te propone un truco:
# 
# "One way to do this is to convert the pickup and dropoff location IDs to strings and join them, separated by a space."
# 
# Si tienes el ID de recogida PULocationID y el ID de destino DOLocationID, los vas a transformar a texto (strings) y los vas a concatenar con un espacio en medio.
# 
# ¿Por qué exige un espacio de separación?
# El texto te da un ejemplo brillante:
# 
# Si vas del punto 12 al 151 y solo los pegas, el resultado sería el texto "12151".
# 
# Si vas del punto 121 al 51 y solo los pegas, el resultado también sería el texto "12151".
# 
# ¡Para Python las dos rutas serían idénticas cuando en la realidad son lugares totalmente distintos de la ciudad! Al ponerle un espacio, las rutas quedan codificadas de forma única y segura como "12 151" y "121 51".

# In[22]:


# 1. Create a mean_distance column that is a copy of the pickup_dropoff helper column
### YOUR CODE HERE ###
df0['pickup_dropoff'] = df0['PULocationID'].astype(str) + ' ' + df0['DOLocationID'].astype(str)

# Create the grouped dictionary with the mean distance per unique route
grouped_dict = df0.groupby('pickup_dropoff')['trip_distance'].mean().to_dict()

# 1. Create a mean_distance column that is a copy of the pickup_dropoff helper column
# Crear la columna mean_distance como una copia de la columna auxiliar pickup_dropoff
df0['mean_distance'] = df0['pickup_dropoff'].copy()

# 2. Map `grouped_dict` to the `mean_distance` column
### YOUR CODE HERE ###
df0['mean_distance'] = df0['mean_distance'].map(grouped_dict)

# Confirm that it worked
### YOUR CODE HERE ###

# Confirmar que la operación funcionó correctamente revisando las nuevas columnas
print("=== FEATURE ENGINEERING VERIFICATION ===")
print(df0[['PULocationID', 'DOLocationID', 'pickup_dropoff', 'trip_distance', 'mean_distance']].head())

print("\n=== MISSING VALUES CHECK IN NEW COLUMN ===")
print(f"NaN values in mean_distance: {df0['mean_distance'].isna().sum()}")


# #### Create `mean_duration` column
# 
# Repeat the process used to create the `mean_distance` column to create a `mean_duration` column.

# In[23]:


# ==============================================================================
# Task 3b. Feature Engineering - Create mean_duration column
# Ingeniería de Características - Crear columna mean_duration
# ==============================================================================

# Create a dictionary where keys are unique pickup_dropoffs and values are
# mean trip duration for all trips with those pickup_dropoff combos
grouped_dict_dur = df0.groupby('pickup_dropoff')['duration'].mean().to_dict()

# Create a mean_duration column that is a copy of the pickup_dropoff helper column
df0['mean_duration'] = df0['pickup_dropoff'].copy()

# Map `grouped_dict_dur` to the `mean_duration` column
df0['mean_duration'] = df0['mean_duration'].map(grouped_dict_dur)

# Confirm that it worked
print("=== DURATION FEATURE ENGINEERING VERIFICATION ===")
print(df0[['pickup_dropoff', 'duration', 'mean_duration']].head())

print("\n=== MISSING VALUES CHECK IN MEAN_DURATION ===")
print(f"NaN values in mean_duration: {df0['mean_duration'].isna().sum()}")


# Por ejemplo, para el viaje en la fila 0 de la zona 100 a la 231, la duración real de ese viaje específico fue de 14.06 minutos, pero tu nueva columna predictor histórica le dice al modelo: "Oye, la media real de tiempo para esta ruta en particular suele ser de 22.84 minutos". Esto le dará una estabilidad tremenda a las predicciones en producción.

# #### Create `day` and `month` columns
# 
# Create two new columns, `day` (name of day) and `month` (name of month) by extracting the relevant information from the `tpep_pickup_datetime` column.

# In[24]:


# ==============================================================================
# Task 3c. Feature Engineering - Create day and month columns
# Ingeniería de Características - Crear columnas de día y mes
# ==============================================================================

# 1. Create 'day' col (Extract the name of the day, e.g., 'Monday')
# Crear la columna 'day' extrayendo el nombre del día
df0['day'] = df0['tpep_pickup_datetime'].dt.day_name().str.lower()

# 2. Create 'month' col (Extract the name of the month, e.g., 'January')
# Crear la columna 'month' extrayendo el nombre del mes
df0['month'] = df0['tpep_pickup_datetime'].dt.month_name().str.lower()

# Confirm that it worked by inspecting the first few rows
# Confirmar que funcionó correctamente inspeccionando las nuevas columnas temporales
print("=== CALENDAR FEATURE ENGINEERING VERIFICATION ===")
print(df0[['tpep_pickup_datetime', 'day', 'month']].head())


# #### Create `rush_hour` column
# 
# Define rush hour as:
# * Any weekday (not Saturday or Sunday) AND
# * Either from 06:00&ndash;10:00 or from 16:00&ndash;20:00
# 
# Create a binary `rush_hour` column that contains a 1 if the ride was during rush hour and a 0 if it was not.

# In[25]:


# Create 'rush_hour' col
### YOUR CODE HERE ###
# 1. Create 'rush_hour' col based on numerical hour ranges
# Extract the hour component to evaluate the rule
pickup_hour = df0['tpep_pickup_datetime'].dt.hour

# Set 1 if hour is between 6-10 (inclusive of 6, up to 9) OR 16-20 (inclusive of 16, up to 19)
# Note: TLC hours are discrete integers (e.g., 6:00 to 9:59 matches hours 6, 7, 8, 9)
df0['rush_hour'] = ((pickup_hour >= 6) & (pickup_hour < 10)) | ((pickup_hour >= 16) & (pickup_hour < 20))

# Convert boolean series (True/False) to binary integers (1/0)
df0['rush_hour'] = df0['rush_hour'].astype(int)

# If day is Saturday or Sunday, impute 0 in `rush_hour` column
### YOUR CODE HERE ###
# 2. If day is Saturday or Sunday, impute 0 in `rush_hour` column
# Si el día es sábado o domingo, forzar un 0 en la columna `rush_hour`
weekend_mask = df0['day'].isin(['saturday', 'sunday'])
df0.loc[weekend_mask, 'rush_hour'] = 0


# In[26]:


### YOUR CODE HERE ###
# 3. Confirm that it worked by inspecting cross-tabulation statistics
# Confirmar la distribución cruzada de la nueva variable explicativa
print("=== RUSH_HOUR VALUE COUNTS ===")
print(df0['rush_hour'].value_counts())

print("\n=== VERIFICATION: RUSH HOUR DISTRIBUTIONS BY DAY ===")
print(pd.crosstab(df0['day'], df0['rush_hour']))


# In[27]:


# ==============================================================================
# 1. Define the rush_hourizer function (Run this cell first!)
# Definir la función rush_hourizer (¡Ejecuta esta celda primero!)
# ==============================================================================

def rush_hourizer(row):
    '''
    Evaluates a row to determine if a ride occurred during weekday rush hours.
    Returns 1 if it was a rush hour trip, 0 otherwise.
    '''
    day = row['day']
    hour = row['tpep_pickup_datetime'].hour
    
    # 1. Condition check: Must be a weekday (not Saturday or Sunday)
    if day not in ['saturday', 'sunday']:
        # 2. Condition check: Morning peak (6-10) or Evening peak (16-20)
        if (6 <= hour < 10) or (16 <= hour < 20):
            return 1
            
    # 3. Safe fallback: If any of the above conditions fail, return 0 (Prevents NaNs)
    return 0

# Apply the `rush_hourizer()` function to the new column
### YOUR CODE HERE ###
# Apply the rush_hourizer function to each row of the dataframe
# Aplicar la función rush_hourizer a cada fila del dataframe (axis=1)
df0['rush_hour'] = df0.apply(rush_hourizer, axis=1)

# Confirm that it worked by inspecting the first few rows
# Confirmar que funcionó correctamente inspeccionando las primeras filas
print("=== RUSH_HOUR FUNCTION APPLICATION VERIFICATION ===")
print(df0[['tpep_pickup_datetime', 'day', 'rush_hour']].head(10))


# ### Task 4. Scatter plot
# 
# Create a scatterplot to visualize the relationship between `mean_duration` and `fare_amount`.

# In[28]:


# Create a scatterplot to visualize the relationship between variables of interest
### YOUR CODE HERE ###
### YOUR CODE HERE ###
# ==============================================================================
# Task 4. Scatter plot
# Crear un gráfico de dispersión para visualizar la relación de las variables
# ==============================================================================

# Set the figure size / Configurar el tamaño del gráfico
plt.figure(figsize=(8, 6))

# Create the scatterplot using seaborn
# Crear el scatterplot vinculando los datos limpios de df0
sns.scatterplot(
    data=df0, 
    x='mean_duration', 
    y='fare_amount', 
    alpha=0.5  # Controls transparency to better visualize dense data points
)

# Set labels and title
# Establecer títulos y etiquetas de los ejes
plt.title('Relationship between Mean Duration and Fare Amount', fontsize=14)
plt.xlabel('Mean Duration (minutes)', fontsize=12)
plt.ylabel('Fare Amount ($)', fontsize=12)

# Display the plot
plt.show()


# In[29]:


# Contar cuántos viajes tienen exactamente la tarifa plana de JFK ($52)
jfk_flat_fares = df0[df0['fare_amount'] == 52.00].shape[0]

# Contar cuántos viajes quedaron en el límite imputado ($62.50)
imputed_fares = df0[df0['fare_amount'] == 62.50].shape[0]

print(f"Viajes con Tarifa Plana JFK ($52.00): {jfk_flat_fares}")
print(f"Viajes con Tarifa Imputada ($62.50): {imputed_fares}")


# The `mean_duration` variable correlates with the target variable. But what are the horizontal lines around fare amounts of 52 dollars and 63 dollars? What are the values and how many are there?
# 
# You know what one of the lines represents. 62 dollars and 50 cents is the maximum that was imputed for outliers, so all former outliers will now have fare amounts of \$62.50. What is the other line?
# 
# English (For the notebook markdown cell):
# The horizontal line at $62.50 represents the maximum upper threshold calculated via the Interquartile Range (Q3+6×IQR) during the data cleansing phase. There are exactly 84 rows in the dataset that were extreme outliers (such as the initial $999.99 maximum) and have been capped at this value, creating a concentration of points along the axis.
# 
# The other distinct horizontal line at $52.00 represents the regulatory JFK Airport Flat Fare. In New York City, any taxi ride between Manhattan and JFK International Airport is charged a flat fee of $52.00, regardless of the trip's actual duration or traffic conditions. There are exactly 514 rows that capture these airport trips. Because the fare remains constant while the mean_duration varies, it maps as a solid horizontal line across the scatter plot.
# 
# Traducción / Explicación en Español:
# La línea horizontal en $62.50 representa el umbral superior máximo calculado a través del Rango Intercuartílico (Q3+6×IQR) durante la fase de limpieza de datos. Hay exactamente 84 filas en el conjunto de datos que eran valores atípicos extremos (como el máximo inicial de $999.99) y que fueron truncados en este valor, creando una concentración de puntos a lo largo del eje.
# 
# La otra línea horizontal evidente en $52.00 representa la Tarifa Plana del Aeropuerto JFK. En la ciudad de Nueva York, cualquier viaje en taxi entre Manhattan y el Aeropuerto Internacional JFK tiene una tarifa fija regulada de $52.00, independientemente de la duración real del viaje o de las condiciones del tráfico. Hay exactamente 514 filas que capturan estos viajes al aeropuerto. Debido a que la tarifa permanece constante mientras la variable mean_duration varía, se dibuja como una línea horizontal perfecta en el gráfico de dispersión.
# 
# Check the value of the rides in the second horizontal line in the scatter plot.

# In[30]:


# ==============================================================================
# Task 4 (Continued): Inspect the second horizontal line ($52.00 rides)
# Inspeccionar los viajes de la segunda línea horizontal ($52.00)
# ==============================================================================

# Create a subset containing only the rows where fare_amount is exactly 52.0
# Crear un subconjunto con los viajes de tarifa igual a $52.00
jfk_lines = df0[df0['fare_amount'] == 52.0]

# Call .describe() on this subset to analyze its characteristics
# Llamar a .describe() para analizar las variables de interés en este grupo
print("=== STATISTICAL SUMMARY FOR $52.00 FLIGHTS/TRIPS ===")
print(jfk_lines[['trip_distance', 'duration', 'mean_distance', 'mean_duration']].describe())


# Examine the first 30 of these trips.

# In[31]:


# Set pandas to display all columns
### YOUR CODE HERE ###
# ==============================================================================
# Task 4 (Continued): Display all columns for the first 30 JFK flat-fare trips
# Configurar Pandas para mostrar todas las columnas e inspeccionar los primeros 30 viajes
# ==============================================================================

# Set pandas to display all columns in the notebook output
# Configurar pandas para que muestre todas las columnas de forma explícita
pd.set_option('display.max_columns', None)

# Examine the first 30 of these trips from the jfk_lines subset
# Examinar los primeros 30 registros del subconjunto jfk_lines
jfk_lines.head(30)


# **Question:** What do you notice about the first 30 trips?
# English:
# By examining the first 30 rows of this subset, several critical operational and data patterns emerge:
# 
# Consistent Rate Code: All the rides explicitly feature a RatecodeID of 2. In the New York City TLC framework, this specific code represents the regulatory JFK Airport Flat Fare, confirming our business logic hypothesis.
# 
# Geographic Association: Nearly all entries contain the location ID 132 (which corresponds to JFK Airport) either as the PULocationID (pickup) or the DOLocationID (dropoff), paired with major Manhattan destination zones (like 163, 234, or 186).
# 
# Imputed Continuous Features: The data shows that our feature engineering worked perfectly. For instance, even when an individual trip record has an anomaly (such as row 2, where duration is only 0.96 minutes and trip_distance is 0.23 miles due to a likely immediate cancellation), the fare remains locked at $52.00 while its engineered mean_distance and mean_duration maintain stable historical route values.
# 
# Traducción / Explicación en Español:
# Al examinar las primeras 30 filas de este subconjunto, emergen varios patrones operativos y de datos cruciales:
# 
# Código de Tarifa Consistente: Todos los viajes presentan explícitamente un RatecodeID de 2. En el marco de la TLC de la ciudad de Nueva York, este código específico representa la Tarifa Plana Regulada del Aeropuerto JFK, confirmando nuestra hipótesis de reglas de negocio.
# 
# Asociación Geográfica: Casi todas las entradas contienen el ID de ubicación 132 (que corresponde al Aeropuerto JFK) ya sea como el PULocationID (recogida) o el DOLocationID (destino), emparejado con zonas de destino importantes de Manhattan (como 163, 234 o 186).
# 
# Características Continuas Imputadas: Los datos demuestran que nuestra ingeniería de características funcionó a la perfección. Por ejemplo, incluso cuando un registro de viaje individual tiene una anomalía (como la fila 2, donde la duration es de solo 0.96 minutos y la trip_distance es de 0.23 millas debido a una probable cancelación inmediata), la tarifa permanece fija en $52.00 mientras que sus variables de ingeniería mean_distance y mean_duration mantienen valores de ruta históricos estables.

# ### Task 5. Isolate modeling variables
# 
# Drop features that are redundant, irrelevant, or that will not be available in a deployed environment.

# In[32]:


### YOUR CODE HERE ###
# ==============================================================================
# Task 5. Isolate modeling variables
# Aislar las variables de modelado seleccionadas
# ==============================================================================

# 1. Create a list of columns to keep for the modeling process
# Crear una lista con las columnas esenciales que usará el modelo
columns_to_keep = ['fare_amount', 'mean_distance', 'mean_duration', 'rush_hour']

# 2. Isolate these variables into a new dataframe called df1
# Aislar estas variables en un nuevo dataframe llamado df1
df1 = df0[columns_to_keep].copy()



# In[33]:


### YOUR CODE HERE ###
# 3. Verify the final structure and columns of df1
# Verificar la estructura final y las columnas del dataframe aislado
print("=== ISOLATED MODELING VARIABLES ===")
print(df1.info())

print("\n=== FIRST ROWS OF THE MODELING DATASET ===")
print(df1.head())


# ### Task 6. Pair plot
# 
# Create a pairplot to visualize pairwise relationships between `fare_amount`, `mean_duration`, and `mean_distance`.

# In[34]:


# Create a pairplot to visualize pairwise relationships between variables in the data
### YOUR CODE HERE ###
# ==============================================================================
# Task 6. Pair plot
# Crear un gráfico de pares para visualizar relaciones cruzadas continuas
# ==============================================================================

# Create the pairplot for the variables of interest in df1
# Nota: Excluimos 'rush_hour' por ser una variable categórica binaria (1/0)
sns.pairplot(
    data=df1[['fare_amount', 'mean_duration', 'mean_distance']],
    plot_kws={'alpha': 0.5, 'size': 1} # Optional: Controls point transparency and size for dense data
)

# Display the plot explicitly
plt.show()


# EN:
# 
# Based on the pairplot visualization, the following key insights can be observed:
# 
# 1. Linear Relationships with the Target: Both mean_duration and mean_distance exhibit a strong, positive linear correlation with fare_amount. As expected, historical routes with longer average distances and times naturally command higher fares, validating their inclusion as primary predictors for the multiple linear regression model.
# 
# 2. High Multicollinearity: There is an exceptionally tight, almost linear relationship between mean_duration and mean_distance (visible in the scatter plots off the diagonal). This indicates that the two independent variables are highly correlated with each other. In a Multiple Linear Regression model, this severe multicollinearity can inflate the variance of coefficient estimates, making it harder to isolate the individual impact of time versus distance.
# 
# 3. Data Distributions: The histograms along the diagonal show that all three variables are heavily right-skewed. Most taxi trips are short in both distance and duration, with lower fares, while a smaller number of long-distance trips form a long tail to the right. Additionally, the flat lines at $52.00 in the fare_amount plots clearly mark the JFK airport fixed-rate trips.
# 
# ES:
# 
# A partir de la visualización del pairplot, se pueden observar las siguientes conclusiones clave:
# 
# 1. Relaciones Lineales con la Variable Objetivo: Tanto mean_duration como mean_distance exhiben una correlación lineal fuerte y positiva con fare_amount. Como es de esperarse, las rutas históricas con mayores promedios de distancia y tiempo conllevan tarifas más altas, validando su inclusión como predictores principales.
# 
# 2. Alta Multicolinealidad: Existe una relación lineal extremadamente estrecha entre mean_duration y mean_distance (visible en los gráficos de dispersión fuera de la diagonal). Esto indica que las dos variables independientes están altamente correlacionadas entre sí. En una Regresión Lineal Múltiple, esta multicolinealidad severa puede inflar la varianza de los coeficientes, dificultando aislar el impacto individual del tiempo versus la distancia.
# 
# 3. Distribución de los Datos: Los histogramas de la diagonal principal muestran que las tres variables están fuertemente sesgadas a la derecha (right-skewed). La mayoría de los viajes en taxi son cortos en distancia y duración, concentrándose en tarifas bajas, mientras que un número menor de viajes largos forma una cola extendida hacia la derecha. Además, las líneas planas en $52.00 vuelven a marcar con claridad los viajes con tarifa fija al aeropuerto JFK.
# 
# These variables all show linear correlation with each other. Investigate this further.

# ### Task 7. Identify correlations

# Next, code a correlation matrix to help determine most correlated variables.

# In[35]:


# Correlation matrix to help determine most correlated variables
### YOUR CODE HERE ###

# ==============================================================================
# Task 7. Identify correlations
# Calcular la matriz de correlación de Pearson y graficar su mapa de calor
# ==============================================================================

# 1. Compute the correlation matrix for the modeling dataframe
# Calcular los coeficientes de correlación entre todas las variables de df1
correlation_matrix = df1.corr()

# 2. Set up the matplotlib figure size
# Configurar el tamaño del gráfico
plt.figure(figsize=(6, 4))

# 3. Create a seaborn heatmap to visualize the correlations
# Crear el mapa de calor con anotaciones numéricas y escala de color
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='Reds', 
    fmt=".2f",
    linewidths=0.5
)

# Add title and adjust layout
# Agregar título al gráfico
plt.title('Correlation Matrix of Modeling Variables', fontsize=12)
plt.tight_layout()

# 4. Display the numerical values in the console output
# Imprimir los coeficientes numéricos exactos en la consola
print("=== NUMERICAL CORRELATION COEFFICIENTS ===")
print(correlation_matrix)

# Display the plot
# Mostrar el mapa de calor
plt.show()


# Visualize a correlation heatmap of the data.

# In[36]:


# Create correlation heatmap
### YOUR CODE HERE ###

# ==============================================================================
# Task 7 (Continued). Visualize a correlation heatmap of the data
# Crear el mapa de calor de correlación
# ==============================================================================

# Set up the matplotlib figure size
# Configurar el tamaño del gráfico
plt.figure(figsize=(6, 4))

# Create the correlation heatmap using seaborn
# Crear el mapa de calor vinculando la matriz calculada en la celda anterior
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='Reds', 
    fmt=".2f",
    linewidths=0.5
)

# Add a professional title and adjust layout
# Agregar título y ajustar el diseño para que no se corten las etiquetas
plt.title('Correlation Heatmap of Modeling Variables', fontsize=12)
plt.tight_layout()

# Display the plot
plt.show()


# **Question:** Which variable(s) are correlated with the target variable of `fare_amount`? 
# 
# English:
# Based on the correlation matrix, the variables that are strongly correlated with the target variable (fare_amount) are:
# 
# mean_distance: It has the highest positive correlation with a coefficient of 0.91. This indicates an exceptionally strong linear relationship, meaning that as the historical average distance of a route increases, the fare amount increases significantly.
# 
# mean_duration: It follows very closely with a strong positive correlation coefficient of 0.86. This shows that the historical average time spent on a route is also a massive driver of the final fare.
# 
# Conversely, rush_hour shows a near-zero correlation of -0.02 with fare_amount. While this suggests no simple linear relationship on its own, its true impact as a categorical control variable will be properly adjusted and revealed once we fit the multiple linear regression model.
# 
# Traducción / Explicación en Español:
# Basado en la matriz de correlación, las variables que están fuertemente correlacionadas con la variable objetivo (fare_amount) son:
# 
# mean_distance: Presenta la correlación positiva más alta con un coeficiente de 0.91. Esto indica una relación lineal excepcionalmente fuerte, lo que significa que a medida que aumenta la distancia promedio histórica de una ruta, el monto de la tarifa aumenta significativamente.
# 
# mean_duration: Le sigue muy de cerca con un fuerte coeficiente de correlación positiva de 0.86. Esto demuestra que el tiempo promedio histórico transcurrido en una ruta también es un predictor masivo para estimar la tarifa.
# 
# Por el contrario, rush_hour muestra una correlación cercana a cero de -0.02 con fare_amount. Aunque esto sugiere que no tiene una relación lineal simple por sí sola, su verdadero impacto como variable categórica de control se ajustará y revelará adecuadamente una vez que ejecutemos el modelo de regresión lineal múltiple.
# 
# Try modeling with both variables even though they are correlated.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## PACE: **Construct**
# 
# After analysis and deriving variables with close relationships, it is time to begin constructing the model. Consider the questions in your PACE Strategy Document to reflect on the Construct stage.
# 

# ### Task 8a. Split data into outcome variable and features

# In[37]:


### YOUR CODE HERE ###
# 1. Define the outcome variable (y)
# Aislar la variable objetivo dependiente: fare_amount
y = df1['fare_amount']

# 2. Define the features/predictors (X)
# Aislar las variables independientes explicativas eliminando el target
X = df1.drop(columns='fare_amount')

# 3. Verify the shapes and columns to ensure alignment
# Verificar las dimensiones de las matrices para confirmar la consistencia
print("=== FEATURE MATRIX (X) SHAPE ===")
print(X.shape)
print("\n=== FEATURE MATRIX COLUMNS ===")
print(X.columns.tolist())

print("\n=== TARGET VECTOR (y) SHAPE ===")
print(y.shape)


# Set your X and y variables. X represents the features and y represents the outcome (target) variable.

# In[38]:


# Remove the target column from the features
# X = df2.drop(columns='fare_amount')
### YOUR CODE HERE ###
# Remover la columna objetivo de las características predictoras
X = df1.drop(columns='fare_amount')

# Set y variable
### YOUR CODE HERE ###
# Set y variable
# Establecer la variable objetivo dependiente
y = df1['fare_amount']

# Display first few rows
### YOUR CODE HERE ###
# Desplegar las primeras filas de las matrices para confirmar la consistencia
print("=== FIRST ROWS OF FEATURE MATRIX (X) ===")
print(X.head())

print("\n=== FIRST ROWS OF TARGET VARIABLE (y) ===")
print(y.head())


# ### Task 8b. Pre-process data
# 

# Dummy encode categorical variables

# In[39]:


# Convert VendorID to string
### YOUR CODE HERE ###

# Convertir VendorID a tipo texto en df0 para que Pandas lo reconozca como categórico
df0['VendorID'] = df0['VendorID'].astype(str)

# Get dummies
### YOUR CODE HERE ###

# Generar las variables dummy para VendorID usando drop_first=True para evitar la trampa de la multicolinealidad
# Nota: Pasamos las variables que ya teníamos en X junto con la nueva columna dummyficada
X = pd.get_dummies(X.join(df0['VendorID']), drop_first=True)

# Display first few rows to verify the new binary columns
# Mostrar las primeras filas de la matriz X actualizada
print("=== FEATURE MATRIX (X) WITH DUMMY VARIABLES ===")
print(X.head())


# ### Split data into training and test sets

# Create training and testing sets. The test set should contain 20% of the total samples. Set `random_state=0`.

# In[40]:


# Create training and testing sets
#### YOUR CODE HERE ####
# Import train_test_split from sklearn
from sklearn.model_selection import train_test_split
# Create training and testing sets
# Dividir X e y fijando el tamaño de prueba en 0.20 y una semilla de aleatoriedad en 0
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20, 
    random_state=0
)


# ### Standardize the data
# 
# Use `StandardScaler()`, `fit()`, and `transform()` to standardize the `X_train` variables. Assign the results to a variable called `X_train_scaled`.

# In[41]:


# Standardize the X variables
### YOUR CODE HERE ###
# Import StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler

# 1. Instantiate the scaler
# Instanciar el escalador estándar
scaler = StandardScaler()

# 2. Fit and transform the training data
# Ajustar el escalador con los datos de entrenamiento (calcula media y std) y transformarlos
X_train_scaled = scaler.fit_transform(X_train)

# 3. Transform the test data using the SAME scaler
# Nota: En los entornos productivos reales, X_test solo se transforma (.transform()), 
# NUNCA se ajusta (.fit()), para evitar la filtración de datos (data leakage).
X_test_scaled = scaler.transform(X_test)

# Verify the result by printing the first row scaled
# Verificar que las variables ahora están en la misma escala (valores alrededor de 0)
print("=== FIRST ROW OF X_TRAIN SCALED ===")
print(X_train_scaled[0])


# ### Fit the model
# 
# Instantiate your model and fit it to the training data.

# In[42]:


# Fit your model to the training data
### YOUR CODE HERE ###
# ==============================================================================
# Task 9. Fit the model
# Instanciar el modelo de Regresión Lineal Múltiple y entrenarlo
# ==============================================================================

# Import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression

# 1. Instantiate the model
# Instanciar el estimador de regresión lineal
model = LinearRegression()

# 2. Fit the model to the training data
# Entrenar el modelo utilizando las características escaladas y el target de entrenamiento
model.fit(X_train_scaled, y_train)

# Print confirmation to ensure the block executed completely
# Confirmación de ejecución exitosa
print("=== MODEL FITTING COMPLETE ===")
print(f"Model instance object: {model}")


# ### Task 8c. Evaluate model

# ### Train data
# 
# Evaluate your model performance by calculating the residual sum of squares and the explained variance score (R^2). Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# In[43]:


# Evaluate the model performance on the training data
### YOUR CODE HERE ###

# Import required metrics from sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Generate predictions on the training data
# Obtener las predicciones del modelo usando la matriz escalada de entrenamiento
y_pred_train = model.predict(X_train_scaled)

# 2. Calculate the metrics
# Coeficiente de Determinación o Varianza Explicada (R^2)
r2_train = r2_score(y_train, y_pred_train)

# Mean Absolute Error (MAE) - Error Absoluto Medio
mae_train = mean_absolute_error(y_train, y_pred_train)

# Mean Squared Error (MSE) - Error Cuadrático Medio
mse_train = mean_squared_error(y_train, y_pred_train)

# Root Mean Squared Error (RMSE) - Raíz del Error Cuadrático Medio
rmse_train = np.sqrt(mse_train)

# Residual Sum of Squares (RSS) - Suma de Cuadrados de los Residuos (calculado manualmente)
rss_train = np.sum((y_train - y_pred_train) ** 2)

# 3. Print the evaluation results professionally
# Mostrar los resultados formateados en la consola
print("=== TRAINING DATA EVALUATION METRICS ===")
print(f"R^2 (Explained Variance Score):  {r2_train:.4f}")
print(f"MAE (Mean Absolute Error):       ${mae_train:.4f}")
print(f"MSE (Mean Squared Error):        {mse_train:.4f}")
print(f"RMSE (Root Mean Squared Error):  ${rmse_train:.4f}")
print(f"RSS (Residual Sum of Squares):   {rss_train:.4f}")


# ### Test data
# 
# Calculate the same metrics on the test data. Remember to scale the `X_test` data using the scaler that was fit to the training data. Do not refit the scaler to the testing data, just transform it. Call the results `X_test_scaled`.

# In[44]:


# Scale the X_test data
### YOUR CODE HERE ###
# 1. Scale the X_test data (Transform only, DO NOT refit)
# Aplicar la misma escala calculada en el entrenamiento sobre los datos de prueba
X_test_scaled = scaler.transform(X_test)

# 2. Generate predictions on the test data
# Obtener las predicciones utilizando la matriz de prueba escalada
y_pred_test = model.predict(X_test_scaled)

# 3. Calculate the evaluation metrics for the test dataset
# Coeficiente de Determinación (R^2)
r2_test = r2_score(y_test, y_pred_test)

# Mean Absolute Error (MAE)
mae_test = mean_absolute_error(y_test, y_pred_test)

# Mean Squared Error (MSE)
mse_test = mean_squared_error(y_test, y_pred_test)

# Root Mean Squared Error (RMSE)
rmse_test = np.sqrt(mse_test)

# Residual Sum of Squares (RSS) calculado manualmente
rss_test = np.sum((y_test - y_pred_test) ** 2)

# 4. Print the evaluation results
# Mostrar los resultados del examen del modelo en consola
print("=== TEST DATA EVALUATION METRICS ===")
print(f"R^2 (Explained Variance Score):  {r2_test:.4f}")
print(f"MAE (Mean Absolute Error):       ${mae_test:.4f}")
print(f"MSE (Mean Squared Error):        {mse_test:.4f}")
print(f"RMSE (Root Mean Squared Error):  ${rmse_test:.4f}")
print(f"RSS (Residual Sum of Squares):   {rss_test:.4f}")


# In[45]:


# Evaluate the model performance on the testing data
### YOUR CODE HERE ###
# ==============================================================================
# Task 8c (Continued). Evaluate the model performance on the testing data
# Calcular y desplegar las métricas oficiales para el conjunto de prueba
# ==============================================================================

# 1. Generate predictions on the test data
# Obtener las predicciones utilizando la matriz de prueba escalada (X_test_scaled)
y_pred_test = model.predict(X_test_scaled)

# 2. Calculate the test metrics
# Coeficiente de Determinación (R^2)
r2_test = r2_score(y_test, y_pred_test)

# Mean Absolute Error (MAE)
mae_test = mean_absolute_error(y_test, y_pred_test)

# Mean Squared Error (MSE)
mse_test = mean_squared_error(y_test, y_pred_test)

# Root Mean Squared Error (RMSE)
rmse_test = np.sqrt(mse_test)

# Residual Sum of Squares (RSS)
rss_test = np.sum((y_test - y_pred_test) ** 2)

# 3. Print the evaluation results for grading
# Imprimir los resultados con formato profesional
print("=== TEST DATA EVALUATION METRICS ===")
print(f"R^2 (Explained Variance Score):  {r2_test:.4f}")
print(f"MAE (Mean Absolute Error):       ${mae_test:.4f}")
print(f"MSE (Mean Squared Error):        {mse_test:.4f}")
print(f"RMSE (Root Mean Squared Error):  ${rmse_test:.4f}")
print(f"RSS (Residual Sum of Squares):   {rss_test:.4f}")


# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## PACE: **Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### Task 9a. Results
# 
# Use the code cell below to get `actual`,`predicted`, and `residual` for the testing set, and store them as columns in a `results` dataframe.

# In[46]:


# Create a `results` dataframe
### YOUR CODE HERE ###

# ==============================================================================
# Task 9a. Results
# Crear el dataframe de resultados para inspeccionar las predicciones y los residuos
# ==============================================================================

# 1. Create a results dataframe with actual and predicted values
# Construir el dataframe alineando los valores reales con las predicciones del modelo
results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test
})

# 2. Calculate the residuals
# Calcular el residuo (valor real menos valor predicho) para cada observación
results['residual'] = results['actual'] - results['predicted']

# 3. Display the first few rows of the results dataframe
# Desplegar las primeras filas de la tabla de resultados para verificación
print("=== FIRST ROWS OF THE RESULTS DATAFRAME ===")
print(results.head())


# las filas muestran con precisión cómo está operando la regresión frente a casos individuales:
# 
# En viajes como el 4655, 7378 y 13914, el modelo anduvo sumamente preciso, con residuos de apenas unos centavos (entre −$0.92 y −$1.49).
# 
# En el viaje 18134, sin embargo, el residuo saltó a +$11.57 (el modelo estimó $16.43 pero la tarifa real fue de $28.00). Este tipo de desviaciones más marcadas suelen ocurrir en trayectos que cruzaron peajes, viajes con propinas muy altas incluidas en el campo registrado o tarifas dinámicas por factores externos puntuales que escapan a las medias de la ruta.

# ### Task 9b. Visualize model results

# Create a scatterplot to visualize `actual` vs. `predicted`.

# In[47]:


# Create a scatterplot to visualize `predicted` over `actual`
### YOUR CODE HERE ###
# ==============================================================================
# Task 9b. Visualize actual vs. predicted
# Crear un gráfico de dispersión para contrastar predicciones vs. valores reales
# ==============================================================================

# 1. Set up the matplotlib figure size
# Configurar el tamaño del gráfico
plt.figure(figsize=(6, 6))

# 2. Create the scatterplot using seaborn
# Graficar 'actual' en el eje X y 'predicted' en el eje Y
sns.scatterplot(
    x='actual',
    y='predicted',
    data=results,
    alpha=0.5,  # Controls transparency to handle overplotting of dense data
    s=15        # Sets point size
)

# 3. Add a 45-degree diagonal line of perfect prediction
# Determinar los límites máximos comunes para trazar la línea de referencia ideal (y = x)
max_val = max(results['actual'].max(), results['predicted'].max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

# 4. Add titles, labels, and legend
# Personalizar etiquetas y títulos profesionales
plt.title('Actual Fares vs. Predicted Fares', fontsize=12)
plt.xlabel('Actual Fare Amount ($)', fontsize=10)
plt.ylabel('Predicted Fare Amount ($)', fontsize=10)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Display the plot
plt.show()


# La visualización confirma de forma contundente la excelente salud estadística del modelo. Ver la enorme densidad de puntos azules siguiendo de manera casi perfecta la trayectoria de la línea segmentada roja de Perfect Prediction es el mejor indicador visual del alto R2(0.8725) que alcanzaste.
# 
# Si observas detenidamente el cuadrante inferior izquierdo (tarifas entre $0 y $30), el ajuste es extremadamente preciso. Hacia la derecha se aprecian las acumulaciones verticales de las tarifas fijas (aeropuertos) y unos pocos valores atípicos que se desvían de la diagonal, lo cual es normal por factores externos como peajes o recargos excepcionales.

# Visualize the distribution of the `residuals` using a histogram.

# In[48]:


# Visualize the distribution of the `residuals`
### YOUR CODE HERE ###

# ==============================================================================
# Task 9b (Continued). Visualize the distribution of the residuals
# Crear un histograma con curva de densidad para evaluar la normalidad de los residuos
# ==============================================================================

# 1. Set up the matplotlib figure size
# Configurar el tamaño del gráfico
plt.figure(figsize=(8, 5))

# 2. Create the histogram using seaborn
# Graficar la columna 'residual' del dataframe results
sns.histplot(
    results['residual'], 
    kde=True,          # Adds the kernel density estimate line (curva de densidad smoothed)
    bins=30            # Number of bins for granularity
)

# 3. Add labels, title, and reference line at zero
# Agregar una línea vertical en 0 (donde el error de predicción es perfecto)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Reference')

# Personalizar etiquetas y título profesionales
plt.title('Distribution of Model Residuals', fontsize=12)
plt.xlabel('Residual Value ($)', fontsize=10)
plt.ylabel('Count / Frequency', fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()

# Display the plot
plt.show()

# 4. Calculate and print the mean of residuals to verify it is close to 0
# Validar matemáticamente que el promedio de los residuos tienda a cero
print(f"Mean of residuals: {results['residual'].mean():.4f}")


# Al observar la gráfica, podemos extraer conclusiones clave para la sección de hallazgos de tu reporte:
# 
# Cumplimiento del Supuesto de Normalidad: La distribución principal tiene una forma de campana (gaussiana) muy alta y estilizada, fuertemente concentrada alrededor de la línea segmentada roja (Zero Error Reference). Esto demuestra matemáticamente que el promedio de los errores está prácticamente centrado en cero, cumpliendo de manera sólida con uno de los supuestos más estrictos de la regresión lineal ordinaria.
# 
# Sesgo Positivo (Right-Skewed Tail): Como anticipábamos, hay una "cola" larga y delgada que se extiende hacia el lado derecho del gráfico (valores positivos). Esto nos indica visualmente que existen algunos viajes específicos donde el residuo es grande y positivo (y− 
# y^>0). Es decir, viajes donde la tarifa real fue significativamente mayor que la predicción del modelo (el modelo se quedó corto).

# In[49]:


# Calculate residual mean
### YOUR CODE HERE ###
# ==============================================================================
# Task 9b (Continued). Calculate residual mean
# Calcular analíticamente la media de los residuos para comprobar que tiende a cero
# ==============================================================================

# Calculate residual mean
# Calcular el promedio de la columna de residuos en el dataframe de resultados
residual_mean = results['residual'].mean()

# Print the result
# Desplegar el valor obtenido con alta precisión decimal
print("=== RESIDUAL MEAN ===")
print(f"The mean of the residuals is: {residual_mean:.15f}")


# El resultado es numéricamente muy sólido. Que la media de los residuos sea −0.0375 (prácticamente 3 centavos de dólar) confirma de manera analítica lo que observamos visualmente en el histograma: los errores del modelo están perfectamente balanceados alrededor de cero.
# 
# En términos de negocio, esto significa que el modelo no tiene un sesgo sistemático de sobreestimación o subestimación general; es un estimador insesgado muy confiable para las tarifas estándar de Automatidata.

# Create a scatterplot of `residuals` over `predicted`.

# In[50]:


# Create a scatterplot of `residuals` over `predicted`
### YOUR CODE HERE ###
# ==============================================================================
# Task 9b (Continued). Scatterplot of residuals over predicted
# Crear un gráfico de dispersión de residuos vs. valores predichos
# ==============================================================================

# 1. Set up the matplotlib figure size
# Configurar el tamaño del gráfico
plt.figure(figsize=(8, 5))

# 2. Create the scatterplot using seaborn
# Graficar 'predicted' en el eje X y 'residual' en el eje Y
sns.scatterplot(
    x='predicted',
    y='residual',
    data=results,
    alpha=0.5,  # Manejar la transparencia por la alta densidad de puntos
    s=15        # Tamaño de los puntos
)

# 3. Add a horizontal reference line at zero error
# Agregar una línea horizontal en Y = 0 (donde el modelo acierta exactamente)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')

# 4. Add titles and labels
# Personalizar etiquetas y títulos profesionales para Automatidata
plt.title('Residuals vs. Predicted Fare Amounts', fontsize=12)
plt.xlabel('Predicted Fare Amount ($)', fontsize=10)
plt.ylabel('Residual ($)', fontsize=10)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Display the plot
plt.show()


# Al mirar la distribución de los puntos azules respecto a la línea roja horizontal de Zero Error (y=0), podemos extraer conclusiones críticas para la sección final de el reporte PACE:
# 
# Validación de la Linealidad (Nube centrada): La mayor densidad de puntos (la gran mancha azul a la izquierda) está perfectamente partida a la mitad por la línea segmentada roja. Esto significa que para la gran mayoría de los viajes habituales (viajes de bajo costo), los errores son mínimos y se distribuyen de forma aleatoria hacia arriba y hacia abajo. El supuesto de linealidad se cumple con éxito.
# 
# Presencia de Heterocedasticidad Ligera (Forma de embudo invertido o dispersión variable): Si observas cómo se esparcen los puntos a medida que nos movemos hacia la derecha, notarás que la varianza no es perfectamente constante. En el extremo izquierdo la nube es compacta, mientras que hacia el centro y la derecha los puntos se dispersan más y aparecen valores atípicos (outliers) altos en el eje Y. Esto es un comportamiento clásico de heterocedasticidad. En el negocio de taxis, significa que el modelo es extremadamente preciso prediciendo viajes cortos y estables, pero que el margen de variación de las tarifas reales se vuelve más volátil e impredecible a medida que los trayectos se hacen más largos y caros.
# 
# La Línea Diagonal Descendente (Tarifas Plana del Aeropuerto): Hacia la derecha del gráfico, se aprecia una línea de puntos muy clara que desciende en diagonal cruzando la línea roja. Esa estructura lineal perfecta representa los viajes al Aeropuerto JFK que tienen una tarifa fija (ej. $52). Como el modelo predice un valor aproximado basado en la distancia y duración promedio de esa ruta, pero la tarifa real fue rígidamente fija, la resta matemática (y− 
# y^) genera esa pendiente perfecta en los residuos.

# ### Task 9c. Coefficients
# 
# Use the `coef_` attribute to get the model's coefficients. The coefficients are output in the order of the features that were used to train the model. Which feature had the greatest effect on trip fare?

# In[51]:


# Output the model's coefficients
# ==============================================================================
# Task 9c. Coefficients
# Extraer e identificar los coeficientes de las variables predictoras
# ==============================================================================

# 1. Create a dataframe to align features with their corresponding coefficients
# Alinear los nombres de las columnas de X con los coeficientes del modelo
coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

# 2. Sort the dataframe by the absolute value of the coefficients to see the magnitude of effect
# Ordenar por el valor absoluto para identificar qué variable altera más la tarifa
coefficients_df['Abs_Coefficient'] = coefficients_df['Coefficient'].abs()
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])

# 3. Output the model's intercept and coefficients
# Desplegar el intercepto y la tabla de coeficientes en la consola
print(f"Model Intercept (Tarifa Base): {model.intercept_:.4f}")
print("\n=== MODEL COEFFICIENTS (ORDERED BY EFFECT MAGNITUDE) ===")
print(coefficients_df.to_string(index=False))


# What do these coefficients mean? How should they be interpreted?

# Intercept (Intercepto) = 12.8917
# Español: Debido a que estandarizamos las características con StandardScaler(), un valor de cero en la matriz representa el promedio de esa variable. Por lo tanto, el intercepto de $12.89 representa la tarifa estimada para un viaje que tiene una distancia promedio, una duración promedio, ocurre en un horario de tráfico promedio y cuenta con una distribución promedio de proveedores.
# 
# English: Because we standardized the features using StandardScaler(), a value of zero represents the mean of that feature. Therefore, the intercept of $12.89 represents the estimated fare for a trip that has an average distance, an average duration, occurs during an average traffic hour, and has an average distribution of vendors.
# 
# mean_distance = 7.477175
# Español: Esta es la variable con mayor impacto en el modelo por un amplio margen. Al ser un coeficiente estandarizado (β), significa que por cada incremento de una desviación estándar en la distancia promedio de la ruta, la tarifa del viaje aumenta en $7.48 dólares, manteniendo las demás variables constantes.
# 
# English: This is the feature with the greatest effect on the model by a wide margin. As a standardized coefficient (β), it means that for every increase of one standard deviation in the mean distance of the route, the trip fare increases by $7.48 dollars, holding all other variables constant.
# 
# mean_duration = 2.424950
# Español: Ocupa el segundo lugar en importancia. Por cada incremento de una desviación estándar en la duración promedio del viaje, la tarifa aumenta en $2.42 dólares, manteniendo el resto constante. Esto demuestra comercialmente que el factor geográfico (las millas) pesa tres veces más que el tiempo transcurrido en el vehículo (7.47 vs 2.42).
# 
# English: This is the second most important feature. For every increase of one standard deviation in the mean trip duration, the fare increases by $2.42 dollars, holding everything else constant. This commercially demonstrates that the geographic factor (miles) weighs three times more than the time spent in the vehicle (7.47 vs 2.42).

# ### Task 9d. Conclusion
# 
# 1. What are the key takeaways from this notebook?
# 
# 
# 
# 2. What results can be presented from this notebook?
# 
# 

# 1. Key Takeaways (Conclusiones Clave)
# Español:
# 
# Poder Predictivo Excepcional: El modelo de regresión lineal múltiple logró un coeficiente de determinación (R2) de 0.8725 en el conjunto de prueba. Esto significa que el algoritmo explica el 87.25% de la variabilidad de las tarifas utilizando solo cuatro características.
# 
# Ausencia de Sobreajuste (Overfitting): El rendimiento en el conjunto de prueba (R2=0.8725, MAE=$2.12) fue ligeramente superior al del conjunto de entrenamiento (R2=0.8399, MAE=$2.20), lo que demuestra que el modelo generaliza de forma excelente ante datos nuevos.
# 
# Validación de Supuestos: Las pruebas gráficas confirmaron que los errores (residuos) están centrados prácticamente en cero (media≈−0.037), validando la linealidad. La presencia de una cola derecha en el histograma y una apertura en el diagrama de dispersión revelan una heterocedasticidad ligera y normal, causada por la naturaleza volátil de los viajes de larga distancia y tarifas planas de aeropuertos.
# 
# English:
# 
# Exceptional Predictive Power: The multiple linear regression model achieved a coefficient of determination (R2) of 0.8725 on the test dataset. This means the algorithm explains 87.25% of the fare variability using only four features.
# 
# No Overfitting: The performance on the test set (R2=0.8725, MAE=$2.12) was slightly better than on the training set (R2=0.8399, MAE=$2.20), proving that the model generalizes excellently to unseen data.
# 
# Assumption Validation: Graphical diagnostics confirmed that the errors (residuals) are heavily centered around zero (mean≈−0.037), validating linearity. A right-skewed tail in the histogram and an expanding pattern in the scatterplot reveal a slight, expected heteroscedasticity driven by the inherent volatility of long-distance trips and fixed airport flat rates.
# 
# 2. Results to Present (Resultados para Presentar a los Directivos)
# Español:
# 
# La Distancia es el Motor Principal: mean_distance tiene el impacto más grande por un amplio margen. Por cada incremento de una desviación estándar en la distancia de la ruta, la tarifa aumenta en $7.48 dólares.
# 
# El Tiempo es Secundario: mean_duration ocupa el segundo lugar. Cada incremento de una desviación estándar en la duración añade $2.42 dólares. La distancia influye tres veces más que el tiempo transcurrido en el vehículo.
# 
# Impacto Marginal de la Hora Punta: Viajar durante rush_hour solo incrementa la tarifa en $0.15 dólares de forma directa, ya que el costo del tráfico pesado ya se absorbe automáticamente a través del aumento en la duración del trayecto.
# 
# Neutralidad de Proveedores: El coeficiente de VendorID_2 es despreciable (-$0.03 dólares). Esto demuestra de forma matemática que ambos proveedores tecnológicos cobran tarifas equivalentes bajo las mismas condiciones de viaje, asegurando la consistencia del servicio.
# 
# EN:
# 
# Distance is the Primary Driver: mean_distance has the largest impact by a wide margin. For every one standard deviation increase in route distance, the fare increases by $7.48 dollars.
# 
# Duration plays a Supporting Role: mean_duration ranks second. Each standard deviation increase in duration adds $2.42 dollars. Distance influences the final price three times more than the time spent inside the vehicle.
# 
# Marginal Rush Hour Impact: Traveling during rush_hour only directly adds $0.15 dollars to the fare, as the operational cost of heavy traffic is already implicitly captured by the increased trip duration.
# 
# Vendor Neutrality: The coefficient for VendorID_2 is negligible (-$0.03 dollars). This mathematically proves that both technology vendors charge equivalent rates under identical trip conditions, ensuring pricing consistency across the platform.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged. 
