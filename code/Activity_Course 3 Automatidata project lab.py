#!/usr/bin/env python
# coding: utf-8

# # **Course 2 Automatidata project**
# **Course 2 - Go Beyond the Numbers: Translate Data into Insights**

# You are the newest data professional in a fictional data consulting firm: Automatidata. The team is in an early stage of the project, having only just completed an initial plan of action and some early Python coding work. 
# 
# Luana Rodriquez, the senior data analyst at Automatidata, is pleased with the work you have already completed and requests your assistance with some EDA and data visualization work for the New York City Taxi and Limousine Commission project (New York City TLC) to get a general understanding of what taxi ridership looks like. The management team is asking for a Python notebook showing data structuring and cleaning, as well as any matplotlib/seaborn visualizations plotted to help understand the data. At the very least, include a box plot of the ride durations and some time series plots, like a breakdown by quarter or month. 
# 
# Additionally, the management team has recently asked all EDA to include Tableau visualizations. For this taxi data, create a Tableau dashboard showing a New York City map of taxi/limo trips by month. Make sure it is easy to understand to someone who isn’t data savvy, and remember that the assistant director at the New York City TLC is a person with visual impairments.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# ### Course 2 End-of-course project: Exploratory data analysis
# 
# In this activity, you will examine data provided and prepare it for analysis. You will also design a professional data visualization that tells a story, and will help data-driven decisions for business needs. 
# 
# Please note that the Tableau visualization activity is optional, and will not affect your completion of the course. Completing the Tableau activity will help you practice planning out and plotting a data visualization based on a specific business need. The structure of this activity is designed to emulate the proposals you will likely be assigned in your career as a data professional. Completing this activity will help prepare you for those career moments.
# 
# **The purpose** of this project is to conduct exploratory data analysis on a provided data set. Your mission is to continue the investigation you began in C1 and perform further EDA on this data with the aim of learning more about the variables. 
#   
# **The goal** is to clean data set and create a visualization.
# <br/>  
# *This activity has 4 parts:*
# 
# **Part 1:** Imports, links, and loading
# 
# **Part 2:** Data Exploration
# *   Data cleaning
# 
# 
# **Part 3:** Building visualizations
# 
# **Part 4:** Evaluate and share results
# 
# <br/> 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work. 
# 
# 

# # **Visualize a story in Tableau and Python**

# # **PACE stages** 
# 
# 
# <img src="images/Pace.png" width="100" height="100" align=left>
# 
#    *        [Plan](#scrollTo=psz51YkZVwtN&line=3&uniqifier=1)
#    *        [Analyze](#scrollTo=mA7Mz_SnI8km&line=4&uniqifier=1)
#    *        [Construct](#scrollTo=Lca9c8XON8lc&line=2&uniqifier=1)
#    *        [Execute](#scrollTo=401PgchTPr4E&line=2&uniqifier=1)

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## PACE: Plan 
# 
# In this stage, consider the following questions where applicable to complete your code response:
# 1. Identify any outliers: 
# 
# 
# *   What methods are best for identifying outliers?
# *   How do you make the decision to keep or exclude outliers from any future models?
# 
# 

# ==> ENTER YOUR RESPONSE HERE
# What methods are best for identifying outliers?
# 
# 
# Box Plots (Diagramas de Caja):
#     EN: Excellent for a quick visual of the distribution and spotting points beyond the whiskers.
#     ES: Excelentes para una visualización rápida de la distribución y para detectar puntos más allá de los "bigotes".Interquartile 
# 
# Range - IQR (Rango Intercuartílico): 
# 
#     EN: A statistical rule where any value below $Q1 - 1.5 \times IQR$ or above $Q3 + 1.5 \times IQR$ is flagged.
# 
#     ES: Una regla estadística donde se marca cualquier valor por debajo de $Q1 - 1.5 \times IQR$ o por encima de $Q3 + 1.5 \times IQR$.Histograms (Histogramas): * 
# 
#     EN: Helpful to see the "tail" of the data and how far it stretches.
# 
#     ES: Útiles para ver la "cola" de los datos y qué tanto se extiende.
# 
# How do you make the decision to keep or exclude outliers from any future models?
# 
# Delete / Eliminar:
# 
#     EN: If they are clearly data entry errors (e.g., a 100-hour taxi ride within Manhattan) or physically impossible values.
# 
#     ES: Si son claramente errores de entrada de datos (ej. un viaje de 100 horas dentro de Manhattan) o valores físicamente imposibles.
#     
#   Reassign / Reasignar (Imputation):
# 
#     EN: If you want to keep the record but reduce the outlier's influence, you can replace it with the median or the boundary value (Capping/Winsorizing).
# 
#     ES: Si quieres conservar el registro pero reducir la influencia del valor atípico, puedes reemplazarlo por la mediana o por el valor del límite (Capping/Winsorizing). 
#     
#     

# ### Task 1. Imports, links, and loading
# Go to Tableau Public
# The following link will help you complete this activity. Keep Tableau Public open as you proceed to the next steps. 
# 
# Link to supporting materials: 
# Tableau Public: https://public.tableau.com/s/ 
# 
# For EDA of the data, import the data and packages that would be most helpful, such as pandas, numpy and matplotlib. 
# 

# In[1]:


# Import packages and libraries
#==> ENTER YOUR CODE HERE

# Import packages and libraries for data manipulation and visualization
# Importar paquetes y librerías para manipulación y visualización de datos

import pandas as pd           # For dataframes / Para manejo de tablas de datos
import numpy as np            # For numerical operations / Para operaciones numéricas
import matplotlib.pyplot as plt # For basic plotting / Para gráficos básicos
import seaborn as sns         # For advanced statistical visuals / Para visualizaciones estadísticas avanzadas


# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe
# Cargar el conjunto de datos (Asegúrate de que el nombre coincida con tu archivo local)

df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## PACE: Analyze 
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### Task 2a. Data exploration and cleaning
# 
# Decide which columns are applicable
# 
# The first step is to assess your data. Check the Data Source page on Tableau Public to get a sense of the size, shape and makeup of the data set. Then answer these questions to yourself: 
# 
# Given our scenario, which data columns are most applicable? 
# Which data columns can I eliminate, knowing they won’t solve our problem scenario? 
# 
# Consider functions that help you understand and structure the data. 
# 
# *    head()
# *    describe()
# *    info()
# *    groupby()
# *    sortby()
# 
# What do you do about missing data (if any)? 
# 
# Are there data outliers? What are they and how might you handle them? 
# 
# What do the distributions of your variables tell you about the question you're asking or the problem you're trying to solve?
# 
# 
# 

# ==> ENTER YOUR RESPONSE HERE
# What do you do about missing data (if any)?
# 
#     EN: I will assess the percentage of missing values using df.isnull().sum(). If the missing data is minimal, I will drop those rows to maintain the integrity of the duration calculations and the Tableau map. If critical columns like fare_amount have gaps, I will consider median imputation to avoid losing too much data.
# 
#     ES: Evaluaré el porcentaje de valores faltantes usando df.isnull().sum(). Si los datos faltantes son mínimos, eliminaré esas filas para mantener la integridad de los cálculos de duración y el mapa de Tableau. Si faltan datos en columnas críticas como fare_amount, consideraré la imputación por la mediana para evitar perder demasiada información.
#    
# Are there data outliers? What are they and how might you handle them?
# 
#     EN: "I expect to find outliers in trip_distance and duration due to traffic or system errors. I will handle negative values and impossible durations by removing them, and I will use the IQR method to identify statistical outliers for further investigation."
# 
#     ES: "Espero encontrar valores atípicos en trip_distance y duration debido al tráfico o errores del sistema. Manejaré los valores negativos y duraciones imposibles eliminándolos, y usaré el método IQR para identificar outliers estadísticos para una investigación más profunda."
# 
# What do the distributions of your variables tell you about the question you're asking or the problem you're trying to solve?  
#    
#     EN: "The distributions reveal the 'typical' behavior of NYC riders. A right-skewed distribution in trip duration suggests that while most rides are short, a small number of long-distance trips significantly impact the total revenue. Identifying peaks (modality) helps determine peak demand hours, which is essential for the management team's goal of understanding ridership patterns."
# 
#     ES: "Las distribuciones revelan el comportamiento 'típico' de los pasajeros de NYC. Una distribución con sesgo a la derecha en la duración del viaje sugiere que, aunque la mayoría de los viajes son cortos, un pequeño número de viajes de larga distancia impacta significativamente en los ingresos totales. Identificar los picos (modalidad) ayuda a determinar las horas de mayor demanda, lo cual es esencial para el objetivo del equipo directivo de entender los patrones de flujo de pasajeros."
#     

# Start by discovering, using head and size. 

# In[3]:


#==> ENTER YOUR CODE HERE
# Check the shape (number of rows and columns) of the dataset
# Revisar la forma (número de filas y columnas) del conjunto de datos
print("Dataset size (rows, columns):", df.shape)

# Alternative: Check the total number of elements
# Alternativa: Revisar el número total de elementos
print("Total number of cells:", df.size)


# In[4]:


#==> ENTER YOUR CODE HERE

# Display the first 10 rows to understand the data makeup
# Mostrar las primeras 10 filas para entender la composición de los datos
df.head(10)


# Use describe... 

# In[5]:


#==> ENTER YOUR CODE HERE
# Generar un resumen estadístico de las columnas numéricas
# Generate a statistical summary of the numerical columns
df.describe()


# And info. 

# In[6]:


#==> ENTER YOUR CODE HERE
# Check the data types and look for non-null counts
# Revisar los tipos de datos y buscar conteos de valores no nulos
df.info()


# ### Task 2b. Assess whether dimensions and measures are correct

# On the data source page in Tableau, double check the data types for the applicable columns you selected on the previous step. Pay close attention to the dimensions and measures to assure they are correct. 
# 
# In Python, consider the data types of the columns. *Consider:* Do they make sense? 

# Review the link provided in the previous activity instructions to create the required Tableau visualization. 

# ### Task 2c. Select visualization type(s)

# Select data visualization types that will help you understand and explain the data.
# 
# Now that you know which data columns you’ll use, it is time to decide which data visualization makes the most sense for EDA of the TLC dataset. What type of data visualization(s) would be most helpful? 
# 
# * Line graph
# * Bar chart
# * Box plot
# * Histogram
# * Heat map
# * Scatter plot
# * A geographic map
# 

# ==> ENTER YOUR RESPONSE HERE
# 
# EN: "For this EDA, I will prioritize Box Plots to identify outliers in trip duration, Histograms to understand the distribution of fares, and Line Graphs for time-series analysis. Additionally, a Geographic Map will be used in Tableau to visualize ridership patterns across NYC zones."
# 
# ES: "Para este EDA, daré prioridad a los Box Plots para identificar valores atípicos en la duración del viaje, Histogramas para entender la distribución de las tarifas y Gráficos de Líneas para el análisis de series temporales. Además, se utilizará un Mapa Geográfico en Tableau para visualizar los patrones de viajes en las distintas zonas de Nueva York."

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## PACE: Construct 
# 
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### Task 3. Data visualization
# 
# You’ve assessed your data, and decided on which data variables are most applicable. It’s time to plot your visualization(s)!
# 

# ### Boxplots

# Perform a check for outliers on relevant columns such as trip distance and trip duration. Remember, some of the best ways to identify the presence of outliers in data are box plots and histograms. 
# 
# **Note:** Remember to convert your date columns to datetime in order to derive total trip duration.  

# In[7]:


# Convert data columns to datetime
#==> ENTER YOUR CODE HERE
# Convert data columns to datetime
# Convertir las columnas de datos a formato datetime
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Derive 'trip_duration' column (in minutes) 
# Crear la columna 'trip_duration' (en minutos)
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

# Check the first few rows to confirm
# Revisar las primeras filas para confirmar
df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_duration']].head()


# **trip distance**

# In[9]:


# Create box plot of trip_distance
#==> ENTER YOUR CODE HERE

import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el estilo visual / Set visual style
sns.set_theme(style="whitegrid")

# Crear una figura con dos subgráficos / Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1. Boxplot para la distancia del viaje / Boxplot for trip distance
sns.boxplot(x=df['trip_distance'], ax=axes[0])
axes[0].set_title('Boxplot: Trip Distance')
axes[0].set_xlabel('Distance (miles)')

# 2. Boxplot para la duración del viaje / Boxplot for trip duration
sns.boxplot(x=df['trip_duration'], ax=axes[1])
axes[1].set_title('Boxplot: Trip Duration')
axes[1].set_xlabel('Duration (minutes)')

plt.show()


# In[10]:


# Create histogram of trip_distance
#==> ENTER YOUR CODE HERE

# Create histogram of trip_distance
# Crear el histograma de trip_distance
plt.figure(figsize=(10, 6))
sns.histplot(df['trip_distance'], bins=range(0, 26, 1), kde=False)

# Personalización / Customization
plt.title('Trip Distance Histogram')
plt.xlabel('Distance (miles)')
plt.ylabel('Frequency (Count)')
plt.show()


# **total amount**

# In[11]:


# Create box plot of total_amount
#==> ENTER YOUR CODE HERE

# Create box plot of total_amount
# Crear el boxplot de total_amount
plt.figure(figsize=(9, 2))
sns.boxplot(x=df['total_amount'])

# Personalización / Customization
plt.title('Total Amount Boxplot')
plt.xlabel('Total Amount ($)')
plt.show()


# In[12]:


# Create histogram of total_amount
#==> ENTER YOUR CODE HERE

# Create histogram of total_amount
# Crear el histograma de total_amount
plt.figure(figsize=(12, 6))
sns.histplot(df['total_amount'], bins=range(0, 101, 5), kde=False)

# Personalización / Customization
plt.title('Total Amount Histogram')
plt.xlabel('Total Amount ($)')
plt.ylabel('Frequency (Count)')
plt.show()


# **tip amount**

# In[13]:


# Create box plot of tip_amount
#==> ENTER YOUR CODE HERE

# Create box plot of tip_amount
# Crear el boxplot de tip_amount
plt.figure(figsize=(9, 2))
sns.boxplot(x=df['tip_amount'])

# Personalización / Customization
plt.title('Tip Amount Boxplot')
plt.xlabel('Tip Amount ($)')
plt.show()


# In[14]:


# Create histogram of tip_amount
#==> ENTER YOUR CODE HERE
# Create histogram of tip_amount
# Crear el histograma de tip_amount
plt.figure(figsize=(10, 6))
ax = sns.histplot(df['tip_amount'], bins=range(0, 21, 1), kde=False)

# Personalización / Customization
plt.title('Tip Amount Histogram')
plt.xlabel('Tip Amount ($)')
plt.ylabel('Frequency (Count)')
plt.xticks(range(0, 21, 2))
plt.show()


# **tip_amount by vendor**

# In[15]:


# Create histogram of tip_amount by vendor
#==> ENTER YOUR CODE HERE

# Create histogram of tip_amount by vendor
# Crear el histograma de tip_amount por proveedor (vendor)
plt.figure(figsize=(12, 7))
ax = sns.histplot(data=df, x='tip_amount', bins=range(0, 21, 1), 
                  hue='VendorID', 
                  multiple='stack', 
                  palette='viridis')

# Personalización / Customization
ax.set_xticks(range(0, 21, 1))
plt.title('Tip Amount by Vendor Histogram')
plt.xlabel('Tip Amount ($)')
plt.ylabel('Frequency (Count)')
plt.show()


# Next, zoom in on the upper end of the range of tips to check whether vendor one gets noticeably more of the most generous tips.

# In[16]:


# Create histogram of tip_amount by vendor for tips > $10 
#==> ENTER YOUR CODE HERE

# Create histogram of tip_amount by vendor for tips > $10
# Crear histograma de tip_amount por proveedor para propinas > $10

# Filtrar el dataframe para propinas mayores a 10
# Filter the dataframe for tips greater than 10
tips_over_10 = df[df['tip_amount'] > 10]

plt.figure(figsize=(12, 7))
ax = sns.histplot(data=tips_over_10, x='tip_amount', bins=range(10, 21, 1), 
                  hue='VendorID', 
                  multiple='stack', 
                  palette='viridis')

# Personalización / Customization
ax.set_xticks(range(10, 21, 1))
plt.title('Tip Amount > $10 by Vendor Histogram')
plt.xlabel('Tip Amount ($)')
plt.ylabel('Frequency (Count)')
plt.show()


# **Mean tips by passenger count**
# 
# Examine the unique values in the `passenger_count` column.

# In[17]:


#==> ENTER YOUR CODE HERE
# Examine the unique values in the passenger_count column
# Examinar los valores únicos en la columna passenger_count
df['passenger_count'].value_counts()


# In[18]:


# Calculate mean tips by passenger_count
#==> ENTER YOUR CODE HERE
# Calculate the mean tip amount for each passenger count
# Calcular el promedio del monto de la propina por cantidad de pasajeros
mean_tips_by_passenger_count = df.groupby('passenger_count')[['tip_amount']].mean()
mean_tips_by_passenger_count


# In[19]:


# Create bar plot for mean tips by passenger count
#==> ENTER YOUR CODE HERE

# Create bar plot for mean tips by passenger count
# Crear gráfico de barras para el promedio de propinas por cantidad de pasajeros
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_tips_by_passenger_count.index, 
            y=mean_tips_by_passenger_count['tip_amount'],
            palette='magma')

# Personalización / Customization
plt.title('Mean Tip Amount by Passenger Count')
plt.xlabel('Passenger Count')
plt.ylabel('Mean Tip Amount ($)')

# Añadir una línea horizontal con el promedio global para referencia
plt.axhline(df['tip_amount'].mean(), color='red', linestyle='--', label='Global Mean')
plt.legend()

plt.show()


# **Create month and day columns**

# In[20]:


# Create a month column
#==> ENTER YOUR CODE HERE

# Create a month column
# Crear una columna de mes
df['month'] = df['tpep_pickup_datetime'].dt.month_name()

# Create a day column
#==> ENTER YOUR CODE HERE

# Create a day column
# Crear una columna de día
df['day'] = df['tpep_pickup_datetime'].dt.day_name()


# **Plot total ride count by month**
# 
# Begin by calculating total ride count by month.

# In[21]:


# Get total number of rides for each month
#==> ENTER YOUR CODE HERE

# 1. Obtener el número total de viajes por mes
monthly_rides = df['month'].value_counts()







# Reorder the results to put the months in calendar order.

# In[22]:


# Reorder the monthly ride list so months go in order
#==> ENTER YOUR CODE HERE
# 2. Reordenar los meses cronológicamente (importante para el análisis de tendencias)
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']



# In[23]:


# Show the index
#==> ENTER YOUR CODE HERE
monthly_rides = monthly_rides.reindex(index=month_order)


# In[24]:


# Create a bar plot of total rides per month
#==> ENTER YOUR CODE HERE

# 3. Visualización
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_rides.index, y=monthly_rides.values, palette='magma')

plt.title('Total Ride Count by Month', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45) # Rotamos para mejor legibilidad

plt.show()


# **Plot total ride count by day**
# 
# Repeat the above process, but now calculate the total rides by day of the week.

# In[ ]:


# Repeat the above process, this time for rides by day
#==> ENTER YOUR CODE HERE


# In[25]:


# Create bar plot for ride count by day
#==> ENTER YOUR CODE HERE
# 1. Obtener el número total de viajes por día de la semana
daily_rides = df['day'].value_counts()

# 2. Reordenar para que empiece en lunes
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_rides = daily_rides.reindex(index=day_order)

# 3. Visualización
plt.figure(figsize=(12, 6))
sns.barplot(x=daily_rides.index, y=daily_rides.values, palette='magma')

plt.title('Total Ride Count by Day', fontsize=14)
plt.xlabel('Day of the Week')
plt.ylabel('Number of Rides')

plt.show()


# **Plot total revenue by day of the week**
# 
# Repeat the above process, but now calculate the total revenue by day of the week.

# In[26]:


# Repeat the process, this time for total revenue by day
#==> ENTER YOUR CODE HERE

# 1. Agrupar por día y sumar el monto total
total_rev_by_day = df.groupby('day')[['total_amount']].sum()

# 2. Reordenar cronológicamente
total_rev_by_day = total_rev_by_day.reindex(index=day_order)

# 3. Visualización
plt.figure(figsize=(12, 6))
sns.barplot(x=total_rev_by_day.index, y=total_rev_by_day['total_amount'], palette='magma')

plt.title('Total Revenue by Day', fontsize=14)
plt.xlabel('Day of the Week')
plt.ylabel('Total Revenue ($)')

plt.show()


# In[ ]:


# Create bar plot of total revenue by day
#==> ENTER YOUR CODE HERE


# **Plot total revenue by month**

# In[27]:


# Repeat the process, this time for total revenue by month
#==> ENTER YOUR CODE HERE

# 1. Obtener el ingreso total por mes
total_rev_by_month = df.groupby('month')[['total_amount']].sum()

# 2. Reordenar los meses cronológicamente
# Asegúrate de haber definido month_order anteriormente
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
total_rev_by_month = total_rev_by_month.reindex(index=month_order)

# 3. Visualización
plt.figure(figsize=(12, 6))
sns.barplot(x=total_rev_by_month.index, y=total_rev_by_month['total_amount'], palette='magma')

plt.title('Total Revenue by Month', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)

plt.show()


# In[27]:


# Create a bar plot of total revenue by month
#==> ENTER YOUR CODE HERE

# 1. Calcular el ingreso total por mes
total_rev_month = df.groupby('month').sum()[['total_amount']]

# 2. Reordenar los meses cronológicamente
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
total_rev_month = total_rev_month.reindex(index=month_order)

# 3. Crear el gráfico de barras
plt.figure(figsize=(12, 7))
sns.barplot(x=total_rev_month.index, y=total_rev_month['total_amount'], palette='magma')

# Personalización
plt.title('Total Revenue by Month', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)

plt.show()


# #### Scatter plot

# You can create a scatterplot in Tableau Public, which can be easier to manipulate and present. If you'd like step by step instructions, you can review the following link. Those instructions create a scatterplot showing the relationship between total_amount and trip_distance. Consider adding the Tableau visualization to your executive summary, and adding key insights from your findings on those two variables.

# [Tableau visualization guidelines](https://docs.google.com/document/d/1pcfUlttD2Y_a9A4VrKPzikZWCAfFLsBAhuKuomjcUjA/template/preview)

# **Plot mean trip distance by drop-off location**

# In[29]:


# Get number of unique drop-off location IDs
#==> ENTER YOUR CODE HERE
# 1. Obtener el número de IDs de ubicación de destino únicos
# Cambiamos 'dropoff_location_id' por 'DOLocationID'

print('Número de ubicaciones de destino únicas:', df['DOLocationID'].nunique())


# In[31]:


# Calculate the mean trip distance for each drop-off location
#==> ENTER YOUR CODE HERE

# 1. Calcular la distancia promedio por ubicación de destino (DOLocationID)
mean_distance_by_dropoff = df.groupby('DOLocationID')[['trip_distance']].mean()

# Sort the results in descending order by mean trip distance
#==> ENTER YOUR CODE HERE
# 2. Ordenar de mayor a menor y tomar las 10 principales para legibilidad
mean_distance_by_dropoff = mean_distance_by_dropoff.sort_values(by='trip_distance', ascending=False).head(10)


# In[32]:


# Create a bar plot of mean trip distances by drop-off location in ascending order by distance
#==> ENTER YOUR CODE HERE

# 3. Visualización
plt.figure(figsize=(14, 6))
sns.barplot(x=mean_distance_by_dropoff.index, 
            y=mean_distance_by_dropoff['trip_distance'], 
            order=mean_distance_by_dropoff.index,
            palette='magma')

# Personalización profesional
plt.title('Mean Trip Distance by Drop-off Location (Top 10)', fontsize=14)
plt.xlabel('Drop-off Location ID (DOLocationID)')
plt.ylabel('Mean Distance (miles)')

plt.show()


# ## BONUS CONTENT
# 
# To confirm your conclusion, consider the following experiment:
# 1. Create a sample of coordinates from a normal distribution&mdash;in this case 1,500 pairs of points from a normal distribution with a mean of 10 and a standard deviation of 5
# 2. Calculate the distance between each pair of coordinates 
# 3. Group the coordinates by endpoint and calculate the mean distance between that endpoint and all other points it was paired with
# 4. Plot the mean distance for each unique endpoint

# In[33]:


#BONUS CONTENT

#1. Generate random points on a 2D plane from a normal distribution
#==> ENTER YOUR CODE HERE
# 1. Generar puntos aleatorios en un plano 2D desde una distribución normal
# Creamos 1,500 pares de puntos (x, y)
mu = 10
sigma = 5
points = np.random.normal(mu, sigma, (1500, 2))

# 2. Calculate Euclidean distances between points in first half and second half of array
#==> ENTER YOUR CODE HERE
# 2. Calcular distancias Euclidianas entre pares de puntos
# Dividimos el array en dos mitades (750 puntos cada una) para simular origen y destino
start_points = points[:750]
end_points = points[750:]

# Calculamos la distancia: sqrt((x2-x1)^2 + (y2-y1)^2)
distances = np.sqrt(np.sum((end_points - start_points)**2, axis=1))

# 3. Group the coordinates by "drop-off location", compute mean distance
#==> ENTER YOUR CODE HERE
# 3. Agrupar por "ubicación de destino" y calcular la distancia promedio
# Creamos un DataFrame auxiliar para este experimento
bonus_df = pd.DataFrame({
    'dropoff_location': np.repeat(np.arange(10), 75), # Simulamos 10 ubicaciones con 75 viajes cada una
    'distance': distances
})
bonus_mean = bonus_df.groupby('dropoff_location').mean()

# 4. Plot the mean distance between each endpoint ("drop-off location") and all points it connected to
#==> ENTER YOUR CODE HERE

# 4. Graficar la distancia promedio por cada punto final ("drop-off location")
plt.figure(figsize=(10, 5))
sns.barplot(x=bonus_mean.index, y=bonus_mean['distance'], palette='magma')

plt.title('Mean Distance to Each Endpoint (Normal Distribution)', fontsize=14)
plt.xlabel('Endpoint ID')
plt.ylabel('Mean Distance')

plt.show()


# **Histogram of rides by drop-off location**

# First, check to whether the drop-off locations IDs are consecutively numbered. For instance, does it go 1, 2, 3, 4..., or are some numbers missing (e.g., 1, 3, 4...). If numbers aren't all consecutive, the histogram will look like some locations have very few or no rides when in reality there's no bar because there's no location. 

# In[34]:


# Check if all drop-off locations are consecutively numbered
#==> ENTER YOUR CODE HERE

# 1. Obtener los valores únicos y ordenarlos
unique_ids = sorted(bonus_df['dropoff_location'].unique())

# 2. Generar el rango esperado desde el mínimo al máximo encontrado
expected_range = list(range(min(unique_ids), max(unique_ids) + 1))

# 3. Comprobar si son idénticos
is_consecutive = unique_ids == expected_range

print(f"¿Los IDs son consecutivos?: {is_consecutive}")

if not is_consecutive:
    missing_ids = set(expected_range) - set(unique_ids)
    print(f"IDs faltantes en la secuencia: {missing_ids}")
else:
    print(f"La secuencia está completa desde el {min(unique_ids)} al {max(unique_ids)}.")


# To eliminate the spaces in the historgram that these missing numbers would create, sort the unique drop-off location values, then convert them to strings. This will make the histplot function display all bars directly next to each other. 

# In[36]:


# 1. Crear la columna de strings directamente en el DataFrame
# Asegúrate de usar el nombre correcto de tu DataFrame (df o bonus_df)
df['DOLocationID_str'] = df['DOLocationID'].astype(str)

# 2. Ordenar el DataFrame por la ubicación numérica para que al convertir a string 
# el eje X mantenga el orden lógico (1, 2, 3...) y no el alfabético ("1", "10", "2")
df = df.sort_values('DOLocationID')

# 3. Graficar
plt.figure(figsize=(16, 6))

# Usamos la nueva columna que acabamos de crear
sns.histplot(data=df, x='DOLocationID_str', discrete=True, color='skyblue', edgecolor='black')

plt.title('Distribución de viajes por ubicación de destino (Tratado como categorías)', fontsize=14)
plt.xlabel('ID de Ubicación (Categoría)', fontsize=12)
plt.ylabel('Cantidad de Viajes', fontsize=12)

# Rotamos los labels porque en Automatidata hay muchísimas zonas
plt.xticks(rotation=90) 

plt.show()


# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## PACE: Execute 
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### Task 4a. Results and evaluation
# 
# Having built visualizations in Tableau and in Python, what have you learned about the dataset? What other questions have your visualizations uncovered that you should pursue? 
# 
# ***Pro tip:*** Put yourself in your client's perspective, what would they want to know? 
# 
# Use the following code fields to pursue any additional EDA based on the visualizations you've already plotted. Also use the space to make sure your visualizations are clean, easily understandable, and accessible. 
# 
# ***Ask yourself:*** Did you consider color, contrast, emphasis, and labeling?
# 
# 

# ==> ENTER YOUR RESPONSE HERE
# 
# I have learned .... 
# 
# EN: We learned that ride distribution is highly skewed; a few locations account for the majority of the traffic. This suggests that fleet management should be centralized in these "hotspots."
# 
# ES: Aprendimos que la distribución de viajes está muy sesgada; unas pocas ubicaciones representan la mayoría del tráfico. Esto sugiere que la gestión de la flota debería centralizarse en estos "puntos calientes".
# 
# EN: We discovered that Euclidean distance is a theoretical baseline, but it doesn't account for NYC's grid complexity or traffic.
# 
# ES: Descubrimos que la distancia euclidiana es una base teórica, pero no tiene en cuenta la complejidad de la cuadrícula de NYC ni el tráfico.
# 
# My other questions are .... 
# 
# EN: Why do some rides with very short distances have high fares?
# ES: ¿Por qué algunos viajes con distancias muy cortas tienen tarifas altas?
# 
# Business Impact: This could reveal hidden costs like tolls (peajes) or surcharges (recargos por hora pico).
# 
# EDA Action: Create a scatter plot of trip_distance vs. fare_amount. Any point far from the diagonal is an "outlier" worth investigating.
# 
# EN: Do the top drop-off locations change depending on the time of day?
# ES: ¿Cambian las ubicaciones de destino principales según la hora del día?
# 
# EN: Are there zones where many people are dropped off but very few are picked up?
# ES: ¿Hay zonas donde se deja a mucha gente pero se recoge a muy poca?
# 
# EN: Is there a correlation between the drop-off location and the tip percentage?
# ES: ¿Existe una correlación entre la ubicación de destino y el porcentaje de propina?
# 
# My client would likely want to know ... 
# 
# EN: The client (TLC) would want to know: "Which zones are the most profitable per mile?" and "Are there discrepancies between the meter distance and the physical distance that suggest fraud or system errors?"
# 
# ES: El cliente (TLC) querría saber: "¿Qué zonas son las más rentables por milla?" y "¿Hay discrepancias entre la distancia del taxímetro y la distancia física que sugieran fraude o errores del sistema?"
# 

# In[37]:


#==> ENTER YOUR CODE HERE

# Create a scatter plot to analyze distance vs. fare
plt.figure(figsize=(12, 6))

sns.scatterplot(x=df['trip_distance'], y=df['fare_amount'], 
                alpha=0.2, # EN: Transparency for density / ES: Transparencia para ver densidad
                color='#34495e', 
                edgecolor=None)

# EN: Adding a reference line (theoretical trend)
# ES: Agregando una línea de referencia (tendencia teórica)
plt.title('Relationship: Trip Distance vs. Fare Amount', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Trip Distance (miles)', fontsize=12)
plt.ylabel('Fare Amount (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()


# In[38]:


#==> ENTER YOUR CODE HERE

# 1. EN: Calculate mean fare per location / ES: Calcular tarifa promedio por ubicación
location_revenue = df.groupby('DOLocationID_str')['fare_amount'].mean().sort_values(ascending=False).head(15)

# 2. Plotting
plt.figure(figsize=(14, 7))
sns.barplot(x=location_revenue.index, y=location_revenue.values, palette='magma')

# EN: Clean labeling and accessibility
# ES: Etiquetado limpio y accesibilidad
plt.title('Top 15 Most Profitable Drop-off Locations (Average Fare)', fontsize=16, fontweight='bold')
plt.xlabel('Location ID', fontsize=12)
plt.ylabel('Average Fare Amount (USD)', fontsize=12)
plt.xticks(rotation=45) # EN: Angle for readability / ES: Ángulo para legibilidad

# EN: Add data labels on top of bars for precision
# ES: Agregar etiquetas de datos sobre las barras para mayor precisión
for i, val in enumerate(location_revenue.values):
    plt.text(i, val + 0.5, f'${val:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# ### Task 4b. Conclusion
# *Make it professional and presentable*
# 
# You have visualized the data you need to share with the director now. Remember, the goal of a data visualization is for an audience member to glean the information on the chart in mere seconds.
# 
# *Questions to ask yourself for reflection:*
# Why is it important to conduct Exploratory Data Analysis? Why are the data visualizations provided in this notebook useful?
# 

# 
# EDA is important because ... 
# ==> ENTER YOUR RESPONSE HERE
# 
# EN: EDA is important because...
# It allows us to audit the data's health before making business decisions. In this project, EDA revealed critical outliers (like $450 fares for 0-mile trips) and non-consecutive location IDs. Without this step, any predictive model would be biased and inaccurate. It turns raw numbers into actionable business intelligence.
# 
# ES: El EDA es importante porque...
# Nos permite auditar la salud de los datos antes de tomar decisiones de negocio. En este proyecto, el EDA reveló valores atípicos críticos (como tarifas de $450 en viajes de 0 millas) e IDs de ubicación no consecutivos. Sin este paso, cualquier modelo predictivo sería sesgado e inexacto. Transforma números brutos en inteligencia de negocios procesable.
# 
# 
# Visualizations helped me understand ..
# ==> ENTER YOUR RESPONSE HERE
# EN: Visualizations helped me understand...
# The spatial and financial patterns of NYC taxi rides. Specifically, I understood that demand is highly concentrated in a few "hotspots" and that the relationship between distance and fare is generally linear, but heavily influenced by fixed-rate zones (like airports). By treating location IDs as categorical strings, I removed visual noise and created a clear, immediate picture of zone performance.
# 
# ES: Las visualizaciones me ayudaron a entender...
# Los patrones espaciales y financieros de los viajes en taxi de NYC. Específicamente, entendí que la demanda está altamente concentrada en unos pocos "puntos calientes" y que la relación entre la distancia y la tarifa es generalmente lineal, pero está fuertemente influenciada por zonas de tarifa fija (como aeropuertos). Al tratar los IDs de ubicación como cadenas categóricas, eliminé el ruido visual y creé una imagen clara e inmediata del rendimiento por zona.

# You’ve now completed professional data visualizations according to a business need. Well done! 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
