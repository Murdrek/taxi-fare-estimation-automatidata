
# Automatidata: Ingeniería de Datos Urbanos y Modelado Predictivo

**ES:** Preparación, estructuración y modelado predictivo de datos masivos de la TLC de NYC. Incluye limpieza de anomalías, ingeniería de variables avanzadas (medias de ruta y factor tráfico), validación empírica de supuestos estadísticos y el desarrollo de un modelo de regresión lineal múltiple para la optimización de tarifas.
**EN:** Data engineering, structuring, and predictive modeling for NYC TLC massive datasets. Features anomaly cleaning, advanced feature engineering (route means and traffic factors), empirical validation of statistical assumptions, and the development of a multiple linear regression model for fare optimization.

---

## 📈 Model Performance & Key Takeaways / Rendimiento del Modelo y Conclusiones Clave

### Evaluation Metrics / Métricas de Evaluación
**ES:** El modelo de regresión lineal múltiple fue entrenado y validado utilizando características estandarizadas (StandardScaler), demostrando una excelente capacidad de generalización y ausencia total de sobreajuste (overfitting):
**EN:** The multiple linear regression model was trained and validated using standardized features (StandardScaler), demonstrating excellent generalization capability and zero overfitting:

* **Train Data:** R2 = 0.8399  | MAE = 2.20  | RMSE = 4.23$
* **Test Data:** R2 = 0.8725  |  MAE = \$2.12 | RMSE = 3.72$

### Feature Importance / Impacto de las Variables
* **mean_distance (β=7.4771):
  
* ** **ES:** El predictor dominante. Cada incremento de una desviación estándar en la distancia de la ruta añade $7.48 a la tarifa.
* **EN:** The dominant predictor. Every one standard deviation increase in route distance adds $7.48 to the fare.
* **mean_duration (β=2.4249):
  
* ** **ES:** Segundo factor clave. La distancia influye tres veces más que el tiempo transcurrido en el vehículo.
* **EN:** Second key driver. Distance influences the final price three times more than the time spent inside the vehicle.
* **rush_hour (β=0.1525) & VendorID_2 (β=−0.0275):

* ** **ES:** Impacto marginal o nulo, demostrando neutralidad de proveedores y que el costo del tráfico pesado ya es absorbido por la duración.
* **EN:** Marginal or negligible impact, proving vendor neutrality and that heavy traffic costs are already captured by duration.

---

## 🛠️ Tools & Libraries / Herramientas y Librerías

**ES:** El proyecto fue desarrollado en Python, utilizando Pandas y NumPy para la manipulación de datos, Scikit-Learn para el modelado estadístico y escalado, y Matplotlib/Seaborn para visualización avanzada y análisis de residuos.
**EN:** The project was developed using Python, leveraging Pandas and NumPy for data manipulation, Scikit-Learn for statistical modeling and scaling, and Matplotlib/Seaborn for advanced data visualization and residual analysis.

---

## 📁 Repository Structure / Estructura del Repositorio

| Folder / Carpeta | Description (EN) | Descripción (ES) |
| :--- | :--- | :--- |
| `/docs/` | Institutional and executive documentation (PDFs, tables, summaries) | Documentación institucional y ejecutiva (PDFs, tablas, resúmenes) |
| `/notebooks/` | Reproducible Jupyter notebooks with simulations, visualizations, and metrics | Notebooks Jupyter reproducibles con simulaciones, visualizaciones y métricas |
| `/code/` | Python scripts for model training, validation, and evaluation | Scripts en Python para entrenamiento, validación y evaluación del modelo |

---

## 🎯 Purpose / Propósito
* **EN:** This repository supports the ethical and strategic development of data science projects through transparent documentation, modular resources, and reproducible statistical workflows designed for corporate decision-making.
* **ES:** Este repositorio respalda el desarrollo ético y estratégico de proyectos de ciencia de datos mediante documentación transparente, recursos modulares y flujos de trabajo estadísticos reproducibles diseñados para la toma de decisiones corporativas.

---

## 🚀 Future Model Improvements / Próximas Mejoras al Modelo
Given the structural dynamics of NYC transit data, the current baseline model can be optimized through the following technical roadmap:
Dado los componentes estructurales de los datos de tránsito en Nueva York, el modelo base actual puede optimizarse a través de la siguiente hoja de ruta técnica:

* **Target Variable Refinement / Refinamiento de la Variable Objetivo:**
    * **EN:** Shift the core predictive target from `total_amount` to `fare_amount` to exclude `tip_amount`. Credit card tips are logged properly, but cash tips are systematically missing (recorded as $0.0), introducing a structural downward bias if left unadjusted.
    * **ES:** Cambiar el objetivo predictivo central de `total_amount` a `fare_amount` para excluir `tip_amount`. Las propinas con tarjeta de crédito se registran correctamente, pero las de efectivo faltan sistemáticamente (registradas como $0.0), introduciendo un sesgo a la baja si no se ajusta.
* **Rule-Based Routing for Flat-Rates / Enrutamiento por Reglas para Tarifas Planas:**
    * **EN:** Implement a hardcoded bypass for known fixed-price commutes (e.g., Manhattan to JFK Airport). Linear models assume continuous cost accumulation and generate massive residuals when encountering rigid flat-rate zone structures.
    * **ES:** Implementar un desvío basado en reglas duras para trayectos de precio fijo conocidos (ej. Manhattan al Aeropuerto JFK). Los modelos lineales asumen una acumulación continua de costos y generan residuos masivos al enfrentar estructuras rígidas de tarifa plana.
* **Architecture Transition to Machine Learning / Transición de Arquitectura a Machine Learning:**
    * **EN:** Migrate from Multiple Linear Regression to tree-based ensemble algorithms like **Random Forest Regressor** or **XGBoost** to natively capture complex, non-linear feature interactions and handle extreme urban outliers without breaking statistical assumptions.
    * **ES:** Migrar de Regresión Linear Múltiple a algoritmos de ensamble basados en árboles como **Random Forest Regressor** o **XGBoost** para capturar de forma nativa interacciones de variables no lineales y complejas sin romper supuestos estadísticos.

---

## ⚖️ Ethical Considerations / Consideraciones Éticas
* **Socioeconomic Equity / Equidad Socioeconómica:**
    * **EN:** Over-indexing on geographic coordinates can lead to algorithmic bias against lower-income neighborhoods, potentially inflating prices or reducing ride availability. Predictive tools must balance passenger affordability with fair driver compensation.
    * **ES:** Indexar en exceso coordenadas geográficas puede generar sesgos algorítmicos contra vecindarios de menores ingresos, inflando los precios o reduciendo la disponibilidad de viajes. Las herramientas predictivas deben equilibrar la accesibilidad del pasajero con una compensación justa para el conductor.
* **Data Privacy / Privacidad de Datos:**
    * **EN:** Combining location IDs with precise timestamps creates a spatiotemporal signature that risks re-identifying individuals' travel routines, requiring strict data governance and purpose limitation principles.
    * **ES:** Combinar IDs de ubicación con marcas de tiempo precisas crea una firma espaciotemporal que corre el riesgo de re-identificar rutinas de viaje individuales, requiriendo una estricta gobernanza de datos y principios de limitación de propósito.ojects through transparent documentation, modular resources, and
