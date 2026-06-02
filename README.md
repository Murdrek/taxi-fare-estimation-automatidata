
# Automatidata: Ingeniería de Datos Urbanos y Modelado Predictivo

**ES:** Preparación, estructuración y modelado predictivo de datos masivos de la TLC de NYC. Incluye limpieza de anomalías, ingeniería de variables avanzadas (medias de ruta y factor tráfico), validación empírica de supuestos estadísticos y el desarrollo de un modelo de regresión lineal múltiple para la optimización de tarifas.

**EN:** Data engineering, structuring, and predictive modeling for NYC TLC massive datasets. Features anomaly cleaning, advanced feature engineering (route means and traffic factors), empirical validation of statistical assumptions, and the development of a multiple linear regression model for fare optimization.

---

## 📈 Model Performance & Key Takeaways / Rendimiento del Modelo y Conclusiones Clave

### Evaluation Metrics / Métricas de Evaluación
**ES:** El modelo de regresión lineal múltiple fue entrenado y validado utilizando características estandarizadas (`StandardScaler`), demostrando una excelente capacidad de generalización y ausencia total de sobreajuste (*overfitting*):
**EN:** The multiple linear regression model was trained and validated using standardized features (`StandardScaler`), demonstrating excellent generalization capability and zero *overfitting*:

* **Train Data:** $R^2 = 0.8399 \ | \ \text{MAE} = \$2.20 \ | \ \text{RMSE} = \$4.23$
* **Test Data:** $R^2 = 0.8725 \ | \ \text{MAE} = \$2.12 \ | \ \text{RMSE} = \$3.72$

### Feature Importance / Impacto de las Variables
* **`mean_distance` ($\beta = 7.4771$):** **ES:** El predictor dominante. Cada incremento de una desviación estándar en la distancia de la ruta añade \$7.48 a la tarifa. **EN:** The dominant predictor. Every one standard deviation increase in route distance adds \$7.48 to the fare.
* **`mean_duration` ($\beta = 2.4249$):** **ES:** Segundo factor clave. La distancia influye tres veces más que el tiempo transcurrido en el vehículo. **EN:** Second key driver. Distance influences the final price three times more than the time spent inside the vehicle.
* **`rush_hour` ($\beta = 0.1525$) & `VendorID_2` ($\beta = -0.0275$):** **ES:** Impacto marginal o nulo, demostrando neutralidad de proveedores y que el costo del tráfico pesado ya es absorbido por la duración. **EN:** Marginal or negligible impact, proving vendor neutrality and that heavy traffic costs are already captured by duration.

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

**EN:** This repository supports the ethical and strategic development of data science projects through transparent documentation, modular resources, and
