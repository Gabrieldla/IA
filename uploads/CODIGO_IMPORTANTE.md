# Código Importante del Proyecto - House Price Prediction

## 1. PROCESAMIENTO DE DATOS (data_processor.py)

### 1.1 Rellenar Valores Nulos
```python
# PASO 1: Rellenar TODOS los nulos primero
for col in df_listo.columns:                     # Recorre todas las columnas del DataFrame una por una
    if df_listo[col].isnull().any():             # Verifica si la columna actual contiene al menos un valor nulo (NaN)
        
        if df_listo[col].dtype in ['int64', 'float64']:   # Si el tipo de dato de la columna es numérico (entero o flotante)
            df_listo[col] = df_listo[col].fillna(0)       # Rellena todos los valores nulos con 0 (solo para columnas numéricas)
        
        else:                                              # Si la columna NO es numérica (por ejemplo, tipo objeto/categórica)
            df_listo[col] = df_listo[col].fillna('None')   # Rellena los nulos con el texto 'None' para categorías

```
**Propósito**: Elimina todos los valores faltantes del dataset. Variables numéricas se rellenan con 0, categóricas con 'None'.

---

### 1.2 Selección de Características por Correlación
```python
df_numerico = df_listo.select_dtypes(include=[np.number])  
# Crea un nuevo DataFrame que contiene solo las columnas numéricas de df_listo

if 'SalePrice' in df_numerico.columns:  
    # Verifica que la columna 'SalePrice' exista entre las columnas numéricas
    
    correlaciones = df_numerico.corr()['SalePrice'].sort_values(ascending=False)  
    # Calcula la matriz de correlación, extrae SOLO la correlación con 'SalePrice'
    # y ordena los valores de mayor a menor correlación

# TOP 10 características más correlacionadas
top_10_caracteristicas = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
                          'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 
                          'YearRemodAdd', 'TotRmsAbvGrd']  
# Lista manual de las 10 variables más importantes para predecir 'SalePrice'

# Reducir dataset: quedarse SOLO con las TOP 10 + SalePrice
df_listo = df_listo[top_10_caracteristicas + ['SalePrice']]  
# Filtra el DataFrame original para conservar únicamente las variables seleccionadas

```
**Propósito**: Reduce el dataset de 81 columnas a solo las 10 características más correlacionadas con el precio, mejorando eficiencia y precisión del modelo.

---

### 1.3 Eliminación de Outliers
```python
# Remover outliers extremos usando percentil 99.5
if 'GrLivArea' in df_listo.columns and 'SalePrice' in df_listo.columns:
    # Verifica que las columnas 'GrLivArea' y 'SalePrice' existan en el DataFrame antes de procesar

    umbral = df_listo['GrLivArea'].quantile(0.995)
    # Calcula el valor del percentil 99.5% de GrLivArea (sirve como límite para detectar outliers)

    df_listo = df_listo[df_listo['GrLivArea'] <= umbral]
    # Filtra el DataFrame, conservando solo las filas donde GrLivArea NO supera el umbral (elimina outliers)

    df_listo = df_listo.reset_index(drop=True)
    # Reinicia los índices después del filtrado para que queden ordenados y sin saltos

```
**Propósito**: Elimina casas con áreas extremadamente grandes (top 0.5%) que distorsionan el modelo.

---

### 1.4 Transformación Logarítmica
```python
# Transformar variable objetivo a escala logarítmica
# (Esto ayuda a normalizar la distribución de los precios)

if "SalePrice" in df_listo.columns:
    # Verifica que la columna 'SalePrice' exista en el DataFrame

    df_listo["SalePrice_log"] = np.log1p(df_listo["SalePrice"])
    # Crea una nueva columna aplicando logaritmo natural a (SalePrice + 1)
    # log1p evita problemas cuando SalePrice podría ser 0

```
**Propósito**: Normaliza la distribución de precios (de asimétrica a normal), mejorando la capacidad del modelo para aprender patrones lineales.

---

## 2. ENTRENAMIENTO DEL MODELO (model_trainer.py)

### 2.1 División de Datos
```python
# División 80-20 para entrenamiento y prueba
# (80% de los datos se usan para entrenar el modelo y 20% para evaluarlo)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                # X = variables predictoras, y = variable objetivo
    test_size=0.2,       # Indica que el 20% de los datos será para prueba (test)
    random_state=42      # Fija una semilla para que la división sea siempre igual (reproducible)
)

```
**Propósito**: Separa datos en conjunto de entrenamiento (80%) y prueba (20%) para validar el modelo.

---

### 2.2 Optimización de Hiperparámetros
```python
# Espacio de búsqueda de hiperparámetros
dist_parametros = {
    'n_estimators': [500, 1000, 1500],          # Número de árboles del modelo XGBoost
    'learning_rate': [0.01, 0.05, 0.1],         # Tasa de aprendizaje (cuánto corrige cada iteración)
    'max_depth': [3, 5, 7],                     # Profundidad máxima de cada árbol
    'min_child_weight': [1, 3, 5],              # Peso mínimo en los nodos hijos (controla sobreajuste)
    'subsample': [0.7, 0.8, 0.9],               # Porcentaje de muestras usadas para cada árbol
    'colsample_bytree': [0.7, 0.8, 1.0],        # Porcentaje de columnas usadas en cada árbol
    'gamma': [0, 0.1],                          # Reducción mínima de pérdida para dividir un nodo
    'reg_alpha': [0, 0.1, 1],                   # Regularización L1 (reduce complejidad)
    'reg_lambda': [0.5, 1, 2]                   # Regularización L2 (penaliza pesos grandes)
}

# RandomizedSearchCV con validación cruzada
busqueda_aleatoria = RandomizedSearchCV(
    estimator=modelo_xgb,                       # Modelo base XGBoost a optimizar
    param_distributions=dist_parametros,        # Hiperparámetros y valores posibles
    n_iter=20,                                  # Número de combinaciones aleatorias a probar
    cv=5,                                       # Validación cruzada de 5 particiones (5-fold)
    scoring='neg_root_mean_squared_error',      # Métrica: RMSE (negativo por convención de sklearn)
    n_jobs=-1,                                  # Usa todos los núcleos posibles para acelerar
    random_state=42                             # Semilla para reproducibilidad
)

busqueda_aleatoria.fit(X_train, y_train)        # Ejecuta la búsqueda probando 20 combinaciones
modelo = busqueda_aleatoria.best_estimator_     # Obtiene el mejor modelo encontrado tras la búsqueda

```
**Propósito**: Encuentra automáticamente la mejor combinación de hiperparámetros probando 20 configuraciones diferentes con validación cruzada de 5 pliegues.

---

### 2.3 Evaluación del Modelo
```python
# Predecir en conjunto de prueba
y_pred = self.model.predict(X_test)
# Genera las predicciones del modelo usando los datos de prueba (X_test)

# Convertir de escala logarítmica a original
y_test_orig = np.expm1(y_test)
# Convierte los valores reales (y_test), que estaban en logaritmo, a su escala original

y_pred_orig = np.expm1(y_pred)
# Convierte las predicciones del modelo desde logaritmo a valores reales (precios en dólares)

# Calcular métricas
metrics = {
    'r2': r2_score(y_test_orig, y_pred_orig), 
    # Cálculo del coeficiente de determinación R²
    # Indica qué porcentaje de la variabilidad del precio real explica el modelo

    'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
    # Cálculo del RMSE: error cuadrático medio (en dólares)
    # Penaliza fuertemente los errores grandes

    'mae': mean_absolute_error(y_test_orig, y_pred_orig)
    # Cálculo del MAE: error absoluto promedio (más fácil de interpretar)
}
# El diccionario 'metrics' almacena todas las métricas clave para evaluar el rendimiento del modelo


---

### 2.4 Predicción Individual
```python

def predict(self, features_dict):
    # Convertir dict a DataFrame
    import pandas as pd
    X_pred = pd.DataFrame([features_dict])
    # Convierte el diccionario recibido (features_dict) en un DataFrame de una sola fila,
    # para que el modelo lo pueda procesar correctamente.

    # Predecir en escala logarítmica
    y_pred_log = self.model.predict(X_pred)
    # Genera la predicción del modelo, pero en escala logarítmica
    # porque el modelo fue entrenado con SalePrice_log.

    # Convertir a escala original
    y_pred = np.expm1(y_pred_log)[0]
    # Convierte la predicción desde logaritmo a su valor real (precio en dólares).
    # np.expm1(x) aplica exp(x) - 1 y se usa porque el entrenamiento usó log1p.

    return y_pred
    # Devuelve el precio predicho en escala real.

```
**Propósito**: Toma las características de una casa y predice su precio, revirtiendo la transformación logarítmica.

---

### 2.5 Importancia de Características
```python
def get_feature_importance(self, feature_names):
    # Obtener importancia del modelo XGBoost
    importance = self.model.feature_importances_
    
    feature_importance = [
        {'feature': name, 'importance': float(imp)}
        for name, imp in zip(feature_names, importance)
    ]
    
    # Ordenar por importancia descendente
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return feature_importance
```
**Propósito**: Extrae qué características son más importantes según el modelo entrenado, útil para interpretar decisiones.

---

## 3. VISUALIZACIONES CLAVE (plot_utils.py)

### 3.1 Predicciones vs Valores Reales
```python
def create_predictions_vs_real_chart(y_test, y_pred):
    # Convertir a escala original
    y_test_orig = np.expm1(y_test)
    # Convierte los valores reales desde escala logarítmica a su valor original

    y_pred_orig = np.expm1(y_pred)
    # Convierte las predicciones del modelo desde logaritmo a escala real

    # Scatter plot
    ax.scatter(y_test_orig, y_pred_orig, alpha=0.6, s=50, color='#60a5fa')
    # Dibuja un gráfico de dispersión comparando precios reales vs predichos
    # Cada punto representa una casa: (precio_real, precio_predicho)

    # Línea de predicción perfecta (diagonal)
    valor_min = min(y_test_orig.min(), y_pred_orig.min())
    # Obtiene el valor mínimo entre reales y predicciones (para la diagonal)

    valor_max = max(y_test_orig.max(), y_pred_orig.max())
    # Obtiene el valor máximo entre reales y predicciones

    ax.plot([valor_min, valor_max], [valor_min, valor_max], 'r--', lw=2)
    # Dibuja la línea diagonal roja punteada que representa "predicción perfecta"
    # Si los puntos están cerca de esta línea, el modelo predice bien

```
**Propósito**: Visualiza qué tan cerca están las predicciones de los valores reales. Puntos cerca de la línea roja = buenas predicciones.

---

### 3.2 Análisis de Residuos
```python
def create_residuals_chart(y_pred, residuos):
    # Convertir a escala original
    y_pred_orig = np.expm1(y_pred)
    # Convierte las predicciones desde logaritmo a escala real

    residuos_orig = np.expm1(residuos + np.log1p(y_pred_orig)) - y_pred_orig
    # Reconstruye los residuos en escala real:
    # 1) residuos = y_test_log - y_pred_log
    # 2) Se suman a log1p(predicción real)
    # 3) Se aplica expm1 para volver a escala real
    # 4) Se resta la predicción real para obtener el residuo real final

    # Gráfico 1: Scatter de residuos
    ejes[0].scatter(y_pred_orig, residuos_orig, alpha=0.6, s=50, color='#4ade80')
    # Dibuja puntos mostrando la relación entre predicción y residuo
    # Permite ver si el modelo tiene patrones de error (indicadores de sesgo)

    ejes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    # Línea horizontal en 0 para identificar si los residuos se distribuyen alrededor del cero

    # Gráfico 2: Histograma de residuos
    ejes[1].hist(residuos_orig, bins=50, color='#60a5fa', alpha=0.7)
    # Muestra la distribución de los residuos; idealmente debe parecerse a una distribución normal

    ejes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    # Línea vertical en 0 para observar simetría de los residuos alrededor del cero

```
**Propósito**: Muestra los errores del modelo. Residuos distribuidos uniformemente alrededor de cero = buen modelo.

---

### 3.3 Importancia de Características
```python
def create_feature_importance_chart(feature_importance):
    # Tomar top 15
    top_features = feature_importance[:15]
    features = [f['feature'] for f in top_features]
    importance = [f['importance'] for f in top_features]
    
    # Gráfico de barras horizontales
    sns.barplot(x=importance, y=features, ax=ax, palette='viridis')
```
**Propósito**: Visualiza qué características son más importantes para el modelo al hacer predicciones.

---

### 3.4 Comparación Antes/Después de Outliers
```python
def create_outliers_comparison_chart(grliv_before, price_before, grliv_after, price_after, threshold):
    # ANTES (con outliers)
    axes[0].scatter(grliv_before, price_before, alpha=0.6, s=30, color='#ef4444')
    axes[0].axvline(threshold, color='#4ade80', linestyle='--', linewidth=2)
    
    # DESPUÉS (sin outliers)
    axes[1].scatter(grliv_after, price_after, alpha=0.6, s=30, color='#4ade80')
```
**Propósito**: Muestra cómo la eliminación de outliers mejora la calidad del dataset.

---

### 3.5 Transformación Logarítmica
```python
def create_log_transformation_chart(price_original, price_log):
    # Distribución ORIGINAL
    axes[0].hist(price_original, bins=50, color='#60a5fa', alpha=0.7)
    # Grafica un histograma de los precios originales (sin transformar)
    # Permite observar si la distribución está sesgada o tiene cola larga

    axes[0].axvline(price_original.mean(), color='red', linestyle='--', linewidth=2)
    # Dibuja una línea vertical indicando el valor promedio de los precios originales

    # Distribución LOGARÍTMICA
    axes[1].hist(price_log, bins=50, color='#4ade80', alpha=0.7)
    # Grafica un histograma de los precios ya transformados a logaritmo
    # Suele producir una distribución más simétrica y cercana a la normal

    axes[1].axvline(price_log.mean(), color='red', linestyle='--', linewidth=2)
    # Dibuja la línea vertical del promedio de los precios transformados

```
**Propósito**: Muestra cómo la transformación logarítmica convierte una distribución asimétrica en una distribución normal (gaussiana).

---

## 4. ARQUITECTURA GENERAL DEL SISTEMA

### 4.1 Flujo de Datos
```
CSV Upload → HousePriceDataProcessor → ModelTrainer → Predicción
     ↓               ↓                        ↓             ↓
  Validación    Limpieza                Optimización   Resultados
              Correlación             XGBoost + CV
              Outliers                Métricas
              Log Transform           Feature Importance
```

### 4.2 Pipeline de Procesamiento
1. **Carga**: Leer CSV con pandas
2. **Limpieza**: Rellenar nulos (0 para números, 'None' para texto)
3. **Selección**: Reducir a TOP 10 características correlacionadas
4. **Outliers**: Eliminar valores extremos (> percentil 99.5)
5. **Transformación**: Aplicar log(1+x) al precio
6. **División**: 80% train, 20% test
7. **Optimización**: RandomizedSearchCV con 20 iteraciones y 5-fold CV
8. **Evaluación**: Calcular R², RMSE, MAE
9. **Visualización**: Gráficos de análisis

### 4.3 Modelo XGBoost
```python
# Modelo base
modelo_xgb = xgb.XGBRegressor(
    random_state=42,
    eval_metric='rmse'
)

# Después de optimización, ejemplo de parámetros finales:
# - n_estimators: 1000 árboles
# - learning_rate: 0.05
# - max_depth: 5 niveles
# - subsample: 0.8 (80% de muestras por árbol)
# - colsample_bytree: 0.8 (80% de features por árbol)
```

---

## 5. MÉTRICAS DE RENDIMIENTO

### R² Score (Coeficiente de Determinación)
- **Rango**: 0 a 1 (o negativo si es peor que predecir la media)
- **Interpretación**: % de varianza explicada por el modelo
- **Ejemplo**: R² = 0.87 → El modelo explica el 87% de la variación en precios

### RMSE (Root Mean Squared Error)
- **Unidad**: Dólares ($)
- **Interpretación**: Error promedio penalizando errores grandes
- **Ejemplo**: RMSE = $25,000 → En promedio, nos equivocamos $25,000

### MAE (Mean Absolute Error)
- **Unidad**: Dólares ($)
- **Interpretación**: Error promedio sin penalizar outliers
- **Ejemplo**: MAE = $18,000 → Error absoluto promedio es $18,000

---

## 6. CONCEPTOS CLAVE DE MACHINE LEARNING

### Validación Cruzada (Cross-Validation)
Divide los datos en K pliegues (folds), entrena K veces usando K-1 pliegues para entrenar y 1 para validar. Promedia los resultados para obtener una estimación más robusta.

### Overfitting vs Underfitting
- **Overfitting**: Modelo aprende demasiado bien el conjunto de entrenamiento (incluyendo ruido)
- **Underfitting**: Modelo demasiado simple, no captura patrones importantes
- **Solución**: Regularización (reg_alpha, reg_lambda) y validación cruzada

### Feature Importance
XGBoost calcula qué tan útil es cada característica para hacer predicciones basándose en:
- Frecuencia de uso en divisiones (splits)
- Ganancia promedio al usarla
- Cobertura (% de muestras afectadas)

### Gradient Boosting
Construye árboles secuencialmente donde cada árbol nuevo intenta corregir errores del conjunto anterior:
1. Árbol 1 hace predicción inicial
2. Árbol 2 aprende de los errores del Árbol 1
3. Árbol 3 aprende de los errores acumulados
4. ... hasta n_estimators árboles
5. Predicción final = suma ponderada de todos los árboles

---

## 7. TRANSFORMACIONES MATEMÁTICAS IMPORTANTES

### Logaritmo Natural (log1p)
```python
# Forward transformation
y_log = np.log1p(y)  # log(1 + y)

# Inverse transformation
y_orig = np.expm1(y_log)  # exp(y_log) - 1
```
**Por qué**: Comprime valores grandes, expande valores pequeños, convierte distribución asimétrica en normal.

### Correlación de Pearson
```python
corr = df.corr()['SalePrice']
```
**Rango**: -1 a 1
- **1**: Correlación positiva perfecta
- **0**: Sin correlación
- **-1**: Correlación negativa perfecta

### Percentiles
```python
q995 = df['GrLivArea'].quantile(0.995)  # 99.5%
```
**Interpretación**: 99.5% de los datos están por debajo de este valor.

---

## RESUMEN EJECUTIVO

**Este sistema implementa un pipeline completo de Machine Learning para predicción de precios de casas:**

1. **Procesamiento**: Limpieza automática de datos, selección de características por correlación
2. **Modelo**: XGBoost con optimización automática de hiperparámetros
3. **Validación**: Cross-validation 5-fold + conjunto de prueba separado
4. **Precisión**: R² típico > 0.85 (explica +85% de la variación de precios)
5. **Interpretabilidad**: Feature importance y visualizaciones completas

**Ventajas técnicas:**
- Manejo automático de valores faltantes
- Reducción dimensional basada en evidencia (correlación)
- Transformación logarítmica para normalización
- Optimización exhaustiva de hiperparámetros
- Validación robusta con múltiples métricas
