# 🏠 Predicción de Precios de Casas con XGBoost

Aplicación web interactiva para entrenar modelos de Machine Learning y predecir precios de casas.

## 📋 Características

- **Entrenamiento Automático**: Sube un CSV y el modelo se entrena automáticamente
- **Visualizaciones**: Gráficos de análisis exploratorio, correlaciones, importancia de features
- **Optimización**: RandomizedSearchCV para encontrar los mejores hiperparámetros
- **Predicciones**: Interfaz amigable para predecir precios de casas nuevas
- **Métricas**: R², RMSE, MAE mostrados en tiempo real

## 🚀 Instalación

### 1. Instalar dependencias

```powershell
cd house_price_app
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación

```powershell
python app.py
```

### 3. Abrir en el navegador

La aplicación estará disponible en: `http://localhost:5001`

## 📁 Estructura del Proyecto

```
house_price_app/
├── app.py                  # Aplicación principal FastHTML
├── data_processor.py       # Procesamiento y limpieza de datos
├── model_trainer.py        # Entrenamiento y optimización del modelo
├── plot_utils.py          # Utilidades para crear gráficos
├── requirements.txt       # Dependencias del proyecto
├── uploads/              # Archivos CSV subidos
├── models/               # Modelos entrenados guardados
└── static/               # Archivos estáticos (si se necesitan)
```

## 🎯 Uso

### Paso 1: Entrenar el Modelo

1. Ve a la pestaña **"Entrenar Modelo"**
2. Sube tu archivo `houseprices.csv`
3. Haz clic en **"Entrenar Modelo"**
4. Espera a que se complete el entrenamiento (puede tomar varios minutos)
5. Revisa las métricas y gráficos generados

### Paso 2: Hacer Predicciones

1. Ve a la pestaña **"Hacer Predicción"**
2. Ingresa las características de la casa:
   - Calidad general (1-10)
   - Área habitable
   - Garaje
   - Año de construcción
   - etc.
3. Haz clic en **"Predecir Precio"**
4. Obtén el precio estimado

## 📊 Métricas del Modelo

- **R² Score**: Mide qué tan bien el modelo explica la variabilidad (~0.90)
- **RMSE**: Error cuadrático medio (~$25,000)
- **MAE**: Error absoluto medio (~$15,000)

## 🔧 Personalización

### Agregar más características

Edita el archivo `create_prediction_form()` en `app.py` para agregar campos adicionales.

### Cambiar parámetros de optimización

Modifica el diccionario `param_dist` en `model_trainer.py`.

### Ajustar visualizaciones

Personaliza las funciones en `plot_utils.py`.

## 📝 Notas

- El modelo se guarda automáticamente después del entrenamiento
- Los archivos se guardan en la carpeta `models/`
- Puedes hacer predicciones sin volver a entrenar el modelo
- Los gráficos se generan en formato base64 para mostrarlos en el navegador

## 🐛 Solución de Problemas

**Error al subir archivo:**
- Verifica que el archivo sea CSV
- Asegúrate de que contenga las columnas esperadas

**Error de memoria:**
- Reduce el parámetro `n_iter` en RandomizedSearchCV
- Usa un dataset más pequeño

**El modelo no se guarda:**
- Verifica permisos de escritura en la carpeta `models/`

## 📄 Licencia

Proyecto educativo - Libre uso
