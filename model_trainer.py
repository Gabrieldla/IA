"""
Módulo para entrenar el modelo XGBoost con optimización
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelTrainer:
    """Entrena y optimiza el modelo XGBoost"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.training_output = []  # Outputs detallados para mostrar en UI
        
    def entrenar_modelo(self, X, y):
        """Entrena el modelo con optimización"""
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.training_output.append({
            'step': 'split',
            'name': 'División de datos',
            'output': f'Train: {len(X_entrenamiento)} muestras | Test: {len(X_prueba)} muestras (80/20 split)'
        })
        
        self.training_output.append({
            'step': 'optimization_start',
            'name': 'Búsqueda de hiperparámetros',
            'output': 'RandomizedSearchCV iniciado con 20 combinaciones y 5-fold cross-validation'
        })
        self.model = self._optimizar_hiperparametros(X_entrenamiento, y_entrenamiento)
        
        # Evaluar modelo
        y_prediccion = self.model.predict(X_prueba)
        self.metrics = self._evaluar_modelo(X_prueba, y_prueba)
        
        return self.model, self.metrics, X_prueba, y_prueba, y_prediccion
    
    def _optimizar_hiperparametros(self, X_entrenamiento, y_entrenamiento):
        """Optimiza hiperparámetros usando RandomizedSearchCV"""
        dist_parametros = {
            'n_estimators': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0.5, 1, 2]
        }
        
        modelo_xgb = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        
        busqueda_aleatoria = RandomizedSearchCV(
            estimator=modelo_xgb,
            param_distributions=dist_parametros,
            n_iter=20,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        busqueda_aleatoria.fit(X_entrenamiento, y_entrenamiento)
        
        modelo = busqueda_aleatoria.best_estimator_
        mejores_params = busqueda_aleatoria.best_params_
        
        self.best_params = mejores_params
        
        # Guardar output de optimización
        params_str = "\n".join([f"  • {param}: {valor}" for param, valor in mejores_params.items()])
        self.training_output.append({
            'step': 'optimization_result',
            'name': 'Mejores hiperparámetros',
            'output': f'OPTIMIZACIÓN COMPLETADA\n\nMejores hiperparámetros:\n{params_str}'
        })
        
        self.training_output.append({
            'step': 'best_score',
            'name': 'Score de validación cruzada',
            'output': f'Mejor RMSE en CV: {-busqueda_aleatoria.best_score_:.4f}'
        })
        
        return modelo
    
    def _evaluar_modelo(self, X_prueba, y_prueba):
        """Evalúa el modelo y calcula métricas"""
        # Predecir
        y_prediccion = self.model.predict(X_prueba)
        
        # Convertir a escala original
        y_prueba_original = np.expm1(y_prueba)
        y_prediccion_original = np.expm1(y_prediccion)
        
        metricas = {
            'r2': r2_score(y_prueba_original, y_prediccion_original),
            'rmse': np.sqrt(mean_squared_error(y_prueba_original, y_prediccion_original)),
            'mae': mean_absolute_error(y_prueba_original, y_prediccion_original)
        }
        
        # Guardar output de evaluación
        self.training_output.append({
            'step': 'evaluation',
            'name': 'Evaluación del modelo',
            'output': f'''Métricas en conjunto de prueba:
  • R² Score: {metricas['r2']:.4f} ({metricas['r2']*100:.2f}% de varianza explicada)
  • RMSE: ${metricas['rmse']:,.0f} (error cuadrático medio)
  • MAE: ${metricas['mae']:,.0f} (error absoluto promedio)'''
        })
        
        return metricas
    
    def predecir(self, dict_caracteristicas):
        """Hace una predicción con el modelo"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Entrenar primero o cargar un modelo guardado.")
        
        # Convertir dict a DataFrame para predicción
        import pandas as pd
        X_pred = pd.DataFrame([dict_caracteristicas])
        
        # Predecir (en escala log)
        y_pred_log = self.model.predict(X_pred)
        
        # Convertir a escala original
        y_pred = np.expm1(y_pred_log)[0]
        
        return y_pred
    
    def obtener_importancia_caracteristicas(self, nombres_caracteristicas):
        """Obtiene la importancia de las características"""
        if self.model is None:
            return []
        
        importancia = self.model.feature_importances_
        importancia_caracteristicas = [
            {'feature': nombre, 'importance': float(imp)}
            for nombre, imp in zip(nombres_caracteristicas, importancia)
        ]
        # Ordenar por importancia
        importancia_caracteristicas.sort(key=lambda x: x['importance'], reverse=True)
        
        return importancia_caracteristicas
