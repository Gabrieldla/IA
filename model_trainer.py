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
        
    def train_model(self, X, y):
        """Entrena el modelo con optimización"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.training_output.append({
            'step': 'split',
            'name': 'División de datos',
            'output': f'Train: {len(X_train)} muestras | Test: {len(X_test)} muestras (80/20 split)'
        })
        
        self.training_output.append({
            'step': 'optimization_start',
            'name': 'Búsqueda de hiperparámetros',
            'output': 'RandomizedSearchCV iniciado con 20 combinaciones y 5-fold cross-validation'
        })
        self.model = self._optimize_hyperparameters(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test)
        self.metrics = self._evaluate_model(X_test, y_test)
        
        return self.model, self.metrics, X_test, y_test, y_pred
    
    def _optimize_hyperparameters(self, X_train, y_train):
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
        
        busqueda_aleatoria.fit(X_train, y_train)
        
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
    
    def _evaluate_model(self, X_test, y_test):
        """Evalúa el modelo y calcula métricas"""
        # Predecir
        y_pred = self.model.predict(X_test)
        
        # Convertir a escala original
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        
        metrics = {
            'r2': r2_score(y_test_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig)
        }
        
        # Guardar output de evaluación
        self.training_output.append({
            'step': 'evaluation',
            'name': 'Evaluación del modelo',
            'output': f'''Métricas en conjunto de prueba:
  • R² Score: {metrics['r2']:.4f} ({metrics['r2']*100:.2f}% de varianza explicada)
  • RMSE: ${metrics['rmse']:,.0f} (error cuadrático medio)
  • MAE: ${metrics['mae']:,.0f} (error absoluto promedio)'''
        })
        
        return metrics
    
    def predict(self, features_dict):
        """Hace una predicción con el modelo"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Entrenar primero o cargar un modelo guardado.")
        
        # Convertir dict a DataFrame para predicción
        import pandas as pd
        X_pred = pd.DataFrame([features_dict])
        
        # Predecir (en escala log)
        y_pred_log = self.model.predict(X_pred)
        
        # Convertir a escala original
        y_pred = np.expm1(y_pred_log)[0]
        
        return y_pred
    
    def get_feature_importance(self, feature_names):
        """Obtiene la importancia de las características"""
        if self.model is None:
            return []
        
        importance = self.model.feature_importances_
        feature_importance = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(feature_names, importance)
        ]
        # Ordenar por importancia
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance
